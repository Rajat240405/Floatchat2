"""FastAPI route definitions."""

import json
import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from floatchat.api.dependencies import (
    get_conversation_manager,
    get_intent_parser,
    get_llm_service,
    get_query_classifier,
    get_query_engine,
)
from floatchat.conversation.base import AbstractConversationManager
from floatchat.exceptions import FloatChatError, IntentParseError
from floatchat.intent_parser.base import AbstractIntentParser
from floatchat.llm_service.base import AbstractLLMService
from floatchat.llm_service.classifier import QueryClassifier
from floatchat.models import ChatResponse, ErrorResponse, ParsedIntent
from floatchat.query_engine.engine import QueryEngine

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    """Incoming POST /chat body."""

    message: str = Field(..., min_length=1, description="Natural language query.")
    session_id: str | None = Field(
        default=None,
        description="Client-generated session ID for conversational continuity.",
    )


def _build_context_hint(
    conversation_manager: AbstractConversationManager,
    session_id: str | None,
) -> str:
    """Build a context hint string for GENERAL_QUERY LLM prompts."""
    if not session_id:
        return ""

    ctx = conversation_manager.get_context(session_id)
    if not ctx:
        return ""

    parts: list[str] = []
    if ctx.last_variables:
        parts.append(f"variables: {', '.join(ctx.last_variables)}")
    if ctx.last_region:
        parts.append(f"region: {ctx.last_region.replace('_', ' ')}")
    if ctx.last_float_id:
        parts.append(f"float: {ctx.last_float_id}")
    if ctx.last_year:
        parts.append(f"year: {ctx.last_year}")

    if not parts:
        return ""

    return (
        "\n\n[Conversation context — the user was previously looking at "
        f"{', '.join(parts)}]"
    )


def _build_full_context_prompt(
    conversation_manager: AbstractConversationManager,
    session_id: str | None,
    message: str,
) -> str:
    """Build a rich context prompt for GENERAL_QUERY explanations."""
    if not session_id:
        return message

    ctx = conversation_manager.get_context(session_id)
    if not ctx:
        return message

    lines = [message, "", "--- Conversation Context ---"]
    if ctx.last_variables:
        lines.append(f"Variable(s): {', '.join(ctx.last_variables)}")
    if ctx.last_region:
        lines.append(f"Region: {ctx.last_region.replace('_', ' ')}")
    if ctx.last_float_id:
        lines.append(f"Float: {ctx.last_float_id}")
    if ctx.last_year:
        lines.append(f"Year: {ctx.last_year}")
    if ctx.last_profile_number:
        lines.append(f"Profile: {ctx.last_profile_number}")
    if ctx.last_intent:
        lines.append(f"Intent: {ctx.last_intent}")
    if ctx.last_message:
        lines.append(f"Previous result: {ctx.last_message}")
    if ctx.last_response_summary:
        summary = ctx.last_response_summary
        if summary.get("matched_records"):
            lines.append(f"Profiles retrieved: {summary['matched_records']}")
        if summary.get("total_measurements"):
            lines.append(f"Total measurements: {summary['total_measurements']}")

    return "\n".join(lines)


def _try_conversational_recovery(
    message: str,
    session_id: str | None,
    conversation_manager: AbstractConversationManager,
    intent_parser: AbstractIntentParser,
) -> ParsedIntent | None:
    """Attempt to recover from a parse failure using conversation context.

    Returns a merged ParsedIntent if recovery succeeds, otherwise None.
    """
    if not session_id:
        logger.debug("No session_id; skipping conversational recovery")
        return None

    ctx = conversation_manager.get_context(session_id)
    if not ctx:
        logger.debug("No context for session %s; skipping recovery", session_id)
        return None

    if ctx.turn_count >= getattr(conversation_manager, "_max_turns", 10):
        logger.debug("Session %s context expired; skipping recovery", session_id)
        return None

    # Create a minimal intent — the merge will fill gaps from context
    try:
        minimal = intent_parser.parse(message)
    except IntentParseError:
        # Even the parser failed — build the most minimal intent possible
        # and let context fill everything
        minimal = ParsedIntent(intent="profile_plot")

    merged = conversation_manager.merge_context(session_id, minimal)
    logger.info(
        "Conversational recovery for session %s: original_vars=%s merged_vars=%s "
        "merged_region=%s merged_float=%s",
        session_id,
        minimal.variables,
        merged.variables,
        merged.region,
        merged.float_id,
    )

    # Recovery succeeds if we now have variables or a float_id
    if merged.variables or merged.float_id:
        return merged

    return None


def _build_suggestion_message(ctx) -> str:
    """Build a helpful suggestion message when parsing fails."""
    parts = ["I couldn't determine which variable or float you meant."]

    if ctx and (ctx.last_variables or ctx.last_region or ctx.last_float_id):
        parts.append("\nPrevious query:")
        if ctx.last_variables:
            parts.append(f"  Variable: {', '.join(ctx.last_variables)}")
        if ctx.last_region:
            parts.append(f"  Region: {ctx.last_region.replace('_', ' ')}")
        if ctx.last_float_id:
            parts.append(f"  Float: {ctx.last_float_id}")
        if ctx.last_year:
            parts.append(f"  Year: {ctx.last_year}")

    parts.append("\nDid you mean:")
    suggestions = ["oxygen", "chlorophyll", "nitrate", "temperature", "salinity"]
    for s in suggestions:
        parts.append(f"  • {s}")
    parts.append("\nOr try: 'same float', 'same region', 'actually chlorophyll'")

    return "\n".join(parts)


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    classifier: Annotated[QueryClassifier, Depends(get_query_classifier)],
    llm_service: Annotated[AbstractLLMService, Depends(get_llm_service)],
    intent_parser: Annotated[AbstractIntentParser, Depends(get_intent_parser)],
    query_engine: Annotated[QueryEngine, Depends(get_query_engine)],
    conversation_manager: Annotated[
        AbstractConversationManager, Depends(get_conversation_manager)
    ],
) -> ChatResponse:
    """Convert a natural-language message into a data visualization.

    Flow:
        1. Classify query as GENERAL_QUERY or DATA_QUERY via LLM.
        2. GENERAL_QUERY → LLM generates a direct answer (no data pipeline).
        3. DATA_QUERY    → intent parser → merge context → query engine → viz.
        4. Parse failure → conversational recovery → merge context → retry.
    """
    request_t0 = time.perf_counter()
    logger.info(
        "POST /chat received: %r session_id=%s",
        request.message,
        request.session_id,
    )

    try:
        # --- Step 1: Classify ------------------------------------------- #
        classify_t0 = time.perf_counter()
        query_type = QueryClassifier.classify(classifier, request.message)
        classify_t1 = time.perf_counter()
        logger.info(
            "Query classified as %s in %.3fs", query_type, classify_t1 - classify_t0
        )

        # --- Step 1.5: Conversational Override -------------------------- #
        # If the classifier says GENERAL_QUERY but it looks like a data follow-up,
        # override to DATA_QUERY to trigger the retrieval pipeline.
        if query_type == "GENERAL_QUERY":
            # We use the parser to check if it's a conversational data request
            try:
                # Use a temporary parser to check for follow-up markers
                # without committing to the final parsed intent yet.
                if intent_parser._is_conversational_follow_up(request.message.lower()):
                    logger.info("Overriding GENERAL_QUERY to DATA_QUERY due to follow-up pattern")
                    query_type = "DATA_QUERY"
            except Exception:
                pass

        # --- Step 2: GENERAL_QUERY — LLM answer only -------------------- #
        if query_type == "GENERAL_QUERY":
            gen_t0 = time.perf_counter()
            augmented_prompt = _build_full_context_prompt(
                conversation_manager, request.session_id, request.message
            )
            answer = llm_service.generate(augmented_prompt)
            gen_t1 = time.perf_counter()
            logger.info("LLM general answer generated in %.3fs", gen_t1 - gen_t0)

            response = ChatResponse(
                intent="general_chat",
                message=answer,
                figure=None,
                data_summary={},
                map_data=[],
            )
            # Preserve data context from previous turns by storing a
            # minimal intent (general_chat has no variables/region/etc.).
            conversation_manager.update_context(
                request.session_id,
                ParsedIntent(intent="general_chat"),
                response,
            )
            _log_response(response, request_t0)
            return response

        # --- Step 3: DATA_QUERY — parse + merge context + pipeline ------ #
        try:
            parsed = intent_parser.parse(request.message)
        except IntentParseError as exc:
            logger.warning(
                "Initial parse failed for %r: %s. Attempting conversational recovery.",
                request.message,
                exc.message,
            )
            recovered = _try_conversational_recovery(
                request.message,
                request.session_id,
                conversation_manager,
                intent_parser,
            )
            if recovered is not None:
                logger.info(
                    "Conversational recovery succeeded: vars=%s region=%s float=%s",
                    recovered.variables,
                    recovered.region,
                    recovered.float_id,
                )
                parsed = recovered
            else:
                # Graceful error with suggestions
                ctx = (
                    conversation_manager.get_context(request.session_id)
                    if request.session_id
                    else None
                )
                suggestion = _build_suggestion_message(ctx)
                logger.info("Conversational recovery failed; returning suggestions")
                return ChatResponse(
                    intent="unknown",
                    message=suggestion,
                    figure=None,
                    data_summary={},
                    map_data=[],
                )

        intent = conversation_manager.merge_context(request.session_id, parsed)
        logger.info(
            "Merged intent for execution: intent=%s vars=%s region=%s year=%s float=%s profile=%s",
            intent.intent,
            intent.variables,
            intent.region,
            intent.year,
            intent.float_id,
            intent.profile_number,
        )
        response = query_engine.execute(intent)
        conversation_manager.update_context(request.session_id, intent, response)
        _log_response(response, request_t0)
        return response

    except FloatChatError:
        raise
    except Exception as exc:
        logger.exception("Unhandled exception in /chat: %s", exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred. Please try again later.",
                details={},
            ).model_dump(),
        )


def _log_response(response: ChatResponse, request_t0: float) -> None:
    """Log response size, serialization time, and total request time."""
    serialize_t0 = time.perf_counter()
    try:
        json_bytes = json.dumps(response.model_dump(mode="json")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        logger.error("JSON serialization failed: %s", exc)
        return
    serialize_t1 = time.perf_counter()

    total_time = time.perf_counter() - request_t0
    logger.info(
        "Response ready: size=%.2f KB, serialize=%.3fs, total=%.3fs, "
        "intent=%s, map_markers=%d, has_figure=%s",
        len(json_bytes) / 1024,
        serialize_t1 - serialize_t0,
        total_time,
        response.intent,
        len(response.map_data),
        response.figure is not None,
    )
