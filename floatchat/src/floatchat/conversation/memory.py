"""In-memory conversation context manager."""

import logging
from datetime import datetime, timezone

from floatchat.config import settings
from floatchat.conversation.base import AbstractConversationManager
from floatchat.models import ChatResponse, ConversationContext, ParsedIntent

logger = logging.getLogger(__name__)


class InMemoryConversationManager(AbstractConversationManager):
    """Thread-safe(ish) in-memory store for conversation contexts.

    Context entries expire after ``max_turns`` turns (default from
    ``settings.conversation_max_turns``).
    """

    def __init__(self, max_turns: int | None = None) -> None:
        self._max_turns = max_turns or settings.conversation_max_turns
        self._store: dict[str, ConversationContext] = {}

    def get_context(self, session_id: str) -> ConversationContext | None:
        return self._store.get(session_id)

    def merge_context(
        self, session_id: str | None, intent: ParsedIntent
    ) -> ParsedIntent:
        if not session_id:
            return intent

        ctx = self._store.get(session_id)
        if not ctx:
            return intent

        if ctx.turn_count >= self._max_turns:
            logger.debug(
                "Session %s context expired (%d >= %d turns)",
                session_id,
                ctx.turn_count,
                self._max_turns,
            )
            return intent

        merged_data = intent.model_dump()

        # Identity filters (float_id, profile_number/cycle) are NOT
        # conversational context; they narrow the search to a specific
        # observation. Inheriting them into a new turn that isn't explicitly
        # asking about the same identity silently converts a region query into
        # a float/cycle query and typically returns 0 records — the
        # intermittent "No Argo profiles matched" bug. Two guards:
        #
        #   Guard A — a region-scoped follow-up (named region or lat/lon bbox)
        #     never inherits float_id or profile_number. Example:
        #       Turn N   : "float 7902250 oxygen"     -> ctx.last_float_id set
        #       Turn N+1 : "oxygen in Arabian Sea"    -> would inherit float_id
        #
        #   Guard B — profile_number is never inherited unless the merged
        #     intent also has a float_id. A cycle number without a float is
        #     meaningless: the metadata filter matches file basenames like
        #     "_052.nc" across ALL floats, which almost always yields 0.
        _has_region_scope = (
            merged_data.get("region") is not None
            or merged_data.get("lat_min") is not None
            or merged_data.get("lat_max") is not None
            or merged_data.get("lon_min") is not None
            or merged_data.get("lon_max") is not None
        )

        # Fill missing fields from context
        if not merged_data.get("variables") and ctx.last_variables:
            merged_data["variables"] = ctx.last_variables.copy()
        if merged_data.get("region") is None and ctx.last_region is not None:
            merged_data["region"] = ctx.last_region
        if (
            merged_data.get("float_id") is None
            and ctx.last_float_id is not None
            and not _has_region_scope
        ):
            merged_data["float_id"] = ctx.last_float_id
        if merged_data.get("year") is None and ctx.last_year is not None:
            merged_data["year"] = ctx.last_year
        if (
            merged_data.get("profile_number") is None
            and ctx.last_profile_number is not None
            and not _has_region_scope
            and merged_data.get("float_id") is not None
        ):
            merged_data["profile_number"] = ctx.last_profile_number

        merged = ParsedIntent(**merged_data)
        logger.info(
            "Merged context for session %s: vars=%s region=%s float=%s year=%s profile=%s",
            session_id,
            merged.variables,
            merged.region,
            merged.float_id,
            merged.year,
            merged.profile_number,
        )
        return merged

    def update_context(
        self,
        session_id: str | None,
        intent: ParsedIntent,
        response: ChatResponse,
    ) -> None:
        if not session_id:
            return

        ctx = self._store.get(session_id)
        if ctx is None:
            ctx = ConversationContext(session_id=session_id)

        ctx.turn_count += 1
        ctx.last_intent = intent.intent
        if intent.float_id is not None:
            ctx.last_float_id = intent.float_id
        if intent.variables:
            ctx.last_variables = intent.variables.copy()
        if intent.region is not None:
            ctx.last_region = intent.region
        if intent.year is not None:
            ctx.last_year = intent.year
        if intent.profile_number is not None:
            ctx.last_profile_number = intent.profile_number
        ctx.last_message = response.message
        ctx.last_response_summary = response.data_summary
        ctx.updated_at = datetime.now(timezone.utc)

        self._store[session_id] = ctx
        logger.debug(
            "Updated context for session %s (turn %d)",
            session_id,
            ctx.turn_count,
        )

    def clear_context(self, session_id: str) -> None:
        self._store.pop(session_id, None)
