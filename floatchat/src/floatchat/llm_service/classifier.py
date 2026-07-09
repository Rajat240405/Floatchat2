"""Query classifier.

Uses an LLM to classify user messages as either DATA_QUERY or GENERAL_QUERY.
Falls back to DATA_QUERY when the LLM is disabled or unreachable.
"""

import logging
from typing import Literal

from floatchat.config import settings
from floatchat.llm_service.base import AbstractLLMService

logger = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM = (
    "You are a query classifier. Your job is to classify the user's request.\n"
    "Output EXACTLY ONE token — nothing else:\n"
    "  DATA_QUERY     — if the user wants data, plots, profiles, visualizations, "
    "or specific float measurements. This includes short follow-ups like "
    "'Actually chlorophyll', 'Now temperature', 'Compare with 2023', "
    "'Now in Bay of Bengal', 'Same float', 'Latest profile'.\n"
    "  GENERAL_QUERY  — if the user asks a general question about oceanography, "
    "Argo, biogeochemistry, or wants an explanation of a previous result.\n"
    "Examples:\n"
    '  "What is Argo?" → GENERAL_QUERY\n'
    '  "Explain dissolved oxygen." → GENERAL_QUERY\n'
    '  "Show oxygen in Arabian Sea." → DATA_QUERY\n'
    '  "Plot nitrate for float 6903091." → DATA_QUERY\n'
    '  "What is chlorophyll?" → GENERAL_QUERY\n'
    '  "Temperature profiles for 2024." → DATA_QUERY\n'
    '  "Actually chlorophyll" → DATA_QUERY\n'
    '  "Now temperature" → DATA_QUERY\n'
    '  "Compare with 2023" → DATA_QUERY\n'
    '  "Now in Bay of Bengal" → DATA_QUERY\n'
    '  "Same float" → DATA_QUERY\n'
    '  "Latest profile" → DATA_QUERY\n'
    '  "Explain this graph" → GENERAL_QUERY\n'
    '  "Why is oxygen decreasing?" → GENERAL_QUERY\n'
)

QueryType = Literal["DATA_QUERY", "GENERAL_QUERY"]


class QueryClassifier:
    """Classify natural-language queries into DATA_QUERY or GENERAL_QUERY."""

    def __init__(self, llm_service: AbstractLLMService) -> None:
        self._llm = llm_service

    def classify(self, message: str) -> QueryType:
        """Classify *message* and return the query type.

        If the LLM is disabled or unreachable, falls back to DATA_QUERY so
        the existing data pipeline continues to work.
        """
        if not settings.llm_enabled:
            logger.debug("LLM disabled; defaulting to DATA_QUERY")
            return "DATA_QUERY"

        prompt = f'Classify this request:\n"{message}"\n\nOutput ONLY: DATA_QUERY or GENERAL_QUERY'

        try:
            raw = self._llm.generate(prompt, system=_CLASSIFIER_SYSTEM)
        except Exception:
            logger.exception("Classifier LLM call failed; falling back to DATA_QUERY")
            return "DATA_QUERY"

        cleaned = raw.strip().upper()
        logger.debug("Classifier raw output: %r → cleaned: %r", raw, cleaned)

        if "GENERAL_QUERY" in cleaned:
            return "GENERAL_QUERY"
        if "DATA_QUERY" in cleaned:
            return "DATA_QUERY"

        # Unexpected output — log and default to data pipeline
        logger.warning(
            "Classifier returned unexpected output %r; defaulting to DATA_QUERY",
            raw,
        )
        return "DATA_QUERY"
