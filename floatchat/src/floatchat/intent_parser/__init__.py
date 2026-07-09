"""Intent Parser implementations.

The backend consumes :class:`ParsedIntent` objects and never knows which
implementation produced them.
"""

from floatchat.intent_parser.base import AbstractIntentParser
from floatchat.intent_parser.mock import MockIntentParser
from floatchat.intent_parser.ollama import OllamaIntentParser
from floatchat.intent_parser.regex import RegexIntentParser
from floatchat.query_normalizer import (
    AbstractQueryNormalizer,
    OllamaQueryNormalizer,
    FallbackQueryNormalizer,
)

__all__ = [
    "AbstractIntentParser",
    "MockIntentParser",
    "RegexIntentParser",
    "OllamaIntentParser",
    "AbstractQueryNormalizer",
    "OllamaQueryNormalizer",
    "FallbackQueryNormalizer",
]
