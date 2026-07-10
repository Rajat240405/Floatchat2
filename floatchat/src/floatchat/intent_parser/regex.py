"""Regex-based intent parser.

Extracts structured intent from natural language using compiled regular
expressions and synonym tables. No external LLM required.

Supports a wide variety of natural-language query patterns including
variable synonyms, optional regions, years, and float IDs.
"""

import logging
import re

from floatchat.exceptions import IntentParseError
from floatchat.intent_parser.base import AbstractIntentParser
from floatchat.intent_parser.fuzzy import correct_variables_with_fuzzy
from floatchat.models import ParsedIntent
from floatchat.query_normalizer import (
    AbstractQueryNormalizer,
    FallbackQueryNormalizer,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Variable synonyms
# --------------------------------------------------------------------------- #
# Maps canonical Argo names → list of natural-language synonyms.
# Synonyms are matched with word boundaries where appropriate.
_VARIABLE_SYNONYMS: dict[str, list[str]] = {
    "DOXY": [
        "oxygen",
        "dissolved oxygen",
        "doxy",
        "o2",
    ],
    "CHLA": [
        "chlorophyll",
        "chlorophyll-a",
        "chla",
        "chlorophyll a",
    ],
    "BBP700": [
        "backscattering",
        "bbp700",
        "particle backscattering",
        "backscatter",
    ],
    "NITRATE": [
        "nitrate",
        "no3",
    ],
    "PH_IN_SITU_TOTAL": [
        "ph",
        "acidity",
        "ph in situ total",
    ],
    "DOWNWELLING_PAR": [
        "par",
        "photosynthetically active radiation",
        "downwelling par",
    ],
    "DOWN_IRRADIANCE380": [
        "irradiance 380",
        "down irradiance 380",
    ],
    "DOWN_IRRADIANCE412": [
        "irradiance 412",
        "down irradiance 412",
    ],
    "DOWN_IRRADIANCE490": [
        "irradiance 490",
        "down irradiance 490",
    ],
    "TEMP": [
        "temperature",
        "temp",
    ],
    "PSAL": [
        "salinity",
        "psal",
    ],
}

# Build regex patterns for each canonical variable.
# Longer synonyms are checked first to avoid partial matches.
_VAR_PATTERNS: list[tuple[str, re.Pattern]] = []
for canonical, synonyms in _VARIABLE_SYNONYMS.items():
    # Sort by length descending so "dissolved oxygen" matches before "oxygen"
    sorted_syns = sorted(synonyms + [canonical.lower()], key=len, reverse=True)
    escaped = [re.escape(s) for s in sorted_syns]
    pattern = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)
    _VAR_PATTERNS.append((canonical, pattern))

# --------------------------------------------------------------------------- #
# Region synonyms
# --------------------------------------------------------------------------- #
_REGION_SYNONYMS: dict[str, list[str]] = {
    "arabian_sea": ["arabian sea"],
    "bay_of_bengal": ["bay of bengal"],
    "north_atlantic": ["north atlantic"],
    "south_atlantic": ["south atlantic"],
    "north_pacific": ["north pacific"],
    "south_pacific": ["south pacific"],
    "indian_ocean": ["indian ocean"],
    "southern_ocean": ["southern ocean"],
    "mediterranean_sea": ["mediterranean", "mediterranean sea"],
    "red_sea": ["red sea"],
    "gulf_of_mexico": ["gulf of mexico"],
    "tasman_sea": ["tasman sea"],
    "caribbean_sea": ["caribbean sea"],
}

# Build regex patterns for regions (phrases with spaces need special handling).
_REGION_PATTERNS: list[tuple[str, re.Pattern]] = []
for canonical, synonyms in _REGION_SYNONYMS.items():
    all_names = sorted(synonyms + [canonical.replace("_", " ")], key=len, reverse=True)
    escaped = [re.escape(s) for s in all_names]
    pattern = re.compile(r"(?:" + "|".join(escaped) + r")", re.IGNORECASE)
    _REGION_PATTERNS.append((canonical, pattern))

# --------------------------------------------------------------------------- #
# Intent detection patterns
# --------------------------------------------------------------------------- #
_INTENT_PROFILE = re.compile(
    r"\b(profile|plot|show|graph|display|visuali[sz]e|get|fetch)\b", re.IGNORECASE
)
_INTENT_TS = re.compile(r"\b(time.?series|trend|over time|temporal|since|from)\b", re.IGNORECASE)
_INTENT_TRAJ = re.compile(r"\b(traject|path|track|route|drift)\b", re.IGNORECASE)
_INTENT_COMP = re.compile(r"\b(compar|vs\.?|versus|difference|diff|against)\b", re.IGNORECASE)

# --------------------------------------------------------------------------- #
# Conversational follow-up patterns
# --------------------------------------------------------------------------- #
# These match natural follow-up phrases that may not contain explicit
# variables or regions.  They signal intent type so context can fill gaps.
_CONVERSATIONAL_VARIABLE = re.compile(
    r"\b(actually|instead|now|what about|how about|and)\b.*?(oxygen|dissolved oxygen|doxy|o2|"
    r"chlorophyll|chlorophyll-a|chla|chlorophyll a|"
    r"backscattering|bbp700|particle backscattering|backscatter|"
    r"nitrate|no3|ph|acidity|ph in situ total|"
    r"par|photosynthetically active radiation|downwelling par|"
    r"irradiance 380|down irradiance 380|irradiance 412|down irradiance 412|"
    r"irradiance 490|down irradiance 490|"
    r"temperature|temp|salinity|psal|explain)",
    re.IGNORECASE,
)
_CONVERSATIONAL_COMPARISON = re.compile(
    r"\b(compar(?:e|ing)?\s+(?:with|against)|vs\.?|versus)\b", re.IGNORECASE
)
_CONVERSATIONAL_REGION = re.compile(
    r"\b(now|instead)\s+in\b", re.IGNORECASE
)
_CONVERSATIONAL_SAME_FLOAT = re.compile(
    r"\b(same float|that float|this float)\b", re.IGNORECASE
)
_CONVERSATIONAL_SAME_REGION = re.compile(
    r"\b(same region|same area|same place)\b", re.IGNORECASE
)
_CONVERSATIONAL_SAME_VARIABLE = re.compile(
    r"\b(same variable|same thing)\b", re.IGNORECASE
)
_CONVERSATIONAL_LATEST_PROFILE = re.compile(
    r"\b(latest|most recent|last)\s+(?:profile|cycle)\b", re.IGNORECASE
)
_CONVERSATIONAL_PREVIOUS_PROFILE = re.compile(
    r"\b(previous|earlier|last)\s+(?:profile|cycle)\b", re.IGNORECASE
)
_CONVERSATIONAL_NEXT_PROFILE = re.compile(
    r"\b(next|following)\s+(?:profile|cycle)\b", re.IGNORECASE
)
_CONVERSATIONAL_FOR_YEAR = re.compile(
    r"\bfor\s+(19|20)\d{2}\b", re.IGNORECASE
)

# --------------------------------------------------------------------------- #
# Value extraction patterns
# --------------------------------------------------------------------------- #
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_FLOAT_RE = re.compile(r"\bfloat\s+(\d{7,})\b", re.IGNORECASE)
_WMO_RE = re.compile(r"\bWMO\s+(\d{7,})\b", re.IGNORECASE)
_PROFILE_NUMBER_RE = re.compile(
    r"\b(?:profile|cycle)\s*#?\s*(\d{1,3})\b", re.IGNORECASE
)


class RegexIntentParser(AbstractIntentParser):
    """Deterministic parser using regular expressions and synonym tables."""

    def __init__(self, normalizer: AbstractQueryNormalizer | None = None) -> None:
        # Phase 20.2: Normalization is opt-in to preserve backward compatibility
        self.normalizer = normalizer  # can be None or explicit instance

    def parse(self, message: str) -> ParsedIntent:
        """Parse *message* via regex heuristics."""
        logger.debug("RegexIntentParser processing: %r", message)

        # Phase 20.2: Query Normalization stage (before any parsing)
        text = message.lower()
        if self.normalizer is not None:
            normalized = self.normalizer.normalize(message)
            if normalized != message:
                logger.info("Original query: %r", message)
                logger.info("Normalized query: %r", normalized)
            text = normalized.lower()

        intent = self._detect_intent(text)
        variables = correct_variables_with_fuzzy(self._extract_variables(text))
        region = self._extract_region(text)
        year = self._extract_year(text)
        float_id = self._extract_float_id(text)
        profile_number = self._extract_profile_number(text)

        # Check for conversational follow-up indicators that signal intent
        # even when variables/float are absent (context will fill gaps).
        is_conversational = self._is_conversational_follow_up(text)

        if not variables and not float_id and not is_conversational:
            logger.warning("Regex parser could not extract variables or float_id")
            raise IntentParseError(
                "Could not determine requested variables or float from message.",
                details={"message": message},
            )

        parsed = ParsedIntent(
            intent=intent,
            region=region,
            variables=variables,
            year=year,
            float_id=float_id,
            profile_number=profile_number,
            limit=5,
        )
        logger.info(
            "RegexIntentParser resolved intent=%s vars=%s region=%s year=%s float=%s profile=%s conversational=%s",
            intent,
            variables,
            region,
            year,
            float_id,
            profile_number,
            is_conversational,
        )
        return parsed

    @staticmethod
    def _is_conversational_follow_up(text: str) -> bool:
        """Return True if *text* is a conversational follow-up phrase."""
        patterns = [
            _CONVERSATIONAL_VARIABLE,
            _CONVERSATIONAL_COMPARISON,
            _CONVERSATIONAL_REGION,
            _CONVERSATIONAL_SAME_FLOAT,
            _CONVERSATIONAL_SAME_REGION,
            _CONVERSATIONAL_SAME_VARIABLE,
            _CONVERSATIONAL_LATEST_PROFILE,
            _CONVERSATIONAL_PREVIOUS_PROFILE,
            _CONVERSATIONAL_NEXT_PROFILE,
            _CONVERSATIONAL_FOR_YEAR,
        ]
        return any(p.search(text) is not None for p in patterns)

    @staticmethod
    def _detect_intent(text: str) -> str:
        if _CONVERSATIONAL_COMPARISON.search(text) or _INTENT_COMP.search(text):
            return "comparison_plot"
        if _INTENT_TS.search(text):
            return "time_series"
        if _INTENT_TRAJ.search(text):
            return "trajectory"
        if _INTENT_PROFILE.search(text):
            return "profile_plot"
        # Default fallback if variables are present
        return "profile_plot"

    @staticmethod
    def _extract_variables(text: str) -> list[str]:
        found: set[str] = set()
        for canonical, pattern in _VAR_PATTERNS:
            if pattern.search(text):
                found.add(canonical)
        return sorted(found)

    @staticmethod
    def _extract_region(text: str) -> str | None:
        for canonical, pattern in _REGION_PATTERNS:
            if pattern.search(text):
                return canonical
        return None

    @staticmethod
    def _extract_year(text: str) -> int | None:
        # For "between 2022 and 2024" or "from 2022 to 2024", use the first year
        # as the primary filter. The downstream engine can expand if needed.
        match = _YEAR_RE.search(text)
        return int(match.group(0)) if match else None

    @staticmethod
    def _extract_float_id(text: str) -> str | None:
        for pattern in (_FLOAT_RE, _WMO_RE):
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _extract_profile_number(text: str) -> int | None:
        match = _PROFILE_NUMBER_RE.search(text)
        return int(match.group(1)) if match else None
