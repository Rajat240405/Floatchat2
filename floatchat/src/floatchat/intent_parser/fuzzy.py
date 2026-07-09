"""Fuzzy / typo recovery utilities (Improvement 4).

Provides lightweight Levenshtein-based correction for common variable typos.
Only suggests when confidence is high.
"""

import difflib
from typing import List, Optional


_VARIABLE_CANONICAL = [
    "TEMP",
    "PSAL",
    "DOXY",
    "CHLA",
    "NITRATE",
    "BBP700",
    "PH_IN_SITU_TOTAL",
    "DOWNWELLING_PAR",
]

# Extra common misspellings (high-confidence corrections)
_TYPO_MAP = {
    "TEMPARATURE": "TEMP",
    "TEMPERATURE": "TEMP",
    "CHLORPHYLL": "CHLA",
    "CHLOROPHYLL": "CHLA",
    "OXIGEN": "DOXY",
    "OXYGEN": "DOXY",
    "SALINTY": "PSAL",
    "SALINITY": "PSAL",
}


def correct_variable_typo(token: str, cutoff: float = 0.75) -> Optional[str]:
    """Return the closest canonical variable if similarity is high enough."""
    matches = difflib.get_close_matches(
        token.upper(), _VARIABLE_CANONICAL, n=1, cutoff=cutoff
    )
    return matches[0] if matches else None


def correct_variables_with_fuzzy(variables: List[str]) -> List[str]:
    """Apply fuzzy correction to a list of extracted variable tokens.

    First checks the high-confidence _TYPO_MAP, then falls back to
    difflib similarity. This ensures 'temparature' and 'chlorphyll'
    are corrected before conversational context is applied.
    """
    corrected = []
    for v in variables:
        upper = v.upper()
        if upper in _VARIABLE_CANONICAL:
            corrected.append(upper)
            continue

        # High-confidence typo map first
        if upper in _TYPO_MAP:
            corrected.append(_TYPO_MAP[upper])
            continue

        # Then fuzzy similarity
        suggestion = correct_variable_typo(v)
        if suggestion:
            corrected.append(suggestion)
        else:
            corrected.append(v)  # keep original
    return corrected