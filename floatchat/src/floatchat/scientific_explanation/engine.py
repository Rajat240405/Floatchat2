"""Scientific Explanation Engine.

Generates rich, context-aware scientific explanations for every successful query.
Uses Argo knowledge base facts and runtime data. Never hallucinates.

Phase 24: Explanations are concise -- each scientific concept appears at most once.
OMZ is only mentioned when DOXY is requested for Arabian Sea / Bay of Bengal.
"""

from typing import Any, Dict, List, Optional

from ..models.intent import ParsedIntent
from ..models.metadata import MetadataRecord
from .reasoning import get_scientific_reasoning


class ScientificExplanationEngine:
    """Generates scientific explanations for query results.

    Designed to be called by QueryEngine after successful data retrieval.
    """

    def __init__(self):
        self.kb = {
            "DOXY": "Dissolved oxygen (DOXY) is measured by optodes. Adjusted values (DOXY_ADJUSTED) are preferred for scientific use.",
            "CHLA": "Chlorophyll-a (CHLA) indicates phytoplankton biomass. Deep Chlorophyll Maximum (DCM) at 50-150 m is common in stratified waters.",
            "TEMP": "Temperature controls density, stratification, oxygen solubility and metabolic rates.",
            "PSAL": "Salinity indicates evaporation-precipitation balance. High surface salinity in Arabian Sea due to evaporation dominance.",
            "QC": "QC flag 1 = good; 2 = probably good; 3 = bad but correctable; 4 = bad. Always prefer adjusted variables in delayed-mode (D) files.",
            "DELAYED_MODE": "Delayed-mode (D) data has expert QC and adjustments. Real-time (R) data is preliminary and may contain sensor drift.",
            "OMZ": "Arabian Sea and Bay of Bengal naturally contain Oxygen Minimum Zones (OMZs) between ~100-1000 m due to high respiration and limited ventilation.",
        }

    def generate_explanation(
        self,
        intent: ParsedIntent,
        records: List[MetadataRecord],
        variables: List[str],
        data_summary: Dict[str, Any],
    ) -> str:
        """Generate a concise scientific explanation (Phase 24).

        Each scientific concept appears at most once. OMZ is only mentioned
        when DOXY is requested for a relevant region.
        """
        parts: List[str] = []
        seen: set[str] = set()

        def _add_once(key: str, text: str):
            if key not in seen:
                seen.add(key)
                parts.append(text)

        # 1. Selection summary
        if records:
            _add_once("selection",
                f"Selected {len(records)} profile(s) from the Argo GDAC index "
                f"based on region, time, and sensor availability."
            )

        # 2. Variable-specific science (once per concept)
        vars_upper = {v.upper() for v in variables}
        region_lower = (intent.region or "").lower().replace("_", " ")

        if "TEMP" in vars_upper or "TEMP_ADJUSTED" in vars_upper:
            _add_once("temp", self.kb["TEMP"])

        if "DOXY" in vars_upper or "DOXY_ADJUSTED" in vars_upper:
            _add_once("doxy", self.kb["DOXY"])
            # OMZ only when DOXY requested AND region is Arabian Sea / Bay of Bengal
            if "arabian sea" in region_lower or "bay of bengal" in region_lower:
                _add_once("omz", self.kb["OMZ"])

        if "CHLA" in vars_upper or "CHLA_ADJUSTED" in vars_upper:
            _add_once("chla", self.kb["CHLA"])

        if "PSAL" in vars_upper or "PSAL_ADJUSTED" in vars_upper:
            _add_once("psal", self.kb["PSAL"])

        # 3. Data mode / QC -- once, only when relevant
        has_real_time = any(
            "R" in (r.parameter_data_mode or "") for r in records
        )
        if has_real_time:
            _add_once("dm", self.kb["DELAYED_MODE"])
        _add_once("qc", self.kb["QC"])

        # 4. Scientific reasoning (from reasoning module)
        reasoning = get_scientific_reasoning(
            intent.region, variables, data_summary.get("data_mode")
        )
        for reason in reasoning:
            _add_once(f"reason_{reason[:30]}", reason)

        return " ".join(parts)