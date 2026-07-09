"""Scientific Explanation Engine.

Generates rich, context-aware scientific explanations for every successful query.
Uses Argo knowledge base facts and runtime data. Never hallucinates.
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
        # Lightweight in-memory knowledge (extracted from attached KB)
        self.kb = {
            "OMZ": "Arabian Sea and Bay of Bengal naturally contain Oxygen Minimum Zones (OMZs) between ~100-1000 m due to high respiration and limited ventilation.",
            "DOXY": "Dissolved oxygen (DOXY) is measured by optodes. Raw values require delayed-mode adjustment using in-air calibrations. Adjusted values (DOXY_ADJUSTED) are preferred for scientific use.",
            "CHLA": "Chlorophyll-a (CHLA) indicates phytoplankton biomass. Deep Chlorophyll Maximum (DCM) at 50-150 m is common in stratified waters.",
            "TEMP": "Temperature controls density, stratification, oxygen solubility and metabolic rates. Strong thermoclines exist in the tropics.",
            "PSAL": "Salinity indicates evaporation-precipitation balance. High surface salinity in Arabian Sea due to evaporation dominance.",
            "QC": "QC flag 1 = good; 2 = probably good; 3 = bad but correctable; 4 = bad. Always prefer adjusted variables in delayed-mode (D) files.",
            "DELAYED_MODE": "Delayed-mode (D) data has expert QC and adjustments. Real-time (R) data is preliminary and may contain sensor drift.",
        }

    def generate_explanation(
        self,
        intent: ParsedIntent,
        records: List[MetadataRecord],
        variables: List[str],
        data_summary: Dict[str, Any],
    ) -> str:
        """Generate a multi-audience scientific explanation."""
        parts: List[str] = []

        # 1. Why floats were selected
        if records:
            parts.append(
                f"Selected {len(records)} profile(s) from the Argo GDAC index based on region, time, and sensor availability."
            )
            if intent.region:
                parts.append(
                    f"Region '{intent.region}' was resolved using realistic ocean boundaries."
                )

        # 2. Variable choice & scientific meaning
        for var in variables:
            var_upper = var.upper()
            if var_upper in ["TEMP", "TEMP_ADJUSTED"]:
                parts.append(
                    "Temperature (TEMP) is a core Argo variable controlling density, stratification, and oxygen solubility."
                )
            elif var_upper in ["DOXY", "DOXY_ADJUSTED"]:
                parts.append(self.kb["DOXY"])
                parts.append(self.kb["OMZ"])
            elif var_upper in ["CHLA", "CHLA_ADJUSTED"]:
                parts.append(self.kb["CHLA"])
            elif var_upper in ["PSAL", "PSAL_ADJUSTED"]:
                parts.append(self.kb["PSAL"])

        # 3. Data mode / QC considerations
        parts.append(self.kb["QC"])
        parts.append(self.kb["DELAYED_MODE"])

        # 4. Expected vs observed (simple heuristics)
        if "DOXY" in [v.upper() for v in variables] and intent.region in [
            "arabian sea",
            "bay of bengal",
        ]:
            parts.append(
                "Deep oxygen minima are expected in this region due to the OMZ."
            )

        # 5. Caveats
        parts.append(
            "Important: Always verify QC flags and prefer ADJUSTED variables for research. "
            "Real-time data may contain uncorrected sensor drift."
        )

        # 6. Scientific reasoning (Improvement 2)
        reasoning = get_scientific_reasoning(
            intent.region, variables, data_summary.get("data_mode")
        )
        if reasoning:
            parts.extend(reasoning)

        return " ".join(parts)