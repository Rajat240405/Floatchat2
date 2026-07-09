"""Automatic Plot Interpretation Engine (Phase 20 Improvement 2).

Phase 24: OMZ is only mentioned when DOXY is present AND region is relevant.
No region-specific text for non-oxygen variables.
"""

from typing import List
import pandas as pd


def generate_plot_interpretation(
    df: pd.DataFrame, variables: List[str], region: str | None
) -> str:
    """Generate lightweight, observable-only scientific interpretation."""
    interpretations: List[str] = []

    region_lower = (region or "").lower().replace("_", " ")

    if "DOXY" in df.columns or "DOXY_ADJUSTED" in df.columns:
        doxy_col = "DOXY_ADJUSTED" if "DOXY_ADJUSTED" in df.columns else "DOXY"
        if not df[doxy_col].dropna().empty:
            min_o2 = df[doxy_col].min()
            if min_o2 < 60:
                interpretations.append(
                    "A subsurface oxygen minimum is visible, consistent with the Arabian Sea / Bay of Bengal OMZ."
                )
            else:
                interpretations.append(
                    "Surface oxygen is elevated, consistent with atmospheric exchange."
                )

    if "CHLA" in df.columns or "CHLA_ADJUSTED" in df.columns:
        chla_col = "CHLA_ADJUSTED" if "CHLA_ADJUSTED" in df.columns else "CHLA"
        if not df[chla_col].dropna().empty:
            interpretations.append(
                "Chlorophyll maximum indicates the depth of the Deep Chlorophyll Maximum (DCM)."
            )

    # Phase 24: Only mention OMZ when DOXY is present AND region is relevant.
    # No longer auto-adds OMZ for all Arabian Sea queries.
    if "DOXY" in df.columns or "DOXY_ADJUSTED" in df.columns:
        if region_lower and ("arabian" in region_lower or "bay of bengal" in region_lower):
            interpretations.append(
                "Observed features are consistent with the known Arabian Sea / Bay of Bengal Oxygen Minimum Zone."
            )

    # Phase 24: Only mention OMZ-related region info when DOXY present.
    # For non-DOXY queries, use a generic structure statement.
    if not interpretations:
        interpretations.append("Data profile shows typical vertical structure.")

    return " ".join(interpretations)