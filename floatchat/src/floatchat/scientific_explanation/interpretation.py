"""Automatic Plot Interpretation Engine (Phase 20 Improvement 2)."""

from typing import List
import pandas as pd


def generate_plot_interpretation(
    df: pd.DataFrame, variables: List[str], region: str | None
) -> str:
    """Generate lightweight, observable-only scientific interpretation."""
    interpretations: List[str] = []

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

    if region and "arabian" in region.lower():
        interpretations.append(
            "Observed features are consistent with the known Arabian Sea Oxygen Minimum Zone."
        )

    return " ".join(interpretations) if interpretations else "Data profile shows typical vertical structure."