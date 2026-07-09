"""Profile plot implementation.

Renders one or more BGC variables versus pressure (depth proxy).
Uses one trace per profile with per-point marker colors for QC,
avoiding the "one trace per point" anti-pattern.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from floatchat.exceptions import VisualizationError
from floatchat.models import ParsedIntent
from floatchat.visualization_engine.base import AbstractVisualizationEngine

logger = logging.getLogger(__name__)

# Argo variable name → axis title mapping.
_VAR_TITLES: dict[str, str] = {
    "DOXY": "Dissolved Oxygen (µmol kg⁻¹)",
    "CHLA": "Chlorophyll-A (mg m⁻³)",
    "BBP700": "Particle Backscattering 700 nm (m⁻¹)",
    "NITRATE": "Nitrate (µmol kg⁻¹)",
    "PH_IN_SITU_TOTAL": "pH (total scale)",
    "DOWNWELLING_PAR": "Downwelling PAR (µmol photons m⁻² s⁻¹)",
    "DOWN_IRRADIANCE380": "Irradiance 380 nm (W m⁻² nm⁻¹)",
    "DOWN_IRRADIANCE412": "Irradiance 412 nm (W m⁻² nm⁻¹)",
    "DOWN_IRRADIANCE490": "Irradiance 490 nm (W m⁻² nm⁻¹)",
    "TEMP": "Temperature (°C)",
    "PSAL": "Practical Salinity",
}

# Default colour cycle.
_COLOURS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]


def _qc_to_alpha(qc: str) -> float:
    """Map Argo QC flags to marker opacity.

    1 = good → 1.0
    2 = probably good → 0.8
    3 = probably bad → 0.4
    4 = bad → 0.2
    others → 0.5
    """
    mapping = {
        "1": 1.0,
        "2": 0.8,
        "3": 0.4,
        "4": 0.2,
    }
    return mapping.get(str(qc).strip(), 0.5)


def _hex_to_rgba(hex_colour: str, alpha: float) -> str:
    """Convert #RRGGBB + alpha to Plotly rgba string."""
    r = int(hex_colour[1:3], 16)
    g = int(hex_colour[3:5], 16)
    b = int(hex_colour[5:7], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


class ProfileVisualizationEngine(AbstractVisualizationEngine):
    """Render profile plots: variable(s) vs pressure."""

    def render(self, intent: ParsedIntent, df: pd.DataFrame) -> dict[str, Any]:
        """Build a Plotly figure for profile data."""
        if df.empty:
            raise VisualizationError("DataFrame is empty; nothing to plot.")

        if "PRES" not in df.columns:
            raise VisualizationError(
                "DataFrame missing 'PRES' column required for profile plots."
            )

        variables = intent.variables or []
        if not variables:
            # Fallback: plot every numeric column except PRES and indices
            exclude = {"PRES", "profile_idx", "level_idx"}
            variables = [
                c
                for c in df.columns
                if c not in exclude
                and not c.endswith("_QC")
                and not c.endswith("_ADJUSTED")
                and not c.endswith("_ADJUSTED_QC")
                and pd.api.types.is_numeric_dtype(df[c])
            ]
            logger.info("No variables specified; auto-selected %s", variables)

        # Filter to variables that actually exist in the DataFrame
        available = [v for v in variables if v in df.columns]
        if not available:
            raise VisualizationError(
                f"None of the requested variables found in data: {variables}",
                details={"columns": list(df.columns)},
            )

        n_vars = len(available)
        fig = make_subplots(
            rows=1,
            cols=n_vars,
            shared_yaxes=True,
            horizontal_spacing=0.05,
            subplot_titles=[_VAR_TITLES.get(v, v) for v in available],
        )

        # Group by float_id if available, otherwise fall back to profile_idx
        group_col = "float_id" if "float_id" in df.columns else "profile_idx"
        groups = sorted(df[group_col].unique())

        for col_idx, var in enumerate(available, start=1):
            qc_col = f"{var}_QC"
            has_qc = qc_col in df.columns

            for group_idx, group_val in enumerate(groups):
                sub = df[df[group_col] == group_val].sort_values("PRES", ascending=True)
                if sub.empty:
                    continue

                pres = sub["PRES"].astype(float).values
                vals = sub[var].astype(float).values

                # Build hover text — include float_id when available
                if group_col == "float_id":
                    hover_text = [
                        f"Float: {group_val}<br>Pressure: {p:.1f} dbar<br>{var}: {v:.3f}"
                        for p, v in zip(pres, vals)
                    ]
                    trace_name = f"Float {group_val}"
                else:
                    hover_text = [
                        f"Profile: {group_val}<br>Pressure: {p:.1f} dbar<br>{var}: {v:.3f}"
                        for p, v in zip(pres, vals)
                    ]
                    trace_name = f"Profile {group_val}"

                marker_colour = _COLOURS[group_idx % len(_COLOURS)]

                if has_qc:
                    # Vectorized per-point marker colors with QC-aware opacity
                    alphas = sub[qc_col].apply(_qc_to_alpha).to_numpy(dtype=float)
                    marker_colors = np.empty(len(pres), dtype=object)
                    for i in range(len(pres)):
                        if np.isnan(vals[i]) or np.isnan(pres[i]):
                            marker_colors[i] = "rgba(0,0,0,0)"
                        else:
                            marker_colors[i] = _hex_to_rgba(marker_colour, alphas[i])

                    fig.add_trace(
                        go.Scatter(
                            x=vals,
                            y=pres,
                            mode="lines+markers",
                            name=trace_name,
                            line=dict(color=marker_colour, width=1.5),
                            marker=dict(
                                color=marker_colors.tolist(),
                                size=6,
                            ),
                            hovertext=hover_text,
                            hoverinfo="text",
                            showlegend=(col_idx == 1),
                        ),
                        row=1,
                        col=col_idx,
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=vals,
                            y=pres,
                            mode="lines+markers",
                            name=trace_name,
                            line=dict(color=marker_colour, width=1.5),
                            marker=dict(size=4),
                            hovertext=hover_text,
                            hoverinfo="text",
                            showlegend=(col_idx == 1),
                        ),
                        row=1,
                        col=col_idx,
                    )

            # X-axis title
            fig.update_xaxes(title_text=_VAR_TITLES.get(var, var), row=1, col=col_idx)

        # Shared Y-axis (pressure) — invert so surface is at top
        fig.update_yaxes(
            title_text="Pressure (dbar)",
            autorange="reversed",
            row=1,
            col=1,
        )

        fig.update_layout(
            title_text=self._build_title(intent),
            height=600,
            width=400 * max(n_vars, 1),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            margin=dict(l=80, r=40, t=80, b=80),
        )

        return fig.to_dict()

    @staticmethod
    def _build_title(intent: ParsedIntent) -> str:
        parts: list[str] = []
        if intent.region:
            parts.append(intent.region.replace("_", " ").title())
        if intent.float_id:
            parts.append(f"Float {intent.float_id}")
        if intent.year:
            parts.append(str(intent.year))
        vars_str = ", ".join(intent.variables) if intent.variables else "Variables"
        return f"{vars_str} Profile — {' '.join(parts)}" if parts else f"{vars_str} Profile"
