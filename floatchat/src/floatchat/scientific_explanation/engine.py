"""Scientific Explanation Engine.

Generates rich, context-aware scientific explanations for every successful query.
Uses Argo knowledge base facts and runtime data. Never hallucinates.

Phase 25.4: Final stabilization for consistency, conversational robustness,
and scientific accuracy.
"""

from typing import Any, Dict, List, Optional
import math
import pandas as pd
import numpy as np

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

    def _get_variable_column(self, df: pd.DataFrame, var_name: str) -> Optional[str]:
        """Find the best available column for a variable, preferring adjusted versions."""
        adj_col = f"{var_name}_ADJUSTED"
        if adj_col in df.columns:
            return adj_col
        if var_name in df.columns:
            return var_name
        return None

    def _compute_stats(self, df: pd.DataFrame, variables: List[str]) -> Dict[str, Any]:
        """Compute descriptive statistics for each requested variable.

        To ensure consistency across different query compositions, statistics are
        computed per profile and then averaged. This avoids 'composite profile'
        artifacts that shift thermocline/halocline depths.
        """
        aggregated_stats: Dict[str, Any] = {}
        
        if "PRES" not in df.columns:
            return aggregated_stats
        
        # Group by profile to compute stats per-profile first.
        # Use 'source_file' as the unique identifier for each profile to avoid
        # 'composite profile' artifacts when multiple profiles are retrieved.
        if "source_file" in df.columns:
            profile_ids = df["source_file"].unique()
        elif "float_id" in df.columns:
            profile_ids = df["float_id"].unique()
        else:
            profile_ids = [0]
        
        all_profile_stats: List[Dict[str, Any]] = []

        for pid in profile_ids:
            if "source_file" in df.columns:
                pdf = df[df["source_file"] == pid]
            elif "float_id" in df.columns:
                pdf = df[df["float_id"] == pid]
            else:
                pdf = df
            pdf = pdf.sort_values("PRES")

            
            p_stats = {}
            for var in variables:
                col = self._get_variable_column(pdf, var)
                if col is None: continue
                
                valid_mask = pdf[col].notna()
                series = pdf.loc[valid_mask, col]
                pres_series = pdf.loc[valid_mask, "PRES"]
                
                if series.empty: continue
                
                v_s = {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                    "mean": float(series.mean()),
                    "count": int(series.count()),
                    "surface": float(series.iloc[:5].mean()),
                    "deep": float(series.iloc[-5:].mean()),
                    "deepest_pres": float(pres_series.max()),
                    "deepest_val": float(series.iloc[-1]),
                }
                
                # Gradient Analysis (ignore surface noise < 20 dbar)
                mask_deep = pres_series > 20
                s_deep = series[mask_deep]
                p_deep = pres_series[mask_deep]
                
                if len(s_deep) > 1:
                    gradient = s_deep.diff() / p_deep.diff()
                    if "TEMP" in var.upper():
                        v_s["grad_depth"] = float(p_deep.loc[gradient.idxmin()])
                    elif "PSAL" in var.upper():
                        v_s["grad_depth"] = float(p_deep.loc[gradient.abs().idxmax()])
                    elif "DOXY" in var.upper():
                        v_s["min_val_depth"] = float(p_deep.loc[s_deep.idxmin()])
                    elif "CHLA" in var.upper():
                        v_s["max_val_depth"] = float(p_deep.loc[s_deep.idxmax()])
                
                p_stats[var] = v_s
            all_profile_stats.append(p_stats)

        # Aggregate per-profile stats into a global average
        for var in variables:
            relevant_stats = [ps[var] for ps in all_profile_stats if var in ps]
            if not relevant_stats: continue
            
            # Average the means
            agg = {
                "min": float(np.nanmin([s["min"] for s in relevant_stats])),
                "max": float(np.nanmax([s["max"] for s in relevant_stats])),
                "median": float(np.nanmedian([s["median"] for s in relevant_stats])),
                "mean": float(np.nanmean([s["mean"] for s in relevant_stats])),
                "surface": float(np.nanmean([s["surface"] for s in relevant_stats])),
                "deep": float(np.nanmean([s["deep"] for s in relevant_stats])),
                "deepest_pres": float(np.nanmean([s["deepest_pres"] for s in relevant_stats])),
                "deepest_val": float(np.nanmean([s["deepest_val"] for s in relevant_stats])),
            }
            
            # Average the depths (weighted or simple mean)
            # Fix: Check if ANY profile had the depth, not just the first one.
            depth_keys = ["grad_depth", "min_val_depth", "max_val_depth"]
            for dk in depth_keys:
                depths = [s[dk] for s in relevant_stats if dk in s]
                if depths: 
                    agg[dk] = float(np.nanmean(depths))
                
            aggregated_stats[var] = agg
            
        return aggregated_stats

    def _generate_data_driven_explanation(
        self, 
        intent: ParsedIntent, 
        records: List[MetadataRecord], 
        variables: List[str], 
        df: pd.DataFrame
    ) -> str:
        """Generate a structured, data-driven scientific explanation."""
        stats = self._compute_stats(df, variables)
        
        # 1. Summary section
        summary_parts: List[str] = []
        for var in variables:
            if var not in stats: continue
            s = stats[var]
            v_name = var.replace("_", " ").title()

            if "TEMP" in var.upper():
                if math.isfinite(s['surface']):
                    summary_parts.append(f"• Surface {v_name}: {s['surface']:.1f}°C")
                if "grad_depth" in s and math.isfinite(s['grad_depth']):
                    summary_parts.append(f"• Thermocline: {s['grad_depth']:.0f} dbar")
            elif "PSAL" in var.upper():
                if math.isfinite(s['surface']):
                    summary_parts.append(f"• Surface {v_name}: {s['surface']:.2f} PSU")
                if "grad_depth" in s and math.isfinite(s['grad_depth']):
                    summary_parts.append(f"• Halocline: {s['grad_depth']:.0f} dbar")
            elif "DOXY" in var.upper():
                if math.isfinite(s['surface']):
                    summary_parts.append(f"• Surface {v_name}: {s['surface']:.1f} µmol/kg")
                if "min_val_depth" in s and math.isfinite(s['min_val_depth']):
                    summary_parts.append(f"• Oxygen Minimum: {s['min_val_depth']:.0f} dbar")
            elif "CHLA" in var.upper():
                if math.isfinite(s['surface']):
                    summary_parts.append(f"• Surface {v_name}: {s['surface']:.3f} mg/m³")
                if "max_val_depth" in s and math.isfinite(s['max_val_depth']):
                    summary_parts.append(f"• DCM Depth: {s['max_val_depth']:.0f} dbar")
            elif "BBP700" in var.upper():
                if math.isfinite(s['surface']):
                    summary_parts.append(f"• Surface {v_name}: {s['surface']:.4f} m^-1")

        summary_text = "\n" + "\n".join(summary_parts) if summary_parts else ""

        # 2. Interpretation section
        interp_parts: List[str] = []

        # Variable-specific narratives
        for var in variables:
            if var not in stats: continue
            s = stats[var]

            if "TEMP" in var.upper():
                if math.isfinite(s['surface']) and math.isfinite(s['deepest_val']) and math.isfinite(s['deepest_pres']):
                    delta = s['surface'] - s['deepest_val']
                    interp_parts.append(
                        f"Surface waters average {s['surface']:.1f}°C, cooling to {s['deepest_val']:.1f}°C "
                        f"at {s['deepest_pres']:.0f} dbar. A total decrease of {delta:.1f}°C "
                        f"indicates strong vertical stratification."
                    )
            elif "PSAL" in var.upper():
                if math.isfinite(s['min']) and math.isfinite(s['max']) and math.isfinite(s['surface']) and math.isfinite(s['deepest_val']):
                    delta = abs(s['surface'] - s['deepest_val'])
                    interp_parts.append(
                        f"Salinity ranges from {s['min']:.2f} to {s['max']:.2f} PSU (global range), "
                        f"with an average change of {delta:.2f} PSU from surface to deep waters, "
                        f"reflecting regional evaporation and precipitation patterns."
                    )

            elif "DOXY" in var.upper():
                if "min_val_depth" in s and math.isfinite(s['min_val_depth']) and math.isfinite(s['min']):
                    if s['min'] < 100:
                        interp_parts.append(
                            f"A pronounced Oxygen Minimum Zone (OMZ) is observed at {s['min_val_depth']:.0f} dbar "
                            f"with concentrations falling to {s['min']:.1f} µmol/kg."
                        )
                    else:
                        interp_parts.append(
                            f"Lowest oxygen ({s['min']:.1f} µmol/kg) occurs near {s['min_val_depth']:.0f} dbar, "
                            f"but concentrations remain relatively high throughout the water column."
                        )
            elif "CHLA" in var.upper():
                if "max_val_depth" in s and math.isfinite(s['max_val_depth']) and s["max_val_depth"] > 15:
                    interp_parts.append(
                        f"A genuine Deep Chlorophyll Maximum (DCM) is detected at {s['max_val_depth']:.0f} dbar, "
                        f"indicating a subsurface peak in primary productivity."
                    )
                else:
                    interp_parts.append("The maximum chlorophyll concentration is located near the surface.")
            elif "BBP700" in var.upper():
                if math.isfinite(s['deepest_val']) and math.isfinite(s['surface']):
                    delta = s['deepest_val'] - s['surface']
                    trend = "increasing" if delta > 0 else "decreasing"
                    interp_parts.append(
                        f"Particle backscatter shows a {trend} trend with depth, "
                        f"changing from {s['surface']:.4f} to {s['deepest_val']:.4f} m^-1, "
                        f"reflecting variations in particle size and concentration."
                    )


        # Integrated Multi-variable reasoning
        core_vars = {}
        for v in variables:
            if "TEMP" in v.upper(): core_vars["TEMP"] = v
            if "PSAL" in v.upper(): core_vars["PSAL"] = v
            if "DOXY" in v.upper(): core_vars["DOXY"] = v
            if "CHLA" in v.upper(): core_vars["CHLA"] = v

        if "TEMP" in core_vars and "DOXY" in core_vars:
            t_var, d_var = core_vars["TEMP"], core_vars["DOXY"]
            if t_var in stats and d_var in stats:
                s_t, s_d = stats[t_var], stats[d_var]
                if math.isfinite(s_t['surface']) and math.isfinite(s_d['min']):
                    grad_depth = s_t.get('grad_depth', 0)
                    grad_str = f"{grad_depth:.0f}" if math.isfinite(grad_depth) else "unknown"
                    interp_parts.append(
                        f"Warm surface waters ({s_t['surface']:.1f}°C) correspond to higher oxygen levels, "
                        f"while the oxygen minimum ({s_d['min']:.1f} µmol/kg) typically emerges below the "
                        f"thermocline ({grad_str} dbar), indicating limited ventilation."
                    )

        if "TEMP" in core_vars and "CHLA" in core_vars:
            t_var, c_var = core_vars["TEMP"], core_vars["CHLA"]
            if t_var in stats and c_var in stats:
                s_t, s_c = stats[t_var], stats[c_var]
                if "grad_depth" in s_t and "max_val_depth" in s_c:
                    t_depth, c_depth = s_t["grad_depth"], s_c["max_val_depth"]
                    if math.isfinite(t_depth) and math.isfinite(c_depth) and abs(t_depth - c_depth) < 50:
                        interp_parts.append(
                            f"The chlorophyll maximum ({c_depth:.0f} dbar) is closely aligned with the "
                            f"thermocline ({t_depth:.0f} dbar), a common feature in stratified oceans."
                        )

        if "TEMP" in core_vars and "PSAL" in core_vars:
            t_var, p_var = core_vars["TEMP"], core_vars["PSAL"]
            if t_var in stats and p_var in stats:
                s_t, s_p = stats[t_var], stats[p_var]
                if (math.isfinite(s_t['surface']) and math.isfinite(s_p['surface'])
                        and s_t['surface'] > 20 and s_p['surface'] > 35):
                    interp_parts.append(
                        "The combination of high surface temperature and high salinity suggests "
                        "strong evaporative forcing in this region."
                    )

        interpretation_text = "\n\nInterpretation\n" + " ".join(interp_parts) if interp_parts else ""

        # 3. Concise Data Quality
        qc_parts: List[str] = []
        has_real_time = any("R" in (r.parameter_data_mode or "") for r in records)
        if has_real_time:
            qc_parts.append("Some profiles contain real-time data (preliminary).")
        else:
            qc_parts.append("Most measurements are delayed-mode quality-controlled.")
        
        qc_text = "\n\nData quality\n" + " ".join(qc_parts)

        return f"{summary_text}\n{interpretation_text}\n{qc_text}"

    def _generate_kb_explanation(
        self,
        intent: ParsedIntent,
        records: List[MetadataRecord],
        variables: List[str],
        data_summary: Dict[str, Any],
    ) -> str:
        """KB-based fallback."""
        parts = ["General scientific context:"]
        vars_upper = {v.upper() for v in variables}
        for v in vars_upper:
            if v in self.kb: parts.append(self.kb[v])
        return " ".join(parts)

    def generate_explanation(
        self,
        intent: ParsedIntent,
        records: List[MetadataRecord],
        variables: List[str],
        data_summary: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
    ) -> str:
        if df is not None and not df.empty:
            return self._generate_data_driven_explanation(intent, records, variables, df)
        return self._generate_kb_explanation(intent, records, variables, data_summary)
