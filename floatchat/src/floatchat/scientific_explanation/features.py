"""Scientific Feature Extractor – Step 2 (shadow mode).

Wraps the legacy ScientificExplanationEngine._compute_stats() output
into a structured ScientificFacts object.

Design constraints for Step 2:
- Do NOT migrate _compute_stats() yet – use it as reference.
- Reproduce EXACTLY: Thermocline, Halocline, Oxygen Minimum, DCM, BBP700 trend
- No new detectors (MLD, Nitracline, pH min, Euphotic Depth) yet.
- Output must be a compact ScientificFacts JSON (1–3 KB).
- Every numeric value originates from Python legacy stats.
- No DataFrames, arrays, or NetCDF objects cross the LLM boundary.

This extractor runs in SHADOW MODE – results are compared to legacy,
but legacy remains authoritative until LLM narration is validated.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from ..models.intent import ParsedIntent
from ..models.metadata import MetadataRecord
from .engine import ScientificExplanationEngine
from .schemas import (
    ProfileMeta,
    QCSummary,
    RetrievalProvenance,
    ScientificFacts,
    VariableStats,
    VerticalFeature,
)

logger = logging.getLogger(__name__)

_FLOAT_ID_RE = re.compile(r"/([\d]{7,})/")

# Units registry – expandable for NITRATE, pH, CDOM, etc. without prompt changes
_UNITS: Dict[str, str] = {
    "TEMP": "°C",
    "TEMP_ADJUSTED": "°C",
    "PSAL": "PSU",
    "PSAL_ADJUSTED": "PSU",
    "DOXY": "µmol/kg",
    "DOXY_ADJUSTED": "µmol/kg",
    "CHLA": "mg/m³",
    "CHLA_ADJUSTED": "mg/m³",
    "BBP700": "m^-1",
    "BBP700_ADJUSTED": "m^-1",
    # Future BGC – schema already supports them, extractor just needs units:
    "NITRATE": "µmol/kg",
    "NITRATE_ADJUSTED": "µmol/kg",
    "PH_IN_SITU_TOTAL": "total scale",
    "PH_IN_SITU_TOTAL_ADJUSTED": "total scale",
    "DOWNWELLING_PAR": "µmol quanta/m²/s",
    "DOWNWELLING_PAR_ADJUSTED": "µmol quanta/m²/s",
    "CDOM": "ppb",
    "CDOM_ADJUSTED": "ppb",
}


def _extract_float_id(file_path: str) -> Optional[str]:
    m = _FLOAT_ID_RE.search(file_path or "")
    return m.group(1) if m else None


def _get_units(var_name: str) -> str:
    # Try exact, then base without _ADJUSTED
    if var_name in _UNITS:
        return _UNITS[var_name]
    base = var_name.replace("_ADJUSTED", "")
    return _UNITS.get(base, "unknown")


def _resolve_column(df: pd.DataFrame, var: str) -> Optional[str]:
    adj = f"{var}_ADJUSTED"
    if adj in df.columns:
        return adj
    if var in df.columns:
        return var
    # case-insensitive fallback
    for c in df.columns:
        if c.upper() == var.upper() or c.upper() == adj.upper():
            return c
    return None


class ScientificFeatureExtractor:
    """
    Wraps legacy _compute_stats() into a typed ScientificFacts object.

    Step 2 policy:
    - use_legacy=True (default) → call ScientificExplanationEngine._compute_stats
    - output is ScientificFacts – array-free, 1–3KB
    - no new scientific detectors yet
    """

    def __init__(self, use_legacy: bool = True):
        self.use_legacy = use_legacy
        # Legacy engine is the reference implementation – do NOT modify it
        self._legacy_engine = ScientificExplanationEngine() if use_legacy else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        df: pd.DataFrame,
        variables: List[str],
        intent: ParsedIntent,
        records: List[MetadataRecord],
        query_id: Optional[str] = None,
    ) -> ScientificFacts:
        """
        Produce a ScientificFacts object from a DataFrame.
        In Step 2 this wraps the legacy _compute_stats() exactly.
        """
        if not variables:
            raise ValueError("variables list must not be empty")

        # --- Step 2: use legacy stats as ground truth ---
        if self.use_legacy and self._legacy_engine is not None:
            legacy_stats = self._legacy_engine._compute_stats(df, variables)
        else:
            # Future native implementation placeholder – not used in Step 2
            raise NotImplementedError("Native extractor not yet enabled in Step 2")

        # Build provenance first (needed for ScientificFacts)
        provenance = self._build_provenance(records, df)

        # Build profile metadata
        profiles = self._build_profiles(records)

        # Convert legacy stats dict → List[VariableStats]
        stats = self._stats_to_variable_stats(legacy_stats, variables, df)

        # Convert legacy feature depths → List[VerticalFeature]
        features = self._stats_to_features(legacy_stats, variables)

        # QC summary
        qc = self._build_qc_summary(records, variables, df)

        # Cross-variable notes – keep empty in Step 2 to match legacy simplicity
        cross_notes: List[str] = []

        facts = ScientificFacts(
            schema_version="1.0.0",
            prompt_version="sci_narrator_v1_2026-07",
            query_id=query_id or uuid.uuid4().hex[:12],
            generated_at=datetime.now(timezone.utc),
            variables_requested=[v.upper() for v in variables],
            region=intent.region,
            float_id=intent.float_id,
            year_filter=intent.year,
            provenance=provenance,
            profiles=profiles,
            stats=stats,
            features=features,
            qc=qc,
            cross_variable_notes=cross_notes,
        )

        # Explicit runtime validation – no assert
        # 1. Ensure no arrays leaked
        try:
            facts.validate_no_arrays()
        except ValueError as e:
            logger.error("ScientificFacts array-leak validation failed: %s", e)
            raise

        # 2. Size check – use configurable limit if available
        try:
            from ..config import settings  # local import to avoid circular
            max_bytes = getattr(settings, "sci_narrator_max_payload_bytes", 4096)
        except Exception:
            max_bytes = 4096

        try:
            payload = facts.to_llm_payload(max_bytes=max_bytes)
            logger.debug(
                "ScientificFacts payload OK: %d bytes, %d variables, %d features",
                len(payload.encode("utf-8")),
                len(stats),
                len(features),
            )
        except ValueError as e:
            logger.warning("ScientificFacts payload exceeds limit: %s", e)
            # Do not crash the pipeline – caller decides fallback
            raise

        return facts

    # ------------------------------------------------------------------
    # Legacy → schema converters
    # ------------------------------------------------------------------

    def _stats_to_variable_stats(
        self, legacy_stats: Dict[str, Any], variables: List[str], df: pd.DataFrame
    ) -> List[VariableStats]:
        out: List[VariableStats] = []
        for var in variables:
            if var not in legacy_stats:
                # Try to find case-insensitive match
                match = next((k for k in legacy_stats if k.upper() == var.upper()), None)
                if not match:
                    continue
                var_key = match
            else:
                var_key = var

            s = legacy_stats[var_key]
            col = _resolve_column(df, var_key) or var_key
            units = _get_units(col)

            # Defensive: legacy stats are always present per Phase 25, but guard anyway
            try:
                vs = VariableStats(
                    variable=col.upper(),
                    units=units,
                    n_obs=int(s.get("count", 0)),
                    min_val=self._safe_float(s.get("min")),
                    max_val=self._safe_float(s.get("max")),
                    mean_val=self._safe_float(s.get("mean")),
                    median_val=self._safe_float(s.get("median")),
                    surface_mean_0_10m=self._safe_float(s.get("surface")),
                    deep_mean_below_200m=self._safe_float(s.get("deep")),
                    deepest_pres_dbar=self._safe_float(s.get("deepest_pres")),
                    deepest_val=self._safe_float(s.get("deepest_val")),
                )
                out.append(vs)
            except Exception as e:
                logger.warning("Skipping VariableStats for %s: %s", var_key, e)
                continue
        return out

    def _stats_to_features(
        self, legacy_stats: Dict[str, Any], variables: List[str]
    ) -> List[VerticalFeature]:
        """
        Map legacy depth keys to VerticalFeature objects.
        Step 2 supported features ONLY:
        - Thermocline (TEMP grad_depth)
        - Halocline (PSAL grad_depth)
        - Oxygen Minimum (DOXY min_val_depth)
        - DCM (CHLA max_val_depth)
        - BBP700: no depth feature in legacy – skip
        """
        features: List[VerticalFeature] = []

        for var in variables:
            # resolve case-insensitive
            stat_key = var if var in legacy_stats else next(
                (k for k in legacy_stats if k.upper() == var.upper()), None
            )
            if not stat_key:
                continue
            s = legacy_stats[stat_key]
            vu = stat_key.upper()

            # Thermocline
            if "TEMP" in vu and "grad_depth" in s:
                depth = self._safe_float(s.get("grad_depth"))
                if depth is not None:
                    features.append(
                        VerticalFeature(
                            feature="thermocline",
                            depth_dbar=depth,
                            strength=None,
                            value_at_feature=None,
                            prominence=None,
                            method="max_gradient_20m_plus",
                        )
                    )

            # Halocline
            if "PSAL" in vu and "grad_depth" in s:
                depth = self._safe_float(s.get("grad_depth"))
                if depth is not None:
                    features.append(
                        VerticalFeature(
                            feature="halocline",
                            depth_dbar=depth,
                            strength=None,
                            value_at_feature=None,
                            prominence=None,
                            method="max_gradient_20m_plus",
                        )
                    )

            # Oxygen Minimum
            if "DOXY" in vu and "min_val_depth" in s:
                depth = self._safe_float(s.get("min_val_depth"))
                val = self._safe_float(s.get("min"))
                if depth is not None:
                    features.append(
                        VerticalFeature(
                            feature="oxygen_minimum",
                            depth_dbar=depth,
                            strength=None,
                            value_at_feature=val,
                            prominence="strong" if val is not None and val < 60 else "moderate",
                            method="min_value_below_20m",
                        )
                    )

            # DCM
            if "CHLA" in vu and "max_val_depth" in s:
                depth = self._safe_float(s.get("max_val_depth"))
                val = self._safe_float(s.get("max"))
                if depth is not None:
                    features.append(
                        VerticalFeature(
                            feature="dcm",
                            depth_dbar=depth,
                            strength=None,
                            value_at_feature=val,
                            prominence="strong" if depth > 15 else "weak",
                            method="max_value_below_20m",
                        )
                    )

            # BBP700 – legacy has no depth feature, only trend text.
            # Intentionally omit VerticalFeature to stay faithful to Step 2 scope.
            # Future steps will add particle_max feature.

        return features

    # ------------------------------------------------------------------
    # Provenance / QC builders
    # ------------------------------------------------------------------

    def _build_provenance(
        self, records: List[MetadataRecord], df: pd.DataFrame
    ) -> RetrievalProvenance:
        dac_list = []
        seen = set()
        for r in records:
            dac = getattr(r, "institution", None)
            if dac and dac not in seen:
                seen.add(dac)
                dac_list.append(dac)

        primary_dac = dac_list[0] if dac_list else None

        # Dates
        dates = [r.date for r in records if getattr(r, "date", None) is not None]
        date_start = min(dates).date().isoformat() if dates else None
        date_end = max(dates).date().isoformat() if dates else None
        average_year = None
        if dates:
            try:
                average_year = sum(d.year for d in dates) / len(dates)
            except Exception:
                average_year = None

        # Data mode counts – parse parameter_data_mode
        mode_counts: Dict[str, int] = {"D": 0, "R": 0, "A": 0}
        for r in records:
            mode_str = getattr(r, "parameter_data_mode", "") or ""
            # Count delayed-mode if 'D' present, else real-time 'R', else adjusted 'A'
            if "D" in mode_str:
                mode_counts["D"] += 1
            elif "R" in mode_str:
                mode_counts["R"] += 1
            else:
                # fallback – treat as adjusted/other
                mode_counts["A"] += 1

        # Remove zero entries for cleanliness (schema allows empty dict default, but we keep keys)
        # keep all keys – schema expects Dict[str,int]

        if mode_counts["D"] >= mode_counts["R"]:
            qc_mode_summary = "delayed-mode dominant"
        elif mode_counts["R"] > 0:
            qc_mode_summary = "real-time present"
        else:
            qc_mode_summary = "mixed"

        gdac_files = []
        seen_files = set()
        for r in records:
            f = getattr(r, "file", None)
            if f and f not in seen_files:
                seen_files.add(f)
                gdac_files.append(f)
            if len(gdac_files) >= 10:
                break

        measurement_count = int(len(df)) if df is not None else 0

        return RetrievalProvenance(
            source="Argo GDAC (https://data-argo.ifremer.fr)",
            dac_list=dac_list,
            primary_dac=primary_dac,
            profile_count=len(records),
            measurement_count=measurement_count,
            date_start=date_start,
            date_end=date_end,
            average_year=average_year,
            data_mode_counts=mode_counts,
            qc_mode_summary=qc_mode_summary,
            gdac_files=gdac_files,
        )

    def _build_profiles(self, records: List[MetadataRecord]) -> List[ProfileMeta]:
        profiles: List[ProfileMeta] = []
        seen_floats = set()
        for r in records:
            # Deduplicate by file to avoid duplicate profile entries
            src = getattr(r, "file", None)
            if src in seen_floats:
                continue
            seen_floats.add(src)

            float_id = _extract_float_id(src or "")
            date_str = r.date.isoformat() if getattr(r, "date", None) else None
            # data_mode – first token of parameter_data_mode
            pdm = getattr(r, "parameter_data_mode", "") or ""
            data_mode = pdm.split()[0] if pdm.split() else None

            try:
                pm = ProfileMeta(
                    float_id=float_id or "unknown",
                    profile_date=date_str,
                    latitude=getattr(r, "latitude", None),
                    longitude=getattr(r, "longitude", None),
                    dac=getattr(r, "institution", None),
                    data_mode=data_mode,
                    profile_number=None,
                    source_file=src,
                )
                profiles.append(pm)
            except Exception as e:
                logger.debug("Skipping profile meta for %s: %s", src, e)
                continue
            if len(profiles) >= 20:  # schema max_length
                break
        return profiles

    def _build_qc_summary(
        self, records: List[MetadataRecord], variables: List[str], df: pd.DataFrame
    ) -> QCSummary:
        total = len(records) or 1
        delayed = 0
        for r in records:
            mode_str = getattr(r, "parameter_data_mode", "") or ""
            if "D" in mode_str:
                delayed += 1
        delayed_pct = round(delayed / total * 100.0, 1)

        # variables_adjusted – check which requested variables have _ADJUSTED column present
        adjusted: List[str] = []
        if df is not None:
            cols_upper = {c.upper(): c for c in df.columns}
            for v in variables:
                adj = f"{v.upper()}_ADJUSTED"
                if adj in cols_upper:
                    adjusted.append(adj)

        return QCSummary(
            delayed_mode_pct=delayed_pct,
            qc_good_pct=None,
            variables_adjusted=adjusted,
        )

    # ------------------------------------------------------------------
    # Shadow comparison helper
    # ------------------------------------------------------------------

    def compare_with_legacy(
        self,
        df: pd.DataFrame,
        variables: List[str],
        intent: ParsedIntent,
        records: List[MetadataRecord],
    ) -> Dict[str, Any]:
        """
        Run extractor and compare numeric outputs to legacy _compute_stats.
        Returns a dict with match status – used for shadow-mode logging.
        Does NOT raise – logs differences only.
        """
        try:
            # Legacy stats
            legacy_stats = self._legacy_engine._compute_stats(df, variables) if self._legacy_engine else {}
            # New facts
            facts = self.extract(df, variables, intent, records, query_id="shadow-compare")

            # Compare each variable stat
            diffs = []
            for vs in facts.stats:
                # map back to legacy key (strip _ADJUSTED)
                base_var = vs.variable.replace("_ADJUSTED", "")
                # find legacy entry – try exact, upper, base
                legacy_entry = (
                    legacy_stats.get(base_var)
                    or legacy_stats.get(vs.variable)
                    or legacy_stats.get(base_var.upper())
                )
                if not legacy_entry:
                    diffs.append(f"{vs.variable}: no legacy entry")
                    continue

                # compare key numeric fields with tolerance
                checks = [
                    ("mean_val", "mean"),
                    ("min_val", "min"),
                    ("max_val", "max"),
                    ("median_val", "median"),
                    ("surface_mean_0_10m", "surface"),
                    ("deep_mean_below_200m", "deep"),
                ]
                for new_key, old_key in checks:
                    new_v = getattr(vs, new_key)
                    old_v = legacy_entry.get(old_key)
                    if new_v is None or old_v is None:
                        continue
                    # tolerance: 1e-6 relative or 1e-9 absolute
                    if abs(new_v - float(old_v)) > 1e-6 * max(1.0, abs(old_v)):
                        diffs.append(
                            f"{vs.variable} {new_key}: facts={new_v} legacy={old_v}"
                        )

            # Feature count check
            legacy_feature_count = sum(
                1
                for v, s in legacy_stats.items()
                if any(k in s for k in ("grad_depth", "min_val_depth", "max_val_depth"))
            )

            result = {
                "match": len(diffs) == 0,
                "differences": diffs,
                "legacy_vars": list(legacy_stats.keys()),
                "facts_vars": [s.variable for s in facts.stats],
                "legacy_feature_hits": legacy_feature_count,
                "facts_features": len(facts.features),
                "payload_bytes": len(facts.to_llm_payload().encode("utf-8")),
            }
            return result
        except Exception as e:
            logger.exception("Shadow comparison failed: %s", e)
            return {"match": False, "error": str(e), "differences": [str(e)]}

    @staticmethod
    def _safe_float(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            import math

            f = float(x)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except Exception:
            return None
