"""Query Engine orchestrator.

Maps :class:`ParsedIntent` through the full pipeline and returns a
:class:`ChatResponse`.
"""

import logging
import re
import time
from typing import Any

import pandas as pd

from floatchat.config import settings
from floatchat.exceptions import FloatChatError
from floatchat.intent_parser.base import AbstractIntentParser
from floatchat.metadata_service.base import AbstractMetadataService
from floatchat.models import ChatResponse, MapData, ParsedIntent, SearchCriteria
from floatchat.netcdf_reader.base import AbstractNetCDFReader
from floatchat.repository_service.base import AbstractRepositoryService
from floatchat.visualization_engine.base import AbstractVisualizationEngine
from floatchat.scientific_explanation.engine import ScientificExplanationEngine
from floatchat.scientific_explanation.verification import (
    build_verification_section,
    build_pipeline_trace,
)
from floatchat.scientific_explanation.interpretation import generate_plot_interpretation

logger = logging.getLogger(__name__)

# Extract WMO float ID from GDAC file path: dac/<dac>/<float_id>/profiles/...
_FLOAT_ID_RE = re.compile(r"/([\d]{7,})/")


def _extract_float_id_from_path(file_path: str) -> str:
    """Extract the 7-digit WMO float ID from a GDAC relative path."""
    match = _FLOAT_ID_RE.search(file_path)
    return match.group(1) if match else "unknown"


class QueryEngine:
    """Orchestrates the data retrieval and visualization pipeline."""

    def __init__(
        self,
        metadata_service: AbstractMetadataService,
        repository_service: AbstractRepositoryService,
        netcdf_reader: AbstractNetCDFReader,
        visualization_engine: AbstractVisualizationEngine,
    ) -> None:
        self.metadata = metadata_service
        self.repository = repository_service
        self.reader = netcdf_reader
        self.viz = visualization_engine
        self.explanation_engine = ScientificExplanationEngine()

    def execute(self, intent: ParsedIntent) -> ChatResponse:
        """Run the full pipeline for a single parsed intent.

        Args:
            intent: Structured intent produced by an intent parser.

        Returns:
            A :class:`ChatResponse` containing the figure and summary.
        """
        pipeline_t0 = time.perf_counter()
        logger.info("Executing intent: %s", intent.intent)

        # --- Step 1: Metadata search -------------------------------------- #
        t0 = time.perf_counter()
        criteria = self._intent_to_criteria(intent)
        records = self.metadata.search(criteria)
        t1 = time.perf_counter()
        logger.info("Metadata search: %.3fs (%d records)", t1 - t0, len(records))

        if not records:
            logger.warning("No metadata records matched criteria: %s", criteria)
            suggestion = self._get_error_suggestion(intent)
            return ChatResponse(
                intent=intent.intent,
                message=f"No Argo profiles matched your query criteria. {suggestion}",
                data_summary={"matched_records": 0},
            )

        # Build map_data from metadata records (no extra backend calls)
        map_data = self._build_map_data(records)

        # --- Step 2: Fetch & read NetCDFs --------------------------------- #
        dataframes: list[pd.DataFrame] = []
        for rec in records:
            float_id = _extract_float_id_from_path(rec.file)
            ncd = self.repository.fetch(rec.file)
            try:
                df = self.reader.read(ncd, intent.variables)
                # Augment with metadata for downstream use
                df["source_file"] = rec.file
                df["profile_date"] = rec.date
                df["latitude"] = rec.latitude
                df["longitude"] = rec.longitude
                df["float_id"] = float_id
                df["dac"] = rec.institution
                dataframes.append(df)
            except FloatChatError:
                logger.exception("Failed to read %s; skipping", rec.file)
            finally:
                ncd.close()

        t2 = time.perf_counter()
        logger.info("NetCDF fetch+read: %.3fs (%d profiles)", t2 - t1, len(records))

        if not dataframes:
            return ChatResponse(
                intent=intent.intent,
                message="Profiles were found but could not be read.",
                data_summary={"matched_records": len(records), "readable": 0},
                map_data=map_data,
            )

        combined = pd.concat(dataframes, ignore_index=True)
        logger.info("Combined DataFrame shape: %s", combined.shape)

        # --- Step 3: Visualization ---------------------------------------- #
        try:
            figure = self.viz.render(intent, combined)
        except FloatChatError:
            logger.exception("Visualization failed")
            return ChatResponse(
                intent=intent.intent,
                message="Data retrieved but visualization failed.",
                data_summary=self._build_summary(combined, records),
                map_data=map_data,
            )

        t3 = time.perf_counter()
        logger.info("Visualization: %.3fs", t3 - t2)
        logger.info("Total pipeline: %.3fs", t3 - pipeline_t0)

        # --- Step 4: Scientific Interpretation + Verification ------------- #
        interpretation = generate_plot_interpretation(
            combined, intent.variables, intent.region
        )
        verification = build_verification_section(
            intent, records, intent.variables, {}
        )
        pipeline_trace = build_pipeline_trace(
            intent,
            {
                "metadata": t1 - t0,
                "netcdf": t2 - t1,
                "viz": t3 - t2,
                "total": t3 - pipeline_t0,
            },
            False,
        )

        base_message = self._build_message(intent, records, combined)
        explanation = self.explanation_engine.generate_explanation(
            intent, records, intent.variables, self._build_summary(combined, records)
        )
        final_message = f"{base_message} {explanation} {interpretation}"

        data_summary = self._build_summary(combined, records)
        data_summary.update(
            {
                "verification": verification,
                "pipeline_trace": pipeline_trace,
                "suggestions": self._generate_suggestions(intent, records),
                "derived_insights": self._calculate_derived_insights(combined, intent.variables),
            }
        )

        return ChatResponse(
            intent=intent.intent,
            message=final_message,
            figure=figure,
            data_summary=data_summary,
            map_data=map_data,
        )

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _intent_to_criteria(intent: ParsedIntent) -> SearchCriteria:
        """Map a :class:`ParsedIntent` to :class:`SearchCriteria`."""
        # When a specific profile number is requested, fetch exactly 1 record.
        limit = (
            1
            if intent.float_id is not None and intent.profile_number is not None
            else min(intent.limit, settings.max_profiles_per_query)
        )
        return SearchCriteria(
            region=intent.region,
            lat_min=intent.lat_min,
            lat_max=intent.lat_max,
            lon_min=intent.lon_min,
            lon_max=intent.lon_max,
            year=intent.year,
            month=intent.month,
            day=intent.day,
            parameters=intent.variables,
            float_id=intent.float_id,
            profile_number=intent.profile_number,
            limit=limit,
        )

    @staticmethod
    def _build_map_data(records: list[Any]) -> list[MapData]:
        """Build geographic marker data from metadata records."""
        markers: list[MapData] = []
        seen_floats: set[str] = set()
        for rec in records:
            float_id = _extract_float_id_from_path(rec.file)
            # Deduplicate by float_id — keep the most recent profile per float
            if float_id in seen_floats:
                continue
            seen_floats.add(float_id)
            markers.append(
                MapData(
                    float_id=float_id,
                    latitude=rec.latitude,
                    longitude=rec.longitude,
                    profile_date=rec.date.isoformat() if rec.date else None,
                    dac=rec.institution,
                    variables=rec.parameters.split() if rec.parameters else [],
                    selected=False,
                )
            )
        return markers

    @staticmethod
    def _build_message(
        intent: ParsedIntent,
        records: list[Any],
        df: pd.DataFrame,
    ) -> str:
        parts = [
            f"Retrieved {len(records)} profile(s)",
            f"with {len(df)} total measurements",
        ]
        if intent.variables:
            parts.append(f"for variables {', '.join(intent.variables)}.")
        else:
            parts.append(".")
        return " ".join(parts)

    @staticmethod
    def _build_summary(df: pd.DataFrame, records: list[Any]) -> dict[str, Any]:
        date_min = df["profile_date"].min() if "profile_date" in df.columns else pd.NaT
        date_max = df["profile_date"].max() if "profile_date" in df.columns else pd.NaT
        return {
            "matched_records": len(records),
            "total_measurements": len(df),
            "unique_profiles": int(df["profile_idx"].nunique()) if "profile_idx" in df.columns else 0,
            "date_range": {
                "min": date_min.isoformat() if pd.notna(date_min) else None,
                "max": date_max.isoformat() if pd.notna(date_max) else None,
            },
            "files": [r.file for r in records],
        }

    @staticmethod
    def _get_error_suggestion(intent: ParsedIntent) -> str:
        """Return a helpful suggestion when no profiles are found."""
        if intent.variables and "TEMP" in intent.variables:
            return "This float may only contain BGC variables. Try requesting DOXY or CHLA instead."
        if intent.year and intent.year < 2015:
            return "Try a more recent year (many BGC floats were deployed after 2015)."
        if intent.region:
            return "Try broadening the region or removing the year filter."
        return "Try another year, different region, or a different variable."

    @staticmethod
    def _generate_suggestions(
        intent: ParsedIntent, records: list[Any]
    ) -> list[str]:
        """Generate context-aware follow-up suggestions (Improvement 7)."""
        suggestions = []
        vars_upper = [v.upper() for v in intent.variables]

        if "DOXY" in vars_upper or "DOXY_ADJUSTED" in vars_upper:
            suggestions.append("Compare with last year")
            suggestions.append("View chlorophyll")
        if "CHLA" in vars_upper or "CHLA_ADJUSTED" in vars_upper:
            suggestions.append("Inspect trajectory")
        if intent.region:
            suggestions.append("Show temperature")
            suggestions.append("Compare another float in same region")
        if not suggestions:
            suggestions = ["View oxygen", "Show salinity profile", "Compare with 2023"]
        return suggestions[:4]

    @staticmethod
    def _calculate_derived_insights(
        df: pd.DataFrame, variables: list[str]
    ) -> dict[str, Any]:
        """Lightweight derived scientific insights (Improvement 6)."""
        insights: dict[str, Any] = {}
        if "PRES" not in df.columns:
            return insights

        for var in variables:
            col = f"{var}_ADJUSTED" if f"{var}_ADJUSTED" in df.columns else var
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            if var.upper().startswith("DOXY"):
                min_idx = series.idxmin()
                insights["min_oxygen_depth_dbar"] = float(df.loc[min_idx, "PRES"])
            if var.upper().startswith("CHLA"):
                max_idx = series.idxmax()
                insights["max_chlorophyll_depth_dbar"] = float(df.loc[max_idx, "PRES"])
            insights[f"surface_{var.lower()}_avg"] = float(series.iloc[:5].mean())
            insights[f"qc_passed_{var.lower()}"] = int((df.get(f"{col}_QC", pd.Series([1]*len(df))) == "1").sum())
        return insights