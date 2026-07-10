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
from floatchat.retrieval_planner.planner import RetrievalPlanner
from floatchat.variable_registry.registry import VariableRegistry

logger = logging.getLogger(__name__)

# Extract WMO float ID from GDAC file path: dac/<dac>/<float_id>/profiles/...
_FLOAT_ID_RE = re.compile(r"/([\d]{7,})/")


def _extract_float_id_from_path(file_path: str) -> str:
    """Extract the 7-digit WMO float ID from a GDAC relative path."""
    match = _FLOAT_ID_RE.search(file_path)
    return match.group(1) if match else "unknown"


def _extract_float_cycle_key(file_path: str) -> tuple[str, str]:
    """Extract (float_id, cycle) key from a GDAC file path for pairing (Phase 24)."""
    _cyc_re = re.compile(r"_(\d{3})\.nc")
    fid = _FLOAT_ID_RE.search(file_path)
    cyc = _cyc_re.search(file_path)
    return (fid.group(1) if fid else "", cyc.group(1) if cyc else "")


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
        self.planner = RetrievalPlanner()

    def execute(self, intent: ParsedIntent) -> ChatResponse:
        """Run the full pipeline for a single parsed intent.

        Args:
            intent: Structured intent produced by an intent parser.

        Returns:
            A :class:`ChatResponse` containing the figure and summary.
        """
        pipeline_t0 = time.perf_counter()

        # --- Phase 26: India-only Deployment Gate --- #
        if settings.deployment_mode == "INDIA_ONLY":
            supported_india_regions = {"arabian_sea", "bay_of_bengal"}
            if intent.region and intent.region not in supported_india_regions:
                return ChatResponse(
                    intent=intent.intent,
                    message=f"Region '{intent.region}' is not supported in the current deployment mode. "
                            f"Please request data for the Arabian Sea or Bay of Bengal.",
                    data_summary={"matched_records": 0},
                )

        logger.info("Executing intent: %s", intent.intent)

        # --- Phase 21: Retrieval Planning --------------------------------- #
        plan = self.planner.plan(intent.variables or [])
        logger.info("Retrieval Plan: %s", plan.reasoning)

        # --- Step 1: Metadata search -------------------------------------- #
        t0 = time.perf_counter()
        criteria = self._intent_to_criteria(intent)
        search_groups = self._search_metadata_groups(intent, criteria, plan)
        records = [record for group_records, _ in search_groups for record in group_records]
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
        fetched_files: set[str] = set()  # Phase 23: dedup same-file downloads
        for group_records, variables in search_groups:
            for rec in group_records:
                float_id = _extract_float_id_from_path(rec.file)

                # Phase 23: Skip duplicate file downloads in one request
                if rec.file in fetched_files:
                    logger.info("Skipping duplicate fetch: %s", rec.file)
                    continue
                fetched_files.add(rec.file)

                t_fetch_t0 = time.perf_counter()
                ncd = self.repository.fetch(rec.file)
                t_fetch_t1 = time.perf_counter()
                logger.info("NetCDF fetch: %.3fs (%s)", t_fetch_t1 - t_fetch_t0, rec.file)

                t_read_t0 = time.perf_counter()
                try:
                    df = self.reader.read(ncd, variables)
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
                t_read_t1 = time.perf_counter()
                logger.info("NetCDF read: %.3fs (%s)", t_read_t1 - t_read_t0, rec.file)

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
        t_sci_t0 = time.perf_counter()
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
            intent, records, intent.variables, self._build_summary(combined, records), df=combined
        )
        final_message = f"{base_message}\n\n{explanation}"

        data_summary = self._build_summary(combined, records)
        data_summary.update(
            {
                "verification": verification,
                "pipeline_trace": pipeline_trace,
                "suggestions": self._generate_suggestions(intent, records),
                "derived_insights": self._calculate_derived_insights(combined, intent.variables),
            }
        )


        t_sci_t1 = time.perf_counter()
        logger.info("Scientific explanation: %.3fs", t_sci_t1 - t_sci_t0)

        total = time.perf_counter()
        logger.info("Total request time: %.3fs", total - pipeline_t0)

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

    def _search_metadata_groups(
        self,
        intent: ParsedIntent,
        criteria: SearchCriteria,
        plan,
    ) -> list[tuple[list[Any], list[str]]]:
        """Search metadata without intersecting Core and Bio indexes.

        Phase 24: For mixed queries, prefers records from the same float+cycle
        before returning independent observations.
        """
        if plan.metadata_index != "both":
            return [(self.metadata.search(criteria), intent.variables)]

        classification = VariableRegistry.classify_variables(intent.variables or [])

        core_vars = classification["core"]
        bgc_vars = classification["bgc"]

        # Request more records than needed so we have candidates for pairing.
        pair_limit = max(criteria.limit, 10)
        core_records: list[Any] = []
        bio_records: list[Any] = []
        if core_vars:
            core_criteria = criteria.model_copy(
                update={"parameters": core_vars, "limit": pair_limit}
            )
            core_records = self.metadata.search(core_criteria)
        if bgc_vars:
            bio_criteria = criteria.model_copy(
                update={"parameters": bgc_vars, "limit": pair_limit}
            )
            bio_records = self.metadata.search(bio_criteria)

        # --- Phase 24: Pair by (float_id, cycle) when both groups exist --- #
        if core_records and bio_records:
            core_records, bio_records = self._pair_by_float_cycle(
                core_records, bio_records, criteria.limit
            )

        groups: list[tuple[list[Any], list[str]]] = []
        if core_records and core_vars:
            groups.append((core_records, core_vars))
        if bio_records and bgc_vars:
            groups.append((bio_records, bgc_vars))

        return groups

    @staticmethod
    def _pair_by_float_cycle(
        core_records: list[Any],
        bio_records: list[Any],
        limit: int,
    ) -> tuple[list[Any], list[Any]]:
        """Phase 24: Reorder records so pairs from the same float+cycle come first.

        Falls back to independent retrieval when no pairs exist.
        """
        # Build lookup: key → list of records
        core_by_key: dict[tuple[str, str], list[Any]] = {}
        for r in core_records:
            key = _extract_float_cycle_key(r.file)
            core_by_key.setdefault(key, []).append(r)

        bio_by_key: dict[tuple[str, str], list[Any]] = {}
        for r in bio_records:
            key = _extract_float_cycle_key(r.file)
            bio_by_key.setdefault(key, []).append(r)

        # Keys present in both indexes = paired floats
        paired_keys = set(core_by_key) & set(bio_by_key)

        if not paired_keys:
            return core_records[:limit], bio_records[:limit]

        # Collect paired records first
        paired_core: list[Any] = []
        paired_bio: list[Any] = []
        for key in sorted(paired_keys):
            paired_core.extend(core_by_key[key])
            paired_bio.extend(bio_by_key[key])

        # Then unpaired (fallback)
        unpaired_core: list[Any] = []
        for key in sorted(set(core_by_key) - paired_keys):
            unpaired_core.extend(core_by_key[key])
        unpaired_bio: list[Any] = []
        for key in sorted(set(bio_by_key) - paired_keys):
            unpaired_bio.extend(bio_by_key[key])

        # Sort all groups by date (newest first) within paired/unpaired
        sort_key = lambda r: r.date
        paired_core.sort(key=sort_key, reverse=True)
        paired_bio.sort(key=sort_key, reverse=True)
        unpaired_core.sort(key=sort_key, reverse=True)
        unpaired_bio.sort(key=sort_key, reverse=True)

        # Prefer paired, cap at limit
        best_core = paired_core + unpaired_core
        best_bio = paired_bio + unpaired_bio

        logger.info(
            "Phase 24 pairing: %d paired floats, returning %d core + %d bio records",
            len(paired_keys), min(len(best_core), limit), min(len(best_bio), limit),
        )

        return best_core[:limit], best_bio[:limit]

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
        """Deprecated: Use ScientificExplanationEngine._compute_stats instead."""
        return {}
