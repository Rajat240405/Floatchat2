"""GDAC metadata service implementation.

Downloads the official ``argo_bio-profile_index.txt.gz``, loads it into a
Pandas DataFrame, and provides fast in-memory filtering.
"""

import gzip
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

from floatchat.config import settings
from floatchat.exceptions import MetadataError
from floatchat.metadata_service.base import AbstractMetadataService
from floatchat.metadata_service.regions import has_polygon, point_in_region, resolve_region
from floatchat.models import MetadataRecord, SearchCriteria
from floatchat.variable_registry.registry import VariableRegistry
from floatchat.retrieval_planner.planner import RetrievalPlanner

logger = logging.getLogger(__name__)

# Columns defined by the Argo GDAC bio-profile index specification.
# See: https://data-argo.ifremer.fr/argo_bio-profile_index.txt.gz
_INDEX_COLUMNS = [
    "file",
    "date",
    "latitude",
    "longitude",
    "ocean",
    "profiler_type",
    "institution",
    "parameters",
    "parameter_data_mode",
    "date_update",
]

# Local cache path (relative to working directory or env override).
_CACHE_DIR = Path(os.environ.get("FLOATCHAT_CACHE_DIR", ".cache"))
_CACHE_FILE = _CACHE_DIR / "argo_bio-profile_index.txt.gz"
_SYNTHETIC_CACHE_FILE = _CACHE_DIR / "argo_synthetic-profile_index.txt.gz"


def _parse_argo_timestamp(val: str) -> datetime:
    """Parse Argo index timestamps: YYYYMMDDHHMMSS."""
    return datetime.strptime(val.strip(), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


class GDACMetadataService(AbstractMetadataService):
    """Metadata service backed by the Ifremer GDAC.

    Phase 21: Supports both bio and synthetic indexes and uses
    the RetrievalPlanner + VariableRegistry for correct routing.
    """

    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None          # bio index
        self._synthetic_df: pd.DataFrame | None = None  # synthetic index
        self._last_load: datetime | None = None
        self.planner = RetrievalPlanner()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def load(self) -> None:
        """Ensure both bio and synthetic indexes are loaded."""
        if self._is_cache_fresh():
            logger.info("Loading metadata indexes from local cache")
            self._load_from_file(_CACHE_FILE, is_synthetic=False)
            self._load_from_file(_SYNTHETIC_CACHE_FILE, is_synthetic=True)
            return

        logger.info("Downloading metadata indexes from GDAC ...")
        self._download_index()
        self._download_synthetic_index()
        self._load_from_file(_CACHE_FILE, is_synthetic=False)
        self._load_from_file(_SYNTHETIC_CACHE_FILE, is_synthetic=True)

    def search(self, criteria: SearchCriteria) -> list[MetadataRecord]:
        """Filter the in-memory index according to *criteria*."""
        if self._df is None:
            raise MetadataError("Metadata index not loaded. Call load() first.")

        # Phase 21: Planning happens once at the beginning
        plan = self.planner.plan(criteria.parameters or [])
        if plan.metadata_index == "synthetic" and self._synthetic_df is not None:
            df = self._synthetic_df
            logger.info("RetrievalPlanner selected synthetic index")
        else:
            df = self._df
            logger.info("RetrievalPlanner selected bio index")

        logger.debug("Starting search on %d rows", len(df))

        # --- Region / bounding box ---------------------------------------- #
        bounds = resolve_region(getattr(criteria, "region", None))
        if bounds:
            criteria = criteria.model_copy(update=bounds)

        if criteria.lat_min is not None:
            df = df[df["latitude"] >= criteria.lat_min]
        if criteria.lat_max is not None:
            df = df[df["latitude"] <= criteria.lat_max]
        if criteria.lon_min is not None:
            df = df[df["longitude"] >= criteria.lon_min]
        if criteria.lon_max is not None:
            df = df[df["longitude"] <= criteria.lon_max]

        # --- Polygon filter (authoritative) ----------------------------- #
        region_name = getattr(criteria, "region", None)
        if has_polygon(region_name):
            logger.debug("Applying polygon filter for region: %s", region_name)
            pre_polygon = len(df)
            mask = df.apply(
                lambda row: point_in_region(row["longitude"], row["latitude"], region_name),
                axis=1,
            )
            df = df[mask]
            logger.info(
                "Polygon filter '%s': %d / %d records retained",
                region_name,
                len(df),
                pre_polygon,
            )

        # --- Date filters ------------------------------------------------- #
        if criteria.year is not None:
            df = df[df["date"].dt.year == criteria.year]
        if criteria.month is not None:
            df = df[df["date"].dt.month == criteria.month]
        if criteria.day is not None:
            df = df[df["date"].dt.day == criteria.day]

        # --- Float ID ----------------------------------------------------- #
        if criteria.float_id is not None:
            # The file path contains the WMO id, e.g. coriolis/6903091/...
            df = df[df["file"].str.contains(f"/{criteria.float_id}/", regex=False)]

        # --- Profile number (exact cycle match) --------------------------- #
        if criteria.profile_number is not None:
            # Argo filenames use zero-padded 3-digit profile numbers:
            # BR3902490_052.nc, BD3902490_052.nc, etc.
            profile_pattern = f"_{criteria.profile_number:03d}.nc"
            df = df[df["file"].str.contains(profile_pattern, regex=False, na=False)]

        # --- Phase 21: Exact parameter matching (token-based) ------------ #
        if criteria.parameters:
            # Use the plan already computed at the top of search()
            if plan.metadata_index == "synthetic" and self._synthetic_df is not None:
                df = self._synthetic_df

            # Exact token matching (not substring)
            mask = pd.Series(True, index=df.index)
            for param in criteria.parameters:
                mask &= df["parameters"].apply(
                    lambda x: param in str(x).split() if pd.notna(x) else False
                )
            df = df[mask]

        # --- Sort & limit ------------------------------------------------- #
        df = df.sort_values("date", ascending=False).head(criteria.limit)
        logger.info("Search returned %d records", len(df))

        return [MetadataRecord(**row) for row in df.to_dict("records")]

    def is_loaded(self) -> bool:
        return self._df is not None

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    def _is_cache_fresh(self) -> bool:
        if not _CACHE_FILE.exists():
            return False
        mtime = datetime.fromtimestamp(_CACHE_FILE.stat().st_mtime, tz=timezone.utc)
        ttl = timedelta(hours=settings.metadata_cache_ttl_hours)
        return datetime.now(timezone.utc) - mtime < ttl

    def _download_index(self) -> None:
        url = f"{settings.gdac_base_url}{settings.metadata_index_path}"
        try:
            limits = httpx.Limits(
                max_connections=settings.http_max_connections,
                max_keepalive_connections=settings.http_max_keepalive,
            )
            with httpx.Client(timeout=settings.http_timeout, limits=limits) as client:
                response = client.get(url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MetadataError(
                f"Failed to download metadata index from {url}",
                details={"exception": str(exc)},
            ) from exc

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_bytes(response.content)
        logger.info("Wrote metadata cache (%d bytes)", len(response.content))

    def _download_synthetic_index(self) -> None:
        """Download the synthetic profile index."""
        synthetic_url = f"{settings.gdac_base_url}/argo_synthetic-profile_index.txt.gz"
        try:
            limits = httpx.Limits(
                max_connections=settings.http_max_connections,
                max_keepalive_connections=settings.http_max_keepalive,
            )
            with httpx.Client(timeout=settings.http_timeout, limits=limits) as client:
                response = client.get(synthetic_url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Failed to download synthetic index: %s", exc)
            return

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _SYNTHETIC_CACHE_FILE.write_bytes(response.content)
        logger.info("Wrote synthetic metadata cache (%d bytes)", len(response.content))

    def _load_from_file(self, path: Path, is_synthetic: bool = False) -> None:
        try:
            df = pd.read_csv(
                path,
                comment="#",
                compression="gzip",
                names=_INDEX_COLUMNS,
                dtype={
                    "file": "string",
                    "ocean": "string",
                    "profiler_type": "string",
                    "institution": "string",
                    "parameters": "string",
                    "parameter_data_mode": "string",
                },
                low_memory=False,
            )
        except Exception as exc:
            raise MetadataError(
                f"Failed to parse metadata index: {path}",
                details={"exception": str(exc)},
            ) from exc

        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

        for col in ("date", "date_update"):
            df[col] = pd.to_datetime(df[col], format="%Y%m%d%H%M%S", errors="coerce", utc=True)

        df = df.dropna(subset=["date", "latitude", "longitude"]).reset_index(drop=True)

        if is_synthetic:
            self._synthetic_df = df
            logger.info("Loaded synthetic metadata index: %d rows", len(df))
        else:
            self._df = df
            self._last_load = datetime.now(timezone.utc)
            logger.info("Loaded bio metadata index: %d rows", len(df))
