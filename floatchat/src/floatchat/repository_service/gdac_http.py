"""GDAC HTTP repository implementation.

Streams NetCDF files directly into RAM without writing to disk.
Uses connection pooling and exponential backoff retries.
"""

import logging
from time import sleep

import httpx
from netCDF4 import Dataset

from floatchat.config import settings
from floatchat.exceptions import RepositoryError
from floatchat.repository_service.base import AbstractRepositoryService
from floatchat.repository_service.dataset_wrapper import NetCDFDataset

logger = logging.getLogger(__name__)


class GDACRepositoryService(AbstractRepositoryService):
    """Fetch profiles from ``https://data-argo.ifremer.fr/dac/``."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        if client is not None:
            self._client = client
        else:
            limits = httpx.Limits(
                max_connections=settings.http_max_connections,
                max_keepalive_connections=settings.http_max_keepalive,
            )
            self._client = httpx.Client(timeout=settings.http_timeout, limits=limits)

    def fetch(self, relative_path: str) -> NetCDFDataset:
        """Download *relative_path* with retries and open it as an in-memory NetCDF dataset."""
        url = f"{settings.gdac_base_url}/dac/{relative_path}"
        logger.debug("Fetching %s", url)

        last_exc: Exception | None = None
        for attempt in range(1, settings.http_max_retries + 1):
            try:
                response = self._client.get(url)
                response.raise_for_status()
                break
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_exc = exc
                if attempt < settings.http_max_retries:
                    backoff = min(2 ** attempt, 30)
                    logger.warning(
                        "Fetch attempt %d/%d failed for %s, retrying in %.1fs: %s",
                        attempt,
                        settings.http_max_retries,
                        relative_path,
                        backoff,
                        exc,
                    )
                    sleep(backoff)
                else:
                    if isinstance(exc, httpx.HTTPStatusError):
                        raise RepositoryError(
                            f"GDAC returned {exc.response.status_code} for {url}",
                            details={
                                "url": url,
                                "status_code": exc.response.status_code,
                            },
                        ) from exc
                    raise RepositoryError(
                        f"Network error fetching {url}",
                        details={"url": url, "exception": str(exc)},
                    ) from exc

        data = response.content
        logger.info(
            "Fetched %s (%d bytes, content-type=%s)",
            relative_path,
            len(data),
            response.headers.get("content-type", "unknown"),
        )

        safe_name = relative_path.replace("/", "-")
        try:
            dataset = Dataset(
                filename=f"in-memory-{safe_name}",
                memory=data,
                mode="r",
                format="NETCDF4",
            )
        except Exception as exc:
            raise RepositoryError(
                f"Failed to open NetCDF dataset from memory: {relative_path}",
                details={"exception": str(exc)},
            ) from exc

        return NetCDFDataset(relative_path, data, dataset)
