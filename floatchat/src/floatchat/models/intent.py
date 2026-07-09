"""Intent model: the single typed object that crosses the NL → backend boundary."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ParsedIntent(BaseModel):
    """Structured representation of a user's natural-language request.

    The intent parser (Mock, Regex, or LLM) is responsible for producing this
    object. All downstream modules consume *only* this model.
    """

    intent: Literal[
        "profile_plot",
        "time_series",
        "comparison_plot",
        "trajectory",
        "general_chat",
        "unknown",
    ] = Field(
        default="unknown",
        description="Deterministic routing key for the visualization engine.",
    )
    region: str | None = Field(default=None, description="Named ocean region.")
    variables: list[str] = Field(
        default_factory=list,
        description="Requested BGC variables (e.g., DOXY, CHLA).",
    )
    year: int | None = Field(default=None, ge=1900, le=2100)
    month: int | None = Field(default=None, ge=1, le=12)
    day: int | None = Field(default=None, ge=1, le=31)
    lat_min: float | None = Field(default=None, ge=-90.0, le=90.0)
    lat_max: float | None = Field(default=None, ge=-90.0, le=90.0)
    lon_min: float | None = Field(default=None, ge=-180.0, le=180.0)
    lon_max: float | None = Field(default=None, ge=-180.0, le=180.0)
    depth_min: float | None = Field(default=None, ge=0)
    depth_max: float | None = Field(default=None, ge=0)
    float_id: str | None = Field(
        default=None,
        description="Argo float WMO identifier.",
    )
    profile_number: int | None = Field(
        default=None,
        ge=1,
        description="Specific profile/cycle number to retrieve.",
    )
    cycle_number: int | None = Field(default=None, ge=1)
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of profiles to retrieve.",
    )

    @field_validator("variables", mode="before")
    @classmethod
    def _uppercase_variables(cls, v: list[str]) -> list[str]:
        """Normalise variable names to uppercase Argo conventions."""
        if isinstance(v, list):
            return [str(item).strip().upper() for item in v]
        return v

    @field_validator("region")
    @classmethod
    def _lowercase_region(cls, v: str | None) -> str | None:
        if v:
            return v.strip().lower().replace(" ", "_")
        return v
