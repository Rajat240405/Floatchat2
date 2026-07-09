"""API response models."""

from typing import Any

from pydantic import BaseModel, Field


class MapData(BaseModel):
    """Geographic marker data for a single Argo float profile."""

    float_id: str = Field(..., description="Argo float WMO identifier.")
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    profile_date: str | None = Field(default=None, description="ISO-8601 profile date.")
    dac: str = Field(..., description="Data assembly centre code.")
    variables: list[str] = Field(default_factory=list, description="Available BGC variables.")
    selected: bool = Field(default=False, description="Whether this marker is selected.")


class ChatResponse(BaseModel):
    """Successful response from POST /chat.

    The ``figure`` field contains a Plotly JSON figure dict when a
    visualization was generated; otherwise it is ``None``.
    """

    intent: str = Field(..., description="Resolved intent type.")
    message: str = Field(..., description="Human-readable summary.")
    figure: dict[str, Any] | None = Field(
        default=None,
        description="Plotly JSON figure object.",
    )
    data_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics or metadata about the result.",
    )
    map_data: list[MapData] = Field(
        default_factory=list,
        description="Geographic markers for returned float profiles.",
    )


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str = Field(..., description="Error type code.")
    message: str = Field(..., description="Human-readable error message.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic information.",
    )
