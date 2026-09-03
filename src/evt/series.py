"""Validated ExtremeSeries construction. Copies input; never mutates the caller."""

from numpy.typing import ArrayLike

from evt.types import ExtremeSeries
from evt.validate import (
    immutable_datetime64_ns,
    immutable_float64,
    validate_series,
    validate_tail,
)


def make_extreme_series(
    values: ArrayLike,
    timestamps: ArrayLike | None = None,
    tail: str = "high",
) -> ExtremeSeries:
    """Build a validated series with owned, write-protected arrays."""
    validate_series(values, timestamps, tail)
    copied_values = immutable_float64(values)
    copied_ts = None if timestamps is None else immutable_datetime64_ns(timestamps)
    return ExtremeSeries(values=copied_values, timestamps=copied_ts, tail=validate_tail(tail))
