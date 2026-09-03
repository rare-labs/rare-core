"""Peaks over / below threshold extraction with optional declustering."""

from typing import Literal

import numpy as np

from evt.constants import DAYS_PER_YEAR
from evt.errors import ErrorCode, ExtractionError, ValidationError
from evt.extract.decluster import decluster_run, decluster_window
from evt.transform import transform_threshold, transform_to_upper_tail
from evt.types import ExtremeSample, ExtremeSeries
from evt.validate import immutable_datetime64_ns, immutable_float64, validate_series

type DeclusterMethod = Literal["run", "window"]


def _exposure_years(timestamps: np.ndarray) -> float | None:
    span_days = (timestamps[-1] - timestamps[0]) / np.timedelta64(1, "D")
    years = float(span_days) / DAYS_PER_YEAR
    if not np.isfinite(years) or years <= 0.0:
        return None
    return years


def extract_peaks_over_threshold(
    series: ExtremeSeries,
    *,
    threshold: float,
    decluster_method: DeclusterMethod | None = None,
    run_length: int = 1,
    window: np.timedelta64 | None = None,
) -> ExtremeSample:
    """Select exceedances and optionally decluster them.

    Excesses are computed on the upper-tail scale so they are non-negative for
    both high and low tails.
    """
    validate_series(series.values, series.timestamps, series.tail)
    if not np.isfinite(threshold):
        raise ValidationError("threshold must be finite", code=ErrorCode.NON_FINITE_VALUES)

    values = series.values
    mask = values > threshold if series.tail == "high" else values < threshold
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ExtractionError("no exceedances above/below threshold", code=ErrorCode.NO_EXCEEDANCES)

    if decluster_method == "run":
        indices = decluster_run(values, indices, run_length=run_length, tail=series.tail)
    elif decluster_method == "window":
        if series.timestamps is None:
            raise ValidationError(
                "window declustering requires timestamps",
                code=ErrorCode.INVALID_SHAPE,
            )
        if window is None:
            raise ValidationError(
                "window declustering requires a positive window",
                code=ErrorCode.INVALID_SHAPE,
            )
        indices = decluster_window(
            values,
            series.timestamps,
            indices,
            window=window,
            tail=series.tail,
        )
    elif decluster_method is not None:
        raise ValidationError(
            "decluster_method must be 'run', 'window', or None",
            code=ErrorCode.INVALID_SAMPLE,
        )

    raw = immutable_float64(values[indices])
    transformed = transform_to_upper_tail(raw, series.tail)
    u_star = transform_threshold(threshold, series.tail)
    excesses = immutable_float64(transformed - u_star)
    timestamps = None
    if series.timestamps is not None:
        timestamps = immutable_datetime64_ns(series.timestamps[indices])

    exposure = _exposure_years(series.timestamps) if series.timestamps is not None else None
    exceedance_rate = None if exposure is None else float(raw.size) / exposure

    return ExtremeSample(
        raw_values=raw,
        transformed_values=transformed,
        timestamps=timestamps,
        method="POT",
        threshold=float(threshold),
        excesses=excesses,
        n_source=int(values.size),
        exposure=exposure,
        exceedance_rate=exceedance_rate,
    )
