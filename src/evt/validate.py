"""Input contract checks. Never mutate caller arrays (SPEC §2)."""

from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike

from evt.constants import GEV_MIN_SAMPLE, GPD_MIN_SAMPLE
from evt.errors import ErrorCode, FitError, ValidationError
from evt.types import ExtremeSample, Tail

_TAILS: frozenset[str] = frozenset({"high", "low"})


def _as_float64_1d(values: ArrayLike, *, name: str) -> np.ndarray:
    """Copy/coerce to float64 for inspection only; does not write back to ``values``."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{name} must be a numeric array",
            code=ErrorCode.INVALID_SHAPE,
        ) from exc
    if array.ndim != 1 or array.size == 0:
        raise ValidationError(
            f"{name} must be a non-empty 1-D array",
            code=ErrorCode.INVALID_SHAPE,
        )
    if not np.all(np.isfinite(array)):
        raise ValidationError(
            f"{name} contains non-finite values",
            code=ErrorCode.NON_FINITE_VALUES,
        )
    return array


def _as_datetime64_ns_1d(timestamps: ArrayLike, *, name: str = "timestamps") -> np.ndarray:
    array = np.asarray(timestamps)
    if array.dtype.kind != "M":
        raise ValidationError(
            f"{name} must have datetime64 dtype",
            code=ErrorCode.INVALID_SHAPE,
        )
    if array.ndim != 1:
        raise ValidationError(
            f"{name} must be a 1-D array",
            code=ErrorCode.INVALID_SHAPE,
        )
    converted = array.astype("datetime64[ns]", copy=False)
    if np.any(np.isnat(converted)):
        raise ValidationError(
            f"{name} contains non-finite values",
            code=ErrorCode.NON_FINITE_VALUES,
        )
    return converted


def _check_timestamps(timestamps: ArrayLike, n_values: int) -> None:
    ts = _as_datetime64_ns_1d(timestamps)
    if ts.size != n_values:
        raise ValidationError(
            "timestamps length must match values",
            code=ErrorCode.TIMESTAMP_LENGTH_MISMATCH,
        )
    if ts.size > 0 and np.unique(ts).size < ts.size:
        raise ValidationError(
            "timestamps contain duplicates",
            code=ErrorCode.DUPLICATE_TIMESTAMPS,
        )
    if ts.size > 1 and not bool(np.all(ts[1:] > ts[:-1])):
        raise ValidationError(
            "timestamps must be strictly increasing",
            code=ErrorCode.UNORDERED_TIMESTAMPS,
        )


def validate_tail(tail: str) -> Tail:
    if tail not in _TAILS:
        raise ValidationError(
            "tail must be 'high' or 'low'",
            code=ErrorCode.INVALID_TAIL,
        )
    return cast(Tail, tail)


def validate_series(
    values: ArrayLike,
    timestamps: ArrayLike | None,
    tail: str,
) -> None:
    """Validate an input series. Raises ``ValidationError`` with a stable code."""
    validate_tail(tail)
    array = _as_float64_1d(values, name="values")
    if timestamps is not None:
        _check_timestamps(timestamps, array.size)


def validate_sample(sample: ExtremeSample) -> None:
    """Enforce ExtremeSample invariants (SPEC §3.2)."""
    raw = _as_float64_1d(sample.raw_values, name="raw_values")
    transformed = _as_float64_1d(sample.transformed_values, name="transformed_values")
    if raw.size != transformed.size:
        raise ValidationError(
            "raw_values and transformed_values must have the same length",
            code=ErrorCode.INVALID_SAMPLE,
        )
    if sample.n_source < raw.size:
        raise ValidationError(
            "n_source must be >= the number of extremes",
            code=ErrorCode.INVALID_SAMPLE,
        )
    if sample.timestamps is not None:
        _check_timestamps(sample.timestamps, raw.size)
    if sample.exposure is not None and (not np.isfinite(sample.exposure) or sample.exposure <= 0):
        raise ValidationError(
            "exposure must be finite and positive when provided",
            code=ErrorCode.INVALID_SAMPLE,
        )
    if sample.exceedance_rate is not None and (
        not np.isfinite(sample.exceedance_rate) or sample.exceedance_rate < 0
    ):
        raise ValidationError(
            "exceedance_rate must be finite and non-negative when provided",
            code=ErrorCode.INVALID_SAMPLE,
        )

    if sample.method == "BM":
        if (
            sample.threshold is not None
            or sample.excesses is not None
            or sample.exceedance_rate is not None
        ):
            raise ValidationError(
                "BM samples must not set threshold, excesses, or exceedance_rate",
                code=ErrorCode.INVALID_SAMPLE,
            )
        return

    if sample.method != "POT":
        raise ValidationError(
            "method must be 'BM' or 'POT'",
            code=ErrorCode.INVALID_SAMPLE,
        )
    if sample.threshold is None or sample.excesses is None:
        raise ValidationError(
            "POT samples require threshold and excesses",
            code=ErrorCode.MISSING_POT_METADATA,
        )
    if not np.isfinite(sample.threshold):
        raise ValidationError(
            "threshold must be finite",
            code=ErrorCode.NON_FINITE_VALUES,
        )
    excesses = _as_float64_1d(sample.excesses, name="excesses")
    if excesses.size != raw.size:
        raise ValidationError(
            "excesses length must match raw_values",
            code=ErrorCode.INVALID_SAMPLE,
        )
    if np.any(excesses < 0):
        raise ValidationError(
            "excesses must be non-negative",
            code=ErrorCode.INVALID_SAMPLE,
        )


def validate_pot_return_level_metadata(sample: ExtremeSample) -> None:
    """POT return levels require original-scale threshold and exceedance rate λ."""
    if sample.method != "POT" or sample.threshold is None or sample.exceedance_rate is None:
        raise ValidationError(
            "GPD return levels require POT threshold and exceedance_rate",
            code=ErrorCode.MISSING_POT_METADATA,
        )


def validate_fit_sample_size(n: int, *, model: Literal["GEV", "GPD"]) -> None:
    minimum = GEV_MIN_SAMPLE if model == "GEV" else GPD_MIN_SAMPLE
    if n < minimum:
        raise FitError(
            f"{model} fit requires n >= {minimum}",
            code=ErrorCode.INSUFFICIENT_SAMPLE,
        )


def immutable_float64(values: ArrayLike) -> np.ndarray:
    """Owned float64 copy that cannot be written (does not mutate ``values``)."""
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def immutable_datetime64_ns(timestamps: ArrayLike) -> np.ndarray:
    """Owned datetime64[ns] copy that cannot be written."""
    array = np.asarray(timestamps).astype("datetime64[ns]", copy=True)
    array.setflags(write=False)
    return array
