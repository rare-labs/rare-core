"""Lower-tail → upper-tail orientation. Negation is an involution."""

import numpy as np

from evt.types import ExtremeSample, Tail
from evt.validate import immutable_float64, validate_tail


def transform_to_upper_tail(values: np.ndarray, tail: str) -> np.ndarray:
    """Map observations so fitting always sees an upper tail.

    High tail is a copy (identity). Low tail is ``-x``. Never writes ``values``.
    """
    orientation = validate_tail(tail)
    owned = immutable_float64(values)
    if orientation == "low":
        flipped = np.negative(owned)
        flipped.setflags(write=False)
        return flipped
    return owned


def inverse_transform_from_upper_tail(values: np.ndarray, tail: str) -> np.ndarray:
    """Map upper-tail quantities back to the original scale."""
    return transform_to_upper_tail(values, tail)


def transform_threshold(threshold: float, tail: Tail) -> float:
    """Threshold on the same scale as ``transform_to_upper_tail``."""
    if tail == "low":
        return float(-threshold)
    return float(threshold)


def infer_tail(sample: ExtremeSample) -> Tail:
    """Recover tail from raw vs transformed values (low tail is exact negation)."""
    raw = sample.raw_values
    transformed = sample.transformed_values
    if raw.size > 0 and np.allclose(transformed, -raw) and not np.allclose(transformed, raw):
        return "low"
    return "high"


def inverse_transform_scalar(value: float, tail: str) -> float:
    mapped = inverse_transform_from_upper_tail(np.asarray([value], dtype=np.float64), tail)
    return float(mapped[0])
