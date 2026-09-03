"""Tail orientation copies input and is an involution."""

import numpy as np
import pytest

from evt.errors import ErrorCode, ValidationError
from evt.transform import inverse_transform_from_upper_tail, transform_to_upper_tail
from evt.validate import immutable_float64


def test_high_tail_is_identity_copy() -> None:
    values = np.array([1.0, -2.0, 3.5], dtype=np.float64)
    original = values.copy()
    out = transform_to_upper_tail(values, "high")
    np.testing.assert_array_equal(out, original)
    np.testing.assert_array_equal(values, original)
    assert not out.flags.writeable


def test_low_tail_negates() -> None:
    values = np.array([1.0, -2.0, 3.5], dtype=np.float64)
    original = values.copy()
    out = transform_to_upper_tail(values, "low")
    np.testing.assert_array_equal(out, -original)
    np.testing.assert_array_equal(values, original)


def test_inverse_is_involution() -> None:
    values = immutable_float64([0.5, 1.5, 4.0])
    for tail in ("high", "low"):
        mapped = transform_to_upper_tail(values, tail)
        restored = inverse_transform_from_upper_tail(mapped, tail)
        np.testing.assert_array_equal(restored, values)


def test_invalid_tail() -> None:
    with pytest.raises(ValidationError) as exc:
        transform_to_upper_tail(np.array([1.0]), "side")
    assert exc.value.code == ErrorCode.INVALID_TAIL
