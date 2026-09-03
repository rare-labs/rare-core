"""Run and window declustering keep one extreme per cluster."""

import numpy as np
import pytest

from evt.errors import ErrorCode, ValidationError
from evt.extract.decluster import decluster_run, decluster_window


def test_run_length_zero_splits_on_any_gap() -> None:
    values = np.array([1.0, 0.0, 5.0, 4.0, 0.0, 9.0])
    idx = np.array([0, 2, 3, 5])
    kept = decluster_run(values, idx, run_length=0, tail="high")
    np.testing.assert_array_equal(kept, [0, 2, 5])


def test_run_length_one_joins_single_non_exceedance() -> None:
    values = np.array([1.0, 0.0, 5.0, 4.0, 0.0, 0.0, 9.0])
    idx = np.array([0, 2, 3, 6])
    kept = decluster_run(values, idx, run_length=1, tail="high")
    np.testing.assert_array_equal(kept, [2, 6])


def test_run_low_tail_keeps_cluster_minimum() -> None:
    values = np.array([3.0, 1.0, 2.0, 8.0, 0.5])
    idx = np.array([0, 1, 2, 4])
    kept = decluster_run(values, idx, run_length=0, tail="low")
    np.testing.assert_array_equal(kept, [1, 4])


def test_negative_run_length() -> None:
    with pytest.raises(ValidationError) as exc:
        decluster_run(np.array([1.0]), np.array([0]), run_length=-1, tail="high")
    assert exc.value.code == ErrorCode.INVALID_SHAPE


def test_window_splits_when_gap_exceeds_window() -> None:
    values = np.array([1.0, 3.0, 2.0, 9.0])
    ts = np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-10"], dtype="datetime64[D]")
    idx = np.array([0, 1, 2, 3])
    kept = decluster_window(
        values,
        ts.astype("datetime64[ns]"),
        idx,
        window=np.timedelta64(2, "D"),
        tail="high",
    )
    np.testing.assert_array_equal(kept, [1, 3])


def test_non_positive_window() -> None:
    ts = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    with pytest.raises(ValidationError) as exc:
        decluster_window(
            np.array([1.0, 2.0]),
            ts,
            np.array([0, 1]),
            window=np.timedelta64(0, "D"),
            tail="high",
        )
    assert exc.value.code == ErrorCode.INVALID_SHAPE
