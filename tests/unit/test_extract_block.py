"""Block maxima / minima extraction."""

import numpy as np
import pytest

from evt.errors import ErrorCode, ExtractionError, ValidationError
from evt.extract import extract_block_maxima
from evt.series import make_extreme_series
from evt.validate import validate_sample


def test_high_tail_block_maxima_drops_trailing_partial_block() -> None:
    values = np.array([1.0, 5.0, 2.0, 9.0, 0.0, 3.0, 100.0])
    series = make_extreme_series(values, tail="high")
    sample = extract_block_maxima(series, block_size=3, min_blocks=2)
    validate_sample(sample)
    np.testing.assert_array_equal(sample.raw_values, [5.0, 9.0])
    np.testing.assert_array_equal(sample.transformed_values, [5.0, 9.0])
    assert sample.method == "BM"
    assert sample.n_source == 7
    assert sample.threshold is None


def test_low_tail_block_minima() -> None:
    values = np.array([1.0, 5.0, 2.0, 9.0, 0.0, 3.0])
    series = make_extreme_series(values, tail="low")
    sample = extract_block_maxima(series, block_size=3)
    validate_sample(sample)
    np.testing.assert_array_equal(sample.raw_values, [1.0, 0.0])
    np.testing.assert_array_equal(sample.transformed_values, [-1.0, 0.0])


def test_block_timestamps_follow_selected_extreme() -> None:
    values = np.array([1.0, 8.0, 2.0, 3.0, 4.0, 9.0])
    ts = np.arange("2020-01-01", "2020-01-07", dtype="datetime64[D]").astype("datetime64[ns]")
    series = make_extreme_series(values, ts, tail="high")
    sample = extract_block_maxima(series, block_size=3)
    assert sample.timestamps is not None
    np.testing.assert_array_equal(sample.timestamps, ts[[1, 5]])


def test_insufficient_blocks() -> None:
    series = make_extreme_series(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), tail="high")
    with pytest.raises(ExtractionError) as exc:
        extract_block_maxima(series, block_size=3, min_blocks=2)
    assert exc.value.code == ErrorCode.INSUFFICIENT_BLOCKS


def test_invalid_block_size() -> None:
    series = make_extreme_series(np.array([1.0, 2.0]), tail="high")
    with pytest.raises(ValidationError) as exc:
        extract_block_maxima(series, block_size=0)
    assert exc.value.code == ErrorCode.INVALID_SHAPE


def test_does_not_mutate_source_series() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    original = values.copy()
    series = make_extreme_series(values, tail="high")
    extract_block_maxima(series, block_size=2)
    np.testing.assert_array_equal(values, original)
    with pytest.raises(ValueError):
        series.values[0] = 99.0
