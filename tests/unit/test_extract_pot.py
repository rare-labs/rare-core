"""Peaks over / below threshold extraction."""

import numpy as np
import pytest

from evt.constants import DAYS_PER_YEAR
from evt.errors import ErrorCode, ExtractionError, ValidationError
from evt.extract import extract_peaks_over_threshold
from evt.series import make_extreme_series
from evt.validate import validate_sample


def _daily(n: int, start: str = "2020-01-01") -> np.ndarray:
    begin = np.datetime64(start, "D")
    return np.arange(begin, begin + np.timedelta64(n, "D"), dtype="datetime64[D]").astype(
        "datetime64[ns]"
    )


def test_high_tail_exceedances_without_declustering() -> None:
    values = np.array([1.0, 5.0, 2.0, 6.0, 3.0])
    series = make_extreme_series(values, tail="high")
    sample = extract_peaks_over_threshold(series, threshold=4.0)
    validate_sample(sample)
    np.testing.assert_array_equal(sample.raw_values, [5.0, 6.0])
    np.testing.assert_array_equal(sample.excesses, [1.0, 2.0])
    assert sample.threshold == 4.0
    assert sample.exceedance_rate is None


def test_low_tail_excesses_are_non_negative() -> None:
    values = np.array([5.0, 1.0, 4.0, 0.5, 3.0])
    series = make_extreme_series(values, tail="low")
    sample = extract_peaks_over_threshold(series, threshold=2.0)
    validate_sample(sample)
    np.testing.assert_array_equal(sample.raw_values, [1.0, 0.5])
    np.testing.assert_array_equal(sample.transformed_values, [-1.0, -0.5])
    np.testing.assert_array_equal(sample.excesses, [1.0, 1.5])


def test_no_exceedances() -> None:
    series = make_extreme_series(np.array([1.0, 2.0, 3.0]), tail="high")
    with pytest.raises(ExtractionError) as exc:
        extract_peaks_over_threshold(series, threshold=10.0)
    assert exc.value.code == ErrorCode.NO_EXCEEDANCES


def test_run_declustering_reduces_count() -> None:
    values = np.array([0.0, 5.0, 6.0, 0.0, 7.0])
    series = make_extreme_series(values, tail="high")
    sample = extract_peaks_over_threshold(
        series, threshold=4.0, decluster_method="run", run_length=0
    )
    np.testing.assert_array_equal(sample.raw_values, [6.0, 7.0])


def test_window_declustering_requires_timestamps() -> None:
    series = make_extreme_series(np.array([1.0, 5.0, 6.0]), tail="high")
    with pytest.raises(ValidationError) as exc:
        extract_peaks_over_threshold(
            series,
            threshold=4.0,
            decluster_method="window",
            window=np.timedelta64(1, "D"),
        )
    assert exc.value.code == ErrorCode.INVALID_SHAPE


def test_window_declustering_with_timestamps() -> None:
    values = np.array([5.0, 6.0, 1.0, 9.0])
    series = make_extreme_series(values, _daily(4), tail="high")
    sample = extract_peaks_over_threshold(
        series,
        threshold=4.0,
        decluster_method="window",
        window=np.timedelta64(1, "D"),
    )
    validate_sample(sample)
    np.testing.assert_array_equal(sample.raw_values, [6.0, 9.0])


def test_exposure_and_exceedance_rate() -> None:
    values = np.array([5.0, 1.0, 6.0, 1.0])
    ts = _daily(4)
    series = make_extreme_series(values, ts, tail="high")
    sample = extract_peaks_over_threshold(series, threshold=4.0)
    span_years = 3.0 / DAYS_PER_YEAR
    assert sample.exposure is not None
    assert sample.exceedance_rate is not None
    assert sample.exposure == pytest.approx(span_years)
    assert sample.exceedance_rate == pytest.approx(2.0 / span_years)


def test_invalid_decluster_method() -> None:
    series = make_extreme_series(np.array([5.0, 6.0]), tail="high")
    with pytest.raises(ValidationError) as exc:
        extract_peaks_over_threshold(series, threshold=1.0, decluster_method="storm")  # type: ignore[arg-type]
    assert exc.value.code == ErrorCode.INVALID_SAMPLE
