"""Validation contracts and stable SPEC §7 error codes."""

import numpy as np
import pytest

from evt.constants import GEV_MIN_SAMPLE, GPD_MIN_SAMPLE
from evt.errors import (
    BootstrapError,
    ErrorCode,
    ExtractionError,
    FitError,
    ValidationError,
)
from evt.types import ExtremeSample
from evt.validate import (
    validate_fit_sample_size,
    validate_pot_return_level_metadata,
    validate_sample,
    validate_series,
)


def _ts(hours: list[int]) -> np.ndarray:
    return np.array(hours, dtype="datetime64[h]").astype("datetime64[ns]")


def _bm_sample(*, n_source: int = 10) -> ExtremeSample:
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    return ExtremeSample(
        raw_values=values,
        transformed_values=values.copy(),
        timestamps=None,
        method="BM",
        threshold=None,
        excesses=None,
        n_source=n_source,
        exposure=None,
        exceedance_rate=None,
    )


def _pot_sample() -> ExtremeSample:
    values = np.array([5.0, 6.0], dtype=np.float64)
    return ExtremeSample(
        raw_values=values,
        transformed_values=values.copy(),
        timestamps=_ts([0, 1]),
        method="POT",
        threshold=4.0,
        excesses=np.array([1.0, 2.0], dtype=np.float64),
        n_source=20,
        exposure=1.0,
        exceedance_rate=2.0,
    )


def test_validate_series_accepts_valid_high_tail() -> None:
    validate_series(np.array([1.0, 2.0]), None, "high")


def test_validate_series_accepts_matching_timestamps() -> None:
    validate_series([1.0, 2.0, 3.0], _ts([0, 1, 2]), "low")


def test_non_finite_values() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_series([1.0, np.nan], None, "high")
    assert exc.value.code == ErrorCode.NON_FINITE_VALUES


def test_infinite_values() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_series([1.0, np.inf], None, "high")
    assert exc.value.code == ErrorCode.NON_FINITE_VALUES


def test_invalid_shape_2d() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_series(np.array([[1.0, 2.0]]), None, "high")
    assert exc.value.code == ErrorCode.INVALID_SHAPE


def test_invalid_shape_empty() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_series(np.array([], dtype=np.float64), None, "high")
    assert exc.value.code == ErrorCode.INVALID_SHAPE


def test_invalid_shape_non_numeric() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_series(["a", "b"], None, "high")
    assert exc.value.code == ErrorCode.INVALID_SHAPE


def test_invalid_tail() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_series([1.0], None, "both")
    assert exc.value.code == ErrorCode.INVALID_TAIL


def test_timestamp_length_mismatch() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_series([1.0, 2.0], _ts([0]), "high")
    assert exc.value.code == ErrorCode.TIMESTAMP_LENGTH_MISMATCH


def test_duplicate_timestamps() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_series([1.0, 2.0], _ts([1, 1]), "high")
    assert exc.value.code == ErrorCode.DUPLICATE_TIMESTAMPS


def test_unordered_timestamps() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_series([1.0, 2.0, 3.0], _ts([0, 2, 1]), "high")
    assert exc.value.code == ErrorCode.UNORDERED_TIMESTAMPS


def test_non_finite_timestamps() -> None:
    ts = np.array(["2020-01-01", "NaT"], dtype="datetime64[ns]")
    with pytest.raises(ValidationError) as exc:
        validate_series([1.0, 2.0], ts, "high")
    assert exc.value.code == ErrorCode.NON_FINITE_VALUES


def test_integer_timestamps_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_series([1.0, 2.0], np.array([1, 2]), "high")
    assert exc.value.code == ErrorCode.INVALID_SHAPE


def test_validate_series_does_not_mutate_input() -> None:
    values = np.array([1.0, 2.0], dtype=np.float64)
    original = values.copy()
    validate_series(values, None, "high")
    np.testing.assert_array_equal(values, original)


def test_bm_sample_ok() -> None:
    validate_sample(_bm_sample())


def test_pot_sample_ok() -> None:
    validate_sample(_pot_sample())


def test_sample_length_mismatch() -> None:
    sample = ExtremeSample(
        raw_values=np.array([1.0, 2.0]),
        transformed_values=np.array([1.0]),
        timestamps=None,
        method="BM",
        threshold=None,
        excesses=None,
        n_source=2,
        exposure=None,
        exceedance_rate=None,
    )
    with pytest.raises(ValidationError) as exc:
        validate_sample(sample)
    assert exc.value.code == ErrorCode.INVALID_SAMPLE


def test_n_source_too_small() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_sample(_bm_sample(n_source=1))
    assert exc.value.code == ErrorCode.INVALID_SAMPLE


def test_bm_rejects_pot_fields() -> None:
    sample = ExtremeSample(
        raw_values=np.array([1.0]),
        transformed_values=np.array([1.0]),
        timestamps=None,
        method="BM",
        threshold=0.0,
        excesses=None,
        n_source=1,
        exposure=None,
        exceedance_rate=None,
    )
    with pytest.raises(ValidationError) as exc:
        validate_sample(sample)
    assert exc.value.code == ErrorCode.INVALID_SAMPLE


def test_pot_missing_excesses() -> None:
    sample = ExtremeSample(
        raw_values=np.array([1.0]),
        transformed_values=np.array([1.0]),
        timestamps=None,
        method="POT",
        threshold=0.0,
        excesses=None,
        n_source=1,
        exposure=None,
        exceedance_rate=None,
    )
    with pytest.raises(ValidationError) as exc:
        validate_sample(sample)
    assert exc.value.code == ErrorCode.MISSING_POT_METADATA


def test_negative_excesses() -> None:
    sample = ExtremeSample(
        raw_values=np.array([1.0]),
        transformed_values=np.array([1.0]),
        timestamps=None,
        method="POT",
        threshold=0.0,
        excesses=np.array([-0.1]),
        n_source=1,
        exposure=None,
        exceedance_rate=None,
    )
    with pytest.raises(ValidationError) as exc:
        validate_sample(sample)
    assert exc.value.code == ErrorCode.INVALID_SAMPLE


def test_missing_pot_return_level_metadata() -> None:
    with pytest.raises(ValidationError) as exc:
        validate_pot_return_level_metadata(_bm_sample())
    assert exc.value.code == ErrorCode.MISSING_POT_METADATA


def test_pot_return_level_metadata_ok() -> None:
    validate_pot_return_level_metadata(_pot_sample())


@pytest.mark.parametrize(
    ("model", "n"),
    [("GEV", GEV_MIN_SAMPLE - 1), ("GPD", GPD_MIN_SAMPLE - 1)],
)
def test_insufficient_sample(model: str, n: int) -> None:
    with pytest.raises(FitError) as exc:
        validate_fit_sample_size(n, model=model)  # type: ignore[arg-type]
    assert exc.value.code == ErrorCode.INSUFFICIENT_SAMPLE


def test_fit_sample_size_ok() -> None:
    validate_fit_sample_size(GEV_MIN_SAMPLE, model="GEV")
    validate_fit_sample_size(GPD_MIN_SAMPLE, model="GPD")


def test_extraction_error_codes() -> None:
    err = ExtractionError("too few blocks", code=ErrorCode.INSUFFICIENT_BLOCKS)
    assert err.code == ErrorCode.INSUFFICIENT_BLOCKS
    err = ExtractionError("no peaks", code=ErrorCode.NO_EXCEEDANCES)
    assert err.code == ErrorCode.NO_EXCEEDANCES


def test_fit_failed_code() -> None:
    err = FitError("optimizer failed", code=ErrorCode.FIT_FAILED)
    assert err.code == ErrorCode.FIT_FAILED


def test_bootstrap_failed_code() -> None:
    err = BootstrapError("retries exhausted", code=ErrorCode.BOOTSTRAP_FAILED)
    assert err.code == ErrorCode.BOOTSTRAP_FAILED


def test_spec_error_codes_are_stable_strings() -> None:
    expected = {
        "NON_FINITE_VALUES",
        "INVALID_SHAPE",
        "TIMESTAMP_LENGTH_MISMATCH",
        "DUPLICATE_TIMESTAMPS",
        "UNORDERED_TIMESTAMPS",
        "INSUFFICIENT_BLOCKS",
        "NO_EXCEEDANCES",
        "MISSING_POT_METADATA",
        "FIT_FAILED",
        "INSUFFICIENT_SAMPLE",
        "BOOTSTRAP_FAILED",
    }
    assert expected <= {member.value for member in ErrorCode}
