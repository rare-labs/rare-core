"""JSON round-trip for every SPEC §3 dataclass."""

import json
from dataclasses import dataclass

import numpy as np
import pytest

from evt.constants import SCHEMA_VERSION
from evt.errors import ErrorCode, SerializationError, ValidationError
from evt.serialize import from_dict, to_dict
from evt.types import (
    MODEL_TYPES,
    DiagnosticResult,
    ExtremeSample,
    ExtremeSeries,
    GEVFit,
    GPDFit,
    ReturnLevelResult,
)


def _series() -> ExtremeSeries:
    return ExtremeSeries(
        values=np.array([1.5, 2.25, 3.0], dtype=np.float64),
        timestamps=np.array(
            ["2020-01-01T00:00:00", "2020-01-02T00:00:00", "2020-01-03T00:00:00"],
            dtype="datetime64[ns]",
        ),
        tail="high",
    )


def _bm_sample() -> ExtremeSample:
    return ExtremeSample(
        raw_values=np.array([10.0, 11.0], dtype=np.float64),
        transformed_values=np.array([10.0, 11.0], dtype=np.float64),
        timestamps=None,
        method="BM",
        threshold=None,
        excesses=None,
        n_source=100,
        exposure=None,
        exceedance_rate=None,
    )


def _pot_sample() -> ExtremeSample:
    return ExtremeSample(
        raw_values=np.array([-3.0, -4.0], dtype=np.float64),
        transformed_values=np.array([3.0, 4.0], dtype=np.float64),
        timestamps=np.array(["2021-06-01", "2021-06-02"], dtype="datetime64[ns]"),
        method="POT",
        threshold=-2.0,
        excesses=np.array([1.0, 2.0], dtype=np.float64),
        n_source=50,
        exposure=2.5,
        exceedance_rate=0.8,
    )


def _gev() -> GEVFit:
    return GEVFit(
        location=1.23,
        scale=0.45,
        shape=0.1,
        log_likelihood=-12.34,
        aic=30.68,
        n_extremes=50,
        converged=True,
    )


def _gpd() -> GPDFit:
    return GPDFit(
        scale=0.8,
        shape=-0.05,
        log_likelihood=-4.0,
        aic=12.0,
        n_extremes=20,
        converged=False,
    )


def _return_level() -> ReturnLevelResult:
    return ReturnLevelResult(
        estimate=12.0,
        return_period=10.0,
        confidence_level=0.95,
        lower=9.0,
        upper=16.0,
        extrapolation_factor=10.0,
        method="GEV",
    )


def _diagnostics() -> DiagnosticResult:
    return DiagnosticResult(
        qq_empirical=np.array([0.1, 0.2], dtype=np.float64),
        qq_model=np.array([0.11, 0.19], dtype=np.float64),
        pp_empirical=np.array([0.25, 0.75], dtype=np.float64),
        pp_model=np.array([0.24, 0.76], dtype=np.float64),
        ks_distance=0.02,
    )


def _roundtrip(obj: object) -> object:
    payload = json.loads(json.dumps(to_dict(obj)))
    return from_dict(payload, type(obj))


@pytest.mark.parametrize(
    "factory",
    [_series, _bm_sample, _pot_sample, _gev, _gpd, _return_level, _diagnostics],
)
def test_json_round_trip(factory: object) -> None:
    original = factory()  # type: ignore[operator]
    restored = _roundtrip(original)
    assert to_dict(restored) == to_dict(original)


def test_every_model_type_has_round_trip_coverage() -> None:
    covered = {
        ExtremeSeries,
        ExtremeSample,
        GEVFit,
        GPDFit,
        ReturnLevelResult,
        DiagnosticResult,
    }
    assert set(MODEL_TYPES) == covered


def test_schema_version_and_type_fields() -> None:
    payload = to_dict(_gev())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["type"] == "GEVFit"
    assert payload["shape"] == 0.1


def test_timestamps_are_integer_nanoseconds() -> None:
    payload = to_dict(_series())
    stamps = payload["timestamps"]
    assert isinstance(stamps, list)
    assert all(isinstance(item, int) for item in stamps)
    restored = from_dict(payload, ExtremeSeries)
    np.testing.assert_array_equal(restored.timestamps, _series().timestamps)


def test_point_estimate_return_level_null_bounds() -> None:
    result = ReturnLevelResult(
        estimate=5.0,
        return_period=2.0,
        confidence_level=None,
        lower=None,
        upper=None,
        extrapolation_factor=2.0,
        method="GPD",
    )
    restored = _roundtrip(result)
    assert isinstance(restored, ReturnLevelResult)
    assert restored.lower is None
    assert restored.method == "GPD"


def test_schema_mismatch() -> None:
    payload = to_dict(_gev())
    payload["schema_version"] = "99.0"
    with pytest.raises(SerializationError) as exc:
        from_dict(payload, GEVFit)
    assert exc.value.code == ErrorCode.SCHEMA_MISMATCH


def test_type_mismatch() -> None:
    payload = to_dict(_gev())
    with pytest.raises(SerializationError) as exc:
        from_dict(payload, GPDFit)
    assert exc.value.code == ErrorCode.UNKNOWN_TYPE


def test_missing_field() -> None:
    payload = to_dict(_gev())
    del payload["location"]
    with pytest.raises(SerializationError) as exc:
        from_dict(payload, GEVFit)
    assert exc.value.code == ErrorCode.INVALID_PAYLOAD


def test_unknown_python_type() -> None:
    @dataclass(frozen=True)
    class Other:
        x: int

    with pytest.raises(SerializationError) as exc:
        to_dict(Other(1))
    assert exc.value.code == ErrorCode.UNKNOWN_TYPE


def test_series_without_timestamps() -> None:
    series = ExtremeSeries(values=np.array([1.0, 2.0]), timestamps=None, tail="low")
    restored = _roundtrip(series)
    assert isinstance(restored, ExtremeSeries)
    assert restored.timestamps is None
    assert restored.tail == "low"


def test_from_dict_rejects_2d_values() -> None:
    payload = to_dict(_series())
    payload["values"] = [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(SerializationError) as exc:
        from_dict(payload, ExtremeSeries)
    assert exc.value.code == ErrorCode.INVALID_PAYLOAD


def test_from_dict_rejects_empty_values() -> None:
    payload = to_dict(_series())
    payload["values"] = []
    payload["timestamps"] = None
    with pytest.raises(ValidationError) as exc:
        from_dict(payload, ExtremeSeries)
    assert exc.value.code == ErrorCode.INVALID_SHAPE


def test_from_dict_rejects_illegal_tail() -> None:
    payload = to_dict(_series())
    payload["tail"] = "sideways"
    with pytest.raises(SerializationError) as exc:
        from_dict(payload, ExtremeSeries)
    assert exc.value.code == ErrorCode.INVALID_PAYLOAD


def test_from_dict_rejects_illegal_method() -> None:
    payload = to_dict(_bm_sample())
    payload["method"] = "block"
    with pytest.raises(SerializationError) as exc:
        from_dict(payload, ExtremeSample)
    assert exc.value.code == ErrorCode.INVALID_PAYLOAD


def test_from_dict_rejects_truncated_timestamp_floats() -> None:
    payload = to_dict(_series())
    payload["timestamps"] = [1.9, 2.9, 3.9]
    with pytest.raises(SerializationError) as exc:
        from_dict(payload, ExtremeSeries)
    assert exc.value.code == ErrorCode.INVALID_PAYLOAD


def test_from_dict_series_arrays_are_not_writeable() -> None:
    restored = from_dict(to_dict(_series()), ExtremeSeries)
    assert not restored.values.flags.writeable
    assert restored.timestamps is not None
    assert not restored.timestamps.flags.writeable


def test_to_dict_rejects_0d_array() -> None:
    series = ExtremeSeries(values=np.array(1.0), timestamps=None, tail="high")
    with pytest.raises(SerializationError) as exc:
        to_dict(series)
    assert exc.value.code == ErrorCode.INVALID_PAYLOAD


def test_to_dict_rejects_complex_array() -> None:
    series = ExtremeSeries(values=np.array([1 + 2j]), timestamps=None, tail="high")
    with pytest.raises(SerializationError) as exc:
        to_dict(series)
    assert exc.value.code == ErrorCode.INVALID_PAYLOAD


def test_to_dict_rejects_non_finite_scalar() -> None:
    fit = GEVFit(
        location=float("nan"),
        scale=0.45,
        shape=0.1,
        log_likelihood=-12.34,
        aic=30.68,
        n_extremes=50,
        converged=True,
    )
    with pytest.raises(SerializationError) as exc:
        to_dict(fit)
    assert exc.value.code == ErrorCode.INVALID_PAYLOAD
