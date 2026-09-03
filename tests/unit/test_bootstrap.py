"""Parametric bootstrap reproducibility and retry exhaustion."""

import numpy as np
import pytest

from evt.bootstrap import bootstrap_return_levels
from evt.errors import BootstrapError, ErrorCode, FitError
from evt.types import ExtremeSample, GEVFit, GPDFit


def _bm(values: np.ndarray) -> ExtremeSample:
    return ExtremeSample(
        raw_values=values,
        transformed_values=values,
        timestamps=None,
        method="BM",
        threshold=None,
        excesses=None,
        n_source=int(values.size),
        exposure=None,
        exceedance_rate=None,
    )


def _pot(excesses: np.ndarray) -> ExtremeSample:
    raw = 1.0 + excesses
    return ExtremeSample(
        raw_values=raw,
        transformed_values=raw,
        timestamps=None,
        method="POT",
        threshold=1.0,
        excesses=excesses,
        n_source=int(excesses.size) + 10,
        exposure=2.0,
        exceedance_rate=5.0,
    )


def test_same_seed_same_intervals() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=10.0, scale=2.0, size=40)
    sample = _bm(x)
    fit = GEVFit(
        location=10.0,
        scale=2.0,
        shape=0.05,
        log_likelihood=0.0,
        aic=0.0,
        n_extremes=40,
        converged=True,
    )
    a = bootstrap_return_levels(sample, fit, [10.0], n_samples=25, seed=123)
    b = bootstrap_return_levels(sample, fit, [10.0], n_samples=25, seed=123)
    assert a[0].lower == b[0].lower
    assert a[0].upper == b[0].upper
    assert a[0].confidence_level == 0.95
    assert a[0].lower < a[0].estimate < a[0].upper


def test_different_seed_can_differ() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=30) + 5.0
    sample = _bm(x)
    fit = GEVFit(
        location=5.0,
        scale=1.0,
        shape=0.0,
        log_likelihood=0.0,
        aic=0.0,
        n_extremes=30,
        converged=True,
    )
    a = bootstrap_return_levels(sample, fit, [5.0], n_samples=20, seed=1)
    b = bootstrap_return_levels(sample, fit, [5.0], n_samples=20, seed=2)
    assert (a[0].lower, a[0].upper) != (b[0].lower, b[0].upper)


def test_gpd_bootstrap_returns_ci() -> None:
    y = np.array([0.2, 0.4, 0.5, 0.8, 1.1, 1.4, 1.6, 2.0], dtype=np.float64)
    sample = _pot(y)
    fit = GPDFit(
        scale=1.0,
        shape=0.05,
        log_likelihood=0.0,
        aic=0.0,
        n_extremes=8,
        converged=True,
    )
    result = bootstrap_return_levels(sample, fit, [10.0, 50.0], n_samples=15, seed=9)
    assert len(result) == 2
    assert result[0].method == "GPD"
    assert result[0].lower < result[0].upper


def test_bootstrap_exhausted_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = _bm(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64))
    fit = GEVFit(
        location=2.0,
        scale=1.0,
        shape=0.0,
        log_likelihood=0.0,
        aic=0.0,
        n_extremes=4,
        converged=True,
    )

    def fail(_sample: ExtremeSample) -> GEVFit:
        raise FitError("forced failure", code=ErrorCode.FIT_FAILED)

    monkeypatch.setattr("evt.bootstrap.fit_gev", fail)
    with pytest.raises(BootstrapError) as exc:
        bootstrap_return_levels(sample, fit, [10.0], n_samples=1, seed=0)
    assert exc.value.code == ErrorCode.BOOTSTRAP_FAILED
