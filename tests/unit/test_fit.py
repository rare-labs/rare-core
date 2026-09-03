"""GEV/GPD MLE orchestration uses canonical likelihood and sample-size guards."""

import numpy as np
import pytest
from scipy.stats import genextreme, genpareto

from evt.errors import ErrorCode, FitError, ValidationError
from evt.fit import fit_gev, fit_gpd
from evt.likelihood import GEV_N_PARAMS, GPD_N_PARAMS, aic, gev_log_likelihood, gpd_log_likelihood
from evt.types import ExtremeSample


def _bm_sample(values: np.ndarray) -> ExtremeSample:
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


def _pot_sample(excesses: np.ndarray) -> ExtremeSample:
    raw = excesses + 1.0
    return ExtremeSample(
        raw_values=raw,
        transformed_values=raw,
        timestamps=None,
        method="POT",
        threshold=1.0,
        excesses=excesses,
        n_source=int(excesses.size) + 5,
        exposure=None,
        exceedance_rate=None,
    )


def test_fit_gev_finite_params_and_canonical_ll() -> None:
    rng = np.random.default_rng(42)
    x = np.asarray(genextreme.rvs(c=-0.1, loc=10.0, scale=2.0, size=80, random_state=rng))
    fit = fit_gev(_bm_sample(x))
    assert fit.converged
    assert np.isfinite(fit.location)
    assert fit.scale > 0
    assert np.isfinite(fit.shape)
    expected_ll = gev_log_likelihood(x, fit.location, fit.scale, fit.shape)
    assert fit.log_likelihood == expected_ll
    assert fit.aic == aic(expected_ll, n_params=GEV_N_PARAMS)
    assert fit.n_extremes == 80


def test_fit_gpd_finite_params_and_canonical_ll() -> None:
    rng = np.random.default_rng(7)
    y = np.asarray(genpareto.rvs(c=0.05, loc=0.0, scale=1.5, size=60, random_state=rng))
    fit = fit_gpd(_pot_sample(y))
    assert fit.converged
    assert fit.scale > 0
    assert np.isfinite(fit.shape)
    expected_ll = gpd_log_likelihood(y, fit.scale, fit.shape)
    assert fit.log_likelihood == expected_ll
    assert fit.aic == aic(expected_ll, n_params=GPD_N_PARAMS)
    assert fit.n_extremes == 60


def test_fit_gev_rejects_small_sample() -> None:
    sample = _bm_sample(np.array([1.0, 2.0], dtype=np.float64))
    with pytest.raises(FitError) as exc:
        fit_gev(sample)
    assert exc.value.code == ErrorCode.INSUFFICIENT_SAMPLE


def test_fit_gpd_rejects_small_sample() -> None:
    sample = _pot_sample(np.array([0.5], dtype=np.float64))
    with pytest.raises(FitError) as exc:
        fit_gpd(sample)
    assert exc.value.code == ErrorCode.INSUFFICIENT_SAMPLE


def test_fit_gpd_requires_excesses() -> None:
    sample = _bm_sample(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    with pytest.raises(ValidationError) as exc:
        fit_gpd(sample)
    assert exc.value.code == ErrorCode.MISSING_POT_METADATA


def test_fit_does_not_mutate_sample() -> None:
    x = np.array([1.0, 2.0, 3.5, 4.0, 5.5], dtype=np.float64)
    sample = _bm_sample(x)
    original = sample.transformed_values.copy()
    fit_gev(sample)
    np.testing.assert_array_equal(sample.transformed_values, original)
