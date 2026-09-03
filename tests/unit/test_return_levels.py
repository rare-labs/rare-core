"""Analytic GEV/GPD return levels, including ξ→0 branches and low-tail mapping."""

import numpy as np
import pytest

from evt.errors import ErrorCode, ValidationError
from evt.return_levels import gev_return_level, gpd_return_level, xi_is_zero
from evt.types import ExtremeSample, GEVFit, GPDFit


def _gev(*, location: float, scale: float, shape: float) -> GEVFit:
    return GEVFit(
        location=location,
        scale=scale,
        shape=shape,
        log_likelihood=0.0,
        aic=0.0,
        n_extremes=10,
        converged=True,
    )


def _pot(
    *,
    threshold: float,
    rate: float,
    raw: np.ndarray,
    transformed: np.ndarray,
) -> ExtremeSample:
    return ExtremeSample(
        raw_values=raw,
        transformed_values=transformed,
        timestamps=None,
        method="POT",
        threshold=threshold,
        excesses=np.abs(raw - threshold),
        n_source=20,
        exposure=1.0,
        exceedance_rate=rate,
    )


def _gpd(*, scale: float, shape: float) -> GPDFit:
    return GPDFit(
        scale=scale,
        shape=shape,
        log_likelihood=0.0,
        aic=0.0,
        n_extremes=10,
        converged=True,
    )


@pytest.mark.parametrize("period", [2.0, 10.0, 50.0])
def test_gev_return_level_matches_formula(period: float) -> None:
    mu, sigma, xi = 5.0, 2.0, 0.2
    p = 1.0 - 1.0 / period
    expected = mu + (sigma / xi) * ((-np.log(p)) ** (-xi) - 1.0)
    result = gev_return_level(_gev(location=mu, scale=sigma, shape=xi), period)
    assert result.estimate == pytest.approx(expected)
    assert result.extrapolation_factor == pytest.approx(period)
    assert result.method == "GEV"


def test_gev_gumbel_limit_and_near_zero_xi() -> None:
    mu, sigma = 0.0, 1.5
    p = 1.0 - 1.0 / 10.0
    gumbel = mu - sigma * np.log(-np.log(p))
    exact = gev_return_level(_gev(location=mu, scale=sigma, shape=0.0), 10.0)
    near = gev_return_level(_gev(location=mu, scale=sigma, shape=1e-12), 10.0)
    assert xi_is_zero(1e-12)
    assert exact.estimate == pytest.approx(gumbel)
    assert near.estimate == pytest.approx(gumbel, rel=1e-6)


def test_gev_low_tail_negates() -> None:
    high = gev_return_level(_gev(location=1.0, scale=1.0, shape=0.0), 10.0, tail="high")
    low = gev_return_level(_gev(location=1.0, scale=1.0, shape=0.0), 10.0, tail="low")
    assert low.estimate == pytest.approx(-high.estimate)


@pytest.mark.parametrize("period", [2.0, 10.0, 50.0])
def test_gpd_return_level_matches_formula(period: float) -> None:
    u, sigma, xi, lam = 10.0, 2.0, 0.1, 3.0
    m = lam * period
    expected = u + (sigma / xi) * (m**xi - 1.0)
    raw = np.array([11.0, 12.0], dtype=np.float64)
    sample = _pot(threshold=u, rate=lam, raw=raw, transformed=raw)
    result = gpd_return_level(_gpd(scale=sigma, shape=xi), sample, period)
    assert result.estimate == pytest.approx(expected)
    assert result.extrapolation_factor == pytest.approx(m)


def test_gpd_exponential_limit() -> None:
    u, sigma, lam, period = 4.0, 1.25, 2.0, 10.0
    m = lam * period
    expected = u + sigma * np.log(m)
    raw = np.array([5.0, 6.0], dtype=np.float64)
    sample = _pot(threshold=u, rate=lam, raw=raw, transformed=raw)
    result = gpd_return_level(_gpd(scale=sigma, shape=0.0), sample, period)
    assert result.estimate == pytest.approx(expected)


def test_gpd_low_tail_maps_back() -> None:
    u, sigma, lam, period = -2.0, 1.0, 1.5, 10.0
    m = lam * period
    z_star = 2.0 + sigma * np.log(m)
    expected = -z_star
    raw = np.array([-3.0, -4.0], dtype=np.float64)
    sample = _pot(threshold=u, rate=lam, raw=raw, transformed=-raw)
    result = gpd_return_level(_gpd(scale=sigma, shape=0.0), sample, period)
    assert result.estimate == pytest.approx(expected)


def test_gev_rejects_period_times_blocks_leq_one() -> None:
    with pytest.raises(ValidationError) as exc:
        gev_return_level(_gev(location=0.0, scale=1.0, shape=0.0), 1.0)
    assert exc.value.code == ErrorCode.INVALID_SHAPE


def test_gpd_missing_metadata() -> None:
    sample = ExtremeSample(
        raw_values=np.array([1.0, 2.0]),
        transformed_values=np.array([1.0, 2.0]),
        timestamps=None,
        method="BM",
        threshold=None,
        excesses=None,
        n_source=2,
        exposure=None,
        exceedance_rate=None,
    )
    with pytest.raises(ValidationError) as exc:
        gpd_return_level(_gpd(scale=1.0, shape=0.0), sample, 10.0)
    assert exc.value.code == ErrorCode.MISSING_POT_METADATA
