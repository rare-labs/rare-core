"""QQ/PP coordinates and KS distance."""

import numpy as np
import pytest

from evt.diagnostics import diagnostic_qq_pp
from evt.return_levels import gev_cdf, gev_quantile
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


def test_qq_matches_model_quantile_at_plotting_positions() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    fit = GEVFit(
        location=0.0,
        scale=1.0,
        shape=0.0,
        log_likelihood=0.0,
        aic=0.0,
        n_extremes=4,
        converged=True,
    )
    diag = diagnostic_qq_pp(_bm(x), fit)
    n = 4
    p = (np.arange(1, n + 1) - 0.5) / n
    np.testing.assert_allclose(diag.qq_empirical, np.sort(x))
    np.testing.assert_allclose(diag.qq_model, gev_quantile(p, 0.0, 1.0, 0.0))
    np.testing.assert_allclose(diag.pp_model, gev_cdf(np.sort(x), 0.0, 1.0, 0.0))
    i = np.arange(1, n + 1)
    expected_ks = np.max(np.abs(i / n - gev_cdf(np.sort(x), 0.0, 1.0, 0.0)))
    assert diag.ks_distance == pytest.approx(float(expected_ks))


def test_gpd_diagnostics_use_excesses() -> None:
    y = np.array([0.1, 0.4, 0.8, 1.5], dtype=np.float64)
    sample = ExtremeSample(
        raw_values=1.0 + y,
        transformed_values=1.0 + y,
        timestamps=None,
        method="POT",
        threshold=1.0,
        excesses=y,
        n_source=10,
        exposure=1.0,
        exceedance_rate=4.0,
    )
    fit = GPDFit(
        scale=1.0,
        shape=0.0,
        log_likelihood=0.0,
        aic=0.0,
        n_extremes=4,
        converged=True,
    )
    diag = diagnostic_qq_pp(sample, fit)
    np.testing.assert_array_equal(diag.qq_empirical, np.sort(y))
    assert diag.ks_distance >= 0.0
    assert diag.pp_empirical.shape == diag.pp_model.shape
