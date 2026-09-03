"""Canonical log-likelihood matches hand-checked Gumbel/exponential cases."""

import numpy as np

from evt.likelihood import GEV_N_PARAMS, GPD_N_PARAMS, aic, gev_log_likelihood, gpd_log_likelihood


def test_gumbel_log_likelihood_at_origin() -> None:
    # Gumbel (ξ=0, μ=0, σ=1) at x=0: z=0, log f = -(z + exp(-z)) = -1.
    x = np.array([0.0], dtype=np.float64)
    ll = gev_log_likelihood(x, location=0.0, scale=1.0, xi=0.0)
    np.testing.assert_allclose(ll, -1.0, atol=1e-12)


def test_exponential_gpd_log_likelihood() -> None:
    # Exponential (ξ=0, σ=1): log f(y) = -y for y >= 0. For y=(0, 1), LL = -1.
    y = np.array([0.0, 1.0], dtype=np.float64)
    ll = gpd_log_likelihood(y, scale=1.0, xi=0.0)
    np.testing.assert_allclose(ll, -1.0, atol=1e-12)


def test_aic_formula() -> None:
    assert aic(-2.0, n_params=GEV_N_PARAMS) == 10.0
    assert aic(-2.0, n_params=GPD_N_PARAMS) == 8.0
