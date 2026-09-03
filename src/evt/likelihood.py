"""Canonical GEV/GPD log-likelihood. AIC and validation must use these functions.

Log-density is evaluated with SciPy ``logpdf`` under the engine ξ convention.
This module must not call SciPy ``fit``.
"""

import numpy as np
from scipy.stats import genextreme, genpareto

GEV_N_PARAMS = 3
GPD_N_PARAMS = 2


def gev_logpdf(x: np.ndarray, location: float, scale: float, xi: float) -> np.ndarray:
    return np.asarray(genextreme.logpdf(x, c=-xi, loc=location, scale=scale), dtype=np.float64)


def gpd_logpdf(excesses: np.ndarray, scale: float, xi: float) -> np.ndarray:
    return np.asarray(genpareto.logpdf(excesses, c=xi, loc=0.0, scale=scale), dtype=np.float64)


def gev_log_likelihood(x: np.ndarray, location: float, scale: float, xi: float) -> float:
    return float(np.sum(gev_logpdf(x, location, scale, xi)))


def gpd_log_likelihood(excesses: np.ndarray, scale: float, xi: float) -> float:
    return float(np.sum(gpd_logpdf(excesses, scale, xi)))


def aic(log_likelihood: float, *, n_params: int) -> float:
    return float(2 * n_params - 2 * log_likelihood)
