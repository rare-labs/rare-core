"""Inverse-CDF sampling independent of fitting (SPEC §6.3).

Formulas match ``evt.return_levels`` quantiles but must not import ``evt.fit``.
"""

import numpy as np


def gev_quantile(p: np.ndarray, location: float, scale: float, xi: float) -> np.ndarray:
    prob = np.asarray(p, dtype=np.float64)
    nlog = -np.log(prob)
    if xi == 0.0:
        return location - scale * np.log(nlog)
    return location + (scale / xi) * (np.power(nlog, -xi) - 1.0)


def gpd_excess_quantile(p: np.ndarray, scale: float, xi: float) -> np.ndarray:
    prob = np.asarray(p, dtype=np.float64)
    survival = 1.0 - prob
    if xi == 0.0:
        return -scale * np.log(survival)
    return (scale / xi) * (np.power(survival, -xi) - 1.0)


def uniform_open(rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw probabilities in (0, 1) so GEV/GPD quantiles stay finite."""
    return rng.uniform(1e-12, 1.0 - 1e-12, size=n)
