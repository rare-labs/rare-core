"""SciPy parameter estimation. The only module allowed to call distribution ``fit``.

Engine convention is ξ (xi). SciPy GEV shape ``c`` is translated here as ``xi = -c``.
SciPy GPD shape ``c`` matches ξ.
"""

import numpy as np
from scipy.stats import genextreme, genpareto

from evt.errors import ErrorCode, FitError


def gev_params_from_scipy(c: float, loc: float, scale: float) -> tuple[float, float, float]:
    """Convert SciPy ``genextreme`` ``(c, loc, scale)`` to ``(location, scale, xi)``."""
    return float(loc), float(scale), float(-c)


def gpd_params_from_scipy(c: float, loc: float, scale: float) -> tuple[float, float]:
    """Convert SciPy ``genpareto`` ``(c, loc, scale)`` to ``(scale, xi)``.

    ``loc`` is unused because GPD location is fixed at 0.
    """
    _ = loc
    return float(scale), float(c)


def _converged(location: float, scale: float, shape: float) -> bool:
    return bool(scale > 0.0 and np.isfinite(location) and np.isfinite(scale) and np.isfinite(shape))


def gev_fit_scipy(x: np.ndarray) -> tuple[float, float, float, bool]:
    """MLE via ``scipy.stats.genextreme.fit``. Returns ``(location, scale, xi, converged)``."""
    try:
        c, loc, scale = genextreme.fit(x)
    except Exception as exc:
        raise FitError("SciPy GEV MLE failed", code=ErrorCode.FIT_FAILED) from exc
    location, scale_f, xi = gev_params_from_scipy(float(c), float(loc), float(scale))
    return location, scale_f, xi, _converged(location, scale_f, xi)


def gpd_fit_scipy(excesses: np.ndarray) -> tuple[float, float, bool]:
    """MLE via ``scipy.stats.genpareto.fit(..., floc=0)``. Returns ``(scale, xi, converged)``."""
    try:
        c, loc, scale = genpareto.fit(excesses, floc=0)
    except Exception as exc:
        raise FitError("SciPy GPD MLE failed", code=ErrorCode.FIT_FAILED) from exc
    scale_f, xi = gpd_params_from_scipy(float(c), float(loc), float(scale))
    return scale_f, xi, _converged(0.0, scale_f, xi)
