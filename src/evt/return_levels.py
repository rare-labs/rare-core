"""GEV/GPD return levels, CDF, and quantiles. Explicit ξ→0 branches live only here."""

import numpy as np

from evt.constants import XI_ZERO_ATOL
from evt.errors import ErrorCode, ValidationError
from evt.transform import infer_tail, inverse_transform_scalar, transform_threshold
from evt.types import ExtremeSample, GEVFit, GPDFit, ReturnLevelResult, Tail
from evt.validate import validate_pot_return_level_metadata, validate_tail


def xi_is_zero(xi: float) -> bool:
    return abs(xi) < XI_ZERO_ATOL


def _require_positive_scale(scale: float) -> None:
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValidationError("scale must be finite and positive", code=ErrorCode.INVALID_SHAPE)


def gev_quantile(
    p: np.ndarray,
    location: float,
    scale: float,
    xi: float,
) -> np.ndarray:
    """GEV quantile function (upper-tail / transformed scale)."""
    _require_positive_scale(scale)
    prob = np.asarray(p, dtype=np.float64)
    nlog = -np.log(prob)
    if xi_is_zero(xi):
        return location - scale * np.log(nlog)
    return location + (scale / xi) * (np.power(nlog, -xi) - 1.0)


def gev_cdf(x: np.ndarray, location: float, scale: float, xi: float) -> np.ndarray:
    _require_positive_scale(scale)
    values = np.asarray(x, dtype=np.float64)
    z = (values - location) / scale
    if xi_is_zero(xi):
        return np.exp(-np.exp(-z))
    t = 1.0 + xi * z
    out = np.empty_like(values)
    ok = t > 0.0
    out[ok] = np.exp(-np.power(t[ok], -1.0 / xi))
    out[~ok] = 0.0 if xi > 0.0 else 1.0
    return out


def gpd_excess_quantile(p: np.ndarray, scale: float, xi: float) -> np.ndarray:
    """GPD quantile of non-negative excesses (location 0)."""
    _require_positive_scale(scale)
    prob = np.asarray(p, dtype=np.float64)
    survival = 1.0 - prob
    if xi_is_zero(xi):
        return -scale * np.log(survival)
    return (scale / xi) * (np.power(survival, -xi) - 1.0)


def gpd_excess_cdf(y: np.ndarray, scale: float, xi: float) -> np.ndarray:
    _require_positive_scale(scale)
    excesses = np.asarray(y, dtype=np.float64)
    if xi_is_zero(xi):
        return 1.0 - np.exp(-excesses / scale)
    t = 1.0 + xi * excesses / scale
    out = np.empty_like(excesses)
    ok = t > 0.0
    out[ok] = 1.0 - np.power(t[ok], -1.0 / xi)
    out[~ok] = 1.0
    return np.clip(out, 0.0, 1.0)


def _gpd_level_transformed(threshold_star: float, scale: float, xi: float, m: float) -> float:
    _require_positive_scale(scale)
    if xi_is_zero(xi):
        return threshold_star + scale * float(np.log(m))
    return float(threshold_star + (scale / xi) * (m**xi - 1.0))


def gev_return_level(
    fit: GEVFit,
    return_period: float,
    *,
    blocks_per_period: float = 1.0,
    tail: str = "high",
) -> ReturnLevelResult:
    """Return level on the original scale. ``tail`` maps low-tail fits back from ``-x``."""
    orientation = validate_tail(tail)
    if return_period <= 0.0 or blocks_per_period <= 0.0:
        raise ValidationError(
            "return_period and blocks_per_period must be positive",
            code=ErrorCode.INVALID_SHAPE,
        )
    extrapolation = float(return_period * blocks_per_period)
    if extrapolation <= 1.0:
        raise ValidationError(
            "T * blocks_per_period must be > 1 so the GEV CDF level is in (0, 1)",
            code=ErrorCode.INVALID_SHAPE,
        )
    p = 1.0 - 1.0 / extrapolation
    z_star = float(gev_quantile(np.asarray(p), fit.location, fit.scale, fit.shape))
    estimate = inverse_transform_scalar(z_star, orientation)
    return ReturnLevelResult(
        estimate=estimate,
        return_period=float(return_period),
        confidence_level=None,
        lower=None,
        upper=None,
        extrapolation_factor=extrapolation,
        method="GEV",
    )


def gpd_return_level(
    fit: GPDFit,
    sample: ExtremeSample,
    return_period: float,
) -> ReturnLevelResult:
    """POT return level on the original scale (inverse-mapped if the sample is low-tail)."""
    validate_pot_return_level_metadata(sample)
    if return_period <= 0.0:
        raise ValidationError("return_period must be positive", code=ErrorCode.INVALID_SHAPE)
    rate = sample.exceedance_rate
    threshold = sample.threshold
    if rate is None or threshold is None:
        raise ValidationError(
            "GPD return levels require POT threshold and exceedance_rate",
            code=ErrorCode.MISSING_POT_METADATA,
        )
    if rate <= 0.0:
        raise ValidationError("exceedance_rate must be positive", code=ErrorCode.INVALID_SHAPE)
    m = rate * float(return_period)
    tail: Tail = infer_tail(sample)
    u_star = transform_threshold(threshold, tail)
    z_star = _gpd_level_transformed(u_star, fit.scale, fit.shape, m)
    estimate = inverse_transform_scalar(z_star, tail)
    return ReturnLevelResult(
        estimate=estimate,
        return_period=float(return_period),
        confidence_level=None,
        lower=None,
        upper=None,
        extrapolation_factor=float(m),
        method="GPD",
    )
