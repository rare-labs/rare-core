"""Parametric bootstrap percentile intervals for return levels."""

from collections.abc import Sequence

import numpy as np

from evt.constants import BOOTSTRAP_MAX_REFIT_RETRIES
from evt.errors import BootstrapError, ErrorCode, FitError, ValidationError
from evt.fit import fit_gev, fit_gpd
from evt.likelihood import gev_rvs, gpd_rvs
from evt.return_levels import gev_return_level, gpd_return_level
from evt.transform import infer_tail, inverse_transform_from_upper_tail, transform_threshold
from evt.types import ExtremeSample, GEVFit, GPDFit, ReturnLevelResult
from evt.validate import immutable_float64


def _with_ci(
    point: ReturnLevelResult,
    draws: np.ndarray,
    confidence_level: float,
) -> ReturnLevelResult:
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    return ReturnLevelResult(
        estimate=point.estimate,
        return_period=point.return_period,
        confidence_level=confidence_level,
        lower=float(lower),
        upper=float(upper),
        extrapolation_factor=point.extrapolation_factor,
        method=point.method,
    )


def _simulated_gev_sample(template: ExtremeSample, draws: np.ndarray) -> ExtremeSample:
    tail = infer_tail(template)
    transformed = immutable_float64(draws)
    raw = inverse_transform_from_upper_tail(transformed, tail)
    return ExtremeSample(
        raw_values=raw,
        transformed_values=transformed,
        timestamps=None,
        method="BM",
        threshold=None,
        excesses=None,
        n_source=template.n_source,
        exposure=None,
        exceedance_rate=None,
    )


def _simulated_gpd_sample(template: ExtremeSample, excesses: np.ndarray) -> ExtremeSample:
    tail = infer_tail(template)
    threshold = template.threshold
    if threshold is None:
        raise ValidationError(
            "GPD bootstrap requires a POT threshold",
            code=ErrorCode.MISSING_POT_METADATA,
        )
    u_star = transform_threshold(threshold, tail)
    y = immutable_float64(excesses)
    transformed = immutable_float64(u_star + y)
    raw = inverse_transform_from_upper_tail(transformed, tail)
    return ExtremeSample(
        raw_values=raw,
        transformed_values=transformed,
        timestamps=None,
        method="POT",
        threshold=threshold,
        excesses=y,
        n_source=template.n_source,
        exposure=template.exposure,
        exceedance_rate=template.exceedance_rate,
    )


def _refit_gev(sample: ExtremeSample) -> GEVFit:
    fit = fit_gev(sample)
    if not fit.converged:
        raise FitError("GEV refit did not converge", code=ErrorCode.FIT_FAILED)
    return fit


def _refit_gpd(sample: ExtremeSample) -> GPDFit:
    fit = fit_gpd(sample)
    if not fit.converged:
        raise FitError("GPD refit did not converge", code=ErrorCode.FIT_FAILED)
    return fit


def bootstrap_return_levels(
    sample: ExtremeSample,
    fit: GEVFit | GPDFit,
    return_periods: Sequence[float],
    *,
    n_samples: int = 500,
    confidence_level: float = 0.95,
    blocks_per_period: float = 1.0,
    seed: int,
) -> list[ReturnLevelResult]:
    """Parametric bootstrap percentile CIs. Reproducible given ``seed``."""
    if n_samples < 1:
        raise ValidationError("n_samples must be >= 1", code=ErrorCode.INVALID_SHAPE)
    if not 0.0 < confidence_level < 1.0:
        raise ValidationError("confidence_level must be in (0, 1)", code=ErrorCode.INVALID_SHAPE)
    periods = [float(t) for t in return_periods]
    if not periods:
        raise ValidationError("return_periods must be non-empty", code=ErrorCode.INVALID_SHAPE)

    rng = np.random.default_rng(seed)
    n = fit.n_extremes
    tail = infer_tail(sample)
    attempts_allowed = BOOTSTRAP_MAX_REFIT_RETRIES + 1
    replicates = np.empty((len(periods), n_samples), dtype=np.float64)

    if isinstance(fit, GEVFit):
        gev_fit = fit
        points = [
            gev_return_level(gev_fit, t, blocks_per_period=blocks_per_period, tail=tail)
            for t in periods
        ]
        for b in range(n_samples):
            new_fit: GEVFit | None = None
            for _ in range(attempts_allowed):
                try:
                    draws = gev_rvs(n, gev_fit.location, gev_fit.scale, gev_fit.shape, rng)
                    new_fit = _refit_gev(_simulated_gev_sample(sample, draws))
                    break
                except (FitError, ValidationError):
                    new_fit = None
            if new_fit is None:
                raise BootstrapError(
                    "parametric bootstrap exhausted refit retries",
                    code=ErrorCode.BOOTSTRAP_FAILED,
                )
            for i, t in enumerate(periods):
                replicates[i, b] = gev_return_level(
                    new_fit, t, blocks_per_period=blocks_per_period, tail=tail
                ).estimate
        return [_with_ci(point, replicates[i], confidence_level) for i, point in enumerate(points)]

    gpd_fit = fit
    points = [gpd_return_level(gpd_fit, sample, t) for t in periods]
    for b in range(n_samples):
        new_gpd: GPDFit | None = None
        for _ in range(attempts_allowed):
            try:
                draws = gpd_rvs(n, gpd_fit.scale, gpd_fit.shape, rng)
                new_gpd = _refit_gpd(_simulated_gpd_sample(sample, draws))
                break
            except (FitError, ValidationError):
                new_gpd = None
        if new_gpd is None:
            raise BootstrapError(
                "parametric bootstrap exhausted refit retries",
                code=ErrorCode.BOOTSTRAP_FAILED,
            )
        for i, t in enumerate(periods):
            replicates[i, b] = gpd_return_level(new_gpd, sample, t).estimate
    return [_with_ci(point, replicates[i], confidence_level) for i, point in enumerate(points)]
