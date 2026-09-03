"""QQ/PP coordinates and descriptive KS distance. No plotting dependency."""

import numpy as np

from evt.errors import ErrorCode, ValidationError
from evt.return_levels import gev_cdf, gev_quantile, gpd_excess_cdf, gpd_excess_quantile
from evt.types import DiagnosticResult, ExtremeSample, GEVFit, GPDFit
from evt.validate import immutable_float64


def diagnostic_qq_pp(
    sample: ExtremeSample,
    fit: GEVFit | GPDFit,
    *,
    n_points: int | None = None,
) -> DiagnosticResult:
    """Plotting position ``(i - 0.5) / n``. KS uses ``max |i/n - F(x_(i))|``."""
    if isinstance(fit, GPDFit):
        if sample.excesses is None:
            raise ValidationError(
                "GPD diagnostics require POT excesses",
                code=ErrorCode.MISSING_POT_METADATA,
            )
        data = np.sort(np.asarray(sample.excesses, dtype=np.float64))
    else:
        data = np.sort(np.asarray(sample.transformed_values, dtype=np.float64))
    n = int(data.size)
    if n < 1:
        raise ValidationError("diagnostic sample is empty", code=ErrorCode.INVALID_SHAPE)
    if n_points is None:
        selected = data
    else:
        if n_points < 1:
            raise ValidationError("n_points must be >= 1", code=ErrorCode.INVALID_SHAPE)
        idx = np.unique(np.linspace(0, n - 1, num=n_points).astype(int))
        selected = data[idx]

    k = int(selected.size)
    i = np.arange(1, k + 1, dtype=np.float64)
    pp_empirical = (i - 0.5) / k
    if isinstance(fit, GEVFit):
        qq_model = gev_quantile(pp_empirical, fit.location, fit.scale, fit.shape)
        pp_model = gev_cdf(selected, fit.location, fit.scale, fit.shape)
    else:
        qq_model = gpd_excess_quantile(pp_empirical, fit.scale, fit.shape)
        pp_model = gpd_excess_cdf(selected, fit.scale, fit.shape)

    i_full = np.arange(1, n + 1, dtype=np.float64)
    if isinstance(fit, GEVFit):
        f_sorted = gev_cdf(data, fit.location, fit.scale, fit.shape)
    else:
        f_sorted = gpd_excess_cdf(data, fit.scale, fit.shape)
    ks_distance = float(np.max(np.abs(i_full / n - f_sorted)))
    return DiagnosticResult(
        qq_empirical=immutable_float64(selected),
        qq_model=immutable_float64(qq_model),
        pp_empirical=immutable_float64(pp_empirical),
        pp_model=immutable_float64(pp_model),
        ks_distance=ks_distance,
    )
