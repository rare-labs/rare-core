"""GPD maximum-likelihood fit on threshold excesses with location fixed at 0."""

from evt.errors import ErrorCode, ValidationError
from evt.fit.scipy_adapter import gpd_fit_scipy
from evt.likelihood import GPD_N_PARAMS, aic, gpd_log_likelihood
from evt.types import ExtremeSample, GPDFit
from evt.validate import validate_fit_sample_size, validate_sample


def fit_gpd(sample: ExtremeSample) -> GPDFit:
    """Fit GPD MLE to ``sample.excesses`` (``floc=0``) and recompute canonical AIC."""
    validate_sample(sample)
    if sample.excesses is None:
        raise ValidationError(
            "GPD fit requires POT excesses",
            code=ErrorCode.MISSING_POT_METADATA,
        )
    y = sample.excesses
    validate_fit_sample_size(int(y.size), model="GPD")
    scale, xi, converged = gpd_fit_scipy(y)
    log_likelihood = gpd_log_likelihood(y, scale, xi)
    return GPDFit(
        scale=scale,
        shape=xi,
        log_likelihood=log_likelihood,
        aic=aic(log_likelihood, n_params=GPD_N_PARAMS),
        n_extremes=int(y.size),
        converged=converged,
    )
