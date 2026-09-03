"""GEV maximum-likelihood fit on upper-tail oriented block maxima."""

from evt.fit.scipy_adapter import gev_fit_scipy
from evt.likelihood import GEV_N_PARAMS, aic, gev_log_likelihood
from evt.types import ExtremeSample, GEVFit
from evt.validate import validate_fit_sample_size, validate_sample


def fit_gev(sample: ExtremeSample) -> GEVFit:
    """Fit GEV MLE to ``sample.transformed_values`` and recompute canonical AIC."""
    validate_sample(sample)
    x = sample.transformed_values
    validate_fit_sample_size(int(x.size), model="GEV")
    location, scale, xi, converged = gev_fit_scipy(x)
    log_likelihood = gev_log_likelihood(x, location, scale, xi)
    return GEVFit(
        location=location,
        scale=scale,
        shape=xi,
        log_likelihood=log_likelihood,
        aic=aic(log_likelihood, n_params=GEV_N_PARAMS),
        n_extremes=int(x.size),
        converged=converged,
    )
