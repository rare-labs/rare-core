"""Evaluate golden samples with the engine. Used to write and compare snapshots."""

from typing import Any, Literal

from evt.fit import fit_gev, fit_gpd
from evt.return_levels import gev_return_level, gpd_return_level
from evt.serialize import to_dict
from evt.types import ExtremeSample

RETURN_PERIODS = (2.0, 10.0, 50.0)


def evaluate_sample(sample: ExtremeSample, model: Literal["GEV", "GPD"]) -> dict[str, Any]:
    """Fit and compute T = 2, 10, 50 return levels plus canonical log-likelihood."""
    if model == "GEV":
        fit = fit_gev(sample)
        levels = {
            str(int(period)): gev_return_level(fit, period).estimate for period in RETURN_PERIODS
        }
        payload = to_dict(fit)
    else:
        fit = fit_gpd(sample)
        levels = {
            str(int(period)): gpd_return_level(fit, sample, period).estimate
            for period in RETURN_PERIODS
        }
        payload = to_dict(fit)
    return {
        "model": model,
        "fit": payload,
        "return_levels": levels,
        "log_likelihood": payload["log_likelihood"],
    }
