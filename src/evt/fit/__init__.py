"""Fitting backends. Public callers should use ``fit_gev`` / ``fit_gpd``."""

from evt.fit.gev import fit_gev
from evt.fit.gpd import fit_gpd

__all__ = ["fit_gev", "fit_gpd"]
