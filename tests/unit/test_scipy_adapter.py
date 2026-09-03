"""SciPy ξ ↔ c translation lives only in the adapter."""

from pathlib import Path

import numpy as np
from scipy.stats import genextreme, genpareto

from evt.fit.scipy_adapter import (
    gev_fit_scipy,
    gev_params_from_scipy,
    gpd_fit_scipy,
    gpd_params_from_scipy,
)


def test_gev_c_to_xi_is_negation() -> None:
    location, scale, xi = gev_params_from_scipy(c=0.25, loc=10.0, scale=2.0)
    assert location == 10.0
    assert scale == 2.0
    assert xi == -0.25


def test_gpd_c_is_xi() -> None:
    scale, xi = gpd_params_from_scipy(c=-0.1, loc=0.0, scale=3.5)
    assert scale == 3.5
    assert xi == -0.1


def test_gev_fit_scipy_round_trip_sign() -> None:
    rng = np.random.default_rng(0)
    # SciPy c = -xi; draw with c=-0.2 so engine xi should be near +0.2.
    x = genextreme.rvs(c=-0.2, loc=5.0, scale=1.5, size=400, random_state=rng)
    location, scale, xi, converged = gev_fit_scipy(x)
    assert converged
    assert scale > 0
    c, loc, sc = genextreme.fit(x)
    loc2, sc2, xi2 = gev_params_from_scipy(float(c), float(loc), float(sc))
    assert location == loc2
    assert scale == sc2
    assert xi == xi2
    assert xi2 == -float(c)


def test_gpd_fit_scipy_matches_c() -> None:
    rng = np.random.default_rng(1)
    y = genpareto.rvs(c=0.15, loc=0.0, scale=2.0, size=400, random_state=rng)
    scale, xi, converged = gpd_fit_scipy(y)
    assert converged
    c, _loc, sc = genpareto.fit(y, floc=0)
    sc2, xi2 = gpd_params_from_scipy(float(c), float(_loc), float(sc))
    assert scale == sc2
    assert xi == xi2
    assert xi2 == float(c)


def test_fit_calls_only_in_adapter() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "evt"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path.name == "scipy_adapter.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "genextreme.fit" in text or "genpareto.fit" in text:
            offenders.append(str(path.relative_to(root)))
    assert offenders == []
