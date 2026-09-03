"""Fit committed golden samples with pyextremes MLE and write JSON snapshots.

Requires the optional ``reference`` extra (pyextremes + pandas). Normal CI does not
run this script. Pinned commit: 81f943e15f4f06246dc0870a14aa3915398d0e6d

Fits **already-extracted** arrays via ``pyextremes.get_model`` (no re-extraction).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import genextreme, genpareto

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "fixtures" / "samples"
CATALOG = ROOT / "fixtures" / "catalog.json"
OUT = ROOT / "fixtures" / "references" / "pyextremes"
RETURN_PERIODS = (2.0, 10.0, 50.0)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gev_return_level(location: float, scale: float, xi: float, period: float) -> float:
    p = 1.0 - 1.0 / period
    nlog = -np.log(p)
    if abs(xi) < 1e-8:
        return float(location - scale * np.log(nlog))
    return float(location + (scale / xi) * (nlog ** (-xi) - 1.0))


def _gpd_return_level(
    threshold: float, scale: float, xi: float, rate: float, period: float
) -> float:
    m = rate * period
    if abs(xi) < 1e-8:
        return float(threshold + scale * np.log(m))
    return float(threshold + (scale / xi) * (m**xi - 1.0))


def _series(values: np.ndarray):
    import pandas as pd

    index = pd.date_range("2000-01-01", periods=int(values.size), freq="D")
    return pd.Series(values, index=index)


def main() -> int:
    try:
        from pyextremes import get_model
    except ImportError:
        print("pyextremes is not installed. pip install -e '.[reference]'", file=sys.stderr)
        return 2

    catalog = _load(CATALOG)
    OUT.mkdir(parents=True, exist_ok=True)
    for entry in catalog["fixtures"]:
        name = str(entry["name"])
        model = str(entry["model"])
        sample = _load(SAMPLES / f"{name}.json")
        if model == "GEV":
            values = np.asarray(sample["transformed_values"], dtype=np.float64)
            fitted = get_model(
                model="MLE",
                extremes=_series(values),
                distribution="genextreme",
            )
            c = float(fitted.fit_parameters["c"])
            location = float(fitted.fit_parameters["loc"])
            scale = float(fitted.fit_parameters["scale"])
            xi = -c
            ll = float(np.sum(genextreme.logpdf(values, c=c, loc=location, scale=scale)))
            fit = {
                "schema_version": "0.1",
                "type": "GEVFit",
                "location": location,
                "scale": scale,
                "shape": xi,
                "log_likelihood": ll,
                "aic": 6.0 - 2.0 * ll,
                "n_extremes": int(values.size),
                "converged": True,
            }
            levels = {
                str(int(period)): _gev_return_level(location, scale, xi, period)
                for period in RETURN_PERIODS
            }
        else:
            excesses = np.asarray(sample["excesses"], dtype=np.float64)
            fitted = get_model(
                model="MLE",
                extremes=_series(excesses),
                distribution="genpareto",
                distribution_kwargs={"floc": 0},
            )
            xi = float(fitted.fit_parameters["c"])
            scale = float(fitted.fit_parameters["scale"])
            ll = float(np.sum(genpareto.logpdf(excesses, c=xi, loc=0.0, scale=scale)))
            fit = {
                "schema_version": "0.1",
                "type": "GPDFit",
                "scale": scale,
                "shape": xi,
                "log_likelihood": ll,
                "aic": 4.0 - 2.0 * ll,
                "n_extremes": int(excesses.size),
                "converged": True,
            }
            threshold = float(sample["threshold"])
            rate = float(sample["exceedance_rate"])
            levels = {
                str(int(period)): _gpd_return_level(threshold, scale, xi, rate, period)
                for period in RETURN_PERIODS
            }
        payload = {
            "model": model,
            "fit": fit,
            "return_levels": levels,
            "log_likelihood": ll,
        }
        (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {OUT / f'{name}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
