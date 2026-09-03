"""Golden-fixture comparison against committed snapshots (SPEC §6.4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
if __package__ is None:
    sys.path.insert(0, str(REPO))

from evt.serialize import from_dict
from evt.types import ExtremeSample
from validation.engine_eval import evaluate_sample

SAMPLES = ROOT / "fixtures" / "samples"
SNAPSHOTS = ROOT / "fixtures" / "snapshots"
REFERENCES = ROOT / "fixtures" / "references"
CATALOG = ROOT / "fixtures" / "catalog.json"


def loc_scale_tol(reference: float) -> float:
    return max(1e-3, 0.005 * abs(reference))


def close_loc_scale(engine: float, reference: float) -> bool:
    return abs(engine - reference) <= loc_scale_tol(reference)


def close_shape(engine: float, reference: float) -> bool:
    return abs(engine - reference) <= 0.005


def close_return_level(engine: float, reference: float) -> bool:
    return abs(engine - reference) <= 0.005 * abs(reference)


def close_log_likelihood(engine: float, reference: float) -> bool:
    return abs(engine - reference) <= 1e-4


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fit_fields(model: Literal["GEV", "GPD"], fit: dict[str, Any]) -> dict[str, float]:
    fields = {"scale": float(fit["scale"]), "shape": float(fit["shape"])}
    if model == "GEV":
        fields["location"] = float(fit["location"])
    return fields


def compare_payloads(
    model: Literal["GEV", "GPD"],
    engine: dict[str, Any],
    reference: dict[str, Any],
    *,
    source: str,
) -> list[str]:
    """Return human-readable mismatch lines using SPEC §6.4 tolerances."""
    failures: list[str] = []
    eng_fit = engine["fit"]
    ref_fit = reference["fit"]
    for name, ref_value in _fit_fields(model, ref_fit).items():
        eng_value = float(eng_fit[name])
        ok = close_shape(eng_value, ref_value) if name == "shape" else close_loc_scale(
            eng_value, ref_value
        )
        if not ok:
            failures.append(f"{source} {name}: engine={eng_value} ref={ref_value}")
    for period, ref_level in reference["return_levels"].items():
        eng_level = float(engine["return_levels"][str(period)])
        if not close_return_level(eng_level, float(ref_level)):
            failures.append(
                f"{source} return_level T={period}: engine={eng_level} ref={ref_level}"
            )
    engine_ll = float(engine["log_likelihood"])
    reference_ll = float(reference["log_likelihood"])
    if not close_log_likelihood(engine_ll, reference_ll):
        failures.append(
            f"{source} log_likelihood: engine={engine['log_likelihood']} "
            f"ref={reference['log_likelihood']}"
        )
    return failures


def compare_all() -> list[str]:
    catalog = _load_json(CATALOG)
    failures: list[str] = []
    for entry in catalog["fixtures"]:
        name = str(entry["name"])
        model: Literal["GEV", "GPD"] = entry["model"]
        sample = from_dict(_load_json(SAMPLES / f"{name}.json"), ExtremeSample)
        engine = evaluate_sample(sample, model)
        snapshot_path = SNAPSHOTS / f"{name}.json"
        if not snapshot_path.exists():
            failures.append(f"{name}: missing engine snapshot {snapshot_path}")
            continue
        failures.extend(compare_payloads(model, engine, _load_json(snapshot_path), source=name))
        for library in ("pyextremes", "extremes_jl"):
            ref_path = REFERENCES / library / f"{name}.json"
            if not ref_path.exists():
                continue
            failures.extend(
                compare_payloads(model, engine, _load_json(ref_path), source=f"{name}/{library}")
            )
    return failures


def main() -> int:
    failures = compare_all()
    if failures:
        print("Golden comparison failed:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("Golden comparison passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
