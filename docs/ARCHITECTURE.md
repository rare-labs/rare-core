# Architecture — rare-core V0.1

Design principles, module boundaries, and extension points for contributors and agents.

---

## 1. Design goals

| Goal | Implication |
|------|-------------|
| Lightweight | Two runtime dependencies; no framework coupling |
| Deterministic substrate | Pure functions, serializable outputs, explicit seeds |
| Agent-friendly | Stable module map, typed contracts, SPEC as source of truth |
| Validated numerics | Single likelihood path; SciPy isolated behind adapter |
| Future-ready | Clean seams for MCP/DuckDB without building them in V0.1 |

The product and distribution name is **rare-core**. The importable Python package is **`evt`**.

---

## 2. Layered architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Public API (evt/__init__.py)                                │
│  extract_* · fit_* · *_return_level · bootstrap · diagnostic │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│  Domain layer                                                │
│  extract/ · transform · return_levels · bootstrap · diag   │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│  Model layer                                                 │
│  fit/ · likelihood.py                                        │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│  Infrastructure                                              │
│  types · validate · errors · serialize · constants           │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│  SciPy adapter (fit/scipy_adapter.py) — only SciPy fit import│
└──────────────────────────────────────────────────────────────┘
```

**Dependency rule:** layers depend downward only. `likelihood.py` may use SciPy `logpdf`/`rvs` but not `fit`. Only `scipy_adapter.py` calls `fit`.

---

## 3. SOLID mapping

### Single Responsibility (S)

| Module | Responsibility |
|--------|----------------|
| `validate.py` | Input contract enforcement |
| `extract/block.py` | Block maxima/minima only |
| `extract/pot.py` | Threshold exceedance selection |
| `extract/decluster.py` | Run and window declustering |
| `transform.py` | Tail orientation |
| `fit/gev.py`, `fit/gpd.py` | Orchestrate fit + likelihood assembly |
| `fit/scipy_adapter.py` | SciPy parameter translation |
| `likelihood.py` | Logpdf and log-likelihood |
| `return_levels.py` | Return-level formulas |
| `bootstrap.py` | Resampling loop and CI aggregation |
| `diagnostics.py` | QQ/PP/KS |
| `serialize.py` | Dict/JSON conversion |

No module combines extraction with fitting or inference with serialization.

### Open/Closed (O)

- New estimation backends (e.g. PWM in V0.2) add modules under `fit/` implementing a small internal protocol without modifying validated MLE paths.
- New declustering methods extend `extract/decluster.py` via registry or strategy functions.

### Liskov Substitution (L)

Internal fit backend protocol (future):

```python
class FitBackend(Protocol):
    def fit_gev(self, x: np.ndarray) -> tuple[float, float, float, bool]: ...
    def fit_gpd(self, excesses: np.ndarray) -> tuple[float, float, bool]: ...
```

Any backend must return parameters interpretable by `likelihood.py` and `return_levels.py`.

### Interface Segregation (I)

- Public exports are minimal (`evt/__init__.py`).
- Validation helpers stay internal unless users need them (`validate_series` may be public for advanced use).
- Bootstrap accepts already-fitted models — it does not re-extract.

### Dependency Inversion (D)

High-level functions depend on dataclass contracts (`ExtremeSample`, `GEVFit`), not SciPy distribution objects. SciPy is an implementation detail of the adapter.

---

## 4. DRY rules

| Concern | Single location |
|---------|-----------------|
| GEV/GPD log-likelihood | `likelihood.py` |
| SciPy ξ ↔ c translation | `fit/scipy_adapter.py` |
| Array/timestamp validation | `validate.py` |
| Return-level math | `return_levels.py` |
| JSON encoding rules | `serialize.py` |
| Physical constants (`DAYS_PER_YEAR`) | `constants.py` |
| Tail transform / inverse | `transform.py` |

**Anti-patterns to reject in review**

- Duplicate likelihood formulas in tests (tests compare to imported functions or analytic references).
- Direct `genextreme.fit` outside `scipy_adapter.py`.
- Inline timestamp validation in extractors.

---

## 5. Module dependency graph

```
types, errors, constants
        ↓
   validate, serialize
        ↓
   transform ← extract/*
        ↓
   scipy_adapter ← fit/*
        ↓
   likelihood
        ↓
return_levels, bootstrap, diagnostics
        ↓
   __init__.py (public API)
```

No cycles. `tests/` and `validation/` depend on public and internal APIs but are not imported by `src/evt`.

---

## 6. Key types and contracts

All result types are **immutable**. Mutations require new instances.

Fitting contract:

1. Input: `ExtremeSample` satisfying SPEC invariants.
2. Output: `GEVFit` or `GPDFit` with `log_likelihood` from `likelihood.py`.
3. Side effects: none.

Extraction contract:

1. Input: validated `ExtremeSeries`.
2. Output: new `ExtremeSample`; original series untouched.
3. Document any dropped trailing block in docstring (not silent).

---

## 7. Constants

```python
# src/evt/constants.py
SCHEMA_VERSION = "0.1"
DAYS_PER_YEAR = 365.25
GEV_MIN_SAMPLE = 3
GPD_MIN_SAMPLE = 2
BOOTSTRAP_MAX_REFIT_RETRIES = 10
DEFAULT_PLOT_POSITION = "blom"  # (i - 0.5) / n
```

---

## 8. Extension points (post-V0.1)

| Seam | Future use |
|------|------------|
| `serialize.py` + `schema_version` | MCP tool payloads, DuckDB struct columns |
| `FitBackend` protocol | PWM, Bayesian backends |
| `extract/` registry | Custom declustering |
| `to_dict()` on all results | REST/WASM without changing core |

Do not implement these in V0.1; keep functions pure and serializable so wrappers stay thin.

---

## 9. Testing architecture

```
tests/unit/           # per-module, fast, no fixtures IO
tests/integration/    # pipeline tests, synthetic data
validation/           # golden JSON, reference comparison (CI subset)
```

Unit tests mock nothing for numerical code — use small deterministic arrays.

Validation tests read committed JSON under `validation/fixtures/` only in CI main path.

---

## 10. CI architecture

| Workflow | Trigger | Dependencies |
|----------|---------|--------------|
| `ci.yml` | push, PR | Python 3.14, numpy, scipy, pytest, ruff, mypy |
| `reference.yml` | manual / schedule | + Julia Extremes.jl, pyextremes |

Normal CI never requires Julia or pyextremes.

---

## 11. File creation order

Implement infrastructure before features:

1. `types`, `errors`, `constants`, `validate`, `serialize`
2. `transform`, `extract/*`
3. `scipy_adapter`, `likelihood`, `fit/*`
4. `return_levels`, `bootstrap`, `diagnostics`
5. `__init__.py` exports
6. `validation/*`

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for phased milestones.
