# Implementation plan — rare-core V0.1

Phased delivery with exit criteria. Update checkboxes as work completes.

**Target:** ~7–8 focused dev days after scaffold  
**Python:** ≥ 3.14  
**Principles:** [ARCHITECTURE.md](ARCHITECTURE.md) (SOLID, DRY)

---

## Phase 0 — Scaffold

**Goal:** Installable package skeleton and CI.

### Tasks

- [x] `pyproject.toml` (Python ≥ 3.14, NumPy ≥ 2.3, SciPy ≥ 1.16)
- [x] Documentation (`README`, `AGENTS`, `docs/*`)
- [x] `src/evt/__init__.py` (version string, empty exports)
- [x] `src/evt/constants.py`
- [x] `.github/workflows/ci.yml`
- [x] `tests/conftest.py` (minimal)

### Exit criteria

```bash
pip install -e ".[dev]"
pytest   # passes (even if zero tests initially)
ruff check .
```

---

## Phase 1 — Types, errors, validation, serialization

**Goal:** Infrastructure layer complete.

### Tasks

- [x] `types.py` — all dataclasses from [SPEC.md](SPEC.md) §3
- [x] `errors.py` — hierarchy + stable codes (SPEC §7)
- [x] `validate.py` — series and array validators
- [x] `serialize.py` — `to_dict` / `from_dict`, `schema_version`
- [x] `tests/unit/test_validate.py`
- [x] `tests/unit/test_serialize.py`

### Exit criteria

- Round-trip JSON for every dataclass type.
- All validation error codes covered by tests.

---

## Phase 2 — Extraction and transform

**Goal:** BM, POT, declustering, exposure, λ.

### Tasks

- [ ] `transform.py`
- [ ] `extract/block.py`
- [ ] `extract/pot.py`
- [ ] `extract/decluster.py`
- [ ] `series.py` — convenience constructor with validation
- [ ] `tests/unit/test_transform.py`
- [ ] `tests/unit/test_extract_block.py`
- [ ] `tests/unit/test_extract_pot.py`
- [ ] `tests/unit/test_decluster.py`

### Exit criteria

- High/low tail paths tested.
- Trailing incomplete block dropped with `ExtractionError` when below `min_blocks`.
- Window declustering requires timestamps (clear error if missing).

---

## Phase 3 — Fitting and likelihood

**Goal:** GEV/GPD MLE with canonical log-likelihood.

### Tasks

- [ ] `fit/scipy_adapter.py` — **only** SciPy `fit` calls
- [ ] `likelihood.py`
- [ ] `fit/gev.py`, `fit/gpd.py`
- [ ] `tests/unit/test_scipy_adapter.py` — xi/c translation
- [ ] `tests/unit/test_likelihood.py`
- [ ] `tests/unit/test_fit.py`

### Exit criteria

- Fits on synthetic samples produce finite parameters.
- `log_likelihood` matches hand-checked small examples.
- No `genextreme.fit` / `genpareto.fit` outside adapter.

---

## Phase 4 — Return levels

**Goal:** GEV and GPD return levels with ξ → 0 branches.

### Tasks

- [ ] `return_levels.py`
- [ ] Inverse tail mapping for low-tail series
- [ ] `tests/unit/test_return_levels.py` — analytic cross-checks
- [ ] `tests/integration/test_gev_pipeline.py` (partial)

### Exit criteria

- Return levels match inverse-CDF simulation for known (μ, σ, ξ).
- T ∈ {2, 10, 50} covered in tests.

---

## Phase 5 — Bootstrap and diagnostics

**Goal:** Reproducible CIs and QQ/PP/KS.

### Tasks

- [ ] `bootstrap.py`
- [ ] `diagnostics.py`
- [ ] `tests/unit/test_bootstrap.py` — same seed → same CI
- [ ] `tests/unit/test_diagnostics.py`

### Exit criteria

- Bootstrap deterministic given `seed`.
- `BootstrapError` after retry exhaustion tested.

---

## Phase 6 — Golden validation

**Goal:** Cross-library parity on committed fixtures.

### Tasks

- [ ] `validation/generate_fixtures.py` — inverse-CDF only
- [ ] `validation/fixtures/samples/*.json`
- [ ] `validation/run_pyextremes.py`
- [ ] `validation/run_extremes_jl.jl`
- [ ] `validation/compare.py`
- [ ] `validation/README.md`
- [ ] `.github/workflows/reference.yml`
- [ ] Commit reference snapshot JSON

### Exit criteria

- `compare.py` passes all fixtures within [SPEC.md](SPEC.md) §6.4 tolerances.
- Any documented mismatch has written justification.

---

## Phase 7 — Public API and release polish

**Goal:** V0.1.0 release candidate.

### Tasks

- [ ] Export stable API from `evt/__init__.py`
- [ ] README quick start matches real signatures
- [ ] `CHANGELOG.md` for 0.1.0
- [ ] Tag `v0.1.0`

### Exit criteria

- [SPEC.md](SPEC.md) §9 acceptance checklist complete.
- CI green on Python 3.14.

---

## Dependency versions (maintain in pyproject.toml)

| Package | Minimum | Notes |
|---------|---------|-------|
| Python | 3.14 | language features, performance |
| NumPy | 2.3 | array API stability |
| SciPy | 1.16 | `genextreme`, `genpareto` |
| pytest | 8.4 | dev |
| ruff | 0.12 | dev |
| mypy | 1.17 | dev, strict |

Bump minors freely; pin only if a regression is discovered.

---

## Risk register

| Risk | Mitigation |
|------|------------|
| SciPy fit optimizer differences vs Julia | Validate on identical samples; document optimizer tolerance |
| ξ ≈ 0 numerical instability | Explicit limit branches; dedicated fixtures |
| Low-tail negation invalid for mixed-sign data | Document in SPEC; validate sign if needed later |
| Bootstrap refit failures for heavy tails | Retry cap + `BootstrapError`; test on heavy-tail fixture |

---

## Definition of done (V0.1)

1. All phases 0–7 complete.
2. Runtime import graph: `numpy`, `scipy` only.
3. Golden validation passes.
4. Documentation aligned: README, AGENTS, SPEC, ARCHITECTURE.
