# Agent guide — rare-core

Instructions for AI coding agents working in this repository.

## Mission

Build **V0.1** of **rare-core**, a lightweight Python extreme-value engine: deterministic, NumPy/SciPy-only at runtime, validated against pinned reference libraries.

The distribution/package name is `rare-core`. The importable Python package is `evt` (`from evt import ...`).

## Read order

1. [docs/SPEC.md](docs/SPEC.md) — authoritative functional and numerical requirements
2. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module boundaries, SOLID/DRY rules
3. [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) — phased delivery and exit criteria

Do **not** implement out-of-scope features listed in SPEC §1.2 unless the user explicitly expands scope.

## Hard constraints

| Rule | Detail |
|------|--------|
| Python | `>=3.14` only |
| Runtime deps | `numpy`, `scipy` — nothing else |
| Shape convention | Public API uses **xi**; SciPy GEV `c` translated as `xi = -c` **only** in `scipy_adapter.py` |
| Data | Never silently drop, filter, or mutate user input |
| Stochastic | Every random procedure takes explicit `seed: int` |
| Likelihood | Single canonical implementation in `likelihood.py`; AIC and validation use it |
| Tolerance changes | Never widen validation tolerances without documented root-cause analysis |

## Repository map

```
src/evt/
  types.py           # Frozen dataclasses — no business logic
  errors.py          # Typed exception hierarchy with stable codes
  validate.py        # All input validation
  series.py          # ExtremeSeries construction
  transform.py       # Lower-tail → upper-tail orientation
  likelihood.py      # Canonical GEV/GPD logpdf / log-likelihood
  return_levels.py     # GEV/GPD return-level formulas (xi→0 branches)
  bootstrap.py       # Parametric bootstrap CIs
  diagnostics.py     # QQ/PP coordinates, KS distance
  serialize.py       # to_dict / from_dict, schema_version
  extract/           # BM, POT, declustering
  fit/               # GEV/GPD MLE; scipy_adapter.py is the only SciPy param bridge
```

## SOLID expectations

- **S** — One module per concern (extract ≠ fit ≠ inference ≠ serialize).
- **O** — Extend via new modules/backends; avoid changing validated numerical paths for new features.
- **L** — Fit backends must produce types satisfying `GEVFit` / `GPDFit` contracts.
- **I** — Small public surface in `evt/__init__.py`; internal modules stay internal.
- **D** — High-level functions depend on dataclass contracts, not SciPy types.

## DRY expectations

- One likelihood implementation (`likelihood.py`).
- One SciPy parameter adapter (`fit/scipy_adapter.py`).
- One serializer (`serialize.py`).
- Shared validators (`validate.py`) — no duplicated array checks.
- Return-level formulas live only in `return_levels.py`.

## Implementation workflow

1. Check current phase in [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).
2. Implement the smallest change that satisfies SPEC acceptance criteria for that phase.
3. Add unit tests alongside code in `tests/unit/`.
4. Run `pytest`, `ruff check`, `mypy src/evt`.
5. Do not add runtime dependencies or plotting libraries.
6. Do not commit reference-regenerated JSON unless running the reference workflow intentionally.

## Common tasks

### Add a new extraction option

- Extend `extract/` only.
- Update `ExtremeSample` invariants in SPEC if metadata changes.
- Add validation tests before integration tests.

### Fix reference mismatch

1. Reproduce with fixture name and parameter diff from `validation/compare.py`.
2. Identify: adapter bug, formula bug, or legitimate library difference.
3. Fix code if engine is wrong; document in SPEC/validation README if reference differs for a explained reason.
4. Do **not** relax tolerance without user approval and written justification.

### Add public API export

- Export from `src/evt/__init__.py` only when stable.
- Update README quick start and SPEC §4 if signatures change.

## Code style

- `@dataclass(frozen=True)` for all result types.
- Type hints on all public functions; `mypy --strict`.
- `ruff` for lint/format; line length 100.
- Prefer pure functions; inject `rng` or `seed` for randomness.
- Comments explain *why*, not *what*, for non-obvious numerical choices.

## Testing commands

```bash
pip install -e ".[dev]"
pytest
ruff check src tests validation
mypy src/evt
```

## Commit messages

Use imperative mood, focus on *why*:

- `Add block maxima extraction with min_blocks guard`
- `Fix GPD return level xi→0 branch for lambda*T near 1`

## Questions to ask the user

Only when SPEC is silent and the choice affects numerical behavior:

- Incomplete block handling policy (default: drop trailing partial block — already in SPEC).
- Minimum sample sizes for fit (default: GEV n≥3, GPD n≥2 — already in SPEC).

Otherwise, follow SPEC defaults.
