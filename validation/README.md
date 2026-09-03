# Validation harness

Cross-library validation for rare-core V0.1.

## Principle

Compare **identical pre-extracted extreme samples**. Each library fits the same `raw_values` / `excesses` arrays from committed fixture JSON — not independent preprocessing.

## Pinned references

| Library | Commit |
|---------|--------|
| Extremes.jl | `fa757a2acb58669067ceef7bf7a3f1ecc6cfe2dc` |
| pyextremes | `81f943e15f4f06246dc0870a14aa3915398d0e6d` |

## Layout (implementation)

```
validation/
├── README.md              # this file
├── generate_fixtures.py   # inverse-CDF sample generation
├── compare.py             # tolerance checks vs snapshots
├── run_pyextremes.py      # optional reference runner
├── run_extremes_jl.jl     # optional reference runner
└── fixtures/
    ├── samples/           # input ExtremeSample JSON
    ├── gev_fit/           # reference fit outputs
    └── gpd_fit/
```

## Workflows

### Normal CI

```bash
pytest validation/compare.py  # when implemented
```

Uses committed reference snapshots only. No Julia or pyextremes required.

### Regenerate references

Manual or scheduled (`reference.yml`):

1. `python validation/generate_fixtures.py` (if samples change)
2. Run Julia and pyextremes reference scripts
3. Review diffs; commit updated snapshots intentionally

## Tolerances

See [docs/SPEC.md](../docs/SPEC.md#64-acceptance-tolerances).

Do not widen tolerances without documented root-cause analysis.

## Status

Fixtures and scripts land in **Phase 6** of the [implementation plan](../docs/IMPLEMENTATION_PLAN.md).
