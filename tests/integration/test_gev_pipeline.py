"""End-to-end GEV extract → fit → return levels."""

import numpy as np
from scipy.stats import genextreme

from evt import extract_block_maxima, fit_gev, gev_return_level, make_extreme_series


def test_gev_pipeline_return_levels_finite() -> None:
    rng = np.random.default_rng(3)
    values = np.asarray(genextreme.rvs(c=-0.1, loc=8.0, scale=1.5, size=120, random_state=rng))
    series = make_extreme_series(values, tail="high")
    sample = extract_block_maxima(series, block_size=1, min_blocks=50)
    fit = fit_gev(sample)
    assert fit.converged
    for period in (2.0, 10.0, 50.0):
        rl = gev_return_level(fit, period)
        assert np.isfinite(rl.estimate)
        assert rl.return_period == period
        assert rl.method == "GEV"
