"""Shared constants. Single source of truth — do not duplicate these values."""

SCHEMA_VERSION = "0.1"
DAYS_PER_YEAR = 365.25
GEV_MIN_SAMPLE = 3
GPD_MIN_SAMPLE = 2
BOOTSTRAP_MAX_REFIT_RETRIES = 10
# Switch to Gumbel/exponential formulas to avoid cancellation for tiny ξ.
XI_ZERO_ATOL = 1e-8
# Plotting position p_i = (i - 0.5) / n (Blom). See SPEC §4.7.
DEFAULT_PLOT_POSITION = "blom"
