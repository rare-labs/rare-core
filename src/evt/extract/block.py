"""Block maxima / minima extraction. Drops a trailing incomplete block (SPEC §4.2)."""

import numpy as np

from evt.errors import ErrorCode, ExtractionError, ValidationError
from evt.transform import transform_to_upper_tail
from evt.types import ExtremeSample, ExtremeSeries
from evt.validate import immutable_datetime64_ns, immutable_float64, validate_series


def extract_block_maxima(
    series: ExtremeSeries,
    *,
    block_size: int,
    min_blocks: int = 1,
) -> ExtremeSample:
    """Extract one extreme per complete block of ``block_size`` observations.

    The trailing incomplete block is dropped. Raises ``ExtractionError`` if the
    number of complete blocks is below ``min_blocks``.
    """
    validate_series(series.values, series.timestamps, series.tail)
    if block_size < 1:
        raise ValidationError("block_size must be >= 1", code=ErrorCode.INVALID_SHAPE)
    if min_blocks < 1:
        raise ValidationError("min_blocks must be >= 1", code=ErrorCode.INVALID_SHAPE)

    n_source = int(series.values.size)
    n_blocks = n_source // block_size
    if n_blocks < min_blocks:
        raise ExtractionError(
            f"complete blocks {n_blocks} is below min_blocks {min_blocks}",
            code=ErrorCode.INSUFFICIENT_BLOCKS,
        )

    usable = n_blocks * block_size
    blocked = np.reshape(series.values[:usable], (n_blocks, block_size))
    local = np.argmax(blocked, axis=1) if series.tail == "high" else np.argmin(blocked, axis=1)
    global_index = np.arange(n_blocks) * block_size + local
    raw = immutable_float64(series.values[global_index])
    transformed = transform_to_upper_tail(raw, series.tail)
    timestamps = None
    if series.timestamps is not None:
        timestamps = immutable_datetime64_ns(series.timestamps[global_index])
    return ExtremeSample(
        raw_values=raw,
        transformed_values=transformed,
        timestamps=timestamps,
        method="BM",
        threshold=None,
        excesses=None,
        n_source=n_source,
        exposure=None,
        exceedance_rate=None,
    )
