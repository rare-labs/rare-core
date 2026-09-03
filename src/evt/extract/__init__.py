"""Extreme extraction: block maxima and peaks over threshold."""

from evt.extract.block import extract_block_maxima
from evt.extract.pot import extract_peaks_over_threshold

__all__ = [
    "extract_block_maxima",
    "extract_peaks_over_threshold",
]
