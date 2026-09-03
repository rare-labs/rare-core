"""Run-length and time-window declustering. Returns indices of cluster representatives."""

import numpy as np

from evt.errors import ErrorCode, ValidationError
from evt.types import Tail


def _representative_index(values: np.ndarray, cluster: np.ndarray, tail: Tail) -> int:
    subset = values[cluster]
    offset = int(np.argmax(subset) if tail == "high" else np.argmin(subset))
    return int(cluster[offset])


def _representatives(values: np.ndarray, clusters: list[np.ndarray], tail: Tail) -> np.ndarray:
    chosen = [_representative_index(values, cluster, tail) for cluster in clusters]
    return np.asarray(chosen, dtype=np.intp)


def decluster_run(
    values: np.ndarray,
    exceedance_indices: np.ndarray,
    *,
    run_length: int,
    tail: Tail,
) -> np.ndarray:
    """Group exceedances separated by at most ``run_length`` non-exceedances."""
    if run_length < 0:
        raise ValidationError(
            "run_length must be >= 0",
            code=ErrorCode.INVALID_SHAPE,
        )
    idx = np.asarray(exceedance_indices, dtype=np.intp)
    if idx.size == 0:
        return idx
    clusters: list[list[int]] = [[int(idx[0])]]
    for raw in idx[1:]:
        current = int(raw)
        previous = clusters[-1][-1]
        gap = current - previous - 1
        if gap <= run_length:
            clusters[-1].append(current)
        else:
            clusters.append([current])
    packed = [np.asarray(cluster, dtype=np.intp) for cluster in clusters]
    return _representatives(values, packed, tail)


def decluster_window(
    values: np.ndarray,
    timestamps: np.ndarray,
    exceedance_indices: np.ndarray,
    *,
    window: np.timedelta64,
    tail: Tail,
) -> np.ndarray:
    """Cluster exceedances whose consecutive timestamps differ by at most ``window``."""
    if window <= np.timedelta64(0, "ns"):
        raise ValidationError(
            "window must be a positive timedelta64",
            code=ErrorCode.INVALID_SHAPE,
        )
    idx = np.asarray(exceedance_indices, dtype=np.intp)
    if idx.size == 0:
        return idx
    times = timestamps[idx]
    clusters: list[np.ndarray] = []
    start = 0
    for i in range(1, idx.size):
        if times[i] - times[i - 1] > window:
            clusters.append(idx[start:i])
            start = i
    clusters.append(idx[start:])
    return _representatives(values, clusters, tail)
