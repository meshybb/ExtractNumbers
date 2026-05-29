"""
Small helper utilities used by the inference pipeline.

These helpers are intentionally minimal and dependency-free so tests
and simple offline pipelines can run without heavy third-party libs.
"""
from typing import Iterable, List, Sequence, Any


def is_sequence(obj: Any) -> bool:
    """Return True if obj is a non-string iterable (list/tuple/etc.)."""
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))


def numeric_values_from_frame(frame: Any) -> List[float]:
    """
    Extract numeric values from a frame-like object for simple statistics.

    Supported frame shapes:
    - single number (int/float) -> [value]
    - iterable of numbers -> flattened list of numbers
    - nested iterables -> flattened recursively
    """
    vals: List[float] = []

    if isinstance(frame, (int, float)):
        return [float(frame)]

    if not is_sequence(frame):
        raise TypeError("Unsupported frame type for numeric extraction")

    def _flatten(obj):
        if isinstance(obj, (int, float)):
            vals.append(float(obj))
        elif is_sequence(obj):
            for x in obj:
                _flatten(x)
        else:
            raise TypeError("Non-numeric element encountered in frame")

    _flatten(frame)
    return vals


def variance_of_frame(frame: Any) -> float:
    """Compute simple population variance of numeric values inside `frame`."""
    vals = numeric_values_from_frame(frame)
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def top_k_indices_by_variance(frames: Sequence[Any], k: int) -> List[int]:
    """Return indices of the top-k frames with highest variance (descending)."""
    if k <= 0:
        return []
    variances = [(i, variance_of_frame(f)) for i, f in enumerate(frames)]
    variances.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in variances[:k]]
