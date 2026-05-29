"""
Frame selection utilities.

Provides a small `FrameSelector` class with selectable strategies:
- 'interval': pick every N-th frame
- 'top_variance': pick top-K frames according to variance (uses pipeline_utils)
"""
from typing import List, Sequence, Any, Optional

from ..pipeline_utils import variance_of_frame, top_k_indices_by_variance


class FrameSelector:
    def __init__(self, strategy: str = "interval", interval: int = 1, top_k: int = 1):
        if strategy not in ("interval", "top_variance"):
            raise ValueError("strategy must be 'interval' or 'top_variance'")
        if interval < 1:
            raise ValueError("interval must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.strategy = strategy
        self.interval = interval
        self.top_k = top_k

    def select_indices(self, frames: Sequence[Any]) -> List[int]:
        """Return indices of frames selected according to the strategy."""
        n = len(frames)
        if self.strategy == "interval":
            return list(range(0, n, self.interval))
        else:  # top_variance
            k = min(self.top_k, n)
            return top_k_indices_by_variance(frames, k)

    def select(self, frames: Sequence[Any]) -> List[Any]:
        """Return the selected frame objects in original order when appropriate."""
        idx = self.select_indices(frames)
        # For top_variance we return frames in the order of the indices returned
        # For interval we already returned increasing indices.
        return [frames[i] for i in idx]
