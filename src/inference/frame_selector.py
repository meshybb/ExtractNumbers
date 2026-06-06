"""
Frame selection utilities.

Provides a small `FrameSelector` class with selectable strategies:
- 'interval': pick every N-th frame
- 'top_variance': pick top-K frames according to variance (uses pipeline_utils)
- 'uniform': evenly spaced `k` frames across the sequence.
- 'motion_and_blur': sample `N` candidate frames, filter by Laplacian variance, pick top `k` based on a combined sharpness and motion score.
"""
import math
from typing import List, Sequence, Any, Optional
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from ..pipeline_utils import top_k_indices_by_variance


class FrameSelector:
    def __init__(
        self, 
        strategy: str = "interval", 
        interval: int = 1, 
        top_k: int = 1,
        lap_thresh: float = 50.0,
        sample_n: int = 500
    ):
        valid_strategies = ("interval", "top_variance", "uniform", "motion_and_blur", "random_1_in_10")
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
        if interval < 1:
            raise ValueError("interval must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
            
        self.strategy = strategy
        self.interval = interval
        self.top_k = top_k
        self.lap_thresh = lap_thresh
        self.sample_n = sample_n

    def select_indices(self, frames: Sequence[Any]) -> List[int]:
        """Return indices of frames selected according to the strategy."""
        n = len(frames)
        if n == 0:
            return []

        if self.strategy == "random_1_in_10":
            import random
            indices = []
            for i in range(0, n, 10):
                chunk_size = min(10, n - i)
                indices.append(i + random.randint(0, chunk_size - 1))
            return indices
            
        elif self.strategy == "interval":
            return list(range(0, n, self.interval))
            
        elif self.strategy == "uniform":
            k = min(self.top_k, n)
            if k == 1:
                return [n // 2]
            return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]
            
        elif self.strategy == "top_variance":
            k = min(self.top_k, n)
            return top_k_indices_by_variance(frames, k)
            
        elif self.strategy == "motion_and_blur":
            if not HAS_CV2:
                raise ImportError("OpenCV (cv2) is required for 'motion_and_blur' strategy.")
                
            # Sample N candidates uniformly
            k = min(self.top_k, n)
            n_candidates = min(self.sample_n, n)
            if n_candidates == 1:
                return [0]
                
            candidate_indices = [int(round(i * (n - 1) / (n_candidates - 1))) for i in range(n_candidates)]
            
            # Compute Laplacian variance and motion energy
            valid_candidates = []
            lap_scores = []
            motion_scores = []
            
            for i, idx in enumerate(candidate_indices):
                frame = frames[idx]
                if not isinstance(frame, np.ndarray):
                    # Attempt to convert to numpy array if possible, or just fail safely
                    frame = np.array(frame)
                    
                if frame.ndim == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                    
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                if lap_var >= self.lap_thresh:
                    # Compute motion energy (absdiff with previous candidate if available)
                    motion = 0.0
                    if i > 0:
                        prev_idx = candidate_indices[i-1]
                        prev_frame = frames[prev_idx]
                        if not isinstance(prev_frame, np.ndarray):
                            prev_frame = np.array(prev_frame)
                        if prev_frame.ndim == 3:
                            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        else:
                            prev_gray = prev_frame
                        # Calculate mean absolute difference as motion energy
                        if gray.shape == prev_gray.shape:
                            motion = cv2.absdiff(gray, prev_gray).mean()
                            
                    valid_candidates.append(idx)
                    lap_scores.append(lap_var)
                    motion_scores.append(motion)
            
            if not valid_candidates:
                # Fallback if all rejected: pick top-k by laplacian from what we computed
                # Re-compute without threshold to avoid empty returns
                # Here we just fallback to uniform
                return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]
                
            # Normalize scores to combine them equally
            lap_arr = np.array(lap_scores)
            mot_arr = np.array(motion_scores)
            
            # Min-Max normalization
            if lap_arr.max() > lap_arr.min():
                lap_norm = (lap_arr - lap_arr.min()) / (lap_arr.max() - lap_arr.min())
            else:
                lap_norm = np.zeros_like(lap_arr)
                
            if mot_arr.max() > mot_arr.min():
                mot_norm = (mot_arr - mot_arr.min()) / (mot_arr.max() - mot_arr.min())
            else:
                mot_norm = np.zeros_like(mot_arr)
                
            combined_scores = lap_norm + mot_norm
            
            # Sort by combined score descending
            scored_indices = list(zip(valid_candidates, combined_scores))
            scored_indices.sort(key=lambda x: x[1], reverse=True)
            
            selected = [idx for idx, _ in scored_indices[:k]]
            return sorted(selected)

    def select(self, frames: Sequence[Any]) -> List[Any]:
        """Return the selected frame objects in original order when appropriate."""
        idx = self.select_indices(frames)
        return [frames[i] for i in idx]
