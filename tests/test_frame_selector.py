import pytest
import numpy as np
import cv2

from src.inference.frame_selector import FrameSelector

def test_interval_selection():
    frames = list(range(10))
    sel = FrameSelector(strategy="interval", interval=3)
    indices = sel.select_indices(frames)
    assert indices == [0, 3, 6, 9]
    selected = sel.select(frames)
    assert selected == [0, 3, 6, 9]

def test_uniform_selection():
    frames = list(range(10))
    sel = FrameSelector(strategy="uniform", top_k=3)
    indices = sel.select_indices(frames)
    assert indices == [0, 4, 9] # round(0 * 9/2), round(1 * 9/2), round(2 * 9/2)
    selected = sel.select(frames)
    assert selected == [0, 4, 9]

def test_top_variance_selection_simple():
    frames = [
        [1, 1, 1],          # variance 0
        [1, 2, 3],          # variance > 0
        [0, 10, -10, 5],    # larger variance
        [2, 2, 2, 2],       # variance 0
    ]
    sel = FrameSelector(strategy="top_variance", top_k=2)
    indices = sel.select_indices(frames)
    assert len(indices) == 2
    assert 2 in indices

def test_motion_and_blur_selection():
    # Create synthetic frames with clear laplacian differences
    # Flat frame
    f1 = np.ones((64, 64, 3), dtype=np.uint8) * 128
    # High frequency frame (checkerboard)
    f2 = np.ones((64, 64, 3), dtype=np.uint8) * 128
    f2[::2, ::2] = 255
    f2[1::2, 1::2] = 0
    
    # f3 is exactly same as f2, so motion energy is 0
    f3 = f2.copy()
    
    # f4 is f2 shifted, high motion energy
    f4 = np.roll(f2, shift=1, axis=0)
    
    frames = [f1, f2, f3, f4]
    
    # We set lap_thresh low enough to capture f2, f3, f4, but maybe not f1
    sel = FrameSelector(strategy="motion_and_blur", top_k=2, lap_thresh=10.0, sample_n=4)
    indices = sel.select_indices(frames)
    
    # f4 has high laplacian AND high motion (difference from f3). It should be selected.
    # f1 should probably be rejected or have low score.
    # We expect 2 frames.
    assert len(indices) == 2
    # f4 should definitely be in there because of high motion and high variance
    assert 3 in indices
