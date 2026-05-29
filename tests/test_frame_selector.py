import pytest

from src.inference.frame_selector import FrameSelector


def test_interval_selection():
    frames = list(range(10))
    sel = FrameSelector(strategy="interval", interval=3)
    indices = sel.select_indices(frames)
    assert indices == [0, 3, 6, 9]
    selected = sel.select(frames)
    assert selected == [0, 3, 6, 9]


def test_top_variance_selection_simple():
    # frames are simple lists where variance increases with number range
    frames = [
        [1, 1, 1],          # variance 0
        [1, 2, 3],          # variance > 0
        [0, 10, -10, 5],    # larger variance
        [2, 2, 2, 2],       # variance 0
    ]
    sel = FrameSelector(strategy="top_variance", top_k=2)
    indices = sel.select_indices(frames)
    assert len(indices) == 2
    # ensure the index with largest variance is included (index 2)
    assert 2 in indices
