"""
A tiny staged pipeline that selects frames and applies a trivial enhancement.

This is a minimal, test-friendly pipeline:
- It uses a FrameSelector to pick frames.
- It applies `_enhance_frame` to each selected frame (identity with a tag).
- It returns a list of result dicts for easy assertions in tests.
"""
from typing import Any, List, Sequence, Dict

from .frame_selector import FrameSelector


class StagedPipeline:
    def __init__(self, selector: FrameSelector):
        self.selector = selector

    def _enhance_frame(self, frame: Any) -> Any:
        """
        Placeholder enhancement step.

        For testing and simple pipelines we return a tuple (frame, "enhanced").
        Real pipelines would replace this with actual processing.
        """
        return (frame, "enhanced")

    def run(self, frames: Sequence[Any]) -> List[Dict[str, Any]]:
        """
        Run the pipeline: select frames, enhance them, and return structured results.

        Returned list elements are dicts:
          {"index": original_index, "input": frame, "output": enhanced_frame}
        """
        selected_indices = self.selector.select_indices(frames)
        results: List[Dict[str, Any]] = []
        for i in selected_indices:
            inp = frames[i]
            out = self._enhance_frame(inp)
            results.append({"index": i, "input": inp, "output": out})
        return results
