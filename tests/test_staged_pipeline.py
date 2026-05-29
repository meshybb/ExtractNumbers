from src.inference.frame_selector import FrameSelector
from src.inference.staged_pipeline import StagedPipeline


def test_staged_pipeline_interval():
    frames = ["a", "b", "c", "d", "e"]
    selector = FrameSelector(strategy="interval", interval=2)
    pipeline = StagedPipeline(selector)
    results = pipeline.run(frames)
    # Expect indices 0,2,4
    assert [r["index"] for r in results] == [0, 2, 4]
    # Each output should be a tuple (input, "enhanced")
    for r in results:
        assert r["output"][1] == "enhanced"
        assert r["output"][0] == r["input"]
