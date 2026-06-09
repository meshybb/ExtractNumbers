import os
import subprocess
import pytest
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def create_mock_video_dataset_for_test(data_root: Path):
    """Generate a quick mock video dataset with annotations for testing."""
    import cv2
    import numpy as np
    import json
    
    dataset_dir = data_root / "mock_video"
    sample_dir = dataset_dir / "sample_001"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = sample_dir / "video.mp4"
    anno_path = sample_dir / "annotations.json"
    
    width, height = 64, 64
    num_frames = 5
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (width, height))
    
    anno_data = {
        "video_metadata": {
            "sample_id": "mock_video/sample_001",
            "width": width,
            "height": height,
            "fps": 10.0
        },
        "frames": {}
    }
    
    for i in range(num_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * (i * 40 % 255)
        out.write(frame)
        
        # Annotate every frame
        anno_data["frames"][str(i)] = {
            "detected_numbers": [
                {
                    "full_value": "78",
                    "full_bounding_box": {
                        "x": 10.0,
                        "y": 10.0,
                        "width": 40.0,
                        "height": 40.0
                    },
                    "digits": [
                        {
                            "label": 7,
                            "bounding_box": {
                                "x": 10.0,
                                "y": 10.0,
                                "width": 20.0,
                                "height": 40.0
                            }
                        },
                        {
                            "label": 8,
                            "bounding_box": {
                                "x": 30.0,
                                "y": 10.0,
                                "width": 20.0,
                                "height": 40.0
                            }
                        }
                    ]
                }
            ]
        }
            
    out.release()
    
    with open(anno_path, 'w') as f:
        json.dump(anno_data, f, indent=4)

def test_video_evaluation_scripts(tmp_path):
    """Test that all video evaluation scripts execute successfully on mock data."""
    data_root = tmp_path / "video_data"
    output_dir = tmp_path / "outputs"
    
    create_mock_video_dataset_for_test(data_root)
    
    scripts = [
        "eval_video_global_bbox.py",
        "eval_video_individual_bbox.py",
        "eval_video_digit_recog.py",
        "eval_video_pipeline.py"
    ]
    
    for script in scripts:
        script_path = BASE_DIR / "src" / "evaluation" / script
        
        cmd = [
            sys.executable, str(script_path),
            "--data-root", str(data_root),
            "--output-dir", str(output_dir),
            "--max-samples", "1",
            "--strategy", "annotated"
        ]
        
        # Run process
        result = subprocess.run(cmd, capture_output=True, text=True)
        # We assert success (exit code 0)
        assert result.returncode == 0, f"Script {script} failed. Stderr:\n{result.stderr}\nStdout:\n{result.stdout}"
        
        # Verify that reports are written
        assert output_dir.exists()
        reports_dir = output_dir / "reports"
        assert reports_dir.exists()
