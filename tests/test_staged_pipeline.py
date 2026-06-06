import os
import subprocess
import pytest
import json
import csv
import cv2
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

def create_mock_video(path: Path, num_frames: int = 5):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, 20.0, (64, 64))
    for i in range(num_frames):
        frame = np.ones((64, 64, 3), dtype=np.uint8) * (i * 50 % 255)
        out.write(frame)
    out.release()

def test_staged_pipeline_dry_run(tmp_path):
    video_path = tmp_path / "mock.mp4"
    out_dir = tmp_path / "output"
    
    create_mock_video(video_path, num_frames=10)
    
    cli_path = BASE_DIR / "src" / "inference" / "run_pipeline.py"
    
    cmd = [
        sys.executable, str(cli_path),
        "--video", str(video_path),
        "--out-dir", str(out_dir),
        "--k", "3",
        "--strategy", "uniform",
        "--dry-run"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed with error:\n{result.stderr}"
    
    # Check outputs
    assert out_dir.exists()
    
    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    
    assert csv_path.exists()
    assert json_path.exists()
    
    # Verify JSON content
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    assert len(data) == 3
    for record in data:
        assert record["status"] == "ok"
        assert record["prediction"] == "dry-run"
        
        # Check that dry-run saved the raw image
        frame_idx = record["frame"]
        img_path = Path(record["paths"]["stage_01_raw"])
        assert img_path.exists()
        assert f"frame_{frame_idx:04d}" in str(img_path)

def test_staged_pipeline_dry_run_comma_list(tmp_path):
    video_path = tmp_path / "mock2.mp4"
    out_dir = tmp_path / "output2"
    
    create_mock_video(video_path, num_frames=10)
    
    cli_path = BASE_DIR / "src" / "inference" / "run_pipeline.py"
    
    cmd = [
        sys.executable, str(cli_path),
        "--video", str(video_path),
        "--out-dir", str(out_dir),
        "--frames", "1,5,8",
        "--dry-run"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed with error:\n{result.stderr}"
    
    # Verify JSON content
    json_path = out_dir / "summary.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    assert len(data) == 3
    frames_selected = [r["frame"] for r in data]
    assert frames_selected == [1, 5, 8]
