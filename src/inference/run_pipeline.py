#!/usr/bin/env python3
"""
CLI Entrypoint for the Staged Video Processing Pipeline.
"""
import argparse
import sys
import os
import csv
import json
import logging
from pathlib import Path
import cv2
import numpy as np

import torch
from ultralytics import YOLO

# Add src to path if needed
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR / "src"))

from inference.frame_selector import FrameSelector
from inference.staged_pipeline import StagedPipeline
from digit_recognizer.digit_recognizer import load_classifier, get_device

def setup_logger(verbose: bool):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

def get_optimal_batch_size() -> int:
    if torch.cuda.is_available():
        return 8
    return 1

def load_video_frames(video_path: Path) -> list:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def handle_paths(path_str: str, default_rel: bool = True) -> Path:
    p = Path(path_str)
    # The user relaxed the strict relative requirement, so we just return the path directly,
    # but we can resolve it against BASE_DIR if it's not absolute and we want a default relative.
    if not p.is_absolute() and default_rel:
        return BASE_DIR / p
    return p

def main():
    parser = argparse.ArgumentParser(description="Staged Video Processing Pipeline")
    
    parser.add_argument("--video", required=True, help="Path to video file")
    
    # Frame selection args
    frame_group = parser.add_mutually_exclusive_group(required=True)
    frame_group.add_argument("--frames", help="Specific frames (comma-list, json, or file)")
    frame_group.add_argument("--k", type=int, help="Number of frames to select")
    parser.add_argument("--strategy", choices=["uniform", "motion_and_blur", "detection-driven"], 
                        default="uniform", help="Frame selection strategy")
                        
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--model-dir", default="outputs/trained_models", help="Directory containing models")
    parser.add_argument("--classifier", default="outputs/trained_models/digit_recognizer.pt", help="Path to classifier model")
    
    parser.add_argument("--batch-size", default="auto", help="Batch size (int or 'auto')")
    parser.add_argument("--workers", default="auto", help="Workers (int or 'auto')")
    
    parser.add_argument("--save-stages", action=argparse.BooleanOptionalAction, default=True, help="Save intermediate stage images")
    parser.add_argument("--summary-format", choices=["csv", "json", "both"], default="both", help="Summary output format")
    
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and extract frames without running models")
    parser.add_argument("--verbose", action="store_true", help="Increase logging")
    
    args = parser.parse_args()
    setup_logger(args.verbose)
    
    video_path = handle_paths(args.video)
    out_dir = handle_paths(args.out_dir)
    model_dir = handle_paths(args.model_dir)
    classifier_path = handle_paths(args.classifier)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Batch size resolution
    if args.batch_size == "auto":
        batch_size = get_optimal_batch_size()
        logging.info(f"Auto batch size selected: {batch_size}")
    else:
        batch_size = int(args.batch_size)

    # Strategy setup
    if args.k is not None:
        selector = FrameSelector(strategy=args.strategy, top_k=args.k)
    else:
        # Simplistic parsing for --frames comma list
        # In a full implementation, you'd parse file/json
        frame_list = [int(x.strip()) for x in args.frames.split(",")]
        # We can simulate this by overriding select_indices on a dummy selector
        class ExactSelector(FrameSelector):
            def select_indices(self, frames):
                return [i for i in frame_list if i < len(frames)]
        selector = ExactSelector()

    logging.info(f"Loading video from {video_path}...")
    frames = load_video_frames(video_path)
    logging.info(f"Loaded {len(frames)} frames.")
    
    if args.dry_run:
        logging.info("DRY RUN: Extracting frames and generating dummy summary...")
        indices = selector.select_indices(frames)
        records = []
        for i in indices:
            records.append({
                "frame": i,
                "status": "ok",
                "prediction": "dry-run",
                "global_conf": 1.0,
                "n_individuals": 2,
                "paths": {}
            })
            if args.save_stages:
                fd = out_dir / f"frame_{i:04d}"
                fd.mkdir(parents=True, exist_ok=True)
                # Save just the raw frame as a dry-run artifact
                cv2.imwrite(str(fd / "stage_01_raw.png"), frames[i])
                records[-1]["paths"]["stage_01_raw"] = str(fd / "stage_01_raw.png")
                
    else:
        logging.info("Loading models...")
        device = get_device()
        global_model = YOLO(str(model_dir / "globalbb.pt"))
        indiv_model = YOLO(str(model_dir / "individualbb.pt"))
        classifier = load_classifier(str(classifier_path), data_dir=str(BASE_DIR / "data" / "digits_data"))
        
        logging.info("Running pipeline...")
        pipeline = StagedPipeline(selector=selector, out_dir=out_dir, batch_size=batch_size, save_stages=args.save_stages)
        records = pipeline.run(frames, global_model, indiv_model, classifier, device)

    # Save summaries
    if args.summary_format in ["csv", "both"]:
        csv_path = out_dir / "summary.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "status", "prediction", "global_conf", "n_individuals", "paths"])
            writer.writeheader()
            for r in records:
                # Stringify paths for CSV
                r_copy = r.copy()
                r_copy["paths"] = json.dumps(r["paths"])
                writer.writerow(r_copy)
        logging.info(f"Saved CSV summary to {csv_path}")

    if args.summary_format in ["json", "both"]:
        json_path = out_dir / "summary.json"
        with open(json_path, 'w') as f:
            json.dump(records, f, indent=2)
        logging.info(f"Saved JSON summary to {json_path}")
        
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
