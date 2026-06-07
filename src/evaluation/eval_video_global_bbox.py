import os
import sys
import pandas as pd
import numpy as np
import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.utils.data_utils import iter_video_samples, get_video_gt_from_anno, create_mock_video_dataset
from src.utils.metrics import calculate_iou
from src.utils.bbox_utils import merge_global_boxes
from src.inference.frame_selector import FrameSelector

def load_video_frames(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def main():
    import argparse
    import random
    from collections import defaultdict
    
    parser = argparse.ArgumentParser(description="Stage 1: Global Bounding Box Video Evaluation")
    parser.add_argument("--max-samples", type=int, default=10, help="Max video samples to evaluate")
    parser.add_argument("--balanced", action="store_true", help="Use equal/balanced split for video categories")
    parser.add_argument("--data-root", type=str, default=os.path.join(BASE_DIR, "data", "video_data"), help="Path to video dataset root")
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "outputs"), help="Base directory for outputs")
    parser.add_argument("--strategy", type=str, default="annotated", choices=["annotated", "uniform", "motion_and_blur", "random_1_in_10"],
                        help="Frame selection strategy")
    parser.add_argument("--k", type=int, default=5, help="Number of frames to select per video (ignored if strategy is 'annotated')")
    args = parser.parse_args()

    # Paths
    TRAINED_DIR = os.path.join(BASE_DIR, "outputs", "trained_models")
    GLOBAL_MODEL_PATH = os.path.join(TRAINED_DIR, "globalbb.pt")
    REPORTS_DIR = os.path.join(args.output_dir, "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if not os.path.exists(GLOBAL_MODEL_PATH):
        print(f"❌ Error: Global model not found at {GLOBAL_MODEL_PATH}")
        sys.exit(1)

    print("\n--- Stage 1: Global Bounding Box Video Evaluation ---")
    model = YOLO(GLOBAL_MODEL_PATH)

    video_samples = list(iter_video_samples(args.data_root))
    if not video_samples:
        print("⚠️ No video samples found. Generating mock video dataset...")
        create_mock_video_dataset(args.data_root)
        video_samples = list(iter_video_samples(args.data_root))

    if not video_samples:
        print("❌ Error: No video samples available.")
        sys.exit(1)

    # Group by category for sampling
    samples_by_cat = defaultdict(list)
    for s in video_samples:
        samples_by_cat[s['category']].append(s)

    random.seed(42)
    eval_videos = []
    if args.balanced:
        target_per_cat = args.max_samples // len(samples_by_cat)
        for cat, samps in samples_by_cat.items():
            random.shuffle(samps)
            eval_videos.extend(samps[:min(target_per_cat, len(samps))])
    else:
        total_vids = sum(len(s) for s in samples_by_cat.values())
        for cat, samps in samples_by_cat.items():
            random.shuffle(samps)
            vids_per_cat = max(1, int(round(args.max_samples * (len(samps) / total_vids))))
            eval_videos.extend(samps[:vids_per_cat])
    random.shuffle(eval_videos)

    print(f"Evaluating {len(eval_videos)} video samples across categories: {list(samples_by_cat.keys())}")

    results = []

    for vid_sample in tqdm(eval_videos, desc="Evaluating Global BB in Videos"):
        vid_path = vid_sample['video_path']
        anno_path = vid_sample['anno_path']
        category = vid_sample['category']
        sample_id = vid_sample['sample_id']
        
        try:
            frames = load_video_frames(vid_path)
        except Exception:
            continue
            
        try:
            with open(anno_path, 'r') as f:
                anno_json = json.load(f)
            annotated_frame_keys = list(anno_json.get("frames", {}).keys())
            annotated_frame_indices = sorted([int(k) for k in annotated_frame_keys])
        except Exception:
            continue
            
        # Determine frame indices to evaluate
        if args.strategy == "annotated":
            eval_frame_indices = [idx for idx in annotated_frame_indices if idx < len(frames)]
        else:
            selector = FrameSelector(strategy=args.strategy, top_k=args.k)
            selected_indices = selector.select_indices(frames)
            eval_frame_indices = [idx for idx in selected_indices if idx in annotated_frame_indices and idx < len(frames)]

        for frame_idx in eval_frame_indices:
            img = frames[frame_idx]
            global_boxes, _, _, _ = get_video_gt_from_anno(anno_path, frame_idx)
            if not global_boxes:
                continue

            res = model.predict(source=img, imgsz=256, verbose=False)
            pred_global = None
            iou = 0.0
            conf = 0.0

            if res and len(res[0].boxes) > 0:
                all_gboxes = res[0].boxes.xyxy.cpu().numpy()
                pred_global = merge_global_boxes(all_gboxes)
                conf = res[0].boxes.conf.max().item()
                if global_boxes:
                    iou = calculate_iou(global_boxes[0], pred_global)

            results.append({
                'video_id':   sample_id,
                'frame_idx':  frame_idx,
                'category':   category,
                'iou':        iou,
                'confidence': conf,
                'detected':   pred_global is not None,
                'hit_05':     iou >= 0.5,
                'hit_075':    iou >= 0.75,
            })

    if not results:
        print("⚠️ No evaluation results generated.")
        sys.exit(0)

    df = pd.DataFrame(results)

    report_lines = []
    def log_print(text=""):
        print(text)
        report_lines.append(str(text))

    log_print("\n" + "="*50)
    log_print("📊 VIDEO STAGE 1: GLOBAL BOUNDING BOX METRICS")
    log_print("="*50)

    detected_df = df[df['detected'] == True]
    overall_map50  = df['hit_05'].mean()
    overall_prec   = detected_df['hit_05'].sum() / len(detected_df) if len(detected_df) > 0 else 0.0
    overall_recall = df['detected'].mean()
    overall_iou    = df['iou'].mean()

    log_print(f"Overall mAP@0.5:  {overall_map50:.2%}")
    log_print(f"Overall Precision:{overall_prec:.2%}")
    log_print(f"Overall Recall:   {overall_recall:.2%}")
    log_print(f"Overall Mean IoU: {overall_iou:.4f}")

    log_print("\n📈 PERFORMANCE BY CATEGORY:")
    log_print(f"{'Category':<15} {'mAP@0.5':>10} {'Precision':>10} {'Recall':>10} {'Mean IoU':>10} {'Count':>7}")
    log_print("-" * 65)

    for cat in sorted(df['category'].unique()):
        c = df[df['category'] == cat]
        c_det = c[c['detected'] == True]
        map50  = c['hit_05'].mean()
        prec   = c_det['hit_05'].sum() / len(c_det) if len(c_det) > 0 else 0.0
        recall = c['detected'].mean()
        miou   = c['iou'].mean()
        log_print(f"{cat:<15} {map50:>10.2%} {prec:>10.2%} {recall:>10.2%} {miou:>10.4f} {len(c):>7}")

    log_print(f"\n{'OVERALL':<15} {overall_map50:>10.2%} {overall_prec:>10.2%} {overall_recall:>10.2%} {overall_iou:>10.4f} {len(df):>7}")

    # Save reports
    report_txt_path = os.path.join(REPORTS_DIR, "video_stage1_global_bbox_summary.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n📝 Text report saved to: {report_txt_path}")

    csv_path = os.path.join(REPORTS_DIR, "video_stage1_global_bbox_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"💾 Detailed CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
