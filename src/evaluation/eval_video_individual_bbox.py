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
from src.utils.bbox_utils import nms_individual_boxes
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
    
    parser = argparse.ArgumentParser(description="Stage 3: Individual Bounding Box Video Evaluation")
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
    INDIV_MODEL_PATH = os.path.join(TRAINED_DIR, "individualbb.pt")
    REPORTS_DIR = os.path.join(args.output_dir, "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if not os.path.exists(INDIV_MODEL_PATH):
        print(f"❌ Error: Individual model not found at {INDIV_MODEL_PATH}")
        sys.exit(1)

    print("\n--- Stage 3: Individual Bounding Box Video Evaluation ---")
    model = YOLO(INDIV_MODEL_PATH)

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

    results = []

    for vid_sample in tqdm(eval_videos, desc="Evaluating Individual BB in Videos"):
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
            gt_global_boxes, digit_info, has_digit_boxes, _ = get_video_gt_from_anno(anno_path, frame_idx)
            if not has_digit_boxes or not gt_global_boxes:
                continue

            # Standard evaluation crops the ground truth global box
            gx1, gy1, gx2, gy2 = map(int, gt_global_boxes[0])
            h, w = img.shape[:2]
            gx1, gy1 = max(0, gx1), max(0, gy1)
            gx2, gy2 = min(w, gx2), min(h, gy2)
            crop = img[gy1:gy2, gx1:gx2]
            
            if crop.size == 0:
                continue

            res = model.predict(source=crop, imgsz=256, verbose=False)
            pred_boxes = []
            if res and len(res[0].boxes) > 0:
                iboxes = res[0].boxes.xyxy.cpu().numpy()
                iconfs = res[0].boxes.conf.cpu().numpy()
                pred_boxes, _ = nms_individual_boxes(iboxes, iconfs, iou_thresh=0.45)

            # Map gt digit boxes to crop coordinates (scale is 1.0 since no upscale here)
            gt_crop_boxes = []
            for digit in digit_info:
                dx1, dy1, dx2, dy2 = digit['bbox']
                nx1, ny1 = dx1 - gx1, dy1 - gy1
                nx2, ny2 = dx2 - gx1, dy2 - gy1
                gt_crop_boxes.append((nx1, ny1, nx2, ny2))

            # Matching logic (standard metrics: IoU, TP, FP, FN)
            matched_gt = set()
            tps = 0
            ious = []

            for p_idx, pbox in enumerate(pred_boxes):
                best_iou = 0
                best_gt_idx = -1
                for g_idx, gbox in enumerate(gt_crop_boxes):
                    if g_idx in matched_gt:
                        continue
                    iou = calculate_iou(gbox, pbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                
                if best_iou >= 0.5:
                    tps += 1
                    matched_gt.add(best_gt_idx)
                    ious.append(best_iou)

            fps = len(pred_boxes) - tps
            fns = len(gt_crop_boxes) - tps
            
            precision = tps / len(pred_boxes) if len(pred_boxes) > 0 else 1.0 if len(gt_crop_boxes) == 0 else 0.0
            recall = tps / len(gt_crop_boxes) if len(gt_crop_boxes) > 0 else 1.0
            
            results.append({
                'video_id':   sample_id,
                'frame_idx':  frame_idx,
                'category':   category,
                'precision':  precision,
                'recall':     recall,
                'mean_iou':   np.mean(ious) if ious else 0.0,
                'gt_count':   len(gt_crop_boxes),
                'pred_count': len(pred_boxes)
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
    log_print("📊 VIDEO STAGE 3: INDIVIDUAL BOUNDING BOX METRICS")
    log_print("="*50)

    overall_prec  = df['precision'].mean()
    overall_rec   = df['recall'].mean()
    overall_iou   = df[df['pred_count'] > 0]['mean_iou'].mean() if len(df[df['pred_count'] > 0]) > 0 else 0.0

    log_print(f"Overall Precision: {overall_prec:.2%}")
    log_print(f"Overall Recall:    {overall_rec:.2%}")
    log_print(f"Overall Mean IoU:  {overall_iou:.4f}")

    log_print("\n📈 PERFORMANCE BY CATEGORY:")
    log_print(f"{'Category':<15} {'Precision':>12} {'Recall':>12} {'Mean IoU':>12} {'Count':>7}")
    log_print("-" * 65)

    for cat in sorted(df['category'].unique()):
        c = df[df['category'] == cat]
        prec = c['precision'].mean()
        rec  = c['recall'].mean()
        miou = c[c['pred_count'] > 0]['mean_iou'].mean() if len(c[c['pred_count'] > 0]) > 0 else 0.0
        log_print(f"{cat:<15} {prec:>12.2%} {rec:>12.2%} {miou:>12.4f} {len(c):>7}")

    log_print(f"\n{'OVERALL':<15} {overall_prec:>12.2%} {overall_rec:>12.2%} {overall_iou:>12.4f} {len(df):>7}")

    # Save reports
    report_txt_path = os.path.join(REPORTS_DIR, "video_stage3_individual_bbox_summary.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n📝 Text report saved to: {report_txt_path}")

    csv_path = os.path.join(REPORTS_DIR, "video_stage3_individual_bbox_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"💾 Detailed CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
