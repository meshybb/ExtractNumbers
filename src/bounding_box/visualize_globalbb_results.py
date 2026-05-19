import os
import sys

# Configure Matplotlib backend for headless environments
import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')

import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, "src"))

def main():
    parser = argparse.ArgumentParser(description="Visualize GlobalBB prediction results on test split.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory containing predictions and dataset splits (default: autodetect outputs/yolo_runs or outputs/bbox_comparison)")
    args = parser.parse_args()

    # Determine paths dynamically
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Autodetect based on which folder actually exists
        yolo_runs_path = os.path.join(BASE_DIR, "outputs", "yolo_runs")
        bbox_comparison_path = os.path.join(BASE_DIR, "outputs", "bbox_comparison")
        if os.path.exists(yolo_runs_path):
            output_dir = yolo_runs_path
        else:
            output_dir = bbox_comparison_path

    csv_path = os.path.join(output_dir, "globalbb_predictions.csv")
    val_images_dir = os.path.join(output_dir, "globalbb_dataset", "images", "val")
    output_image = os.path.join(output_dir, "globalbb_comparison_summary.png")

    if not os.path.exists(csv_path):
        print(f"Error: Could not find CSV at {csv_path}")
        return
    
    if not os.path.exists(val_images_dir):
        print(f"Error: Val directory not found at {val_images_dir}. Please run training first.")
        return

    # 1. Load prediction results.
    df = pd.read_csv(csv_path)
    
    # 2. Keep only samples that exist in the validation split.
    # Validation filenames follow: category_id.jpg (e.g., natural_10.jpg).
    val_filenames = set(os.listdir(val_images_dir))
    
    def is_test_image(row):
        # Extract the sample index (the part after '/').
        idx = row['sample_id'].split('/')[-1]
        target_filename = f"{row['category']}_{idx}.jpg"
        return target_filename in val_filenames

    df['is_test'] = df.apply(is_test_image, axis=1)
    test_df = df[df['is_test'] == True].copy()

    print(f"Filtering complete: Found {test_df['image_path'].nunique()} unique test images.")

    categories = ['svhn', 'race_numbers', 'handwritten']
    samples_per_cat = 2
    
    fig, axes = plt.subplots(len(categories), samples_per_cat, figsize=(10, 8))
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for i, cat in enumerate(categories):
        cat_df = test_df[test_df['category'] == cat]
        unique_images = cat_df['image_path'].unique()
        
        if len(unique_images) < samples_per_cat:
            print(f"Warning: Not enough test images for {cat}")
            continue
            
        selected_img_paths = np.random.choice(unique_images, samples_per_cat, replace=False)
        
        for j, img_path in enumerate(selected_img_paths):
            anno_path = img_path.replace("original.png", "annotations.json")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax = axes[i, j]
            ax.imshow(img)
            
            # Draw ground-truth box (green).
            from utils.data_utils import get_gt_from_anno
            if os.path.exists(anno_path):
                global_boxes, _, _, _ = get_gt_from_anno(anno_path)
                for x1, y1, x2, y2 in global_boxes:
                    ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=3))
            
            # Draw GlobalBB predicted boxes (red).
            predictions = cat_df[cat_df['image_path'] == img_path]
            for _, pred in predictions.iterrows():
                if pd.notna(pred['pred_x1']):
                    p_x1, p_y1, p_x2, p_y2 = pred['pred_x1'], pred['pred_y1'], pred['pred_x2'], pred['pred_y2']
                    ax.add_patch(plt.Rectangle((p_x1, p_y1), p_x2-p_x1, p_y2-p_y1, fill=False, edgecolor='red', linewidth=2, linestyle='--'))
                    ax.text(p_x1, p_y1-5, f"{pred['pred_conf']:.2f}", color='red', fontsize=8, weight='bold')

            ax.set_title(f"TEST IMAGE: {cat}")
            ax.axis('off')

    plt.suptitle("GlobalBB Final Test Results (Unseen Data)\nGreen = Truth | Red = Prediction", fontsize=10)
    plt.savefig(output_image, bbox_inches='tight')
    print(f"Test visualization saved to: {output_image}")
    
    if 'DISPLAY' in os.environ:
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot interactively: {e}")

if __name__ == "__main__":
    main()