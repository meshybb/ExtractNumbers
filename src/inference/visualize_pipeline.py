import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from image_preprocessing.digit_preprocessor import enhance_digit
from digit_recognizer.digit_recognizer import build_digit_model, get_device, preprocess_crop
from utils.bbox_utils import merge_global_boxes, nms_individual_boxes

def visualize_pipeline(image_path, model_dir, output_path=None):
    device = get_device()
    
    # Individual model components
    GLOBAL_MODEL_PATH = os.path.join(model_dir, "globalbb.pt")
    INDIV_MODEL_PATH = os.path.join(model_dir, "individualbb.pt")
    CLASSIFIER_PATH = os.path.join(model_dir, "digit_classifier.pth")
    
    # Load Models
    print("Loading models...")
    global_model = YOLO(GLOBAL_MODEL_PATH)
    indiv_model = YOLO(INDIV_MODEL_PATH)
    classifier = build_digit_model()
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier.to(device).eval()
    
    # Read Image
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Global Detection
    res1 = global_model.predict(source=img, imgsz=256, verbose=False)
    if not res1 or len(res1[0].boxes) == 0:
        print("No numbers detected in global detection stage.")
        return
    
    # Merge all detected global boxes (handling duplicates)
    all_gboxes = res1[0].boxes.xyxy.cpu().numpy()
    gbox = merge_global_boxes(all_gboxes)
    gx1, gy1, gx2, gy2 = map(int, gbox)
    h, w = img.shape[:2]
    gx1, gy1, gx2, gy2 = max(0, gx1), max(0, gy1), min(w, gx2), min(h, gy2)
    crop = img[gy1:gy2, gx1:gx2]
    
    # 2. Enhancement
    sharp = enhance_digit(crop, upscale_factor=2.0)
    
    # 3. Individual Detection
    res2 = indiv_model.predict(source=sharp, imgsz=256, verbose=False)
    if not res2 or len(res2[0].boxes) == 0:
        print("Detected container but no individual digits found.")
        return
    
    # Sort and NMS boxes
    iboxes = res2[0].boxes.xyxy.cpu().numpy()
    iconfs = res2[0].boxes.conf.cpu().numpy()
    iboxes, iconfs = nms_individual_boxes(iboxes, iconfs, iou_thresh=0.45)
    
    iboxes = sorted(iboxes, key=lambda b: b[0])
    
    # 4. Classification
    digits = []
    for ibox in iboxes:
        inputs = preprocess_crop(sharp, (ibox[0], ibox[1], ibox[2], ibox[3])).unsqueeze(0).to(device)
        with torch.no_grad():
            digit = classifier(inputs).argmax(dim=1).item()
            digits.append(str(digit))
            
    final_number = "".join(digits)
    print(f"Final Predicted Number: {final_number}")
    
    # ---------------- VISUALIZATION ----------------
    print("Generating pipeline visualization...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(f"FULL EXTRACTION PIPELINE: Multi-Stage Progression\nPredicted: {final_number}", fontsize=20, fontweight='bold')
    
    # Plot 1: Original Image
    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Original Image")
    axes[0].axis('off')
    
    # Plot 2: Global Detection
    axes[1].imshow(img_rgb)
    rect_global = Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1, linewidth=2, edgecolor='g', facecolor='none', linestyle='--')
    axes[1].add_patch(rect_global)
    axes[1].set_title("2. Global Detection (GlobalBB)")
    axes[1].axis('off')
    
    # Plot 3: Raw Crop
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    axes[2].imshow(crop_rgb)
    axes[2].set_title("3. Raw Crop (Unsharpened)")
    axes[2].axis('off')
    
    # Plot 4: Image Enhancement
    sharp_rgb = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
    axes[3].imshow(sharp_rgb)
    axes[3].set_title("4. Image Enhancement (Sharpening)")
    axes[3].axis('off')
    
    # Plot 5: Individual Detection
    axes[4].imshow(sharp_rgb)
    for i, ibox in enumerate(iboxes):
        ix1, iy1, ix2, iy2 = map(int, ibox)
        rect_indiv = Rectangle((ix1, iy1), ix2 - ix1, iy2 - iy1, linewidth=2, edgecolor='r', facecolor='none')
        axes[4].add_patch(rect_indiv)
        # Add digit label
        axes[4].text(ix1, max(0, iy1-5), digits[i], color='white', fontsize=12, fontweight='bold', bbox=dict(facecolor='red', alpha=0.5, pad=1))
        
    axes[4].set_title("5. Individual Detection (IndividualBB)")
    axes[4].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize the multi-stage extraction pipeline for a single image.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--model-dir", default=os.path.join(BASE_DIR, "outputs", "trained_models"), help="Directory containing models")
    parser.add_argument("--output", "-o", help="Path to save the generated visualization image (e.g., pipeline.png)")
    args = parser.parse_args()
    
    visualize_pipeline(args.image_path, args.model_dir, args.output)

if __name__ == "__main__":
    main()
