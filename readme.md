# ExtractNumbers

A comprehensive image recognition and segmentation dataset generation pipeline for digit extraction from noisy environments.

## Initial Setup

1. **Install Dependencies**:
   Ensure you have Python 3.12+ installed. Create a virtual environment and install the requirements:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Data Preparation**:
   The entire data fetching and processing pipeline is automated. Just run the following command from the project root:

   ```bash
   python src/prep_data.py
   ```

---

## 📂 Project Structure
The source code is organized into specialized modules:

* **[`src/training/`](src/training/README.md)**: Full pipeline training orchestrators.
* **[`src/inference/`](src/inference/README.md)**: Production prediction scripts.
* **[`src/data/`](src/data/README.md)**: Dataset loading and normalization.
* **[`src/bounding_box/`](src/bounding_box/README.md)**: Stage 1 & 3 YOLO detection.
* **[`src/image_preprocessing/`](src/image_preprocessing/README.md)**: Optional image preprocessing utilities.
* **[`src/digit_recognizer/`](src/digit_recognizer/README.md)**: Stage 3 ResNet18 classification.
* **[`src/evaluation/`](src/evaluation/README.md)**: Multi-stage benchmarking suite.
* **[`src/utils/`](src/utils/README.md)**: Shared helper functions.

For a comprehensive technical reference of all scripts, see the **[Source API Documentation](src/API.md)**.

---

## Pipeline Workflow

The extraction process is divided into three core stages:

1.  **Global Bounding-Box Detection (GlobalBB):** Localizes the entire number sequence within the noisy source image and extracts the crop.
2.  **Individual Digit Localization (IndividualBB):** Detects and segments each digit individually within the cropped sequence.
3.  **Neural Character Recognition (Classification):** ResNet18-based classification of localized digits into final values (0-9).

![Process Pipeline](assets/diagram.PNG)

---

### Core Pipeline Execution

The system is designed for high-performance batch processing and seamless model synchronization.

**To train and run the full batch pipeline:**
```bash
python src/training/train_pipeline.py
```

**To run prediction on a single image:**
```bash
python src/inference/predict_single.py path/to/image.png
```

#### Control Flags
- `--skip-train`: Automatically skips training if valid weights already exist.
- `--force-train`: Forces a fresh training cycle for both YOLO stages.
- `--analyze-only`: Skips heavy detection/training and generates reports from previous results.
- `--viz-only`: Regenerates the progression visualizations from existing predictions.

---

## Evaluation & Insights

The pipeline is evaluated across four isolated stages and one comprehensive end-to-end benchmark.

### 🔍 Metric Definitions
To ensure clarity across all reports, the following metrics are used:
*   **Mean IoU (Intersection over Union)**: Measures the spatial overlap between the predicted bounding box and the ground truth. A score of 1.0 is a perfect match.
*   **Detection Rate**: The percentage of samples where the model successfully proposed at least one bounding box.
*   **mAP@0.5**: "Mean Average Precision" at a 50% IoU threshold. This is the standard accuracy metric for object detection.
*   **Precision**: The percentage of positive predictions that were actually correct (Quality).
*   **Recall**: The percentage of actual ground truth objects that were successfully detected (Quantity).
*   **Full Sequence Accuracy**: The percentage of images where the **entire** predicted number string exactly matches the ground truth.
*   **Mean Digit Accuracy (Pos)**: The percentage of digits correctly identified at their specific index in the sequence.
*   **Succession Rate**: The probability that a digit is correct given that the *previous* digit was correct. This measures the model's ability to maintain consistency across a sequence.

### 📊 Stage 1: Global Bounding Box Detection
*Evaluates the ability to localize the entire number sequence.*

| Category | Mean IoU | Detection Rate | mAP@0.5 |
| :--- | :--- | :--- | :--- |
| **Overall** | **0.7977** | **99.15%** | **94.40%** |
| **Handwritten** | 0.6670 | 84.48% | 79.31% |
| **SVHN** | 0.8016 | 99.59% | 94.85% |

### 📊 Stage 2: Individual Digit Localization
*Evaluates digit segmentation within cropped sequences.*

| Category | Mean IoU | Recall |
| :--- | :--- | :--- |
| **Overall** | **0.8409** | **100.00%** |
| **Handwritten** | 0.8688 | 100.00% |
| **SVHN** | 0.8401 | 100.00% |

### 📊 Stage 3: Digit Classification
*Isolated classification performance (ResNet18).*

| Category | Accuracy | Support |
| :--- | :--- | :--- |
| **Overall** | **98.99%** | **4461 digits** |
| **Handwritten** | **99.51%** | 205 digits |
| **SVHN** | **98.97%** | 4256 digits |

### 🏆 Full End-to-End Pipeline Performance
*Master benchmark: Raw pixels → Final predicted string.*

| Metric | Overall | Handwritten | SVHN |
| :--- | :--- | :--- | :--- |
| **Full Sequence Accuracy** | **84.17%** | **69.39%** | **84.54%** |
| **Mean Digit Accuracy (Pos)**| **91.04%** | **85.99%** | **91.17%** |
| **Succession Rate** | **95.35%** | **96.70%** | **95.31%** |
| **Stage 1 Mean IoU** | **0.8046** | **0.7895** | **0.8050** |
| **Stage 2 Mean IoU** | **0.7355** | **0.7703** | **0.7346** |

---


### How to Run Evaluations
The suite is divided into scripts for isolated performance analysis. You can now specify custom data sources for evaluation and choose between proportional stratified sampling (default) or balanced equal-split sampling:

```bash
# Run ALL evaluations (Stages 1-3 + Full End-to-End Pipeline) with default proportional sampling
python src/evaluation/eval_all.py --max-samples 100

# Run ALL evaluations with a perfectly balanced 50/50 split between SVHN and Handwritten
python src/evaluation/eval_all.py --max-samples 116 --balanced

# Full End-to-End pipeline benchmark with error analysis dashboard
python src/evaluation/eval_pipeline.py --max-samples 500 --save-viz --analyze-errors

# Evaluate on custom datasets (e.g., the Trains OCR dataset)
python src/evaluation/eval_pipeline.py --data-root data/ocr_trains --output-dir outputs/trains_eval
```

## 📊 Dataset Integration
The pipeline now supports "Weakly Labeled" datasets—data that contains global number/plate bounding boxes and sequence labels but lacks fine-grained individual digit annotations.

| Dataset | Type | Samples | Command to Prepare |
| :--- | :--- | :---: | :--- |
| **Trains OCR** | Weakly Labeled | 13 | `python src/data/ocr_trains.py` |
| **Race Numbers** | Fully Labeled | 10,000+ | `python src/prep_data.py --datasets race_numbers` |
| **Handwritten** | Fully Labeled | 10,000+ | `python src/prep_data.py --datasets handwritten` |
| **SVHN / Digits** | Fully Labeled | 200,000+ | `python src/prep_data.py --datasets svhn` |

### Handling Weakly Labeled Data
When a dataset is identified as weakly labeled (`has_digit_boxes=False` in `annotations.json`):
1.  **Stage 1 (Global Detection)**: Evaluated as normal using Mean IoU.
2.  **Stage 2 (Individual Detection)**: Skipped for metric calculation to avoid statistical contamination.
3.  **End-to-End Accuracy**: Calculated by comparing the final OCR output with the ground truth sequence label.

### Pipeline Progression

![Full Pipeline Dashboard](assets/full_pipeline_progression.png)

### Error Analysis
Detailed breakdown of how the model succeeds or fails at each individual step:
![Detailed Error Analysis](assets/detailed_error_analysis.png)



---

## 🎬 Video Asset Generation

Runs the full 4-stage pipeline on **9 representative images** — 3 randomly selected from each data type (SVHN, Race Numbers, Handwritten) — and saves the per-stage visual output for use in a demo video.

```bash
python src/generate_video_assets.py \
    --model-dir outputs/trained_models \
    --data-root data/digits_data \
    --out-dir   video_assets
```

This produces the folder `video_assets/` with one sub-folder per pipeline stage:

| Folder | Contents |
| :--- | :--- |
| `01_samples/` | 9 raw input images (3 per data type) |
| `02_global_bb/` | Each image with the GlobalBB rectangle drawn |
| `03_sharpened/` | Real-ESRGAN enhanced crop |
| `04_individual_bb/` | Sharpened crop with per-digit boxes |
| `05_classification/` | Digit labels overlaid + predicted number banner |

> **Note:** Statistics and accuracy metrics are not computed by this script.
> Use the evaluation suite (`src/evaluation/`) for benchmarking.

---