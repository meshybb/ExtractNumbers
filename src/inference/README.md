# Inference & Production

Scripts for running the extraction pipeline on production data.

## Key Files
- `predict_single.py`: Runs the complete 4-stage pipeline on a single image and returns the final predicted number string.
- `visualize_pipeline.py`: Runs the complete pipeline on a single image and generates a visual plot showing each step of the progression (Original -> Global Detection -> Raw Crop -> Enhanced Crop -> Individual Detection).
