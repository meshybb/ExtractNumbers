"""
Staged Video Processing Pipeline.

Implements the 4-stage pipeline for selected video frames:
1. Global Detection
2. Enhancement (Pass-through)
3. Individual Detection
4. Classification
"""
import os
import cv2
import json
import csv
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Sequence, Tuple

from .frame_selector import FrameSelector
from ..utils.bbox_utils import merge_global_boxes, nms_individual_boxes
from ..digit_recognizer.digit_recognizer import preprocess_crop

def infer_batch_yolo(model, images: List[np.ndarray]) -> List[Any]:
    """Adaptor for batched inference with YOLO models."""
    if not images:
        return []
    # YOLO natively supports list of numpy arrays for batching
    try:
        results = model.predict(source=images, imgsz=256, verbose=False)
        return results
    except Exception as e:
        # Fallback to per-image if batching fails
        print(f"Batched YOLO inference failed ({e}), falling back to sequential...")
        return [model.predict(source=img, imgsz=256, verbose=False)[0] for img in images]

def infer_batch_classifier(model, tensors: List[torch.Tensor], device: torch.device) -> List[int]:
    """Adaptor for batched inference with the PyTorch classifier."""
    if not tensors:
        return []
    try:
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            outputs = model(batch)
            preds = outputs.argmax(dim=1).cpu().tolist()
        return preds
    except Exception as e:
        print(f"Batched classifier inference failed ({e}), falling back to sequential...")
        preds = []
        for t in tensors:
            with torch.no_grad():
                out = model(t.unsqueeze(0).to(device))
                preds.append(out.argmax(dim=1).item())
        return preds


class StagedPipeline:
    def __init__(self, selector: FrameSelector, out_dir: Path, batch_size: int = 8, save_stages: bool = True):
        self.selector = selector
        self.out_dir = Path(out_dir)
        self.batch_size = batch_size
        self.save_stages = save_stages
        
    def _save_stage_image(self, frame_idx: int, stage_name: str, img: np.ndarray, meta: Dict[str, Any]) -> str:
        if not self.save_stages:
            return ""
        frame_dir = self.out_dir / f"frame_{frame_idx:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        out_path = frame_dir / f"{stage_name}.png"
        cv2.imwrite(str(out_path), img)
        if "paths" not in meta:
            meta["paths"] = {}
        meta["paths"][stage_name] = str(out_path)
        return str(out_path)

    def run(self, frames: Sequence[np.ndarray], global_model, indiv_model, classifier, device) -> List[Dict[str, Any]]:
        selected_indices = self.selector.select_indices(frames)
        
        # Initialize metadata records
        records: Dict[int, Dict[str, Any]] = {
            i: {
                "frame": i,
                "status": "ok",
                "prediction": "",
                "global_conf": None,
                "n_individuals": 0,
                "paths": {},
                # Temporary internal state
                "_raw": frames[i],
                "_crop": None,
                "_iboxes": [],
                "_digits": []
            }
            for i in selected_indices
        }
        
        active_indices = list(records.keys())
        
        # --- Stage 1: Global Detection ---
        for i in range(0, len(active_indices), self.batch_size):
            batch_idx = active_indices[i:i+self.batch_size]
            batch_imgs = [records[idx]["_raw"] for idx in batch_idx]
            
            # Save raw stage
            for idx, img in zip(batch_idx, batch_imgs):
                self._save_stage_image(idx, "stage_01_raw", img, records[idx])
                
            results = infer_batch_yolo(global_model, batch_imgs)
            
            for idx, res, img in zip(batch_idx, results, batch_imgs):
                if not res or len(res.boxes) == 0:
                    records[idx]["status"] = "no-global"
                    continue
                    
                all_gboxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                gbox = merge_global_boxes(all_gboxes)
                
                records[idx]["global_conf"] = float(np.mean(confs))
                
                # --- Stage 2: Enhancement (Pass-through) ---
                gx1, gy1, gx2, gy2 = map(int, gbox)
                h, w = img.shape[:2]
                gx1, gy1, gx2, gy2 = max(0, gx1), max(0, gy1), min(w, gx2), min(h, gy2)
                crop = img[gy1:gy2, gx1:gx2]
                
                records[idx]["_crop"] = crop
                
                # Save stage 2 & 3 images
                self._save_stage_image(idx, "stage_02_globalbb", img[gy1:gy2, gx1:gx2], records[idx]) # Optional full with box drawn, but standard is just saving
                self._save_stage_image(idx, "stage_03_sharpened", crop, records[idx])
                
        # Filter active
        active_indices = [idx for idx in active_indices if records[idx]["status"] == "ok"]
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # --- Stage 3: Individual Detection ---
        for i in range(0, len(active_indices), self.batch_size):
            batch_idx = active_indices[i:i+self.batch_size]
            batch_crops = [records[idx]["_crop"] for idx in batch_idx]
            
            results = infer_batch_yolo(indiv_model, batch_crops)
            
            for idx, res in zip(batch_idx, results):
                if not res or len(res.boxes) == 0:
                    records[idx]["status"] = "no-individuals"
                    continue
                    
                iboxes = res.boxes.xyxy.cpu().numpy()
                iconfs = res.boxes.conf.cpu().numpy()
                iboxes, iconfs = nms_individual_boxes(iboxes, iconfs, iou_thresh=0.45)
                iboxes = sorted(iboxes, key=lambda b: b[0])
                
                records[idx]["_iboxes"] = iboxes
                records[idx]["n_individuals"] = len(iboxes)
                self._save_stage_image(idx, "stage_04_individualbb", records[idx]["_crop"], records[idx])
                
        # Filter active
        active_indices = [idx for idx in active_indices if records[idx]["status"] == "ok"]
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        # --- Stage 4: Classification ---
        # Flatten all individual crops into a single batching flow
        flat_crops = []
        flat_meta = [] # (idx, pos)
        
        for idx in active_indices:
            crop_img = records[idx]["_crop"]
            for pos, ibox in enumerate(records[idx]["_iboxes"]):
                try:
                    tensor = preprocess_crop(crop_img, (ibox[0], ibox[1], ibox[2], ibox[3]))
                    flat_crops.append(tensor)
                    flat_meta.append((idx, pos))
                except Exception as e:
                    records[idx]["status"] = "error"
                    records[idx]["prediction"] = f"Crop error: {e}"
        
        # Filter again just in case
        active_indices = [idx for idx in active_indices if records[idx]["status"] == "ok"]
        
        # Run classification in batches
        all_preds = []
        for i in range(0, len(flat_crops), self.batch_size):
            batch_tensors = flat_crops[i:i+self.batch_size]
            preds = infer_batch_classifier(classifier, batch_tensors, device)
            all_preds.extend(preds)
            
        # Reconstruct predictions
        for (idx, pos), pred in zip(flat_meta, all_preds):
            records[idx]["_digits"].append(str(pred))
            
        for idx in active_indices:
            records[idx]["prediction"] = "".join(records[idx]["_digits"])
            self._save_stage_image(idx, "stage_05_classification", records[idx]["_crop"], records[idx])
            
        if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        # Clean up internal state before returning
        out_records = []
        for idx in selected_indices:
            rec = records[idx]
            # remove private keys
            keys_to_remove = [k for k in rec.keys() if k.startswith("_")]
            for k in keys_to_remove:
                del rec[k]
            out_records.append(rec)
            
        return out_records

