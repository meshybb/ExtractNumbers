import os
import shutil
import cv2
import json
import xml.etree.ElementTree as ET
import requests
import zipfile
from tqdm import tqdm

def convert_moving_mnist(src_dir, dest_dir):
    print("Converting Moving MNIST to unified format...")
    if not os.path.exists(src_dir):
        print(f"[!] Source directory {src_dir} does not exist.")
        return
        
    sequences = [d for d in os.listdir(src_dir) if d.startswith("sequence_") and os.path.isdir(os.path.join(src_dir, d))]
    
    for seq_name in tqdm(sequences, desc="Moving MNIST Sequences"):
        seq_src_path = os.path.join(src_dir, seq_name)
        seq_dest_path = os.path.join(dest_dir, seq_name)
        os.makedirs(seq_dest_path, exist_ok=True)
        
        video_dest_path = os.path.join(seq_dest_path, "video.mp4")
        anno_dest_path = os.path.join(seq_dest_path, "annotations.json")
        
        # 1. Compile PNG frames to MP4
        frame_files = sorted([f for f in os.listdir(seq_src_path) if f.startswith("frame_") and f.endswith(".png")],
                             key=lambda x: int(os.path.splitext(x)[0].split("_")[1]))
        
        if not frame_files:
            continue
            
        first_frame = cv2.imread(os.path.join(seq_src_path, frame_files[0]))
        height, width, _ = first_frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_dest_path, fourcc, 10.0, (width, height))
        
        for f_file in frame_files:
            frame_img = cv2.imread(os.path.join(seq_src_path, f_file))
            out.write(frame_img)
        out.release()
        
        # 2. Parse XML GT annotations
        xml_file = f"{seq_name}_GT.xml"
        xml_path = os.path.join(seq_src_path, xml_file)
        
        anno_data = {
            "video_metadata": {
                "sample_id": f"moving_mnist/{seq_name}",
                "width": width,
                "height": height,
                "fps": 10.0
            },
            "frames": {}
        }
        
        if os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                for frame in root.findall('frame'):
                    frame_id = int(frame.get('ID')) - 1  # 0-indexed conversion
                    detected_numbers = []
                    
                    boxes = []
                    digits = []
                    for obj in frame.findall('object'):
                        points = obj.findall('Point')
                        if len(points) >= 4:
                            xs = [int(pt.get('x')) for pt in points]
                            ys = [int(pt.get('y')) for pt in points]
                            x1, y1 = min(xs), min(ys)
                            x2, y2 = max(xs), max(ys)
                            w = x2 - x1
                            h = y2 - y1
                            
                            boxes.append({
                                "x": float(x1),
                                "y": float(y1),
                                "width": float(w),
                                "height": float(h)
                            })
                            digits.append({
                                "label": "digit",
                                "bounding_box": {
                                    "x": float(x1),
                                    "y": float(y1),
                                    "width": float(w),
                                    "height": float(h)
                                }
                            })
                            
                    if boxes:
                        xs = [b["x"] for b in boxes]
                        ys = [b["y"] for b in boxes]
                        x_min = min(xs)
                        y_min = min(ys)
                        x_max = max([b["x"] + b["width"] for b in boxes])
                        y_max = max([b["y"] + b["height"] for b in boxes])
                        
                        detected_numbers.append({
                            "full_value": "digit_seq",
                            "full_bounding_box": {
                                "x": float(x_min),
                                "y": float(y_min),
                                "width": float(x_max - x_min),
                                "height": float(y_max - y_min)
                            },
                            "digits": digits
                        })
                        
                    if detected_numbers:
                        anno_data["frames"][str(frame_id)] = {
                            "detected_numbers": detected_numbers
                        }
            except Exception as e:
                print(f"[!] Error parsing XML for {seq_name}: {e}")
                
        with open(anno_dest_path, 'w') as f:
            json.dump(anno_data, f, indent=4)

def convert_dstext_v2(src_dir, dest_dir):
    print("Converting DSText V2 to unified format...")
    if not os.path.exists(src_dir):
        print(f"[!] Source directory {src_dir} does not exist.")
        return
        
    # Check if there are any XML files in src_dir
    has_xml = False
    for root, dirs, files in os.walk(src_dir):
        if any(f.endswith(".xml") for f in files):
            has_xml = True
            break
            
    if not has_xml:
        print("XML annotations not found under DSText_V2 folder.")
        print("Downloading annotations archive from Zenodo (record 10010840)...")
        ann_url = "https://zenodo.org/records/10010840/files/V2_Ann_Train.zip?download=1"
        ann_zip = os.path.join(src_dir, "V2_Ann_Train.zip")
        try:
            r = requests.get(ann_url, stream=True)
            r.raise_for_status()
            with open(ann_zip, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Extracting annotations...")
            with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
                zip_ref.extractall(src_dir)
            os.remove(ann_zip)
            print("Annotations extracted successfully!")
        except Exception as e:
            print(f"[!] Failed to download or extract annotations: {e}")
        
    found_videos = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".mp4"):
                found_videos.append((root, file))
                
    for root, file in tqdm(found_videos, desc="DSText V2 Videos"):
        video_src_path = os.path.join(root, file)
        video_name = os.path.splitext(file)[0]
        
        # Associated xml file
        xml_name = f"{video_name}_GT.xml"
        xml_path = os.path.join(root, xml_name)
        if not os.path.exists(xml_path):
            # Try finding any XML in the same folder
            xml_files = [f for f in os.listdir(root) if f.endswith(".xml")]
            if xml_files:
                xml_path = os.path.join(root, xml_files[0])
                
        if os.path.exists(xml_path):
            sample_folder = os.path.join(dest_dir, video_name)
            os.makedirs(sample_folder, exist_ok=True)
            
            video_dest_path = os.path.join(sample_folder, "video.mp4")
            anno_dest_path = os.path.join(sample_folder, "annotations.json")
            
            # Copy video file
            shutil.copy(video_src_path, video_dest_path)
            
            # Parse video metadata
            cap = cv2.VideoCapture(video_dest_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            
            anno_data = {
                "video_metadata": {
                    "sample_id": f"dstext_v2/{video_name}",
                    "width": width,
                    "height": height,
                    "fps": fps
                },
                "frames": {}
            }
            
            try:
                tree = ET.parse(xml_path)
                xml_root = tree.getroot()
                
                for frame in xml_root.findall('frame'):
                    frame_id = int(frame.get('ID')) - 1  # 0-indexed
                    detected_numbers = []
                    
                    for obj in frame.findall('object'):
                        transcription = obj.get('Transcription', '')
                        if not transcription or transcription == "###":
                            continue
                            
                        points = obj.findall('Point')
                        if len(points) >= 4:
                            xs = [int(pt.get('x')) for pt in points]
                            ys = [int(pt.get('y')) for pt in points]
                            x1, y1 = min(xs), min(ys)
                            x2, y2 = max(xs), max(ys)
                            w = x2 - x1
                            h = y2 - y1
                            
                            gx = float(x1)
                            gy = float(y1)
                            gw = float(w)
                            gh = float(h)
                            
                            # Clean transcription for digit extraction
                            cleaned_trans = "".join([c for c in transcription if c.isdigit()])
                            digits = []
                            if cleaned_trans:
                                dw = gw / len(cleaned_trans)
                                for idx, char in enumerate(cleaned_trans):
                                    digits.append({
                                        "label": int(char),
                                        "bounding_box": {
                                            "x": float(gx + idx * dw),
                                            "y": float(gy),
                                            "width": float(dw),
                                            "height": float(gh)
                                        }
                                    })
                                    
                            detected_numbers.append({
                                "full_value": cleaned_trans if cleaned_trans else transcription,
                                "full_bounding_box": {
                                    "x": gx,
                                    "y": gy,
                                    "width": gw,
                                    "height": gh
                                },
                                "digits": digits
                            })
                            
                    if detected_numbers:
                        anno_data["frames"][str(frame_id)] = {
                            "detected_numbers": detected_numbers
                        }
            except Exception as e:
                print(f"[!] Error parsing XML for {video_name}: {e}")
                
            with open(anno_dest_path, 'w') as f:
                json.dump(anno_data, f, indent=4)

def convert_roadtext(src_dir, dest_dir):
    print("Converting RoadText to unified format...")
    if not os.path.exists(src_dir):
        print(f"[!] Source directory {src_dir} does not exist.")
        return
        
    annotation_path = os.path.join(src_dir, "roadtext-annotation-fixed.json")
    if not os.path.exists(annotation_path):
        print("[!] Master annotation file not found.")
        return
        
    with open(annotation_path, 'r') as f:
        data = json.load(f)
        
    # Find all mp4 files
    found_videos = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".mp4"):
                found_videos.append((root, file))
                
    for root, file in tqdm(found_videos, desc="RoadText Videos"):
        video_src_path = os.path.join(root, file)
        seq_id = os.path.splitext(file)[0]
        
        # Check if we have annotations for this sequence ID
        if seq_id not in data:
            # Maybe the sequence ID in JSON is an integer or formatted differently
            matching_key = None
            for key in data.keys():
                if str(key) == str(seq_id) or str(key).endswith(str(seq_id)) or str(seq_id).endswith(str(key)):
                    matching_key = key
                    break
            if matching_key:
                seq_data = data[matching_key]
            else:
                continue
        else:
            seq_data = data[seq_id]
            
        sample_folder = os.path.join(dest_dir, f"sequence_{seq_id}")
        os.makedirs(sample_folder, exist_ok=True)
        
        video_dest_path = os.path.join(sample_folder, "video.mp4")
        anno_dest_path = os.path.join(sample_folder, "annotations.json")
        
        shutil.copy(video_src_path, video_dest_path)
        
        cap = cv2.VideoCapture(video_dest_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        anno_out = {
            "video_metadata": {
                "sample_id": f"roadtext/sequence_{seq_id}",
                "width": width,
                "height": height,
                "fps": fps
            },
            "frames": {}
        }
        
        for frame_key, frame_content in seq_data.items():
            frame_id = int(frame_key) - 1 # 0-indexed conversion
            detected_numbers = []
            
            if frame_content and isinstance(frame_content, dict):
                for label in frame_content.get("labels", []) or []:
                    box = label.get("box2d")
                    ocr_text = label.get("ocr") or ""
                    cleaned_ocr = "".join([c for c in ocr_text if c.isdigit()])
                    
                    if not cleaned_ocr:
                        continue
                        
                    x1 = float(box.get("x1", 0))
                    y1 = float(box.get("y1", 0))
                    x2 = float(box.get("x2", 0))
                    y2 = float(box.get("y2", 0))
                    
                    gx = x1
                    gy = y1
                    gw = x2 - x1
                    gh = y2 - y1
                    
                    digits = []
                    dw = gw / len(cleaned_ocr)
                    for idx, char in enumerate(cleaned_ocr):
                        digits.append({
                            "label": int(char),
                            "bounding_box": {
                                "x": float(gx + idx * dw),
                                "y": float(gy),
                                "width": float(dw),
                                "height": float(gh)
                            }
                        })
                        
                    detected_numbers.append({
                        "full_value": cleaned_ocr,
                        "full_bounding_box": {
                            "x": gx,
                            "y": gy,
                            "width": gw,
                            "height": gh
                        },
                        "digits": digits
                    })
                
            if detected_numbers:
                anno_out["frames"][str(frame_id)] = {
                    "detected_numbers": detected_numbers
                }
                
        with open(anno_dest_path, 'w') as f:
            json.dump(anno_out, f, indent=4)

def convert_bovtext_or_icdar(src_dir, dest_dir, dataset_name):
    print(f"Converting {dataset_name} to unified format...")
    if not os.path.exists(src_dir):
        print(f"[!] Source directory {src_dir} does not exist.")
        return
        
    videos_dir = os.path.join(src_dir, "videos")
    annos_dir = os.path.join(src_dir, "annotations")
    
    if not os.path.exists(videos_dir) or not os.path.exists(annos_dir):
        return
        
    anno_files = [f for f in os.listdir(annos_dir) if f.endswith(".json")]
    
    for anno_file in anno_files:
        anno_path = os.path.join(annos_dir, anno_file)
        
        with open(anno_path, 'r') as f:
            data = json.load(f)
            
        video_name = data.get("video_name", "")
        if not video_name:
            video_name = os.path.splitext(anno_file)[0].replace("_gt", "")
            
        video_file = f"{video_name}.mp4"
        video_src_path = os.path.join(videos_dir, video_file)
        
        if not os.path.exists(video_src_path):
            # Try any video ending with .mp4
            mp4_files = [f for f in os.listdir(videos_dir) if f.endswith(".mp4")]
            if mp4_files:
                video_src_path = os.path.join(videos_dir, mp4_files[0])
                
        if os.path.exists(video_src_path):
            sample_folder = os.path.join(dest_dir, video_name)
            os.makedirs(sample_folder, exist_ok=True)
            
            video_dest_path = os.path.join(sample_folder, "video.mp4")
            anno_dest_path = os.path.join(sample_folder, "annotations.json")
            
            shutil.copy(video_src_path, video_dest_path)
            
            cap = cv2.VideoCapture(video_dest_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            
            anno_out = {
                "video_metadata": {
                    "sample_id": f"{dataset_name}/{video_name}",
                    "width": width,
                    "height": height,
                    "fps": fps
                },
                "frames": {}
            }
            
            for frame_data in data.get("frames", []):
                frame_idx = frame_data.get("frame_index")
                detected_numbers = []
                
                for anno in frame_data.get("annotations", []):
                    box = anno.get("box", [0, 0, 0, 0])
                    text = anno.get("text") or ""
                    cleaned_trans = "".join([c for c in text if c.isdigit()])
                    
                    if not cleaned_trans:
                        continue
                        
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    gx, gy = float(x1), float(y1)
                    gw, gh = float(x2 - x1), float(y2 - y1)
                    
                    digits = []
                    dw = gw / len(cleaned_trans)
                    for idx, char in enumerate(cleaned_trans):
                        digits.append({
                            "label": int(char),
                            "bounding_box": {
                                "x": float(gx + idx * dw),
                                "y": float(gy),
                                "width": float(dw),
                                "height": float(gh)
                            }
                        })
                        
                    detected_numbers.append({
                        "full_value": cleaned_trans,
                        "full_bounding_box": {
                            "x": gx,
                            "y": gy,
                            "width": gw,
                            "height": gh
                        },
                        "digits": digits
                    })
                    
                if detected_numbers:
                    anno_out["frames"][str(frame_idx)] = {
                        "detected_numbers": detected_numbers
                    }
                    
            with open(anno_dest_path, 'w') as f:
                json.dump(anno_out, f, indent=4)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_data_root = os.path.join(base_dir, "data")
    dest_data_root = os.path.join(base_dir, "data", "video_data")
    
    os.makedirs(dest_data_root, exist_ok=True)
    
    # Clean previous video_data to ensure fresh conversion
    # Skip mock_video so we don't have to regenerate it
    for d in os.listdir(dest_data_root):
        if d != "mock_video":
            dir_to_clean = os.path.join(dest_data_root, d)
            if os.path.isdir(dir_to_clean):
                shutil.rmtree(dir_to_clean)
                
    # 1. Moving MNIST
    convert_moving_mnist(os.path.join(raw_data_root, "Moving_MNIST"), os.path.join(dest_data_root, "moving_mnist"))
    
    # 2. DSText V2
    convert_dstext_v2(os.path.join(raw_data_root, "DSText_V2"), os.path.join(dest_data_root, "dstext_v2"))
    
    # 3. RoadText
    convert_roadtext(os.path.join(raw_data_root, "RoadText"), os.path.join(dest_data_root, "roadtext"))
    
    # 4. BOVText
    convert_bovtext_or_icdar(os.path.join(raw_data_root, "BOVText"), os.path.join(dest_data_root, "bovtext"), "bovtext")
    
    # 5. ICDAR SVT
    convert_bovtext_or_icdar(os.path.join(raw_data_root, "ICDAR_SVT"), os.path.join(dest_data_root, "icdar_svt"), "icdar_svt")
    
    print("\n=== All video datasets converted and integrated successfully! ===")

if __name__ == "__main__":
    main()
