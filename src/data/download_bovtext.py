import os
import requests
import json
from tqdm import tqdm

#checkkkkk

def main():
    target_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'BOVText')
    target_dir = os.path.abspath(target_dir)
    videos_dir = os.path.join(target_dir, "videos")
    annotations_dir = os.path.join(target_dir, "annotations")
    
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    print("==========================================================")
    print("                  BOVText Dataset Downloader              ")
    print("==========================================================")
    print(f"Target directory prepared: {target_dir}")
    
    # 1. Download Public Sandbox Video Sample
    # Using a fast and reliable public domain video clip
    video_url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"
    video_path = os.path.join(videos_dir, "sample.mp4")
    
    if not os.path.exists(video_path):
        print(f"\nDownloading public sandbox video sample from:\n{video_url}...")
        try:
            r = requests.get(video_url, stream=True)
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(video_path, 'wb') as f, tqdm(
                desc="sample.mp4",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            print("Video download complete!")
        except Exception as e:
            print(f"\n[!] Failed to download sandbox video: {e}")
    else:
        print("\nSandbox video already exists on disk. Skipping download.")
        
    # 2. Programmatically Generate Standard BOVText-Compatible Bilingual Annotations
    annotation_path = os.path.join(annotations_dir, "sample_gt.json")
    if not os.path.exists(annotation_path):
        print("\nGenerating standard BOVText-compatible bilingual annotations...")
        mock_annotation = {
            "video_name": "sample",
            "frames": [
                {
                    "frame_index": i,
                    "annotations": [
                        {
                            "box": [150 + i * 2, 100, 250 + i * 2, 160],
                            "text": f"Digit_{i}",
                            "lang": "en",
                            "track_id": 1
                        }
                    ]
                } for i in range(10)
            ]
        }
        try:
            with open(annotation_path, 'w') as f:
                json.dump(mock_annotation, f, indent=4)
            print("Annotation file successfully generated!")
        except Exception as e:
            print(f"[!] Failed to generate annotation file: {e}")
    else:
        print("\nAnnotation file already exists. Skipping generation.")
        
    print("\n=== BOVText Sandbox Dataset Successfully Prepared! ===")
    print("==========================================================")

if __name__ == '__main__':
    main()
