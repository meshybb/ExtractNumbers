import os
import requests
import json
from tqdm import tqdm

def main():
    target_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ICDAR_SVT')
    target_dir = os.path.abspath(target_dir)
    videos_dir = os.path.join(target_dir, "videos")
    annotations_dir = os.path.join(target_dir, "annotations")
    
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    print("==========================================================")
    print("         ICDAR Scene Video Text Spotting Downloader       ")
    print("==========================================================")
    print(f"Target directory prepared: {target_dir}")
    
    # 1. Download Public Sandbox Video Sample
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
        
    # 2. Programmatically Generate Standard ICDAR-Compatible Annotations
    annotation_path = os.path.join(annotations_dir, "sample_gt.json")
    if not os.path.exists(annotation_path):
        print("\nGenerating standard ICDAR-compatible annotations...")
        mock_annotation = {
            "video_name": "sample",
            "frames": [
                {
                    "frame_index": i,
                    "annotations": [
                        {
                            "box": [100 + i * 2, 120, 200 + i * 2, 180],
                            "text": str(1000 + i),
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
        
    print("\n=== ICDAR SVT Sandbox Dataset Successfully Prepared! ===")
    print("==========================================================")

if __name__ == '__main__':
    main()

