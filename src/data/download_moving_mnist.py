import os
import requests
import numpy as np
import cv2
from tqdm import tqdm

def main(num_sequences=50):
    target_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Moving_MNIST')
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    print("==========================================================")
    print("                 Moving MNIST Dataset Pipeline            ")
    print("==========================================================")
    print(f"Target directory prepared: {target_dir}")
    
    npy_url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
    npy_path = os.path.join(target_dir, "mnist_test_seq.npy")
    
    # 1. Download official Moving MNIST .npy file
    if not os.path.exists(npy_path):
        print(f"\nDownloading Moving MNIST from {npy_url}...")
        try:
            r = requests.get(npy_url, stream=True)
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(npy_path, 'wb') as f, tqdm(
                desc="mnist_test_seq.npy",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            print("Download completed successfully!")
        except Exception as e:
            print(f"\n[!] Failed to download Moving MNIST: {e}")
            return
    else:
        print("\nMoving MNIST dataset array already exists on disk. Skipping download.")

    # 2. Process sequences and generate images + XML annotations
    print(f"\nLoading dataset array and extracting first {num_sequences} sequences...")
    try:
        data = np.load(npy_path) # Shape: (20, 10000, 64, 64)
        num_frames, total_sequences, height, width = data.shape
        print(f"Loaded dataset with {total_sequences} sequences of {num_frames} frames ({height}x{width}).")
        
        # We will process the first `num_sequences`
        for seq_idx in range(min(num_sequences, total_sequences)):
            seq_dir = os.path.join(target_dir, f"sequence_{seq_idx}")
            os.makedirs(seq_dir, exist_ok=True)
            
            xml_path = os.path.join(seq_dir, f"sequence_{seq_idx}_GT.xml")
            
            # Start XML string
            xml_lines = [
                '<?xml version="1.0" ?>',
                '<Frames>'
            ]
            
            for frame_idx in range(num_frames):
                frame_data = data[frame_idx, seq_idx] # 64x64 array
                
                # Save frame as PNG image
                img_path = os.path.join(seq_dir, f"frame_{frame_idx + 1}.png")
                cv2.imwrite(img_path, frame_data)
                
                # XML frame block
                xml_lines.append(f'  <frame ID="{frame_idx + 1}">')
                
                # Bounding box detection via thresholding & contours
                # Since background is black (0) and digits are bright (up to 255)
                _, thresh = cv2.threshold(frame_data, 20, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                obj_id = 1
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Filter out tiny noise contours
                    if w > 3 and h > 3:
                        xml_lines.append(f'    <object ID="{obj_id}" Transcription="digit" category="moving" language="alphanumeric">')
                        xml_lines.append(f'      <Point x="{x}" y="{y}"/>')
                        xml_lines.append(f'      <Point x="{x + w}" y="{y}"/>')
                        xml_lines.append(f'      <Point x="{x + w}" y="{y + h}"/>')
                        xml_lines.append(f'      <Point x="{x}" y="{y + h}"/>')
                        xml_lines.append('    </object>')
                        obj_id += 1
                
                xml_lines.append('  </frame>')
                
            xml_lines.append('</Frames>')
            
            # Save XML file
            with open(xml_path, 'w') as xml_f:
                xml_f.write('\n'.join(xml_lines))
                
        print(f"Successfully processed {num_sequences} sequences with XML ground-truth annotations!")
        
    except Exception as e:
        print(f"[!] Error processing sequences: {e}")
        
    print("==========================================================")

if __name__ == '__main__':
    main()
