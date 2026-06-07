import os
import requests
import tarfile
from tqdm import tqdm

def main():
    target_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'RoadText')
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    print("==========================================================")
    print("                 RoadText Dataset Pipeline                ")
    print("==========================================================")
    print(f"Target directory prepared: {target_dir}")
    
    annotation_url = "https://datasets.cvc.uab.es/roadtext3k/roadtext-annotation-fixed.json"
    annotation_path = os.path.join(target_dir, "roadtext-annotation-fixed.json")
    
    # 1. Download Master JSON Annotation File
    if not os.path.exists(annotation_path):
        print(f"\nDownloading master annotation file from {annotation_url}...")
        try:
            r = requests.get(annotation_url, stream=True, verify=False) # Disable SSL verify if needed due to university certs
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(annotation_path, 'wb') as f, tqdm(
                desc="roadtext-annotation-fixed.json",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            print("Master annotation download complete!")
        except Exception as e:
            print(f"\n[!] Failed to download RoadText annotations: {e}")
    else:
        print("\nRoadText master annotation file already exists on disk. Skipping download.")
        
    # 2. Download sample archive (e.g. 0.tar.gz) for verification/sandbox use
    sample_url = "https://datasets.cvc.uab.es/roadtext3k/I_0.tar.gz"
    sample_archive = os.path.join(target_dir, "I_0.tar.gz")
    extracted_sentinel = os.path.join(target_dir, ".I_0_extracted")
    
    if not os.path.exists(extracted_sentinel):
        print(f"\nDownloading sample video archive from {sample_url}...")
        try:
            r = requests.get(sample_url, stream=True, verify=False)
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(sample_archive, 'wb') as f, tqdm(
                desc="I_0.tar.gz",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
                        
            print("Extracting sample video files...")
            with tarfile.open(sample_archive, "r:gz") as tar:
                tar.extractall(path=target_dir)
            
            # Clean up archive
            os.remove(sample_archive)
            
            # Write extraction marker
            with open(extracted_sentinel, 'w') as sf:
                sf.write("extracted")
                
            print("Sample video files successfully downloaded and extracted!")
            
        except Exception as e:
            print(f"\n[!] Failed to download/extract sample RoadText video clips: {e}")
            print("[i] You can still manually download video archives (.tar.gz) from https://datasets.cvc.uab.es/roadtext3k/")
    else:
        print("\nSample RoadText video clips already downloaded and extracted.")
        
    # 3. Print Instructions for Full Scale Run
    print("\n[i] RoadText Dataset Setup Information:")
    print("  * Master annotations: data/RoadText/roadtext-annotation-fixed.json")
    print("  * Video archives are hosted at: https://datasets.cvc.uab.es/roadtext3k/")
    print("  * To download the complete dataset, retrieve the remaining .tar.gz archives")
    print(f"    and extract their contents into: {target_dir}")
    print("==========================================================")

if __name__ == '__main__':
    main()
