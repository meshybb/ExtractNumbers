import os
import requests
import zipfile
from tqdm import tqdm

def main():
    target_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'DSText_V2')
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    print("==========================================================")
    print("                  DSText V2 Dataset Download              ")
    print("==========================================================")
    print(f"Target directory prepared: {target_dir}")
    print("\nThis dataset is hosted publicly on Zenodo (Record 10009984).")
    print("It consists of 13 large split files (split_1.zip to split_13.zip).")
    
    zenodo_api_url = "https://zenodo.org/api/records/10009984"
    print(f"\nFetching file list from {zenodo_api_url}...")
    
    try:
        response = requests.get(zenodo_api_url)
        response.raise_for_status()
        data = response.json()
        
        files = data.get('files', [])
        if not files:
            print("No files found in the Zenodo record.")
            return

        print(f"\nFound {len(files)} files to download.")
        
        for f in files:
            file_url = f['links']['self']
            file_name = f['key']
            file_size = f.get('size', 0)
            dest_path = os.path.join(target_dir, file_name)
            
            extracted_marker = dest_path + ".extracted"
            if os.path.exists(extracted_marker):
                print(f"Skipping {file_name}, already downloaded and extracted.")
                continue
                
            print(f"\nDownloading {file_name} ({file_size / (1024*1024):.2f} MB)...")
            
            r = requests.get(file_url, stream=True)
            r.raise_for_status()
            
            with open(dest_path, 'wb') as out_f, tqdm(
                desc=file_name,
                total=file_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        out_f.write(chunk)
                        bar.update(len(chunk))
            
            print(f"Extracting {file_name} to {target_dir}...")
            try:
                with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                print(f"Successfully extracted {file_name}.")
                
                # Create the extracted marker
                with open(extracted_marker, 'w') as marker_f:
                    marker_f.write("extracted")
                
                # Delete zip file to save space
                os.remove(dest_path)
                print(f"Cleaned up zip archive: {file_name}")
            except Exception as extract_err:
                print(f"[!] Error extracting {file_name}: {extract_err}")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
        
    except Exception as e:
        print(f"\n[!] Failed to fetch metadata from Zenodo: {e}")
        print("Please visit manually: https://zenodo.org/records/10009984")
        
    print("==========================================================")

if __name__ == '__main__':
    main()
