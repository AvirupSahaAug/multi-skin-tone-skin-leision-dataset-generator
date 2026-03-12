import os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import hashlib

# Configuration
CSV_PATH = 'data/fitzpatrick17k/fitzpatrick17k-main/fitzpatrick17k.csv'
OUTPUT_DIR = 'data/fitzpatrick17k/images'
NUM_WORKERS = 16

def download_image(row):
    url = row['url']
    md5_hash = row['md5hash']
    filename = f"{md5_hash}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(filepath):
        return "exists"

    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Verify MD5 (Optional but good practice)
            # file_hash = hashlib.md5(response.content).hexdigest()
            # if file_hash != md5_hash:
            #     return "hash_mismatch"
            
            return "success"
        else:
            return f"error_{response.status_code}"
    except Exception as e:
        return f"exception_{str(e)}"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading CSV from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"Found {len(df)} images to download.")

    # Convert dataframe to list of dicts for faster iteration
    rows = df.to_dict('records')

    success_count = 0
    error_count = 0
    exists_count = 0

    print(f"Starting download with {NUM_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(download_image, rows), total=len(rows)))

    for res in results:
        if res == "success":
            success_count += 1
        elif res == "exists":
            exists_count += 1
        else:
            error_count += 1

    print("\nDownload Complete.")
    print(f"Success: {success_count}")
    print(f"Existing: {exists_count}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    main()
