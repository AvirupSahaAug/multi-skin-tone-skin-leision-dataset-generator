import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# Define base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
HAM_DIR = os.path.join(DATA_DIR, 'ham10000')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

def prepare_ham10000():
    print("Processing HAM10000 dataset...")
    
    # Load metadata
    metadata_path = os.path.join(HAM_DIR, 'HAM10000_metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    df = pd.read_csv(metadata_path)
    
    # Check image directories
    part1_dir = os.path.join(HAM_DIR, 'HAM10000_images_part_1')
    part2_dir = os.path.join(HAM_DIR, 'HAM10000_images_part_2')
    
    # Function to find image path
    def get_image_path(image_id):
        fname = f"{image_id}.jpg"
        if os.path.exists(os.path.join(part1_dir, fname)):
            return os.path.join(part1_dir, fname)
        elif os.path.exists(os.path.join(part2_dir, fname)):
            return os.path.join(part2_dir, fname)
        return None

    # Add image paths
    print("Mapping image paths...")
    df['image_path'] = df['image_id'].apply(get_image_path)
    
    # Filter missing images
    missing_mask = df['image_path'].isna()
    if missing_mask.any():
        print(f"Warning: {missing_mask.sum()} images not found. Dropping them.")
        df = df[~missing_mask]

    # Perform stratified split (train/test) based on diagnosis ('dx')
    print("Splitting dataset...")
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['dx'], 
        random_state=42
    )

    # Save to processed directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    train_path = os.path.join(PROCESSED_DIR, 'ham10000_train.csv')
    test_path = os.path.join(PROCESSED_DIR, 'ham10000_test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved split dataset:")
    print(f"  Train: {len(train_df)} samples -> {train_path}")
    print(f"  Test:  {len(test_df)} samples -> {test_path}")

    # Print distribution
    print("\nClass Distribution (Train):")
    print(train_df['dx'].value_counts(normalize=True))

if __name__ == "__main__":
    prepare_ham10000()
