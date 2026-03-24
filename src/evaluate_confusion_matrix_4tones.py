import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models_unified import StrongSkinToneClassifier
from train_tone_classifier_strong_4tones import FitzpatrickDataset4Tones

def evaluate_4tones_confusion_matrix():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'fitzpatrick17k', 'fitzpatrick17k-main', 'fitzpatrick17k.csv')
    img_dir = os.path.join(base_dir, 'data', 'fitzpatrick17k', 'images')
    model_path = os.path.join(base_dir, 'checkpoints', 'unified', 'C_strong_4tones_best.pth')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train it first.")
        return

    # Basic transforms without heavy augmentation for evaluation
    transform = A.Compose([
        A.Resize(128, 128),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    print("Loading Dataset...")
    dataset = FitzpatrickDataset4Tones(csv_filepath=csv_path, img_dir=img_dir, transform=transform)
    # Using larger batch size and no drop_last/shuffling for clean evaluation
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # Initialize Model and load weights
    model = StrongSkinToneClassifier(num_tones=4, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating Model...")
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Calculating Confusion Matrix"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plotting using Seaborn
    plt.figure(figsize=(8, 6))
    
    # Class labels are Tone 1 to Tone 4
    class_names = ['Tone 1', 'Tone 2', 'Tone 3', 'Tone 4']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix: Strong Skin Tone Classifier (4 Tones)')
    plt.ylabel('True Fitzpatrick Tone')
    plt.xlabel('Predicted Fitzpatrick Tone')
    
    # Save the plot
    save_path = os.path.join(base_dir, "samples", "strong_unified_4tones", "confusion_matrix_4tones.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\nConfusion matrix plot saved to: {save_path}")
    
    # Print numerical matrix to console
    print("\nNumerical Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    evaluate_4tones_confusion_matrix()
