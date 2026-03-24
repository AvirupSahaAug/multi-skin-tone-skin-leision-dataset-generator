import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix

# Import our setup from existing scripts
from models_unified import SkinToneClassifier
from train_tone_classifier import FitzpatrickDataset

def evaluate_model():
    # 1. Config
    BATCH_SIZE = 32
    IMG_SIZE = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'fitzpatrick17k', 'fitzpatrick17k-main', 'fitzpatrick17k.csv')
    img_dir = os.path.join(base_dir, 'data', 'fitzpatrick17k', 'images')
    weights_path = os.path.join(base_dir, 'checkpoints', 'unified', 'C_pretrained.pth')

    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return

    # 2. Data Loader (Test transforms - no random augmentation)
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    
    # We use the full dataset here, or you could split it in the future
    dataset = FitzpatrickDataset(csv_filepath=csv_path, img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Evaluating on {len(dataset)} images...")

    # 3. Load Model
    model = SkinToneClassifier(num_tones=6, img_size=IMG_SIZE).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 4. Prediction Loop
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Predicting"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 5. Report Results
    target_names = [f"Tone {i+1}" for i in range(6)]
    print("\n--- Model Evaluation (Per Skin Tone) ---")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    print("\n--- Confusion Matrix (Row: Actual, Col: Predicted) ---")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate_model()
