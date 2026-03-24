import torch
from torchvision.utils import save_image
import argparse
import os
import sys

# Since we are in src/, we can import directly
from models import MultiHeadGenerator
from diversity import ResNetEncoder

def check_diversity():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checking checkpoint: {args.checkpoint}")
    
    # Load Model
    # Note: Generator must match training config
    G = MultiHeadGenerator(z_dim=100, num_classes=7, img_size=128).to(device)
    
    # Handle loading path
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.eval()

    # Feature Extractor
    Enc = ResNetEncoder().to(device)
    Enc.eval()
    
    # Generate one batch
    torch.manual_seed(42)
    # 32 Samples
    z = torch.randn(32, 100).to(device) 
    # Class 0 (NV) for consistency
    labels = torch.zeros(32).long().to(device) 
    
    with torch.no_grad():
        x1, x2, x3 = G(z, labels)
        
        # 1. Diversity Check (Head 1 vs 2)
        pix_diff = (x1 - x2).abs().mean()
        
        f1 = Enc(x1)
        f2 = Enc(x2)
        feat_diff = (f1 - f2).abs().mean()
        
        print(f"\n--- Diversity Check (Head 1 vs Head 2) ---")
        print(f"Pixel Mean Difference:   {pix_diff.item():.6f} (Range 0-2)")
        print(f"Feature Mean Difference: {feat_diff.item():.6f}")
        
        if pix_diff < 0.01:
            print(">> CONCLUSION: Heads are PIXEL-IDENTICAL. Diversity Failed.")
        elif feat_diff > 1.0 and pix_diff < 0.05:
            print(">> CONCLUSION: High Feature Diff but Low Pixel Diff -> ADVERSARIAL ATTACK on Loss!")
        else:
            print(">> CONCLUSION: Heads are distinct.")

        # 2. Mode Collapse Check (Sample 0 vs Sample 1)
        print(f"\n--- Mode Collapse Check (Sample 0 vs Sample 1) ---")
        batch_pix_diff = (x1[0] - x1[1]).abs().mean()
        print(f"Batch Pixel Difference:  {batch_pix_diff.item():.6f}")
        
        if batch_pix_diff < 0.02:
            print(">> CONCLUSION: Sample 0 and 1 are identical -> MODE COLLAPSE.")
        else:
            print(">> CONCLUSION: Batch samples are distinct.")
            
    # Save Grid for User
    # Save first 8 rows
    vis_list = []
    for i in range(8):
        vis_list.append(x1[i])
        vis_list.append(x2[i])
        vis_list.append(x3[i])
    
    vis_tensor = torch.stack(vis_list) # (24, 3, 128, 128)
    # Save to parent dir (dl skin thing)
    out_path = "../debug_diversity_grid.png"
    save_image(vis_tensor, out_path, nrow=3, normalize=True)
    print(f"\nSaved debug grid to {out_path}")
    print("Grid Layout: 8 Rows. Each Row = [Head 1, Head 2, Head 3] for SAME input.")

if __name__ == "__main__":
    check_diversity()
