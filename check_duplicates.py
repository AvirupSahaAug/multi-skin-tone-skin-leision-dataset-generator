import torch
import torch.nn as nn
from torchvision.utils import save_image
from models import MultiHeadGenerator
import argparse
import os

def check_diversity():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    G = MultiHeadGenerator(z_dim=100, num_classes=7, img_size=128).to(device)
    G.load_state_dict(torch.load(args.checkpoint, map_location=device))
    G.eval()
    
    # Generate one batch
    z = torch.randn(8, 100).to(device) # Just 8 samples
    labels = torch.zeros(8).long().to(device) # All Class 0 (NV)
    
    with torch.no_grad():
        x1, x2, x3 = G(z, labels)
        
    # Check pixel differences
    diff12 = (x1 - x2).abs()
    diff23 = (x2 - x3).abs()
    diff13 = (x1 - x3).abs()
    
    print(f"Mean Diff (1 vs 2): {diff12.mean().item():.4f}")
    print(f"Mean Diff (2 vs 3): {diff23.mean().item():.4f}")
    print(f"Mean Diff (1 vs 3): {diff13.mean().item():.4f}")
    
    if diff12.mean() < 0.01:
        print("WARNING: Heads 1 and 2 are almost identical!")
    else:
        print("Heads 1 and 2 are distinct.")
        
    # Save a visualization of the differences
    # We will normalize the difference maps to be visible (0 to 1)
    # The diffs might be small, so multiply by 5 to enhance visibility
    vis = torch.cat([
        x1, x2, x3,
        diff12 * 5, diff23 * 5, diff13 * 5
    ], dim=0) 
    # Current shape: (8*6, 3, 128, 128)
    
    os.makedirs("debug_output", exist_ok=True)
    save_image(vis, "debug_output/diversity_debug.png", nrow=8, normalize=True)
    print("Saved debug visualization to debug_output/diversity_debug.png")
    
    # Check duplicates across batch (Mode Collapse)
    # Compare Sample 0 vs Sample 1
    batch_diff = (x1[0] - x1[1]).abs().mean()
    print(f"Batch Diff (Sample 0 vs 1): {batch_diff.item():.4f}")
    if batch_diff < 0.05:
         print("WARNING: Sample 0 and 1 look very similar (Potential Mode Collapse)")

if __name__ == "__main__":
    check_diversity()
