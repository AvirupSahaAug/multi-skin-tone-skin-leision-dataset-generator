import torch
import os
from torchvision.utils import save_image
import argparse

# Import our models
from models_unified import DynamicMultiHeadGenerator, ConditionalDiscriminator

def verify_consistency(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Generator and Discriminator
    # We load them to check how they react to different inputs
    G = DynamicMultiHeadGenerator(z_dim=args.z_dim, num_classes=args.num_classes, 
                                  num_heads=args.num_tones, img_size=args.img_size).to(device)
    
    D = ConditionalDiscriminator(num_classes=args.num_classes, img_size=args.img_size).to(device)

    # Load weights if path provided
    if args.g_path and os.path.exists(args.g_path):
        G.load_state_dict(torch.load(args.g_path, map_location=device))
        print(f"Loaded Generator from {args.g_path}")
    
    if args.d_path and os.path.exists(args.d_path):
        D.load_state_dict(torch.load(args.d_path, map_location=device))
        print(f"Loaded Discriminator from {args.d_path}")

    G.eval()
    D.eval()

    # 2. Setup Test Scenarios
    # Scenario A: Multiple noises for the SAME class
    # Scenario B: Multiple classes for the SAME noise
    
    test_noise_count = 5
    test_z = torch.randn(test_noise_count, args.z_dim).to(device)
    
    os.makedirs(args.output_dir, exist_ok=True)

    print("Running consistency checks...")

    with torch.no_grad():
        for cls in range(args.num_classes):
            # For each class, generate images using different noises
            labels = torch.full((test_noise_count,), cls, dtype=torch.long).to(device)
            
            # G returns list of length num_tones, each item (test_noise_count, 3, H, W)
            fake_imgs_list = G(test_z, labels)
            
            # Check Discriminator scores for each head
            print(f"\nChecking Class {cls}:")
            for t, head_imgs in enumerate(fake_imgs_list):
                scores = D(head_imgs, labels)
                mean_score = torch.sigmoid(scores).mean().item()
                print(f"  Head {t} (Tone {t+1}) Avg D-Score: {mean_score:.4f}")

            # Save a grid: Rows = Noises, Cols = Skin Tones
            # Stack heads horizontally
            combined = torch.cat(fake_imgs_list, dim=3) # Dim 3 is width
            save_path = os.path.join(args.output_dir, f"consistency_class_{cls}.png")
            save_image(combined, save_path, normalize=True, nrow=1)
            print(f"  Saved consistency grid to {save_path}")

    print("\nVerification complete. Check the output directory for visual grids.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--g_path", type=str, default="checkpoints/unified/G_unified_latest.pth")
    parser.add_argument("--d_path", type=str, default="checkpoints/unified/D_unified_latest.pth")
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--num_tones", type=int, default=6)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="samples/verification")
    
    args = parser.parse_args()
    
    # Try to find the latest checkpoint if default doesn't exist
    if not os.path.exists(args.g_path):
        checkpoint_dir = "checkpoints/unified"
        if os.path.exists(checkpoint_dir):
            files = [f for f in os.listdir(checkpoint_dir) if f.startswith("G_unified_") and f.endswith(".pth")]
            if files:
                # Get the one with the highest epoch number
                latest = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
                args.g_path = os.path.join(checkpoint_dir, latest)
                # Do same for D
                args.d_path = args.g_path.replace("G_unified_", "D_unified_")

    verify_consistency(args)
