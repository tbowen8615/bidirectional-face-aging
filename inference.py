# inference.py
import torch
import torchvision.transforms as T
from PIL import Image
import argparse
from models.generator import Generator
import os
from torchvision.utils import save_image

transform = T.Compose([
    T.Resize((128, 128)),  # Match training size
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

@torch.no_grad()
def load_generators_from_checkpoint(ckpt_path, device):
    """Load generators from a Lightning checkpoint"""
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict']
    
    G_y2o = Generator().to(device)
    G_o2y = Generator().to(device)
    
    # Extract generator weights (Lightning prefixes with "G_y2o." and "G_o2y.")
    g_y2o_state = {k.replace('G_y2o.', ''): v for k, v in state_dict.items() if k.startswith('G_y2o.')}
    g_o2y_state = {k.replace('G_o2y.', ''): v for k, v in state_dict.items() if k.startswith('G_o2y.')}
    
    G_y2o.load_state_dict(g_y2o_state)
    G_o2y.load_state_dict(g_o2y_state)
    
    G_y2o.eval()
    G_o2y.eval()
    
    return G_y2o, G_o2y

@torch.no_grad()
def test_aging(input_path, ckpt_path, output_dir, direction="both"):
    """
    Test face aging/de-aging
    
    direction: "age" (young→old), "deage" (old→young), or "both" (full cycle)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load generators
    G_y2o, G_o2y = load_generators_from_checkpoint(ckpt_path, device)
    print(f"Loaded checkpoint: {ckpt_path}")
    
    # Load and transform image
    img = Image.open(input_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_path))[0]
    
    if direction in ["age", "both"]:
        # Young → Old
        fake_old = G_y2o(img_tensor)
        save_image((fake_old + 1) / 2, os.path.join(output_dir, f"{basename}_aged.png"))
        print(f"Saved aged image")
        
    if direction in ["deage", "both"]:
        # Old → Young
        fake_young = G_o2y(img_tensor)
        save_image((fake_young + 1) / 2, os.path.join(output_dir, f"{basename}_deaged.png"))
        print(f"Saved de-aged image")
    
    if direction == "both":
        # Full cycle: Young → Old → Young (reconstruction)
        fake_old = G_y2o(img_tensor)
        reconstructed = G_o2y(fake_old)
        
        # Create comparison grid: Original | Aged | Reconstructed
        grid = torch.cat([img_tensor, fake_old, reconstructed], dim=0)
        grid = (grid + 1) / 2  # Denormalize
        save_image(grid, os.path.join(output_dir, f"{basename}_cycle.png"), nrow=3)
        print(f"Saved cycle comparison")
    
    print(f"\nOutputs saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test face aging/de-aging")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--output_dir", default="results/test")
    parser.add_argument("--direction", choices=["age", "deage", "both"], default="both",
                        help="age: young→old, deage: old→young, both: full cycle")
    args = parser.parse_args()

    test_aging(args.input, args.checkpoint, args.output_dir, args.direction)