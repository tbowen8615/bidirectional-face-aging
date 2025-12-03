# inference.py
import torch
import torchvision.transforms as T
from PIL import Image
import argparse
from models.generator import Generator
import os
from torchvision.utils import save_image

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

@torch.no_grad()
def generate_cycle(input_path, G_y2o_path, G_o2y_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G_y2o = Generator().to(device)
    G_o2y = Generator().to(device)
    G_y2o.load_state_dict(torch.load(G_y2o_path, map_location=device))
    G_o2y.load_state_dict(torch.load(G_o2y_path, map_location=device))
    G_y2o.eval()
    G_o2y.eval()

    img = transform(Image.open(input_path).convert("RGB")).unsqueeze(0).to(device)

    fake_old = G_y2o(img)
    rec_young = G_o2y(fake_old)
    deep_cycle = G_o2y(G_y2o(rec_young))

    grid = torch.cat([img, fake_old, rec_young, deep_cycle], dim=0)
    grid = (grid + 1) / 2  # denormalize
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, os.path.basename(input_path))
    save_image(grid, save_path, nrow=4)
    print(f"Saved cycle → {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to young image")
    parser.add_argument("--g_y2o", required=True, help="Path to G_y2o checkpoint")
    parser.add_argument("--g_o2y", required=True, help="Path to G_o2y checkpoint")
    parser.add_argument("--output_dir", default="results/cycles")
    args = parser.parse_args()

    generate_cycle(args.input, args.g_y2o, args.g_o2y, args.output_dir)