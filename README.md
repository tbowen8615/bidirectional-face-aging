# Bidirectional Face Aging GAN

A PyTorch Lightning implementation of bidirectional face aging using CycleGAN with deep cycle consistency. This model can transform young faces to look older and old faces to look younger.

## Architecture

The model uses two generators and two discriminators:

| Component | Purpose |
|-----------|---------|
| G_y2o | Transforms young faces → old faces (aging) |
| G_o2y | Transforms old faces → young faces (de-aging) |
| D_young | Discriminates real vs. fake young faces |
| D_old | Discriminates real vs. fake old faces |

### Key Features

- **Cycle Consistency Loss**: Ensures young → old → young ≈ original
- **Deep Cycle Loss**: Applies the cycle twice for better identity preservation (young → old → young → old → young)
- **Identity Loss**: Uses ArcFace embeddings to preserve facial identity
- **Perceptual Loss**: LPIPS-based loss for realistic textures

## Requirements

- Python 3.11 (recommended)
- NVIDIA GPU with CUDA support
- ~6GB VRAM minimum

## Installation

1. Clone the repository: the repository does not include the training data. 
   ```bash
   git clone https://github.com/tbowen8615/bidirectional-face-aging.git
   cd bidirectional-face-aging
   ```

2. Create a virtual environment with Python 3.11:
   ```bash
   py -3.11 -m venv venv
   
   # Windows
   .\venv\Scripts\Activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Dataset Setup

Organize your dataset in the following structure:

```
data/
├── young/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── old/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

The images do not need to be paired. The model learns from unpaired collections of young and old faces.

## Configuration

Edit `config.yaml` to adjust training parameters:

```yaml
data:
  young_dir: "data/young"
  old_dir: "data/old"
  batch_size: 8
  num_workers: 4
  img_size: 128

training:
  max_epochs: 200
  lr: 0.0002
  betas: [0.5, 0.999]
  lambda_cycle: 10.0
  lambda_deep_cycle: 5.0
  lambda_identity: 2.0
  lambda_lpips: 1.0
```

## Training

```bash
python train.py
```

Training progress:
- Checkpoints are saved to `lightning_logs/version_X/checkpoints/`
- Logs are saved to `wandb/` (offline by default)
- With an RTX 3060, expect ~2 minutes per epoch (~7 hours total for 200 epochs)

## Inference

After training, test the model on new images:

```bash
# Age a young face
python inference.py --input "path/to/young_face.jpg" --checkpoint "lightning_logs/version_0/checkpoints/epoch=199-step=XXXX.ckpt" --direction age

# De-age an old face
python inference.py --input "path/to/old_face.jpg" --checkpoint "lightning_logs/version_0/checkpoints/epoch=199-step=XXXX.ckpt" --direction deage

# Full cycle (age then de-age, creates comparison grid)
python inference.py --input "path/to/face.jpg" --checkpoint "lightning_logs/version_0/checkpoints/epoch=199-step=XXXX.ckpt" --direction both
```

Results are saved to `results/test/` by default.

## Project Structure

```
bidirectional-face-aging/
├── data/
│   ├── young/              # Young face images
│   └── old/                # Old face images
├── models/
│   ├── __init__.py
│   ├── generator.py        # Generator architecture
│   └── discriminator.py    # Discriminator architecture
├── results/
│   └── cycles/             # Training visualizations
├── config.yaml             # Training configuration
├── train.py                # Training script
├── inference.py            # Testing script
├── requirements.txt        # Python dependencies
└── README.md
```

## Expected Results

This implementation produces visible aging/de-aging effects suitable for demonstration purposes. Due to the 128×128 resolution and dataset size, results may include:

- Noticeable aging effects (wrinkles, skin texture)
- Some identity drift on certain faces
- Occasional artifacts in hair or background

For production-quality results, consider using larger datasets, higher resolutions, and more sophisticated architectures.

## Acknowledgments

- CycleGAN: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- LPIPS: [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924)
- ArcFace: [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
