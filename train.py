import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torchvision.transforms as T
from PIL import Image
import yaml
import wandb
import lpips
from facenet_pytorch import InceptionResnetV1
from models.generator import Generator
from models.discriminator import Discriminator
import os

# --------------------- Config Helper ---------------------
class DictNamespace:
    """Convert nested dict to object with attribute access"""
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, DictNamespace(value))
            else:
                setattr(self, key, value)

# --------------------- Dataset ---------------------
class YoungOldDataset(Dataset):
    def __init__(self, young_dir, old_dir, transform=None):
        self.young_paths = list(Path(young_dir).glob("*.png")) + list(Path(young_dir).glob("*.jpg"))
        self.old_paths = list(Path(old_dir).glob("*.png")) + list(Path(old_dir).glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return max(len(self.young_paths), len(self.old_paths))

    def __getitem__(self, idx):
        young = Image.open(self.young_paths[idx % len(self.young_paths)]).convert("RGB")
        old = Image.open(self.old_paths[idx % len(self.old_paths)]).convert("RGB")
        if self.transform:
            young = self.transform(young)
            old = self.transform(old)
        return {"young": young, "old": old}

# --------------------- Lightning Module ---------------------
class BidirectionalAgingGAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = DictNamespace(cfg) if isinstance(cfg, dict) else cfg
        self.save_hyperparameters(cfg)
        
        # Enable manual optimization for multiple optimizers
        self.automatic_optimization = False

        self.G_y2o = Generator()   # young → old
        self.G_o2y = Generator()   # old → young
        self.D_young = Discriminator()
        self.D_old = Discriminator()

        self.lpips_loss = lpips.LPIPS(net='vgg').eval()
        self.arcface = InceptionResnetV1(pretrained='vggface2').eval()
        for p in self.arcface.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.G_y2o(x)

    def adversarial_loss(self, pred, is_real):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return F.mse_loss(pred, target)

    def identity_loss(self, real, cycled):
        with torch.no_grad():
            real_feat = self.arcface((real + 1) / 2)
        fake_feat = self.arcface((cycled + 1) / 2)
        return 1 - F.cosine_similarity(real_feat, fake_feat).mean()

    def training_step(self, batch, batch_idx):
        young = batch["young"]
        old = batch["old"]
        
        # Get optimizers
        opt_g, opt_dy, opt_do = self.optimizers()

        # ========== Train Generators ==========
        opt_g.zero_grad()
        
        # Forward translations
        fake_old = self.G_y2o(young)
        fake_young = self.G_o2y(old)

        # Cycle reconstructions
        rec_young = self.G_o2y(fake_old)
        rec_old = self.G_y2o(fake_young)

        # Deep cycles (your core contribution)
        deep_young = self.G_o2y(self.G_y2o(rec_young))
        deep_old = self.G_y2o(self.G_o2y(rec_old))

        # Adversarial loss
        g_loss_adv = self.adversarial_loss(self.D_old(fake_old), True) + \
                     self.adversarial_loss(self.D_young(fake_young), True)

        # Standard cycle
        cycle_loss = F.l1_loss(rec_young, young) + F.l1_loss(rec_old, old)

        # Deep cycle (the key idea from the slides)
        deep_cycle_loss = F.l1_loss(deep_young, young) + F.l1_loss(deep_old, old)

        # Identity preservation
        id_loss = self.identity_loss(young, rec_young) + self.identity_loss(old, rec_old)

        # Perceptual
        perceptual = self.lpips_loss(rec_young, young).mean() + \
                     self.lpips_loss(rec_old, old).mean()

        g_loss = g_loss_adv + \
                 self.cfg.training.lambda_cycle * cycle_loss + \
                 self.cfg.training.lambda_deep_cycle * deep_cycle_loss + \
                 self.cfg.training.lambda_identity * id_loss + \
                 self.cfg.training.lambda_lpips * perceptual

        self.manual_backward(g_loss)
        opt_g.step()

        # ========== Train Discriminator Young ==========
        opt_dy.zero_grad()
        
        real_loss = self.adversarial_loss(self.D_young(young), True)
        fake_loss = self.adversarial_loss(self.D_young(self.G_o2y(old).detach()), False)
        d_young_loss = (real_loss + fake_loss) / 2
        
        self.manual_backward(d_young_loss)
        opt_dy.step()

        # ========== Train Discriminator Old ==========
        opt_do.zero_grad()
        
        real_loss = self.adversarial_loss(self.D_old(old), True)
        fake_loss = self.adversarial_loss(self.D_old(self.G_y2o(young).detach()), False)
        d_old_loss = (real_loss + fake_loss) / 2
        
        self.manual_backward(d_old_loss)
        opt_do.step()

        # Logging
        self.log("g_loss", g_loss, prog_bar=True)
        self.log("cycle_loss", cycle_loss)
        self.log("deep_cycle_loss", deep_cycle_loss)
        self.log("id_loss", id_loss)
        self.log("d_young_loss", d_young_loss)
        self.log("d_old_loss", d_old_loss)

    def configure_optimizers(self):
        lr = self.cfg.training.lr
        b1, b2 = self.cfg.training.betas
        opt_g = torch.optim.Adam(list(self.G_y2o.parameters()) + list(self.G_o2y.parameters()), lr=lr, betas=(b1, b2))
        opt_dy = torch.optim.Adam(self.D_young.parameters(), lr=lr, betas=(b1, b2))
        opt_do = torch.optim.Adam(self.D_old.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_dy, opt_do]

    def train_dataloader(self):
        transform = T.Compose([
            T.Resize((128, 128)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        dataset = YoungOldDataset(self.cfg.data.young_dir, self.cfg.data.old_dir, transform)
        return DataLoader(dataset, batch_size=self.cfg.data.batch_size,
                          num_workers=self.cfg.data.num_workers, shuffle=True)

# --------------------- Main ---------------------
if __name__ == "__main__":
    # Optimize for Tensor Cores
    torch.set_float32_matmul_precision('medium')
    
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], config=cfg)

    model = BidirectionalAgingGAN(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        accelerator="gpu",
        devices=1,
        log_every_n_steps=cfg["training"]["log_every_n_steps"],
        precision="16-mixed",
        callbacks=[
            pl.callbacks.ModelCheckpoint(every_n_epochs=cfg["training"]["checkpoint_every_n_epochs"],
                                         save_top_k=-1),
            pl.callbacks.LearningRateMonitor(logging_interval='step')
        ]
    )

    trainer.fit(model)