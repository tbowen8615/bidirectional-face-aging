# models/generator.py
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )
    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    U-Net style generator with 9 ResBlocks (same as CycleGAN / Contrastive Unpaired Translation)
    Input: 3x256x256 → Output: 3x256x256
    """
    def __init__(self, in_channels=3, out_channels=3, nf=64):
        super().__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, nf, kernel_size=7),
            nn.InstanceNorm2d(nf),
            nn.ReLU(inplace=True),
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(nf*2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(nf*4),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.resblocks = nn.Sequential(*[ResBlock(nf*4) for _ in range(9)])

        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(nf*2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(nf),
            nn.ReLU(inplace=True),
        )

        # Output layer
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf, out_channels, kernel_size=7),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.resblocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        return x