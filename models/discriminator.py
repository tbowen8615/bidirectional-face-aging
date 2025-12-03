# models/discriminator.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator (70x70 receptive field)
    Same architecture used in CycleGAN, outputs a feature map instead of single scalar
    """
    def __init__(self, in_channels=3, nf=64):
        super().__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, nf, stride=2, normalize=False),        # (nf) x 128 x 128
            *discriminator_block(nf, nf*2, stride=2),                                 # (nf*2) x 64 x 64
            *discriminator_block(nf*2, nf*4, stride=2),                               # (nf*4) x 32 x 32
            *discriminator_block(nf*4, nf*8, stride=1),                               # (nf*8) x 31 x 31
            nn.Conv2d(nf*8, 1, kernel_size=4, stride=1, padding=1),                  # 1 x 30 x 30
        )

    def forward(self, x):
        return self.model(x)