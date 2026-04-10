# autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(True),   # 112×112
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True),  # 56×56
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(True), # 28×28
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(True), # 14×14
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruction_error(self, x):
        recon = self(x)
        return F.mse_loss(recon, x, reduction='none').mean(dim=[1, 2, 3])
