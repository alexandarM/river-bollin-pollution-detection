# train_autoencoder.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from config import DEVICE, AE_EPOCHS, BATCH_SIZE, LR
from dataset_utils import SimpleDataset, AE_TF
from autoencoder import ConvAutoencoder


def train_autoencoder(clean_train_paths, clean_val_paths, save_path, output_dir=None):
    #Train the autoencoder on clean images only
    ae_train_loader = DataLoader(
        SimpleDataset(clean_train_paths, AE_TF),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    ae_val_loader = DataLoader(
        SimpleDataset(clean_val_paths, AE_TF),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    model = ConvAutoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=AE_EPOCHS)

    best_val = float('inf')
    train_hist, val_hist = [], []

    for epoch in range(1, AE_EPOCHS + 1):
        # Train
        model.train()
        epoch_loss = 0
        for imgs, _ in tqdm(ae_train_loader, desc=f'AE {epoch}/{AE_EPOCHS}', leave=False):
            imgs = imgs.to(DEVICE)
            recon = model(imgs)
            loss = F.mse_loss(recon, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
        train_hist.append(epoch_loss / len(clean_train_paths))

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, _ in ae_val_loader:
                imgs = imgs.to(DEVICE)
                recon = model(imgs)
                val_loss += F.mse_loss(recon, imgs).item() * imgs.size(0)
        val_hist.append(val_loss / len(clean_val_paths))
        scheduler.step()

        if val_hist[-1] < best_val:
            best_val = val_hist[-1]
            torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0:
            print(f'  Epoch {epoch:3d} | train={train_hist[-1]:.6f} | val={val_hist[-1]:.6f}')

    print(f'\nAE training done | best val loss = {best_val:.6f}')
    return model, train_hist, val_hist
