# classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm.notebook import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score

from config import DEVICE, CLF_EPOCHS, BATCH_SIZE, LR
from dataset_utils import RiverDataset, make_weighted_sampler
from focal_loss import FocalLoss


def build_efficientnet(num_classes=1, pretrained=True):
    
    model = models.efficientnet_b0(pretrained=pretrained)
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_classifier(model, train_loader, val_loader, epochs=CLF_EPOCHS, lr=LR, device=DEVICE):
   
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    best_val_loss = float('inf')
    best_model_state = None
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for imgs, labels, _ in tqdm(train_loader, desc=f'CLF {epoch}/{epochs}', leave=False):
            imgs, labels = imgs.to(device), labels.float().to(device)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.float().to(device)
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if epoch % 5 == 0:
            print(f'  Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}')

    model.load_state_dict(best_model_state)
    print(f'\nClassifier training done | best val loss = {best_val_loss:.4f}')
    return model, train_losses, val_losses


def evaluate_classifier(model, data_loader, device=DEVICE):
    """Evaluate classifier and return predictions, labels, and metrics."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _ in data_loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    pred_binary = (all_probs >= 0.5).astype(int)

    metrics = {
        'auc': roc_auc_score(all_labels, all_probs),
        'avg_precision': average_precision_score(all_labels, all_probs),
        'classification_report': classification_report(all_labels, pred_binary, target_names=['Clean', 'Pollution']),
        'confusion_matrix': confusion_matrix(all_labels, pred_binary),
        'probs': all_probs,
        'labels': all_labels
    }
    return metrics
