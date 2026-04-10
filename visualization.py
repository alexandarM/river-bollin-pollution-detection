# visualization.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc


def plot_training_history(train_losses, val_losses, title='Training History', save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_pr_curve(labels, scores, title='Precision-Recall Curve', save_path=None):
    precision, recall, _ = precision_recall_curve(labels, scores)
    avg_precision = average_precision_score(labels, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'AP = {avg_precision:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_roc_curve(labels, scores, title='ROC Curve', save_path=None):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_reconstructions(ae_model, dataset, device, num_samples=5, save_path=None):
    """Plot original vs reconstructed images."""
    import torch
    from torchvision.utils import make_grid
    from PIL import Image

    ae_model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    orig_imgs = []
    recon_imgs = []
    with torch.no_grad():
        for idx in indices:
            img, _ = dataset[idx]
            orig_imgs.append(img)
            img_batch = img.unsqueeze(0).to(device)
            recon = ae_model(img_batch).cpu().squeeze(0)
            recon_imgs.append(recon)

    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(orig_imgs[i].permute(1, 2, 0).clamp(0, 1))
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        # Reconstruction
        axes[1, i].imshow(recon_imgs[i].permute(1, 2, 0).clamp(0, 1))
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()