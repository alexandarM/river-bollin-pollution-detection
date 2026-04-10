# dataset_utils.py
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from config import IMAGENET_MEAN, IMAGENET_STD


def split_list(lst, val_frac, test_frac):
    """Stratified train/val/test split."""
    lst = lst.copy()
    random.shuffle(lst)
    n = len(lst)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    return lst[n_test + n_val:], lst[n_test:n_test + n_val], lst[:n_test]


# Base transforms
BASE_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

AUG_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# No‑normalise transform for autoencoder (MSE in pixel space)
AE_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # [0, 1]
])


class RiverDataset(Dataset):
    """Labelled dataset for the classifier."""
    def __init__(self, img_paths, positive_set, augment=False):
        self.img_paths = img_paths
        self.positive_set = positive_set
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        label = 1 if path.name in self.positive_set else 0
        img = Image.open(path).convert('RGB')
        tf = AUG_TF if (self.augment and label == 1) else BASE_TF
        return tf(img), label, str(path)


class SimpleDataset(Dataset):
    # Unlabelled dataset for autoencoder (returns image + dummy 0
    def __init__(self, img_paths, tf):
        self.img_paths = img_paths
        self.tf = tf

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.tf(img), 0


def make_weighted_sampler(paths, positive_set):
    """Balance batches so positives appear ~50% of the time."""
    labels = [1 if p.name in positive_set else 0 for p in paths]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    w_pos = 1.0 / n_pos if n_pos else 0
    w_neg = 1.0 / n_neg if n_neg else 0
    weights = [w_pos if l == 1 else w_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
