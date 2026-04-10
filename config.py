# config.py
import random
import numpy as np
import torch

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper‑parameters
AE_EPOCHS = 30
CLF_EPOCHS = 20
BATCH_SIZE = 64
LR = 3e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
