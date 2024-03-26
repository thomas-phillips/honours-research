from one_stage.dataset import MetaSpectrogramDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

EPOCHS = 40


train_data = MetaSpectrogramDataset(
    "/home/dev/dataset/inclusion_2000_exclusion_4000/train",
    "gammatone",
    included_classes=["background", "cargo", "passengership", "tanker", "tug"],
)
val_data = MetaSpectrogramDataset(
    "/home/dev/dataset/inclusion_2000_exclusion_4000/validation",
    "gammatone",
    included_classes=["background", "cargo", "passengership", "tanker", "tug"],
)
train_loader = DataLoader(train_data, 4, num_workers=6, pin_memory=True)
val_loader = DataLoader(val_data, 4, num_workers=6, pin_memory=True)
