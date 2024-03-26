from one_stage.dataset import MetaSpectrogramDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from maml.meta import Meta

EPOCHS = 40
N_WAY = 5
K_SHOT_SUPPORT = 1
K_SHOT_QUERY = 15

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

img_cqt = np.load(
    "/home/dev/dataset/inclusion_2000_exclusion_4000/train/cqt/cargo/19020.npy"
)
print(img_cqt.shape)


config = [
    ("conv2d", [16, 1, 3, 3, 1, 2]),
    ("bn", [16]),
    ("relu", [True]),
    ("max_pool2d", [2, 2, 0]),
    ###################
    ("conv2d", [32, 16, 3, 3, 1, 2]),
    ("bn", [32]),
    ("relu", [True]),
    ("max_pool2d", [2, 2, 0]),
    ###################
    ("conv2d", [64, 32, 3, 3, 1, 2]),
    ("bn", [64]),
    ("relu", [True]),
    ("max_pool2d", [2, 2, 0]),
    ###################
    ("conv2d", [128, 64, 3, 3, 1, 2]),
    ("bn", [128]),
    ("relu", [True]),
    ("max_pool2d", [2, 2, 0]),
    ###################
    ("flatten", []),
    ("linear", [N_WAY, 128 * 7 * 9]),
]

args = {
    "update_lr": 0.01,
    "meta_lr": 1e-3,
    "n_way": N_WAY,
    "k_spt": K_SHOT_SUPPORT,
    "k_qry": K_SHOT_QUERY,
    "task_num": 4,
    "update_step": 5,
    "update_step_test": 10,
    "imgsz": (95, 126),
    "imgc": 1,
}

device = torch.device("cuda")
maml = Meta(config, **args).to(device)
print(maml)

train_loader = DataLoader(train_data, 4, num_workers=6, pin_memory=True)
val_loader = DataLoader(val_data, 4, num_workers=6, pin_memory=True)

for epoch in range(EPOCHS):
    train_loader = DataLoader(train_data, 4, num_workers=6, pin_memory=True)

    for step, data in enumerate(train_loader):
        x_shot, x_qry, y_shot, y_qry = data[0], data[1], data[2], data[3]

        x_shot, x_qry, y_shot, y_qry = (
            x_shot.to("cuda"),
            x_qry.to("cuda"),
            y_shot.to("cuda"),
            y_qry.to("cuda"),
        )

        accs = maml(x_shot, y_shot, x_qry, y_qry)
        exit()
