from maml.models.maml import MAML, make
from one_stage.dataset import MetaSpectrogramDataset
from torch.utils.data import DataLoader
from maml.datasets import collate_fn
import maml.utils.optimizers as optimizers
import maml.utils as utils
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

EPOCH = 40

enc_args = {"bn_args": {"track_running_stats": False, "n_episode": 4}}

model = make("convnet4", enc_args, "logistic", {"n_way": 5})

# print(model.encoder)
# print(model.encoder.channels)
# exit()

optimizer, lr_scheduler = optimizers.make("adam", model.parameters(), lr=0.001)

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
train_loader = DataLoader(
    train_data, 4, collate_fn=collate_fn, num_workers=1, pin_memory=True
)
val_loader = DataLoader(
    val_data, 4, collate_fn=collate_fn, num_workers=1, pin_memory=True
)

start_epoch = 1
timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()
eval_val = True

aves_keys = ["tl", "ta", "vl", "va"]
trlog = dict()
for k in aves_keys:
    trlog[k] = []

inner_args = {
    "n_step": 5,
    "encoder_lr": 0.01,
    "classifier_lr": 0.01,
    "first_order": False,  # set to True for FOMAML
    "frozen": ["bn"],
    "momentum": 0.9,
}

for epoch in range(start_epoch, 40 + 1):
    timer_epoch.start()
    aves = {k: utils.AverageMeter() for k in aves_keys}

    # meta-train
    model.train()
    np.random.seed(epoch)

    for data in tqdm(train_loader, desc="meta-train", leave=False):
        x_shot, x_query, y_shot, y_query = data
        print(x_shot.shape, x_query.shape)
        # exit()
        x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
        x_query, y_query = x_query.cuda(), y_query.cuda()

        logits = model(x_shot, x_query, y_shot, inner_args, meta_train=True)
        logits = logits.flatten(0, 1)
        labels = y_query.flatten()

        pred = torch.argmax(logits, dim=-1)
        acc = utils.compute_acc(pred, labels)
        loss = F.cross_entropy(logits, labels)
        aves["tl"].update(loss.item(), 1)
        aves["ta"].update(acc, 1)

        optimizer.zero_grad()
        loss.backward()
        for param in optimizer.param_groups[0]["params"]:
            nn.utils.clip_grad_value_(param, 10)
        optimizer.step()

    # meta-val
    if eval_val:
        model.eval()
        np.random.seed(0)

        for data in tqdm(val_loader, desc="meta-val", leave=False):
            x_shot, x_query, y_shot, y_query = data
            x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
            x_query, y_query = x_query.cuda(), y_query.cuda()

            logits = model(x_shot, x_query, y_shot, inner_args, meta_train=False)
            logits = logits.flatten(0, 1)
            labels = y_query.flatten()

            pred = torch.argmax(logits, dim=-1)
            acc = utils.compute_acc(pred, labels)
            loss = F.cross_entropy(logits, labels)
            aves["vl"].update(loss.item(), 1)
            aves["va"].update(acc, 1)

    if lr_scheduler is not None:
        lr_scheduler.step()

    for k, avg in aves.items():
        aves[k] = avg.item()
        trlog[k].append(aves[k])

    t_epoch = utils.time_str(timer_epoch.end())
    t_elapsed = utils.time_str(timer_elapsed.end())
    t_estimate = utils.time_str(
        timer_elapsed.end() / (epoch - start_epoch + 1) * (EPOCH - start_epoch + 1)
    )

    # formats output
    log_str = "epoch {}, meta-train {:.4f}|{:.4f}".format(
        str(epoch), aves["tl"], aves["ta"]
    )

    if eval_val:
        log_str += ", meta-val {:.4f}|{:.4f}".format(aves["vl"], aves["va"])

    log_str += ", {} {}/{}".format(t_epoch, t_elapsed, t_estimate)
    utils.log(log_str)
