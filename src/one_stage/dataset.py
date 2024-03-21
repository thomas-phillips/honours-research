import os
import numpy as np
import random
import torch
import copy
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(
        self, data_dir, preprocessing_method="mel", included_classes=[], shot=None
    ):
        self.data_dir = os.path.join(data_dir, preprocessing_method)
        self.included_classes = included_classes

        if len(self.included_classes) == 0:
            self.included_classes = copy.deepcopy(self.classes)
        else:
            self.classes = self._get_classes()

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.data = self._load_data(shot)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        spectrogram, label = self.data[idx]
        spectrogram = np.expand_dims(spectrogram, axis=0)

        return spectrogram, label

    def _get_classes(self):
        return [c for c in os.listdir(self.data_dir) if c in self.included_classes]

    def _load_data(self, shot):
        data = []
        for i, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(self.data_dir, cls_name)
            file_names = os.listdir(cls_dir)
            for file in file_names:
                file_path = os.path.join(cls_dir, file)
                spectrogram = np.load(file_path)
                label = self.class_to_idx[cls_name]
                data.append((spectrogram, label))

        if shot is not None:
            return random.sample(data, shot)
        return data
