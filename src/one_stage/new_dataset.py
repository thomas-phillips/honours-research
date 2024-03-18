import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, preprocessing_method="mel", excluded_classes=[]):
        self.data_dir = os.path.join(data_dir, preprocessing_method)
        self.excluded_classes = excluded_classes
        self.classes = self._get_classes()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.data = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        spectrogram, label = self.data[idx]
        spectrogram = np.expand_dims(spectrogram, axis=0)
        # if len(spectrogram.shape) == 2:
        #     # If single-channel, convert to three-channel by repeating the channel dimension
        #     spectrogram = np.repeat(spectrogram[:, :, np.newaxis], 3, axis=2)
        #     spectrogram = np.transpose(spectrogram, (2, 0, 1))

        return spectrogram, label

    def _get_classes(self):
        return [c for c in os.listdir(self.data_dir) if c not in self.excluded_classes]

    def _load_data(self):
        data = []
        for i, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(self.data_dir, cls_name)
            file_names = os.listdir(cls_dir)
            for file in file_names:
                file_path = os.path.join(cls_dir, file)
                spectrogram = np.load(file_path)
                label = self.class_to_idx[cls_name]
                data.append((spectrogram, label))

        return data
