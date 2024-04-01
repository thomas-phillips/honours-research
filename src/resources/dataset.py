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

        self.classes = self._get_classes()

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.data = self._load_data(shot)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        spectrogram, label = self.data[index]
        spectrogram = np.expand_dims(spectrogram, axis=0)

        return spectrogram, label

    def _get_classes(self):
        return [c for c in os.listdir(self.data_dir) if c in self.included_classes]

    def _load_data(self, shot):
        temp_data = {}
        data = []
        for label in self.class_to_idx:
            temp_data[self.class_to_idx[label]] = []

        for i, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(self.data_dir, cls_name)
            file_names = os.listdir(cls_dir)
            for file in file_names:
                file_path = os.path.join(cls_dir, file)
                spectrogram = np.load(file_path)
                label = self.class_to_idx[cls_name]
                temp_data[label].append(spectrogram)

        for label in temp_data:
            if shot is not None:
                temp_data[label] = random.sample(temp_data[label], shot)

            for spectrum in temp_data[label]:
                data.append((spectrum, label))

        return data


class MaMLSpectrogramDataset(SpectrogramDataset):
    def __init__(
        self,
        data_dir,
        preprocessing_method="mel",
        included_classes=[],
        shot=None,
        n_batch=200,
        n_episode=4,
        n_shot=1,
        n_query=15,
    ):
        super().__init__(data_dir, preprocessing_method, included_classes, shot)
        self.n_batch = n_batch
        self.n_episode = n_episode
        self.n_way = len(self.classes)
        self.n_shot = n_shot
        self.n_query = n_query
        self.catlocs = tuple()
        for cat in range(len(self.classes)):
            self.catlocs += (np.argwhere(np.array(self.data)[:, 1] == cat).reshape(-1),)

    def __len__(self):
        return self.n_batch * self.n_episode

    def __getitem__(self, index):
        shot, query = [], []
        classes = np.random.choice(len(self.classes), self.n_way, replace=False)
        for c in classes:
            c_shot, c_query = [], []
            idx_list = np.random.choice(
                self.catlocs[c], self.n_shot + self.n_query, replace=False
            )
            shot_idx, query_idx = idx_list[: self.n_shot], idx_list[-self.n_query :]
            for idx in shot_idx:
                c_shot.append(torch.tensor(np.expand_dims(self.data[idx][0], axis=0)))
            for idx in query_idx:
                c_query.append(torch.tensor(np.expand_dims(self.data[idx][0], axis=0)))

            shot.append(torch.stack(c_shot))
            query.append(torch.stack(c_query))

        shot = torch.cat(shot, dim=0)
        query = torch.cat(query, dim=0)
        cls = torch.arange(self.n_way)[:, None]
        shot_labels = cls.repeat(1, self.n_shot).flatten()
        query_labels = cls.repeat(1, self.n_query).flatten()

        return shot, query, shot_labels, query_labels
