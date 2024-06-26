import os
import numpy as np
import random
import torch
import copy
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(
        self, data_dir, preprocessing_method="mel", included_classes=None, siamese=False
    ):
        self.data_dir = os.path.join(data_dir, preprocessing_method)
        self.included_classes = included_classes

        self.classes = self._get_classes()
        self.siamese = siamese

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.data = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        spectrogram, label = self.data[index]

        if not self.siamese:
            if len(spectrogram.shape) == 2:
                spectrogram = np.expand_dims(spectrogram, axis=0)
            return spectrogram, label

        target = torch.tensor(1, dtype=torch.float)
        data_len = len(self.data)
        idx = np.random.randint(0, data_len - 1)
        spec_two, label_two = self.data[idx]

        if index % 2 == 1:
            target = torch.tensor(0, dtype=torch.float)
            while (
                idx == index
                and np.array_equal(spectrogram, spec_two)
                and label == label_two
            ):
                idx = np.random.randint(0, data_len - 1)
                spec_two, label_two = self.data[idx]

            if len(spectrogram.shape) == 2:
                spectrogram = np.expand_dims(spectrogram, axis=0)
                spec_two = np.expand_dims(spec_two, axis=0)

            return spectrogram, spec_two, target

        while (
            idx == index
            and np.array_equal(spectrogram, spec_two)
            and label != label_two
        ):
            idx = np.random.randint(0, data_len - 1)
            spec_two, label_two = self.data[idx]

        if len(spectrogram.shape) == 2:
            spectrogram = np.expand_dims(spectrogram, axis=0)
            spec_two = np.expand_dims(spec_two, axis=0)

        return spectrogram, spec_two, target

    def _get_classes(self):
        return [c for c in os.listdir(self.data_dir) if c in self.included_classes]

    def _load_data(self):
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
            # if shot is not None:
            #     temp_data[label] = random.sample(temp_data[label], shot)

            for spectrum in temp_data[label]:
                data.append((spectrum, label))

        return data


class FewShotSpectrogramDataset(SpectrogramDataset):
    def __init__(
        self,
        data_dir,
        preprocessing_method="mel",
        included_classes=[],
        shot=10,
        shuffle=False,
        siamese=False,
    ):
        super().__init__(data_dir, preprocessing_method, included_classes, siamese)
        self.shot = shot
        if not shuffle:
            self.data = self._sample_dataset()
        self.shuffle = shuffle
        self.labels = [label for _, label in self.data]
        self.indices = {
            cls: np.where(np.array(self.labels) == cls)[0]
            for cls in range(len(included_classes))
        }

    def _sample_dataset(self):
        store = {c: [] for _, c in self.class_to_idx.items()}

        for spectrogram, label in self.data:
            store[label].append(spectrogram)

        data = []
        for key in store:
            indices = np.random.choice(len(store[key]), self.shot, replace=False)
            sampled_spectrograms = [(store[key][idx], key) for idx in indices]
            data += sampled_spectrograms

        return data

    def __len__(self):
        return len(self.included_classes) * self.shot

    def __getitem__(self, index):
        if not self.shuffle:
            spectrogram, label = self.data[index]
            if len(spectrogram.shape) == 2:
                spectrogram = np.expand_dims(spectrogram, axis=0)

            return spectrogram, label

        cls = index // self.shot
        sample_index = index % self.shot

        class_indices = np.random.choice(
            self.indices[cls], size=self.shot, replace=False
        )
        selected_idx = class_indices[sample_index]

        spectrogram, label = self.data[selected_idx]
        if len(spectrogram.shape) == 2:
            spectrogram = np.expand_dims(spectrogram, axis=0)

        if not self.siamese:
            return spectrogram, label

        second_class_indices = np.random.choice(
            self.indices[cls], size=self.shot, replace=False
        )
        while np.array_equal(class_indices, second_class_indices):
            second_class_indices = np.random.choice(
                self.indices[cls], size=self.shot, replace=False
            )

        target = torch.tensor(1, dtype=torch.float)

        if index % 2 == 1:
            new_cls = random.randint(0, len(self.classes) - 1)

            while cls == new_cls:
                new_cls = random.randint(0, len(self.classes) - 1)

            cls = new_cls
            target = torch.tensor(0, dtype=torch.float)

        second_sampled_index = random.randint(0, self.shot - 1)

        while sample_index == second_sampled_index:
            second_sampled_index = random.randint(0, self.shot - 1)

        second_selected_idx = second_class_indices[second_sampled_index]

        second_spectrogram, _ = self.data[second_selected_idx]

        if len(second_spectrogram.shape) == 2:
            second_spectrogram = np.expand_dims(second_spectrogram, axis=0)

        return spectrogram, second_spectrogram, target


class MaMLSpectrogramDataset(SpectrogramDataset):
    def __init__(
        self,
        data_dir,
        preprocessing_method="mel",
        included_classes=[],
        n_batch=200,
        n_episode=4,
        n_shot=1,
        n_query=15,
        siamese=False,
    ):
        super().__init__(data_dir, preprocessing_method, included_classes, siamese)
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
