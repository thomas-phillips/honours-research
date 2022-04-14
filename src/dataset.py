import torchaudio
import torch

import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DeeperShipDataset(Dataset):
    """A class describing the complete DeeperShip Dataset.
    """

    def __init__(self, metadata_file, target_sample_rate,
                 num_samples, transform=None, target_transform=None):
        """Initialize the DeeperShipDataset class.

        Args:
            metadata_file (os.path): The path to the metadata csv file.
            target_sample_rate (int): The sample rate to convert the read samples.
            num_samples (int): The number of samples to be considered.
            transform (torch transform, optional): A transform to be used on the signal data. Defaults to None.
            target_transform (torch transform, optional): A transform to be used on the target data. Defaults to None.
        """
        self.metadata = self._get_metadata(metadata_file)
        self.transform = transform
        self.target_transform = target_transform
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.class_mapping = {'tug':0, 'tanker':1, 'cargo':2, 'passengership':3, 'background':4}

    def __len__(self):
        """Returns the lenght of the dataset.

        Returns:
            int: The lenght of the dataset.
        """
        return len(self.metadata)

    def __getitem__(self, index):
        """Returns the item from the desired index.

        Args:
            index (int): The index of the desired data.

        Returns:
            tuple: The (signal,label) tuple
        """
        audio_sample_path = self.metadata.path.iloc[index]
        label = self._get_audio_sample_label(index)
        if self.target_transform:
            label = self.target_transform(label)

        signal, sr = torchaudio.load(
            audio_sample_path,
            frame_offset=self.metadata.sub_init.iloc[index],
            num_frames=self.num_samples,
        )
        signal = self._resample_to_target_sr(signal, sr)
        signal = self._mix_down_to_one_channel(signal)
        signal = self._cut_bigger_samples(signal)
        signal = self._right_pad_small_samples(signal)
        if self.transform:
            signal = self.transform(signal)

        return signal, label

    def _right_pad_small_samples(self, signal):
        """Insert a pad at the right side of the data

        Args:
            signal (tensor): The input signal.

        Returns:
            tensor: The processed signal.
        """
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_to_target_sr(self, signal, sr):
        """Resample audio to desired sample rate.

        Args:
            signal (tensor): The input signal.
            sr (int): The desired sample rate.

        Returns:
            tensor: The processed signal.
        """
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_to_one_channel(self, signal):
        """Unify the data into ione channel.

        Args:
            signal (tensor): The input signal.

        Returns:
            tensor: The processed signal.
        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_bigger_samples(self, signal):
        """Cut the signal to the desired num of samples.

        Args:
            signal (tensor): The input signal.

        Returns:
            tensor: The processed signal.
        """
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _get_audio_sample_label(self, index):
        """Gets the audio sample target label.

        Args:
            index (int): The index of the desired audio.

        Returns:
            tensor: The label of the data.
        """
        label = self.metadata.label.iloc[index]
        return torch.tensor(self.class_mapping[label.lower()])

    def _get_metadata(self, metadata_file):
        """Reads the csv metadata into a dataframe. 

        Args:
            metadata_file (os.path): The path to the csv file.

        Returns:
            pd.DataFrame: The metadata DataFrame.
        """
        metadata = pd.read_csv(metadata_file)
        return metadata


def create_data_loader(data, batch_size, shuffle=True):
    """Creates a pytorch dataloader from a Dataset.

    Args:
        data (Dataset): The desired dataset.
        batch_size (int): The size of the mini batch.
        shuffle (bool, optional): Indicates if the data needs to be shuffled. Defaults to True.

    Returns:
        DataLoader: The generated dataloader.
    """
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    return loader
