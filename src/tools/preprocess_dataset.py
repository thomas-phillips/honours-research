import os
import torch
import torchaudio
import pandas as pd


class DeeperShipPreprocessor:
    def __init__(
        self,
        metadata_file,
        target_sample_rate,
        num_samples,
        transform=None,
        target_transform=None,
    ) -> None:
        self.metadata = self._get_metadata(metadata_file)
        self.transform = transform
        self.target_transform = target_transform
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.class_mapping = {
            "tug": 0,
            "tanker": 1,
            "cargo": 2,
            "passengership": 3,
            "background": 4,
            "other": 5,
        }

    def _get_metadata(self, metadata_file):
        return pd.read_csv(metadata_file)

    def preprocess_data(self):
        for i in range(len(self.metadata)):
            base_path = "/home/dev/dataset"
            audio_sample_path = base_path + self.metadata.path.iloc[i]
            label = self._get_audio_sample_label(i)
            print(audio_sample_path, label)

            if self.target_transform:
                label = self.target_transform(label)

            signal, sr = torchaudio.load(
                audio_sample_path,
                frame_offset=self.metadata.sub_init.iloc[i]
                * self.metadata.sample_rate.iloc[i],
                num_frames=self.num_samples,
            )

            signal = self._resample_to_target_sr(signal, sr)
            signal = self._mix_down_to_one_channel(signal)
            signal = self._cut_bigger_samples(signal)
            signal = self._right_pad_small_samples(signal)
            if self.transform:
                signal = self.transform(signal)
            print(signal)

    def _right_pad_small_samples(self, signal):
        print("Signal: ", signal)
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_to_target_sr(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_to_one_channel(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_bigger_samples(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def _get_audio_sample_label(self, index):
        label = self.metadata.label.iloc[index]
        return torch.tensor(self.class_mapping[label.lower()])
