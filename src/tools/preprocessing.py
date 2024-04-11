from nnAudio import Spectrogram
import torch
import torchaudio
import pandas as pd

FREQ_BINS = 95  # This number was based on the CQT, which have 95 freq bins for 4186hz
HOP_LENGTH = 256  # Used to generate an output of 128 on x axis
N_FFT = 2048  # This value is UNUSED because of the freq bins is mandatory
FMAX = 4186  # Correspond to a C8 note (Most High on a piano) (empirical)
FMIN = 18.0  # Minimun accepted value on CQT for audios of 1s


def right_pad_small_samples(signal, num_samples):
    """Insert a pad at the right side of the data

    Args:
        signal (tensor): The input signal.

    Returns:
        tensor: The processed signal.
    """
    length_signal = signal.shape[1]
    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal


def resample_to_target_sr(signal, sr, target_sample_rate):
    """Resample audio to desired sample rate.

    Args:
        signal (tensor): The input signal.
        sr (int): The desired sample rate.

    Returns:
        tensor: The processed signal.
    """
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        signal = resampler(signal)
    return signal


def mix_down_to_one_channel(signal):
    """Unify the data into ione channel.

    Args:
        signal (tensor): The input signal.

    Returns:
        tensor: The processed signal.
    """
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def cut_bigger_samples(signal, num_samples):
    """Cut the signal to the desired num of samples.

    Args:
        signal (tensor): The input signal.

    Returns:
        tensor: The processed signal.
    """
    if signal.shape[1] > num_samples:
        signal = signal[:, :num_samples]
    return signal


def get_full_audio(audio_sample_path, target_sample_rate):
    signal, sr = torchaudio.load(audio_sample_path)

    signal = resample_to_target_sr(signal, sr, target_sample_rate)
    signal = mix_down_to_one_channel(signal)
    return signal


def get_audio_chunk(signal, seconds_init, sample_rate, num_samples):
    initial_offset = int(seconds_init * sample_rate)
    final_offset = initial_offset + num_samples
    signal_chunk = signal[:, initial_offset:final_offset]
    signal_chunk = cut_bigger_samples(signal_chunk, num_samples)
    signal_chunk = right_pad_small_samples(signal_chunk, num_samples)
    return signal_chunk


def get_interleaved_metadata(metadata_path):
    original_metadata = pd.read_csv(metadata_path)
    interleaved_metadata = original_metadata.copy()
    interleaved_metadata["sub_init"] = interleaved_metadata["sub_init"].apply(
        lambda row: row + 0.5
    )
    metadata = pd.concat([original_metadata, interleaved_metadata])
    metadata = metadata.reset_index(drop=True)
    return metadata


def define_mel_spectrogram(sample_rate):
    """Returns a MelSpectrogram transforms object.

    Args:
        sample_rate (int): The desired sample rate.

    Returns:
        torchaudio.transforms: The MelSpectrogram object initialized.
    """
    mel_spectrogram = Spectrogram.MelSpectrogram(
        sr=sample_rate,
        n_fft=N_FFT,
        n_mels=FREQ_BINS,
        hop_length=HOP_LENGTH,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        htk=False,
        fmin=FMIN,
        fmax=FMAX,
        norm=1,
        trainable_mel=False,
        trainable_STFT=False,
        verbose=False,
    )
    return mel_spectrogram


def define_gamma_spectrogram(sample_rate):
    """Returns a Gammatonegram object.

    Args:
        sample_rate (int): The desired sample rate.

    Returns:
        Spectrogram: The Gammatonegram object initialized.
    """
    gamma_spectrogram = Spectrogram.Gammatonegram(
        sr=sample_rate,
        n_fft=N_FFT,
        n_bins=FREQ_BINS,
        hop_length=HOP_LENGTH,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        htk=False,
        fmin=FMIN,
        fmax=FMAX,
        norm=1,
        trainable_bins=False,
        trainable_STFT=False,
        verbose=False,
    )
    return gamma_spectrogram


def define_cqt_spectrogram(sample_rate):
    """Returns a CQT object.

    Args:
        sample_rate (int): The desired sample rate.

    Returns:
        Spectrogram: The CQT object initialized.
    """
    cqt_spectrogram = Spectrogram.CQT(
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
        n_bins=FREQ_BINS,
        bins_per_octave=12,
        filter_scale=1,
        norm=1,
        window="hann",
        center=True,
        pad_mode="reflect",
        trainable=False,
        output_format="Magnitude",
        verbose=False,
    )
    return cqt_spectrogram


def define_stft_spectrogram(sample_rate):
    stft_spectrogram = Spectrogram.STFT(
        sr=sample_rate,
        n_fft=N_FFT,
        freq_bins=FREQ_BINS,
        hop_length=HOP_LENGTH,
        window="hann",
        center=True,
        pad_mode="reflect",
        fmin=FMIN,
        fmax=FMAX,
        trainable=False,
        verbose=False,
        output_format="Magnitude",
    )

    return stft_spectrogram


_pre_processing_layers = {
    "mel": define_mel_spectrogram,
    "gammatone": define_gamma_spectrogram,
    "cqt": define_cqt_spectrogram,
    "stft": define_stft_spectrogram,
}


def get_preprocessing_layer(pre_processing_type, sample_rate):
    return _pre_processing_layers[pre_processing_type](sample_rate)
