import os
from tools.preprocessing import get_preprocessing_layer, right_pad_small_samples
from tools.utils import create_dir
import numpy as np
import torchaudio
from tqdm import tqdm

FILTERS = [
    "mel",
    "cqt",
    "gammatone",
    "stft",
    "combined1",
    "combined2",
    "combined3",
    "combined4",
]
CLASSES = ["background", "cargo", "passengership", "tanker", "tug"]


def main():
    zones = [
        (2000, 4000),
        (3000, 5000),
        (4000, 6000),
    ]
    sets = ["test", "train", "validation"]

    for zone in zones:
        for s in sets:
            preprocess_dataset(
                dataset_dir=f"/home/dev/dataset/inclusion_{zone[0]}_exclusion_{zone[1]}/{s}/audio",
                target_sample_rate=32000,
            )


def preprocess_dataset(dataset_dir, target_sample_rate=32000):
    transformation_mel = get_preprocessing_layer("mel", target_sample_rate)
    transformation_cqt = get_preprocessing_layer("cqt", target_sample_rate)
    transformation_gamma = get_preprocessing_layer("gammatone", target_sample_rate)
    transformation_stft = get_preprocessing_layer("stft", target_sample_rate)

    for f in FILTERS:
        filter_path = dataset_dir.replace("audio", f)
        create_dir(filter_path)
        for c in CLASSES:
            class_dir = os.path.join(filter_path, c)
            create_dir(class_dir)

    file_count = 0
    for root, dirs, files in os.walk(dataset_dir, topdown=False):
        file_count += len(files)

    with tqdm(total=file_count, desc="Processing files") as pbar:
        for root, dirs, files in os.walk(dataset_dir, topdown=False):
            for name in files:
                original_path = os.path.join(root, name)
                try:
                    audio, sr = torchaudio.load(original_path)
                except Exception as err:
                    print(err)
                    continue

                audio = right_pad_small_samples(audio, target_sample_rate)

                original_path_stripped = original_path[:-4]

                img_mel = transformation_mel(audio).cpu().numpy()[0]
                if img_mel.shape[1] < 126:
                    print(original_path)
                    print(audio.shape)
                    print(sr)
                np.save(original_path_stripped.replace("audio", "mel"), img_mel)

                img_cqt = transformation_cqt(audio).cpu().numpy()[0]
                np.save(original_path_stripped.replace("audio", "cqt"), img_cqt)

                img_gamma = transformation_gamma(audio).cpu().numpy()[0]
                np.save(original_path_stripped.replace("audio", "gammatone"), img_gamma)

                img_stft = transformation_stft(audio).cpu().numpy()[0]
                np.save(original_path_stripped.replace("audio", "stft"), img_stft)

                img_combined1 = np.stack((img_mel, img_cqt, img_gamma))
                np.save(
                    original_path_stripped.replace("audio", "combined1"), img_combined1
                )

                img_combined2 = np.stack((img_mel, img_cqt, img_stft))
                np.save(
                    original_path_stripped.replace("audio", "combined2"), img_combined2
                )

                img_combined3 = np.stack((img_mel, img_stft, img_gamma))
                np.save(
                    original_path_stripped.replace("audio", "combined3"), img_combined3
                )

                img_combined4 = np.stack((img_cqt, img_stft, img_gamma))
                np.save(
                    original_path_stripped.replace("audio", "combined4"), img_combined4
                )

                pbar.update(1)


if __name__ == "__main__":
    main()
