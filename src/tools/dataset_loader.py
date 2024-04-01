from torch.utils.data import DataLoader
import os

from resources.dataset import SpectrogramDataset, MaMLSpectrogramDataset

CLASSES = ["background", "cargo", "passengership", "tanker", "tug"]


def create_data_loader(data, batch_size, shuffle=True):
    """Creates a pytorch dataloader from a Dataset.

    Args:
        data (Dataset): The desired dataset.
        batch_size (int): The size of the mini batch.
        shuffle (bool, optional): Indicates if the data needs to be shuffled. Defaults to True.

    Returns:
        DataLoader: The generated dataloader.
    """
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=6)

    return loader


def get_standard_dataset(
    data_dir, preprocessing_method, batch_size=25, included_classes=CLASSES
):
    """#todo: Update description
    Returns the desired dataloaders for validation and train.

    Args:
        config (dict, required): The dict resulting from the YAML config file.

    Returns:
        DataLoader, DataLoader : The train and the validation dataloaders, respectively.
    """
    train_path = os.path.join(data_dir, "train")
    train_dataset = SpectrogramDataset(
        train_path, preprocessing_method, included_classes
    )

    validation_path = os.path.join(data_dir, "validation")
    validation_dataset = SpectrogramDataset(
        validation_path, preprocessing_method, included_classes
    )

    test_path = os.path.join(data_dir, "test")

    test_dataset = SpectrogramDataset(test_path, preprocessing_method, included_classes)

    train_dataloader = create_data_loader(train_dataset, batch_size=batch_size)
    validation_dataloader = create_data_loader(
        validation_dataset, batch_size=batch_size
    )
    test_dataloader = create_data_loader(test_dataset, batch_size=batch_size)

    return train_dataloader, validation_dataloader, test_dataloader


def get_maml_dataset(
    data_dir,
    preprocessing_method,
    n_episode,
    n_shot,
    n_query,
    n_batch,
    task_num=4,
    included_classes=CLASSES,
):
    """#todo: Update description
    Returns the desired dataloaders for validation and train.

    Args:
        config (dict, required): The dict resulting from the YAML config file.

    Returns:
        DataLoader, DataLoader : The train and the validation dataloaders, respectively.
    """
    train_path = os.path.join(data_dir, "train")
    train_dataset = MaMLSpectrogramDataset(
        train_path,
        preprocessing_method,
        included_classes,
        n_episode=n_episode,
        n_batch=n_batch,
        n_shot=n_shot,
        n_query=n_query,
    )

    validation_path = os.path.join(data_dir, "validation")
    validation_dataset = MaMLSpectrogramDataset(
        validation_path,
        preprocessing_method,
        included_classes,
        n_episode=n_episode,
        n_batch=n_batch,
        n_shot=n_shot,
        n_query=n_query,
    )

    test_path = os.path.join(data_dir, "test")

    test_dataset = MaMLSpectrogramDataset(
        test_path,
        preprocessing_method,
        included_classes,
        n_episode=n_episode,
        n_batch=n_batch,
        n_shot=n_shot,
        n_query=n_query,
    )

    train_dataloader = create_data_loader(train_dataset, batch_size=task_num)
    validation_dataloader = create_data_loader(validation_dataset, batch_size=None)
    test_dataloader = create_data_loader(test_dataset, batch_size=None)

    return train_dataloader, validation_dataloader, test_dataloader
