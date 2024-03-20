from torch.utils.data import DataLoader
import os

from one_stage.dataset import SpectrogramDataset


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


def get_dataset(data_dir, preprocessing_method, batch_size=25, excluded_classes=[]):
    """Returns the desired dataloaders for validation and train.

    Args:
        config (dict, required): The dict resulting from the YAML config file.

    Returns:
        DataLoader, DataLoader : The train and the validation dataloaders, respectively.
    """
    train_path = os.path.join(data_dir, "train")
    train_dataset = SpectrogramDataset(
        train_path, preprocessing_method, excluded_classes
    )

    validation_path = os.path.join(data_dir, "validation")
    validation_dataset = SpectrogramDataset(
        validation_path, preprocessing_method, excluded_classes
    )

    test_path = os.path.join(data_dir, "test")
    test_dataset = SpectrogramDataset(test_path, preprocessing_method, excluded_classes)

    train_dataloader = create_data_loader(train_dataset, batch_size=batch_size)
    validation_dataloader = create_data_loader(
        validation_dataset, batch_size=batch_size
    )
    test_dataloader = create_data_loader(test_dataset, batch_size=batch_size)

    return train_dataloader, validation_dataloader, test_dataloader
