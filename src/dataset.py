from torch.utils.data import DataLoader

from preprocessing import get_preprocessing_layer

from one_stage.dataset import DeeperShipFeature, DeeperShip
from one_stage.new_dataset import SpectrogramDataset


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


def get_split_dataloader(config, split="test", shuffle=False):
    """Returns the desired dataloader for the selected split.

    Args:
        config (dict, required): The dict resulting from the YAML config file.

    Returns:
        DataLoader : The desired dataloader object.
    """
    if config["dataset"]["type"] == "deepershipfeature":
        batch_size = config["hyperparameters"]["batch_size"]
        dataset_path = config["dataset"][f"{split}_root_path"]
        preprocessings = config["dataset"]["preprocess"]
        num_of_classes = config["model"]["num_of_classes"]

        # Get the dataset and dataloader.
        dataset = DeeperShipFeature(
            dataset_path, num_of_classes=num_of_classes, preprocessing=preprocessings
        )
        dataloader = create_data_loader(dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        sample_rate = config["hyperparameters"]["sample_rate"]
        number_of_samples = sample_rate * config["hyperparameters"]["number_of_samples"]
        batch_size = config["hyperparameters"]["batch_size"]

        metadata_path = config["dataset"][f"{split}_metadata"]

        pre_processing_type = config["preprocessing"]["type"].lower()
        transformation = get_preprocessing_layer(pre_processing_type, sample_rate)

        # Get the dataset and dataloader.
        dataset = DeeperShip(
            metadata_path, sample_rate, number_of_samples, transform=transformation
        )
        dataloader = create_data_loader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def get_dataset(config):
    """Returns the desired dataloaders for validation and train.

    Args:
        config (dict, required): The dict resulting from the YAML config file.

    Returns:
        DataLoader, DataLoader : The train and the validation dataloaders, respectively.
    """
    if config["dataset"]["type"] == "deepershipfeature":
        sample_rate = config["hyperparameters"]["sample_rate"]
        batch_size = config["hyperparameters"]["batch_size"]
        train_dataset_path = config["dataset"]["train_root_path"]
        validation_dataset_path = config["dataset"]["validation_root_path"]
        preprocessings = config["dataset"]["preprocess"]
        num_of_classes = config["model"]["num_of_classes"]
        number_of_samples = sample_rate * config["hyperparameters"]["number_of_samples"]

        train_dataset = SpectrogramDataset(train_dataset_path)
        validation_dataset = SpectrogramDataset(validation_dataset_path)

        train_dataloader = create_data_loader(train_dataset, batch_size=batch_size)
        validation_dataloader = create_data_loader(
            validation_dataset, batch_size=batch_size
        )

        # Get the training and validation.
        # train_dataset = DeeperShipFeature(
        #     train_dataset_path,
        #     num_samples=number_of_samples,
        #     num_of_classes=num_of_classes,
        #     preprocessing=preprocessings,
        # )
        # print(type(train_dataset))
        # train_dataloader = create_data_loader(train_dataset, batch_size=batch_size)

        # validation_dataset = DeeperShipFeature(
        #     validation_dataset_path,
        #     num_samples=number_of_samples,
        #     num_of_classes=num_of_classes,
        #     preprocessing=preprocessings,
        # )
        # print(type(validation_dataset))
        # validation_dataloader = create_data_loader(
        #     validation_dataset, batch_size=batch_size, shuffle=False
        # )
        return train_dataloader, validation_dataloader
    else:
        sample_rate = config["hyperparameters"]["sample_rate"]
        number_of_samples = sample_rate * config["hyperparameters"]["number_of_samples"]
        batch_size = config["hyperparameters"]["batch_size"]

        train_metadata_path = config["dataset"]["train_metadata"]
        validation_metadata_path = config["dataset"]["validation_metadata"]

        pre_processing_type = config["preprocessing"]["type"].lower()
        transformation = get_preprocessing_layer(pre_processing_type, sample_rate)

        # Get the training, validation and test dataloaders.
        train_dataset = DeeperShip(
            train_metadata_path,
            sample_rate,
            number_of_samples,
            transform=transformation,
        )
        print(type(train_dataset))
        train_dataloader = create_data_loader(train_dataset, batch_size=batch_size)

        validation_dataset = DeeperShip(
            validation_metadata_path,
            sample_rate,
            number_of_samples,
            transform=transformation,
        )
        print(type(validation_dataset))
        validation_dataloader = create_data_loader(
            validation_dataset, batch_size=batch_size, shuffle=False
        )
        return train_dataloader, validation_dataloader
