import argparse
import os
import random
import torch
import yaml
import numpy as np

from torch import nn
from torchsummary import summary
from torchmetrics import Accuracy, Precision, Recall, F1, ConfusionMatrix

from tools.utils import create_dir
from tools.dataset_loader import get_standard_dataset, get_maml_dataset
from tools.checkpoint import CheckpointManager, Checkpoint
from tools.train_manager import TrainManagerCNN, TrainManagerMaML
from resources.model import VGGNet, ResNet18
from resources.maml_configurations import make_config
from maml.meta import Meta

ZONES = [
    (2000, 4000),
    (3000, 5000),
    (4000, 6000),
]
UPLOAD_RESULTS = True


def main():
    """The core of the training execution.
    Initializes all the variables and call the respective methods.
    """
    torch.manual_seed(8)
    random.seed(8)
    np.random.seed(8)

    parser = argparse.ArgumentParser(description="Execute the training routine.")

    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="",
        required=True,
    )
    args = parser.parse_args()

    train(args.config_file)


def train(config_file):
    with open(config_file) as file:
        paths = yaml.load(file, Loader=yaml.FullLoader)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Start training - using {device}\n")

    model_info = paths["model"]
    optimiser_info = paths.get("optimiser", None)
    hyperparameter_info = paths["hyperparameters"]
    path_info = paths["paths"]
    preprocessing_methods = paths["preprocessing"]
    method = paths["method"]
    upload_results = paths["upload_results"]

    for key, value in paths.items():
        print(f"{key}: {value}", sep="\n")

    for z in ZONES:
        for preprocessing in preprocessing_methods:
            for model_data in model_info:
                if method == "cnn":
                    cnn_method(
                        z,
                        model_data["name"],
                        model_data["input_channels"],
                        preprocessing,
                        optimiser_info,
                        hyperparameter_info,
                        path_info,
                        device,
                        upload_results,
                    )

                elif method == "maml":
                    maml_method(
                        z,
                        model_data["name"],
                        model_data["input_channels"],
                        preprocessing,
                        hyperparameter_info,
                        path_info,
                        device,
                        upload_results,
                    )


def cnn_method(
    zone,
    model_name,
    input_channels,
    preprocessing,
    optimiser_info,
    hyperparameter_info,
    path_info,
    device="cuda",
    upload_results=False,
):

    output_directory = os.path.join(
        path_info["output_base_path"],
        f"in_{zone[0]}_ex_{zone[1]}_{preprocessing}_{model_name}_{input_channels}_{optimiser_info['type']}_{optimiser_info['early_stop']}",
    )

    print(output_directory)

    final_model_dir = create_dir(os.path.join(output_directory, "final_model"))
    checkpoint_dir = create_dir(os.path.join(output_directory, "checkpoints"))

    model = None
    if model_name == "resnet18":
        model = ResNet18(input_channels)
    elif model_name == "vggnet":
        model = VGGNet(input_channels)

    model = model.cuda()
    print(next(model.parameters()).device)
    print("Model Architecture")
    print(summary(model, (input_channels, 95, 126)))

    dataset_path = os.path.join(
        path_info["dataset_base_path"], f"inclusion_{zone[0]}_exclusion_{zone[1]}"
    )

    train_dataloader, validation_dataloader, _ = get_standard_dataset(
        dataset_path,
        preprocessing,
        hyperparameter_info["batch_size"],
    )

    print(len(train_dataloader.dataset.data))

    # Initialise loss funtion + optimizer.
    loss_fn = nn.CrossEntropyLoss()

    if optimiser_info["type"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hyperparameter_info["learning_rate"]
        )
    elif optimiser_info["type"] == "nadam":
        optimizer = torch.optim.NAdam(
            model.parameters(), lr=hyperparameter_info["learning_rate"]
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=hyperparameter_info["learning_rate"]
        )

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=hyperparameter_info["lr_schd_gamma"]
    )

    num_of_classes = len(train_dataloader.dataset.classes)
    print(f"Num of classes: {num_of_classes}")
    print(f"Num of examples: {len(train_dataloader.dataset.data)}")

    # Initialize metrics.
    accuracy = Accuracy("multiclass", average="macro", num_classes=num_of_classes)
    accuracy_micro = Accuracy(average="micro", num_classes=num_of_classes)
    accuracy_weight = Accuracy(average="weighted", num_classes=num_of_classes)
    precision = Precision(average="macro", num_classes=num_of_classes)
    recall = Recall(average="macro", num_classes=num_of_classes)
    f1 = F1(average="macro", num_classes=num_of_classes)
    confusion_matrix = ConfusionMatrix(num_classes=num_of_classes)

    metrics = {
        "Accuracy": accuracy,
        "AccuracyMicro": accuracy_micro,
        "AccuracyWeighted": accuracy_weight,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ConfusionMatrix": confusion_matrix,
    }

    # Create a checkpoint manager.
    checkpoint_manager = CheckpointManager(
        Checkpoint(model, optimizer),
        checkpoint_dir,
        device,
        max_to_keep=5,
        keep_best=True,
    )
    last_epoch = checkpoint_manager.restore_or_initialize()
    init_epoch = last_epoch + 1 if last_epoch != 0 else 0

    upload_info = {
        "optimiser": optimiser_info["type"],
        "input_channels": input_channels,
        "classes": train_dataloader.dataset.classes,
        "inclusion": zone[0],
        "exclusion": zone[1],
        "preprocessing": preprocessing,
        "type": "cnn",
        **hyperparameter_info,
    }
    print(upload_info)

    # Call train routine.
    train_manager = TrainManagerCNN(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        epochs=hyperparameter_info["epochs"],
        initial_epoch=init_epoch,
        metrics=metrics,
        reference_metric="Accuracy",
        device=device,
        early_stop=optimiser_info["early_stop"],
        upload_results=upload_results,
        upload_info=upload_info,
    )

    train_manager.train_model(checkpoint_manager)

    # Save the last checkpoint model.
    torch.save(model.state_dict(), os.path.join(final_model_dir, "last.pth"))

    # Save the best accuraccy model.
    checkpoint_manager.load_best_checkpoint()
    torch.save(model.state_dict(), os.path.join(final_model_dir, "best.pth"))


def maml_method(
    zone,
    model_name,
    input_channels,
    preprocessing,
    hyperparameter_info,
    path_info,
    device,
    upload_results=False,
):
    maml_config = {
        "update_lr": hyperparameter_info["update_learning_rate"],
        "meta_lr": hyperparameter_info["learning_rate"],
        "n_way": 5,
        "k_spt": hyperparameter_info["n_shot"],
        "k_qry": hyperparameter_info["n_query"],
        "task_num": hyperparameter_info["task_num"],
        "update_step": hyperparameter_info["update_step"],
        "update_step_test": hyperparameter_info["update_step_test"],
        "imgsz": (95, 126),
        "imgc": input_channels,
    }

    model_config = make_config(model_name, input_channels, 5)
    model = Meta(model_config, **maml_config).to(device)

    dataset_path = os.path.join(
        path_info["dataset_base_path"], f"inclusion_{zone[0]}_exclusion_{zone[1]}"
    )
    train_dataloader, validation_dataloader, _ = get_maml_dataset(
        dataset_path,
        preprocessing,
        hyperparameter_info["episodes"],
        hyperparameter_info["n_shot"],
        hyperparameter_info["n_query"],
        hyperparameter_info["batch_size"],
        hyperparameter_info["task_num"],
    )
    upload_info = {
        "model": model_name,
        "input_channels": input_channels,
        "classes": train_dataloader.dataset.classes,
        "inclusion": zone[0],
        "exclusion": zone[1],
        "preprocessing": preprocessing,
        "type": "maml",
        **hyperparameter_info,
    }

    train_mananger = TrainManagerMaML(
        model,
        train_dataloader,
        validation_dataloader,
        hyperparameter_info["epochs"],
        upload_results=upload_results,
        upload_info=upload_info,
    )

    train_mananger.train_model()


if __name__ == "__main__":
    main()
