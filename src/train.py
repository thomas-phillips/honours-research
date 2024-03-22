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
from tools.dataset_loader import get_dataset
from tools.model import get_model
from tools.checkpoint import CheckpointManager, Checkpoint
from tools.train_manager import TrainManager
from one_stage.model import VGGNet, CustomResNet18

ZONES = [
    (2000, 4000),
    (3000, 5000),
    (4000, 6000),
]
UPLOAD_RESULTS = True


def create_parser():
    """Create the parser object.

    Returns:
        parser: The generated parser object with arguments
    """
    parser = argparse.ArgumentParser(description="Execute the training routine.")

    parser.add_argument(
        "config_file",
        type=str,
        help="",
    )

    return parser


def main():
    """The core of the training execution.
    Initializes all the variables and call the respective methods.
    """
    torch.manual_seed(8)
    random.seed(8)
    np.random.seed(8)

    parser = create_parser()
    args = parser.parse_args()
    with open(args.config_file) as file:
        paths = yaml.load(file, Loader=yaml.FullLoader)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Start training - using {device}\n")

    model_info = paths["model"]
    optimiser_info = paths["optimiser"]
    hyperparameter_info = paths["hyperparameters"]
    path_info = paths["paths"]
    preprocessing_methods = paths["preprocessing"]

    few_shot_info = None
    if "few_shot" in paths:
        few_shot_info = paths["few_shot"]

    print(
        model_info,
        optimiser_info,
        hyperparameter_info,
        path_info,
        preprocessing_methods,
        sep="\n",
    )

    early_stop = False if optimiser_info["early_stop"] == 0 else True

    for z in ZONES:
        for preprocessing in preprocessing_methods:
            for model_data in model_info:
                train(
                    z,
                    model_data["name"],
                    model_data["input_channels"],
                    preprocessing,
                    optimiser_info,
                    hyperparameter_info,
                    path_info,
                    early_stop,
                    device,
                    few_shot_info,
                )


def train(
    zone,
    model_name,
    input_channels,
    preprocessing,
    optimiser_info,
    hyperparameter_info,
    path_info,
    early_stop=0,
    device="cuda",
    few_shot_info=None,
):

    output_directory = os.path.join(
        path_info["output_base_path"],
        f"in_{zone[0]}_ex_{zone[1]}_{preprocessing}_{model_name}_{input_channels}_{optimiser_info['type']}_{optimiser_info['early_stop']}",
    )

    if few_shot_info is not None:
        output_directory = os.path.join(
            path_info["output_base_path"],
            f"in_{zone[0]}_ex_{zone[1]}_{preprocessing}_{model_name}_{input_channels}_{optimiser_info['type']}_{optimiser_info['early_stop']}_few-shot_{len(few_shot_info['way'])}_{few_shot_info['shot']}",
        )

    print(output_directory)

    final_model_dir = create_dir(os.path.join(output_directory, "final_model"))
    checkpoint_dir = create_dir(os.path.join(output_directory, "checkpoints"))

    model = None
    if model_name == "resnet18":
        model = CustomResNet18(input_channels)
    elif model_name == "vggnet":
        model = VGGNet(input_channels)

    model = model.cuda()
    print(next(model.parameters()).device)
    print("Model Architecture")
    print(summary(model, (input_channels, 95, 126)))

    dataset_path = os.path.join(
        path_info["dataset_base_path"], f"inclusion_{zone[0]}_exclusion_{zone[1]}"
    )

    if few_shot_info is None:
        train_dataloader, validation_dataloader, _ = get_dataset(
            dataset_path,
            preprocessing,
            hyperparameter_info["batch_size"],
        )
    else:
        train_dataloader, validation_dataloader, _ = get_dataset(
            dataset_path,
            preprocessing,
            hyperparameter_info["batch_size"],
            included_classes=few_shot_info["way"],
            shot=few_shot_info["shot"],
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
        "early_stop": optimiser_info["early_stop"],
        "model": model_name,
        "input_channels": input_channels,
        "epochs": hyperparameter_info["epochs"],
        "batch_size": hyperparameter_info["batch_size"],
        "learning_rate": hyperparameter_info["learning_rate"],
        "lr_schd_gamma": hyperparameter_info["lr_schd_gamma"],
        "classes": train_dataloader.dataset.classes,
        "inclusion": zone[0],
        "exclusion": zone[1],
        "preprocessing": preprocessing,
    }
    print(upload_info)

    # Call train routine.
    train_manager = TrainManager(
        model,
        loss_fn,
        optimizer,
        lr_scheduler,
        train_dataloader,
        validation_dataloader,
        hyperparameter_info["epochs"],
        initial_epoch=init_epoch,
        metrics=metrics,
        reference_metric="Accuracy",
        device=device,
        early_stop=early_stop,
        upload_info=upload_info,
        upload_results=UPLOAD_RESULTS,
        few_shot=few_shot_info,
    )

    train_manager.start_train(checkpoint_manager)

    # Save the last checkpoint model.
    torch.save(model.state_dict(), os.path.join(final_model_dir, "last.pth"))

    # Save the best accuraccy model.
    checkpoint_manager.load_best_checkpoint()
    torch.save(model.state_dict(), os.path.join(final_model_dir, "best.pth"))


if __name__ == "__main__":
    main()
