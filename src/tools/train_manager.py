import math
from tqdm import tqdm
import boto3
from tools.utils import plot_confusion_matrix
from decimal import Decimal
from datetime import datetime
import json
import hashlib
import copy
import numpy as np


class TrainManager:
    def __init__(
        self,
        model,
        train_dataloader,
        validation_dataloader,
        epochs,
        device="cuda",
        upload_results=False,
        upload_info={},
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.epochs = epochs
        self.device = device
        self.upload_results = upload_results
        self.upload_info = upload_info

    def train_model(self):
        print("Starting training...")

    def upload_results_to_dynamodb(self, timestamp, results, **kwargs):
        dynamodb = boto3.resource("dynamodb")
        table_name = "StorageStack-ResultsTable"
        table = dynamodb.Table(table_name)

        upload_info = copy.deepcopy(self.upload_info)

        for key, item in upload_info.items():
            if isinstance(item, float):
                upload_info[key] = Decimal(str(item))
            if isinstance(item, list):
                upload_info[key] = json.dumps(item)

        upload_info_str = json.dumps(self.upload_info)
        results_str = json.dumps(results)
        record_id = hashlib.sha256(
            f"{results_str}{upload_info_str}".encode()
        ).hexdigest()

        item = {
            "id": record_id,
            "timestamp": timestamp,
            "model": self.model.name,
            "measurements": results_str,
            "type": "training",
            **kwargs,
            **upload_info,
        }

        print(item)
        # table.put_item(Item=item)


class TrainManagerCNN(TrainManager):
    def __init__(
        self,
        model,
        train_dataloader,
        validation_dataloader,
        epochs,
        loss_fn,
        optimizer,
        lr_scheduler,
        initial_epoch=0,
        early_stop=True,
        metrics={},
        reference_metric="",
        device="cuda",
        upload_results=False,
        upload_info={},
    ):
        super().__init__(
            model,
            train_dataloader,
            validation_dataloader,
            epochs,
            device,
            upload_results,
            upload_info,
        )

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metrics = metrics
        self.reference_metric = reference_metric
        self.initial_epoch = initial_epoch

        self.best_measure = 0

        self.current_validation_loss = 0
        self.last_validation_loss = 0

        self.early_stop = early_stop
        self.trigger_times = 0
        self.patience = 4
        return

    def _efficient_zero_grad(self, model):
        for param in model.parameters():
            param.grad = None

    def _train_single_epoch(self, epoch):
        self.model.train()
        step = epoch * len(self.train_dataloader)
        train_loss = 0
        for batch_idx, (batch_spectrograms, labels) in enumerate(
            tqdm(
                self.train_dataloader,
                desc=f"Train",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            )
        ):

            batch_spectrograms = batch_spectrograms.to(self.device)
            labels = labels.to(self.device)

            # calculate loss
            self._efficient_zero_grad(self.model)
            prediction = self.model(batch_spectrograms)
            loss = self.loss_fn(prediction, labels)
            train_loss += loss.item()

            step += 1

            # backpropagate error and update weights
            loss.backward()
            self.optimizer.step()

        train_loss = train_loss / len(self.train_dataloader)

        print(f"Loss: {train_loss:.4f}")
        return train_loss

    def _validate_single_epoch(self, epoch):
        self.model.eval()
        self.last_validation_loss = self.current_validation_loss
        self.current_validation_loss = 0
        display_values = []
        for batch_idx, (batch_spectrograms, labels) in enumerate(
            tqdm(
                self.validation_dataloader,
                desc=f"Validation",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            )
        ):
            batch_spectrograms = batch_spectrograms.to(self.device)
            labels = labels.to(self.device)

            prediction = self.model(batch_spectrograms)
            loss = self.loss_fn(prediction, labels)
            self.current_validation_loss += loss.item()

            labels = labels.to("cpu")
            prediction = prediction.to("cpu")
            for metric in self.metrics:
                self.metrics[metric](prediction, labels)

        use_reference = False
        if self.reference_metric in self.metrics:
            use_reference = True

        self.current_validation_loss = self.current_validation_loss / len(
            self.validation_dataloader
        )
        display_values.append(f"Loss: {self.current_validation_loss:.4f}")

        metrics_out = {}

        for idx, metric in enumerate(self.metrics):
            value = self.metrics[metric].compute()
            if idx == 0:
                ref_metric = value
            if use_reference:
                if metric == self.reference_metric:
                    ref_metric = value
            if metric == "ConfusionMatrix":
                cm_fig = plot_confusion_matrix(
                    value.numpy(),
                    class_names=self.validation_dataloader.dataset.classes,
                )
            else:
                display_values.append(f"{metric}: {value:.4f}")
                metrics_out[metric] = value.item()
            self.metrics[metric].reset()

        print("  ".join(display_values))

        return ref_metric, metrics_out

    def upload_results_to_dynamodb(self, timestamp, results, **kwargs):
        return super().upload_results_to_dynamodb(timestamp, results, **kwargs)

    def train_model(self, checkpoint_manager=None):
        super().train_model()
        timestamp = int(datetime.now().timestamp())
        results = {}
        for epoch in range(self.initial_epoch, self.epochs):
            print(f"Epoch {epoch+1}")
            loss = self._train_single_epoch(epoch)
            measure, metric = self._validate_single_epoch(epoch)

            self.lr_scheduler.step()

            metric["training_loss"] = loss

            results[epoch + 1] = metric

            if measure.cpu().detach().numpy() > self.best_measure:
                self.best_measure = measure.cpu().detach().numpy()

            # Save a checkpoint.
            if checkpoint_manager is not None:
                checkpoint_manager.save(
                    epoch, measure=math.floor(self.best_measure * 1000000)
                )

            print("---------------------------")

            # Early stopping
            if self.early_stop:
                if self.current_validation_loss > self.last_validation_loss:
                    self.trigger_times += 1
                    if self.trigger_times >= self.patience:
                        print("Early stopping!\n")
                        break
                else:
                    self.trigger_times = 0
        print("Finished training")

        if self.upload_results:
            print("Uploading results")
            self.upload_results_to_dynamodb(
                timestamp, results, early_stop=self.early_stop
            )


class TrainManagerMaML(TrainManager):
    def __init__(
        self,
        model,
        train_dataloader,
        validation_dataloader,
        epochs,
        device="cuda",
        upload_results=False,
        upload_info={},
    ) -> None:
        super().__init__(
            model,
            train_dataloader,
            validation_dataloader,
            epochs,
            device,
            upload_results,
            upload_info,
        )

    def _validate_model(self) -> list:
        accs_all_test = []
        for x_shot, x_qry, y_shot, y_qry in self.validation_dataloader:
            x_shot, y_shot, x_qry, y_qry = (
                x_shot.to(self.device),
                y_shot.to(self.device),
                x_qry.to(self.device),
                y_qry.to(self.device),
            )

            accs = self.model.finetunning(x_shot, y_shot, x_qry, y_qry)
            accs_all_test.append(accs)

        return accs_all_test

    def train_model(self):
        for epoch in range(self.epochs + 1):
            print(f"Epoch: {epoch + 1}")
            for step, data in enumerate(self.train_dataloader):
                x_shot, x_qry, y_shot, y_qry = data[0], data[1], data[2], data[3]

                x_shot, x_qry, y_shot, y_qry = (
                    x_shot.to(self.device),
                    x_qry.to(self.device),
                    y_shot.to(self.device),
                    y_qry.to(self.device),
                )

                # https://github.com/dragen1860/MAML-Pytorch/issues/41#issuecomment-600604345
                # accuracy before the first update, accuracy after the first update, accuracies for each update steps (set in args)
                accs = self.model(x_shot, y_shot, x_qry, y_qry)

                if step % 30 == 0:
                    print(f"step: {step}, \ttraining accuracy: {accs}")

                if step % 500 == 0:  # evaluation
                    accs_all_test = self._validate_model()

                    accs = np.array(accs_all_test)
                    print("test accuracy:", accs)
