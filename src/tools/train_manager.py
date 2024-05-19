from tqdm import tqdm
from tools.utils import plot_confusion_matrix, get_db_connection
from decimal import Decimal
from datetime import datetime
import json
import hashlib
import copy
import pprint
import csv
import psycopg2
import torch


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

    def train_model(self):
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

        pprint.pp(results)
        if self.upload_results:
            self.upload_results_to_db(results)

    def upload_results_to_db(self, results):
        conn = get_db_connection()
        cur = conn.cursor()

        try:
            print("Upload info:", self.upload_info)
            cur.execute(
                "INSERT INTO model (name, epochs, inclusion, exclusion, input_channels, preprocessing, type) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    self.upload_info["model_name"],
                    self.epochs,
                    self.upload_info["inclusion"],
                    self.upload_info["exclusion"],
                    self.upload_info["input_channels"],
                    self.upload_info["preprocessing"],
                    self.upload_info["type"],
                ),
            )

            cur.execute("SELECT LASTVAL()")
            model_id = cur.fetchone()

            print(model_id)

            for key, val in results.items():
                cur.execute(
                    "INSERT INTO epoch (epoch, model_id) VALUES (%s, %s)",
                    (key, model_id),
                )
                cur.execute("SELECT LASTVAL()")
                epoch_id = cur.fetchone()
                for k in val:
                    cur.execute(
                        "INSERT INTO cnn_result (value, type, epoch_id) VALUES (%s, %s, %s)",
                        (val[k], k, epoch_id),
                    )

            cur.execute(
                "INSERT INTO cnn_data (learning_rate, lr_schd_gamma, optimiser, early_stop, batch_size, shot, shuffle, model_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    self.upload_info["learning_rate"],
                    self.upload_info["lr_schd_gamma"],
                    self.upload_info["optimiser"],
                    self.early_stop,
                    self.upload_info["batch_size"],
                    self.upload_info.get("shot", None),
                    self.upload_info.get("shuffle", None),
                    model_id,
                ),
            )

            for c in self.upload_info["classes"]:
                cur.execute(
                    "INSERT INTO class (name, model_id) VALUES (%s, %s)",
                    (c, model_id),
                )

            conn.commit()

        except Exception as e:
            conn.rollback()
            print("Transaction rolled back: ", e)

        finally:
            cur.close()
            conn.close()


class TrainManagerSiamese(TrainManager):
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
        self.initial_epoch = initial_epoch
        return

    def _efficient_zero_grad(self, model):
        for param in model.parameters():
            param.grad = None

    def _train_single_epoch(self, epoch):
        self.model.train()
        step = epoch * len(self.train_dataloader)
        train_loss = 0

        for batch_idx, (first, second, targets) in enumerate(
            tqdm(
                self.train_dataloader,
                desc=f"Train",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            )
        ):
            first, second, targets = (
                first.to("cuda"),
                second.to("cuda"),
                targets.to("cuda"),
            )

            self._efficient_zero_grad(self.model)
            prediction = self.model(first, second).squeeze()
            loss = self.loss_fn(prediction, targets)
            train_loss += loss.item()

            step += 1

            loss.backward()
            self.optimizer.step()

        train_loss /= len(self.train_dataloader.dataset)
        print(f"Loss: {train_loss:.4f}")
        return train_loss

    def _validate_single_epoch(self, epoch):
        self.model.eval()
        validation_loss = 0
        correct = 0

        for batch_idx, (first, second, targets) in enumerate(
            tqdm(
                self.validation_dataloader,
                desc=f"Validation",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            )
        ):
            first, second, targets = (
                first.to("cuda"),
                second.to("cuda"),
                targets.to("cuda"),
            )

            self._efficient_zero_grad(self.model)
            prediction = self.model(first, second).squeeze()
            loss = self.loss_fn(prediction, targets)
            validation_loss += loss.item()

            pred = torch.where(prediction > 0.5, 1, 0)
            correct += pred.eq(targets.view_as(pred)).sum().item()

            loss.backward()
            self.optimizer.step()

        validation_loss /= len(self.validation_dataloader.dataset)
        accuracy = correct / len(self.validation_dataloader.dataset)
        print("Correct", correct)

        print(f"Validation Loss: {validation_loss:.4f}, Accuracy: {accuracy}")
        return validation_loss, accuracy

    def train_model(self):
        super().train_model()
        timestamp = int(datetime.now().timestamp())
        results = {}

        for epoch in range(self.initial_epoch, self.epochs):
            print(f"Epoch {epoch + 1}")
            loss = self._train_single_epoch(epoch)
            validation_loss, accuracy = self._validate_single_epoch(epoch)

            self.lr_scheduler.step()

            results[epoch + 1] = {"training_loss": loss, "accuracy": accuracy}

            print("---------------------------")

        print("Finished training")

        pprint.pp(results)
        if self.upload_results:
            self.upload_results_to_db(results)

    def upload_results_to_db(self, results):
        conn = get_db_connection()
        cur = conn.cursor()

        try:
            print("Upload info:", self.upload_info)
            cur.execute(
                "INSERT INTO model (name, epochs, inclusion, exclusion, input_channels, preprocessing, type) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    self.upload_info["model_name"],
                    self.epochs,
                    self.upload_info["inclusion"],
                    self.upload_info["exclusion"],
                    self.upload_info["input_channels"],
                    self.upload_info["preprocessing"],
                    self.upload_info["type"],
                ),
            )

            cur.execute("SELECT LASTVAL()")
            model_id = cur.fetchone()

            print(model_id)

            for key, val in results.items():
                cur.execute(
                    "INSERT INTO epoch (epoch, model_id) VALUES (%s, %s)",
                    (key, model_id),
                )
                cur.execute("SELECT LASTVAL()")
                epoch_id = cur.fetchone()
                for k in val:
                    cur.execute(
                        "INSERT INTO cnn_result (value, type, epoch_id) VALUES (%s, %s, %s)",
                        (val[k], k, epoch_id),
                    )

            cur.execute(
                "INSERT INTO cnn_data (learning_rate, lr_schd_gamma, optimiser, early_stop, batch_size, shot, shuffle, model_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    self.upload_info["learning_rate"],
                    self.upload_info["lr_schd_gamma"],
                    self.upload_info["optimiser"],
                    0,
                    self.upload_info["batch_size"],
                    self.upload_info.get("shot", None),
                    self.upload_info.get("shuffle", None),
                    model_id,
                ),
            )

            conn.commit()

        except Exception as e:
            conn.rollback()
            print("Transaction rolled back: ", e)

        finally:
            cur.close()
            conn.close()


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
            # print(x_shot.shape, x_qry.shape, y_shot.shape, y_qry.shape)
            x_shot, y_shot, x_qry, y_qry = (
                x_shot.to(self.device),
                y_shot.to(self.device),
                x_qry.to(self.device),
                y_qry.to(self.device),
            )

            accs = self.model.finetunning(x_shot, y_shot, x_qry, y_qry)
            accs_dict = {}
            for index, acc in enumerate(accs.tolist()):
                if index == 0:
                    accs_dict[0] = acc
                else:
                    accs_dict[index] = acc

            accs_all_test.append(accs_dict)

        return accs_all_test

    def train_model(self):
        super().train_model()
        timestamp = int(datetime.now().timestamp())
        results = {}
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch + 1}")
            results[epoch + 1] = []

            for step, data in enumerate(self.train_dataloader):
                x_shot, x_qry, y_shot, y_qry = data[0], data[1], data[2], data[3]

                x_shot, x_qry, y_shot, y_qry = (
                    x_shot.to(self.device),
                    x_qry.to(self.device),
                    y_shot.to(self.device),
                    y_qry.to(self.device),
                )

                print(f"Step: {step}")
                # https://github.com/dragen1860/MAML-Pytorch/issues/41#issuecomment-600604345
                # accuracy before the first update, accuracy after the first update, accuracies for each update steps (set in args)
                accs = self.model(x_shot, y_shot, x_qry, y_qry)

                accs_dict = {}
                for index, acc in enumerate(accs.tolist()):
                    accs_dict[index] = acc

                # results[epoch + 1].append({step: accs.tolist(), "type": "training"})
                results[epoch + 1].append(
                    {"accuracies": accs_dict, "type": "training", "step": step}
                )

                if step % 30 == 0:
                    print(f"step: {step}, \ttraining accuracy: {accs}")

                if step == 99:  # evaluation
                    accs_all_test = self._validate_model()

                    results[epoch + 1].append(
                        {
                            "accuracies": accs_all_test,
                            "type": "evaluation",
                            "step": step,
                        }
                    )
                    print("test accuracy:", accs_all_test[0])
                # print("test accuracy:", accs)

        pprint.pp(results)
        if self.upload_results:
            self.upload_results_to_db(results)

    def upload_results_to_db(self, results):
        conn = get_db_connection()
        cur = conn.cursor()

        try:
            print("Upload info:", self.upload_info)
            cur.execute(
                "INSERT INTO model (name, epochs, inclusion, exclusion, input_channels, preprocessing, type) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    self.upload_info["model"]["name"],
                    self.epochs,
                    self.upload_info["inclusion"],
                    self.upload_info["exclusion"],
                    self.upload_info["input_channels"],
                    self.upload_info["preprocessing"],
                    self.upload_info["type"],
                ),
            )

            cur.execute("SELECT LASTVAL()")
            model_id = cur.fetchone()

            print(model_id)

            for key, val in results.items():
                cur.execute(
                    "INSERT INTO epoch (epoch, model_id) VALUES (%s, %s)",
                    (key, model_id),
                )
                cur.execute("SELECT LASTVAL()")
                epoch_id = cur.fetchone()
                for s in val:
                    cur.execute(
                        "INSERT INTO maml_step (step, type, epoch_id) VALUES (%s, %s, %s)",
                        (s["step"], s["type"], epoch_id),
                    )
                    cur.execute("SELECT LASTVAL()")
                    step_id = cur.fetchone()

                    if s["type"] == "evaluation":
                        for eval_acc in s["accuracies"]:
                            for i, acc in eval_acc.items():
                                cur.execute(
                                    "INSERT INTO maml_update_acc (update, accuracy, maml_step_id) VALUES (%s, %s, %s)",
                                    (i, acc, step_id),
                                )
                    else:
                        for i, acc in s["accuracies"].items():
                            cur.execute(
                                "INSERT INTO maml_update_acc (update, accuracy, maml_step_id) VALUES (%s, %s, %s)",
                                (i, acc, step_id),
                            )

            cur.execute(
                "INSERT INTO maml_data (update_lr, meta_lr, n_way, k_spt, k_qry, task_num, update_step, update_step_test, model_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    self.upload_info["update_learning_rate"],
                    self.upload_info["learning_rate"],
                    5,
                    self.upload_info["n_shot"],
                    self.upload_info["n_query"],
                    self.upload_info["task_num"],
                    self.upload_info["update_step"],
                    self.upload_info["update_step_test"],
                    model_id,
                ),
            )

            for c in self.upload_info["classes"]:
                cur.execute(
                    "INSERT INTO class (name, model_id) VALUES (%s, %s)",
                    (c, model_id),
                )

            conn.commit()

        except Exception as e:
            conn.rollback()
            print("Transaction rolled back: ", e)

        finally:
            cur.close()
            conn.close()
