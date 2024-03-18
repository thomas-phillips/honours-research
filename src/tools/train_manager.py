import math
from tqdm import tqdm

from tools.utils import plot_confusion_matrix


class TrainManager:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        lr_scheduler,
        train_dataloader,
        validation_dataloader,
        epochs,
        initial_epoch=0,
        metrics={},
        reference_metric="",
        writer=None,
        device="cpu",
        early_stop=True,
    ):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device
        self.metrics = metrics
        self.reference_metric = reference_metric
        self.epochs = epochs
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
            self.metrics[metric].reset()

        print("  ".join(display_values))

        return ref_metric

    def start_train(self, checkpoint_manager=None):
        for epoch in range(self.initial_epoch, self.epochs):
            print(f"Epoch {epoch+1}")
            loss = self._train_single_epoch(epoch)
            measure = self._validate_single_epoch(epoch)

            self.lr_scheduler.step()

            print(measure)
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
