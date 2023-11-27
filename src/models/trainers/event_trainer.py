import copy
from typing import List

import numpy as np
import torch
import wandb
from numpy import ndarray
from timm.scheduler import CosineLRScheduler
from torch import nn, log_softmax, softmax
from tqdm import tqdm

from ... import data_info
from ...logger.logger import logger


def masked_loss(criterion, outputs, y):
    labels = y[:, :, :3]

    unlabeled_mask = y[:, :, 3]
    unlabeled_mask = 1 - unlabeled_mask
    unlabeled_mask = unlabeled_mask.unsqueeze(1).repeat(1, 1, 3)

    loss_unreduced = criterion(outputs, labels)

    loss_masked = loss_unreduced * unlabeled_mask

    loss = torch.sum(loss_masked) / torch.sum(unlabeled_mask)
    return loss


class EventTrainer:
    """
    Trainer class for the models that predict events.
    :param epochs: The number of epochs to train for.
    :param criterion: The loss function to use.
    :param maskUnlabeled: Whether to mask the unlabeled data or not. (If true shape should be (batch_size, 4, seq_len))
    """

    def __init__(
            self,
            epochs: int = 10,
            criterion: nn.Module = nn.CrossEntropyLoss(),
            mask_unlabeled: bool = False,
            early_stopping: int = 10
    ) -> None:
        self.criterion = criterion
        self.mask_unlabeled = mask_unlabeled
        self.dataset = None
        self.train_data = None
        self.test_data = None
        self.n_epochs = epochs
        self.early_stopping = early_stopping
        cuda_dev = "0"
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")

    def fit(
            self,
            trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            name: str,
            scheduler: CosineLRScheduler = None,
            activation_delay: int = None

    ) -> tuple[ndarray[float], ndarray[float], int]:
        """
        Train the model on the training set and validate on the test set.
        :param trainloader: The training set.
        :param testloader: The test set.
        :param model: The model to train.
        :param optimizer: The optimizer to use.
        :param name: The name of the model.
        :param scheduler: The optional LR scheduler to use.
        :param activation_delay: The optional activation delay to use.
        """

        # Setup model for training
        model = model.to(self.device)
        model.float()

        # Wandb logging
        if wandb.run is not None:
            wandb.define_metric("epoch")
            wandb.define_metric(
                f"{data_info.substage} - Train {str(self.criterion)} of {name}", step_metric="epoch")
            wandb.define_metric(
                f"{data_info.substage} - Validation {str(self.criterion)} of {name}", step_metric="epoch")

        # Check if full training or not
        full_train = False
        if testloader is None:
            full_train = True

        # Train and validate
        avg_train_losses = []
        avg_val_losses = []
        lowest_val_loss = np.inf
        best_model = model.state_dict()
        counter = 0
        max_counter = self.early_stopping
        trained_epochs = 0
        for epoch in range(self.n_epochs):
            if activation_delay is None:
                use_activation = None
            else:
                use_activation = epoch >= activation_delay
            # Training loss
            train_losses = self.train_one_epoch(
                dataloader=trainloader, epoch_no=epoch, optimizer=optimizer, model=model, scheduler=scheduler, use_activation=use_activation)
            train_loss = sum(train_losses) / (len(train_losses) + 1)
            avg_train_losses.append(train_loss)

            # Validation
            if not full_train:
                val_losses = self.val_loss(
                    testloader, epoch, model, use_activation=use_activation)
                val_loss = sum(val_losses) / (len(val_losses) + 1)
                avg_val_losses.append(val_loss)

            if wandb.run is not None:
                if not full_train:
                    wandb.log({f"{data_info.substage} - Train {str(self.criterion)} of {name}": train_loss,
                               f"{data_info.substage} - Validation {str(self.criterion)} of {name}": val_loss, "epoch": epoch})
                else:
                    wandb.log(
                        {f"{data_info.substage} - Train {str(self.criterion)} of {name}": train_loss, "epoch": epoch})

            # Save model if validation loss is lower than previous lowest validation loss
            if not full_train and val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                counter = 0
            elif not full_train:
                counter += 1
                if counter >= max_counter:
                    model.load_state_dict(best_model)
                    trained_epochs = (epoch - counter + 1)
                    logger.info(
                        f"--- Early stopping achieved at {epoch} ---, loading model from epoch {trained_epochs}")
                    break

            trained_epochs = epoch + 1

        # Load best model
        if not full_train:
            model.load_state_dict(best_model)

        return avg_train_losses, avg_val_losses, trained_epochs

    def train_one_epoch(
            self, dataloader: torch.utils.data.DataLoader,
            epoch_no: int,
            optimizer: torch.optim.Optimizer,
            model: nn.Module,

            disable_tqdm=False,
            scheduler: CosineLRScheduler = None,
            use_activation: bool = None

    ) -> ndarray[float]:
        """
        Train the model on the training set for one epoch and return training losses
        :param dataloader: The training set.
        :param epoch_no: The epoch number.
        :param optimizer: The optimizer to use.
        :param model: The model to train.
        :param disable_tqdm: Whether to disable tqdm or not.
        :param scheduler: The optional LR scheduler to use.
        :param use_activation: The optional activation delay to use.
        """

        # Loop through batches and return losses
        losses = []

        # Set model to train mode
        model.train()

        # Step the scheduler
        if scheduler is not None:
            scheduler.step(epoch_no)

        with tqdm(dataloader, unit="batch", disable=disable_tqdm) as tepoch:
            for _, data in enumerate(tepoch):
                losses = self._train_one_loop(
                    data=data, losses=losses, model=model, optimizer=optimizer, use_activation=use_activation)
                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=sum(losses) / (len(losses) + 1))
        return losses

    def _train_one_loop(self, data: torch.utils.data.DataLoader, losses: List[float], model: nn.Module, optimizer: torch.optim.Optimizer,
                        use_activation: bool = None) -> List[float]:
        """
        Train the model on one batch and return the loss.
        :param data: The batch to train on.
        :param losses: The list of losses.
        :param model: The model to train.
        :param optimizer: The optimizer to use.
        :param use_activation: The optional activation delay to use.
        :return: The updated list of losses.
        """

        # Retrieve target and output
        data[0] = data[0].to(self.device).float()
        data[1] = data[1].to(self.device).float()

        # Set gradients to zero
        optimizer.zero_grad()

        # Forward pass with model and optional activation delay
        if use_activation is not None:
            # If it is an GRU Model, ignore the second output
            if str(model).startswith("MultiResidualBiGRU"):
                output, _ = model(data[0].to(self.device),
                                  use_activation=use_activation)
            else:
                output = model(data[0].to(self.device),
                               use_activation=use_activation)
        else:
            if str(model).startswith("MultiResidualBiGRU"):
                output, _ = model(data[0].to(self.device))
            else:
                output = model(data[0].to(self.device))

        # Assert output is in correct format
        assert output.shape[1] == data[1].shape[
            1], f"Output shape {tuple(output.shape)} is not equal to target shape {tuple(data[1].shape)}"
        assert output.shape[1] == data_info.window_size, \
            f"Output shape {tuple(output.shape[1])} is not equal to window size {data_info.window_size}, check if model output is correct"
        # TODO: Describe magic number 2
        assert output.shape[2] == 2, f"Output shape {tuple(output.shape)} does not have 2 classes"

        # Calculate loss
        if self.mask_unlabeled:
            assert data[1].shape[
                2] == 3, f"Masked loss only works with y shape (batch_size, seq_len, 3), but shape is {tuple(data[1].shape)}"
            loss = self.masked_loss(output, data[1])
        else:
            # TODO: Describe magic number 2
            assert data[1].shape[2] == 2, f"Data shape {tuple(data[1].shape[2])} does not have 2 classes"
            if str(self.criterion) == "KLDivLoss()":
                loss = self.criterion(log_softmax(
                    output, dim=1), softmax(data[1], dim=1))
            else:
                loss = self.criterion(output, data[1])

        # Backpropagate loss and update weights with gradient clipping
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-1)
        optimizer.step()

        # Append loss to list
        losses.append(loss.item())

        return losses

    def val_loss(
            self,
            dataloader: torch.utils.data.DataLoader,
            epoch_no: int, model: nn.Module,
            disable_tqdm: bool = False,
            use_activation: bool = True
    ) -> List[float]:
        """
        Run the model on the test set and return validation loss
        :param dataloader: The test set.
        :param epoch_no: The epoch number.
        :param model: The model to train.
        :param disable_tqdm: Whether to disable tqdm or not.
        :return: The validation loss.
        """

        # Loop through batches and return losses
        losses = []

        # Set model to eval mode
        model.eval()

        with tqdm(dataloader, unit="batch", disable=disable_tqdm) as tepoch:
            for _, data in enumerate(tepoch):
                losses = self._val_one_loop(
                    data=data, losses=losses, model=model, use_activation=use_activation)
                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=sum(losses) /
                                   (len(losses) + 0.0000001))
        return losses

    def _val_one_loop(
            self,
            data: torch.utils.data.DataLoader,
            losses: List[float],
            model: nn.Module,
            use_activation: bool = True
    ) -> List[float]:
        """
        Validate the model on one batch and return the loss.
        :param data: The batch to validate on.
        :param losses: The list of losses.
        :param model: The model to validate.
        :return: The updated list of losses.
        """

        # Use torch.no_grad() to disable gradient calculation and calculate loss
        with torch.no_grad():
            # Retrieve target and output
            data[0] = data[0].to(self.device).float()
            data[1] = data[1].to(self.device).float()

            # Forward pass with model
            if use_activation is not None:
                # If it is an GRU Model, ignore the second output
                if str(model).startswith("MultiResidualBiGRU"):
                    output, _ = model(data[0].to(self.device),
                                      use_activation=use_activation)
                else:
                    output = model(data[0].to(self.device),
                                   use_activation=use_activation)
            else:
                if str(model).startswith("MultiResidualBiGRU"):
                    output, _ = model(data[0].to(self.device))
                else:
                    output = model(data[0].to(self.device))

            # Assert output is in correct format
            assert output.shape[1] == data[1].shape[
                1], f"Output shape {tuple(output.shape)} is not equal to target shape {tuple(data[1].shape)}"
            assert output.shape[1] == data_info.window_size, \
                f"Output shape {tuple(output.shape[1])} is not equal to window size {data_info.window_size}, check if model output is correct"
            # TODO: Describe magic number 2
            assert output.shape[2] == 2, f"Output shape {tuple(output.shape)} does not have 2 classes"

            # Calculate loss
            if self.mask_unlabeled:
                loss = self.masked_loss(output, data[1])
            else:
                if str(self.criterion) == "KLDivLoss()":
                    loss = self.criterion(log_softmax(
                        output, dim=1), softmax(data[1], dim=1))
                else:
                    loss = self.criterion(output, data[1])
            losses.append(loss.item())
        return losses

    def masked_loss(self, outputs, y):
        assert y.shape[2] == 3, "Masked loss only works with y shape (batch_size, seq_len, 3)"
        assert y.shape[1] == data_info.window_size, "Output shape is not equal to window size, check if targets is correct"
        assert outputs.shape[1] == data_info.window_size, "Output shape is not equal to window size, check if model output is correct"
        assert outputs.shape[0] == y.shape[
            0], "Output shape is not equal to target shape (0)"
        assert outputs.shape[2] == 2, "Output shape is not equal to 2 (2 classes)"

        # Get the event labels
        labels = y[:, :, 1:]
        assert labels.shape[1] == data_info.window_size, "Output shape is not equal to window size, check if targets is correct"
        assert labels.shape[2] == 2, "Output shape is not equal to 2 (2 classes)"

        # Get the mask from y (shape (batch_size, seq_len, 2))
        unlabeled_mask = torch.stack([y[:, :, 0], y[:, :, 0]], dim=2)
        assert unlabeled_mask.shape == labels.shape, "Unlabeled mask shape is not equal to labels shape"

        # If the mask is 1, keep data, else set to 0
        # Do this if value is 3 (unlabeled), else set to 1
        unlabeled_mask = unlabeled_mask == 3
        # Set true to 0 and false to 1
        unlabeled_mask = unlabeled_mask ^ 1

        if str(self.criterion) == "KLDivLoss()":
            # Outputs should be given the label when the mask is 0
            outputs = outputs * unlabeled_mask + labels * (1 - unlabeled_mask)

            loss_unreduced = self.criterion(log_softmax(
                outputs, dim=1), softmax(labels, dim=1))
            loss_unreduced = loss_unreduced.sum() / loss_unreduced.shape[0]
            return loss_unreduced
        else:
            loss_unreduced = self.criterion(outputs, labels)

        loss_masked = loss_unreduced * unlabeled_mask
        loss = torch.sum(loss_masked) / (torch.sum(unlabeled_mask) + 1)
        return loss
