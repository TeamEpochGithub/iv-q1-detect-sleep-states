from typing import List

import numpy as np
import torch
import wandb
from numpy import ndarray
from torch import nn, log_softmax, softmax
from tqdm import tqdm

from ... import data_info


def masked_loss(criterion, outputs, y):
    assert y.shape[1] > 1, "Masked loss only works with shape (batch_size, 2 | 3 depending on both awake and onset, seq_len)"

    if y.shape[1] == data_info.window_size:
        y = y.permute(0, 2, 1)
    labels = y[:, 1:, :]
    labels = labels.squeeze()

    # Get the mask from y (shape (batch_size, 2, seq_len)) if y.shape[1] == 3 else (batch_size, 1, seq_len)
    if y.shape[1] == 3:
        # Mask is should be two times y[:,0,:] so shape is (batch_size, 2, seq_len)
        unlabeled_mask = torch.stack([y[:, 0, :], y[:, 0, :]], dim=1)
    else:
        # Mask is should be one time y[:,0,:] so shape is (batch_size, 1, seq_len)
        unlabeled_mask = y[:, 0, :]

    # If the mask is 1, keep data, else set to 0
    # Do this if value is 3 (unlabeled), else set to 1
    unlabeled_mask = unlabeled_mask == 3
    # Set true to 0 and false to 1
    unlabeled_mask = unlabeled_mask ^ 1

    if str(criterion) == "KLDivLoss()":
        loss_unreduced = criterion(log_softmax(outputs, dim=1), softmax(labels, dim=1))
    else:
        loss_unreduced = criterion(outputs, labels)

    loss_masked = loss_unreduced * unlabeled_mask

    loss = torch.sum(loss_masked) / (torch.sum(unlabeled_mask) + 1)
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
            name: str
    ) -> tuple[ndarray[float], ndarray[float], int]:
        """
        Train the model on the training set and validate on the test set.
        :param trainloader: The training set.
        :param testloader: The test set.
        :param model: The model to train.
        :param optimizer: The optimizer to use.
        :param name: The name of the model.
        """

        # Setup model for training
        model = model.to(self.device)
        model.train()
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

            # Training loss
            train_losses = self.train_one_epoch(
                dataloader=trainloader, epoch_no=epoch, optimizer=optimizer, model=model)
            train_loss = sum(train_losses) / (len(train_losses) + 1)
            avg_train_losses.append(train_loss.cpu())

            # Validation
            if not full_train:
                val_losses = self.val_loss(testloader, epoch, model)
                val_loss = sum(val_losses) / (len(val_losses) + 1)
                avg_val_losses.append(val_loss.cpu())

            if wandb.run is not None:
                if not full_train:
                    wandb.log({f"{data_info.substage} - Train {str(self.criterion)} of {name}": train_loss,
                               f"{data_info.substage} - Validation {str(self.criterion)} of {name}": val_loss, "epoch": epoch})
                else:
                    wandb.log({f"{data_info.substage} - Train {str(self.criterion)} of {name}": train_loss, "epoch": epoch})

            # Save model if validation loss is lower than previous lowest validation loss
            if not full_train and val_loss < lowest_val_loss:
                lowest_val_loss = val_loss
                best_model = model.state_dict()
                counter = 0
            elif not full_train:
                counter += 1
                if counter >= max_counter:
                    model.load_state_dict(best_model)
                    trained_epochs = (epoch - counter + 1)
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
            disable_tqdm=False
    ) -> ndarray[float]:
        """
        Train the model on the training set for one epoch and return training losses
        :param dataloader: The training set.
        :param epoch_no: The epoch number.
        :param optimizer: The optimizer to use.
        :param model: The model to train.
        """

        # Loop through batches and return losses
        losses = []
        with tqdm(dataloader, unit="batch", disable=disable_tqdm) as tepoch:
            for _, data in enumerate(tepoch):
                losses = self._train_one_loop(
                    data=data, losses=losses, model=model, optimizer=optimizer)
                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=sum(losses) / (len(losses) + 1))
        return losses

    def _train_one_loop(
            self, data: torch.utils.data.DataLoader, losses: List[float], model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> List[float]:
        """
        Train the model on one batch and return the loss.
        :param data: The batch to train on.
        :param losses: The list of losses.
        :param model: The model to train.
        :param optimizer: The optimizer to use.
        :return: The updated list of losses.
        """

        # Retrieve target and output
        optimizer.zero_grad()
        data[0] = data[0].to(self.device).float()
        data[1] = data[1].to(self.device).float()
        output = model(data[0].to(self.device))
        output = output.squeeze()

        # Calculate loss
        if self.mask_unlabeled:
            loss = masked_loss(self.criterion, output, data[1])
        else:
            if str(self.criterion) == "KLDivLoss()":
                loss = self.criterion(log_softmax(output, dim=1), softmax(data[1], dim=1))
            else:
                loss = self.criterion(output, data[1])

        # Backpropagate loss and update weights
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        return losses

    def val_loss(
            self,
            dataloader: torch.utils.data.DataLoader,
            epoch_no: int, model: nn.Module,
            disable_tqdm: bool = False
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
        with tqdm(dataloader, unit="batch", disable=disable_tqdm) as tepoch:
            for _, data in enumerate(tepoch):
                losses = self._val_one_loop(
                    data=data, losses=losses, model=model)
                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=sum(losses) / (len(losses) + 0.0000001))
        return losses

    def _val_one_loop(
            self,
            data: torch.utils.data.DataLoader,
            losses: List[float],
            model: nn.Module
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
            output = model(data[0].to(self.device))
            output = output.squeeze()

            # Calculate loss
            if self.mask_unlabeled:
                loss = masked_loss(self.criterion, output, data[1])
            else:
                if str(self.criterion) == "KLDivLoss()":
                    loss = self.criterion(log_softmax(output, dim=1), softmax(data[1], dim=1))
                else:
                    loss = self.criterion(output, data[1])
            losses.append(loss.detach())
        return losses
