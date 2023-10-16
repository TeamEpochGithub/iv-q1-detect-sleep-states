import numpy as np
from torch import nn
import torch
from tqdm import tqdm
from typing import List, Tuple
import wandb
from src.logger.logger import logger


class StackedTrainer:
    """Trainer class for the transformer model.

    Args:
        model: The model to train.
        batch_size: The batch size.
        lr: The learning rate.
        betas: The betas for the Adam optimiser.
        eps: The epsilon for the Adam optimiser.
        epochs: The number of epochs to train for.
    """

    def __init__(
        self,
        epochs: int = 10,
        criterion: nn.Module = nn.CrossEntropyLoss()
    ):
        self.criterion = criterion

        self.dataset = None
        self.train_data = None
        self.test_data = None
        self.n_epochs = epochs
        cuda_dev = "0"
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")

    def fit(
        self, dataloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader, model: nn.Module, optimiser: torch.optim.Optimizer, name: str
    ) -> Tuple[List[float], List[float]]:
        """
        Train the model on the training set and validate on the test set.

        Args:
            dataloader: The training set.
            testloader: The test set.
            model: The model to train.
            optimiser: The optimiser to use.
            name: The name of the model.
        """

        # Setup model for training
        model = model.to(self.device)
        model.train()
        model.double()

        # Wandb logging
        if wandb.run is not None:
            wandb.define_metric("epoch")
            wandb.define_metric(
                f"Train {str(self.criterion)} of {name}", step_metric="epoch")
            wandb.define_metric(
                f"Validation {str(self.criterion)} of {name}", step_metric="epoch")

        # Train the model
        avg_train_losses = []
        avg_val_losses = []
        for epoch in range(self.n_epochs):
            val_losses = []
            train_losses = self.train_one_epoch(
                dataloader=dataloader, epoch_no=epoch, optimiser=optimiser, model=model)
            val_losses = self.val_loss(testloader, epoch, model)
            train_loss = sum(train_losses) / len(train_losses)
            val_loss = sum(val_losses) / len(val_losses)
            logger.debug(f"Epoch {epoch} train loss: {train_loss}")
            logger.debug(f"Epoch {epoch} val loss: {val_loss}")
            avg_train_losses.append(train_loss.cpu())
            avg_val_losses.append(val_loss.cpu())
            if wandb.run is not None:
                wandb.log({f"Train {str(self.criterion)} of {name}": train_loss,
                           f"Validation {str(self.criterion)} of {name}": val_loss, "epoch": epoch})

        return avg_train_losses, avg_val_losses

    def train_one_epoch(self, dataloader: torch.utils.data.DataLoader, epoch_no: int, optimiser: torch.optim.Optimizer, model: nn.Module, disable_tqdm: bool = False):
        """
        Train the model on the training set for one epoch and return training losses

        Args:
            dataloader: The training set.
            epoch_no: The epoch number.
            optimiser: The optimiser to use.
            model: The model to train.
            disable_tqdm: Disable tqdm.
        """

        # Loop through batches and return losses
        losses = []
        i = 0
        with tqdm(dataloader, unit="batch", disable=disable_tqdm) as tepoch:
            for idx, data in enumerate(tepoch):
                loss, losses = self._train_one_loop(
                    data=data, losses=losses, model=model, optimiser=optimiser)
                if loss.detach() > 0:
                    i += 1
                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=sum(losses).item() / i)
        return losses

    def _train_one_loop(
        self, data: torch.utils.data.DataLoader, losses: List[float], model: nn.Module, optimiser: torch.optim.Optimizer
    ) -> Tuple[float, List[float]]:
        """
        Train the model on one batch and return the loss.

        Args:
            data: The batch to train on.
            losses: The list of losses.
            model: The model to train.
            optimiser: The optimiser to use.
        """

        # Retrieve target and output
        optimiser.zero_grad()
        data[0] = data[0].double()
        padding_mask = torch.ones((data[0].shape[0], data[0].shape[1])) > 0
        output = model(data[0].to(self.device), padding_mask.to(self.device))

        # Create mask to ignore nan values
        mask = torch.ones_like(data[1]).to(self.device)
        mask[:, 0] = (1 - data[1][:, 2])
        mask[:, 1] = (1 - data[1][:, 3])

        # Calculate loss
        loss = self.criterion(output, data[1].type(
            torch.DoubleTensor).to(self.device), mask)

        # Backpropagate loss if not nan
        if not np.isnan(loss.item()):
            loss.backward()
            optimiser.step()
            losses.append(loss.detach())

        return loss.detach(), losses

    def val_loss(self, dataloader: torch.utils.data.DataLoader, epoch_no, model: nn.Module, disable_tqdm=False):
        """
        Run the model on the test set and return validation loss

        Args:
            dataloader: The test set.
            epoch_no: The epoch number.
            model: The model to train.
            disable_tqdm: Disable tqdm.
        """

        # Loop through batches and return losses
        losses = []
        with tqdm(dataloader, unit="batch", disable=disable_tqdm) as tepoch:
            for idx, data in enumerate(tepoch):
                losses = self._val_one_loop(
                    data=data, losses=losses, model=model)
                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=sum(losses) / len(losses))
        return losses

    def _val_one_loop(self, data: torch.utils.data.DataLoader, losses: List[float], model: nn.Module):
        """
        Validate the model on one batch and return the loss.

        Args:
            data: The batch to train on.
            losses: The list of losses.
            model: The model to train.
        """

        # Use torch.no_grad() to disable gradient calculation and calculate loss
        with torch.no_grad():
            data[0] = data[0].double()
            padding_mask = torch.ones((data[0].shape[0], data[0].shape[1])) > 0
            output = model(data[0].to(self.device),
                           padding_mask.to(self.device))

            mask = torch.ones_like(data[1]).to(self.device)
            mask[:, 0] = (1 - data[1][:, 2])
            mask[:, 1] = (1 - data[1][:, 3])

            loss = self.criterion(output, data[1].type(
                torch.DoubleTensor).to(self.device), mask)
            if not np.isnan(loss.item()):
                losses.append(loss.detach())
        return losses