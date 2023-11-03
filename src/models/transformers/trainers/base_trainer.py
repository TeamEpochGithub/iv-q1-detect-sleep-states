import numpy as np
from torch import nn
import torch
from tqdm import tqdm
from typing import List
import wandb

from src import data_info


class Trainer:
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

    def fit(self, dataloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader, model: nn.Module, optimiser: torch.optim.Optimizer, name: str):
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
        max_counter = 10
        trained_epochs = 0
        for epoch in range(self.n_epochs):

            # Training loss
            train_losses = self.train_one_epoch(
                dataloader=dataloader, epoch_no=epoch, optimiser=optimiser, model=model)
            if len(train_losses) > 0:
                train_loss = sum(train_losses) / len(train_losses)
                avg_train_losses.append(train_loss.cpu())

            # Validation
            if not full_train:
                val_losses = self.val_loss(testloader, epoch, model)
                if len(val_losses) > 0:
                    val_loss = sum(val_losses) / len(val_losses)
                    avg_val_losses.append(val_loss.cpu())

            if wandb.run is not None and not full_train:
                wandb.log({f"{data_info.substage} - Train {str(self.criterion)} of {name}": train_loss,
                           f"{data_info.substage} - Validation {str(self.criterion)} of {name}": val_loss, "epoch": epoch})

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

    def train_one_epoch(self, dataloader, epoch_no, optimiser, model, disable_tqdm=False) -> List[float]:
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
        with tqdm(dataloader, unit="batch", disable=disable_tqdm) as tepoch:
            for _, data in enumerate(tepoch):
                losses = self._train_one_loop(
                    data=data, losses=losses, model=model, optimiser=optimiser)
                tepoch.set_description(f"Epoch {epoch_no}")
                if len(losses) > 0:
                    tepoch.set_postfix(loss=sum(losses) / len(losses))
                else:
                    tepoch.set_postfix(loss=-1)
        return losses

    def _train_one_loop(
        self, data: torch.utils.data.DataLoader, losses: List[float], model: nn.Module, optimiser: torch.optim.Optimizer
    ) -> List[float]:
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
        data[0] = data[0].float()
        output = model(data[0].to(self.device))

        # Create mask to ignore nan values
        mask = torch.ones_like(data[1]).to(self.device)
        mask[:, 0] = (1 - data[1][:, 2])
        mask[:, 1] = (1 - data[1][:, 3])

        # Calculate loss
        loss = self.criterion(output, data[1].type(
            torch.FloatTensor).to(self.device), mask)

        # Backpropagate loss if not nan
        if not np.isnan(loss.item()):
            loss.backward()
            optimiser.step()
            losses.append(loss.detach())

        return losses

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
                if len(losses) > 0:
                    tepoch.set_postfix(loss=sum(losses) / len(losses))
                else:
                    tepoch.set_postfix(loss=-1)
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
            data[0] = data[0].float()
            output = model(data[0].to(self.device))

            # Create mask to ignore nan values
            mask = torch.ones_like(data[1]).to(self.device)
            mask[:, 0] = (1 - data[1][:, 2])
            mask[:, 1] = (1 - data[1][:, 3])

            loss = self.criterion(output, data[1].type(
                torch.FloatTensor).to(self.device), mask)
            if not np.isnan(loss.item()):
                losses.append(loss.detach())
        return losses
