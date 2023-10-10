from torch import nn
import torch
from tqdm import tqdm
from typing import List, Tuple
import wandb


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

    def fit(self, dataloader: torch.utils.data.DataLoader, testloader, model: nn.Module, optimiser: torch.optim.Optimizer, name: str):
        model = model.to(self.device)
        model.train()
        model.double()

        # Wandb logging
        wandb.define_metric("epoch")
        wandb.define_metric(
            f"Train {str(self.criterion)} of {name}", step_metric="epoch")
        wandb.define_metric(
            f"Validation {str(self.criterion)} of {name}", step_metric="epoch")

        avg_train_losses = []
        avg_val_losses = []
        for epoch in range(self.n_epochs):
            val_losses = []
            train_losses = self.train_one_epoch(
                dataloader=dataloader, epoch_no=epoch, optimiser=optimiser, model=model)
            val_losses = self.val_loss(testloader, epoch, model)
            train_loss = sum(train_losses) / len(train_losses)
            val_loss = sum(val_losses) / len(val_losses)
            avg_train_losses.append(train_loss.cpu())
            avg_val_losses.append(val_loss.cpu())
            wandb.log({f"Train {str(self.criterion)} of {name}": train_loss,
                      f"Validation {str(self.criterion)} of {name}": val_loss, "epoch": epoch})

        return avg_train_losses, avg_val_losses

    def train_one_epoch(self, dataloader, epoch_no, optimiser, model, disable_tqdm=False):
        losses = []
        epoch_loss = 0
        i = 0
        with tqdm(dataloader, unit="batch", disable=disable_tqdm) as tepoch:
            for idx, data in enumerate(tepoch):
                i += 1
                loss, losses = self._train_one_loop(
                    data=data, losses=losses, model=model, optimiser=optimiser)
                epoch_loss += loss.detach()
                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=epoch_loss.item() / i)
        return losses

    def _train_one_loop(
        self, data: torch.utils.data.DataLoader, losses: List[float], model: nn.Module, optimiser: torch.optim.Optimizer
    ) -> Tuple[float, List[float]]:

        optimiser.zero_grad()
        data[0] = data[0].double()
        padding_mask = torch.ones((data[0].shape[0], data[0].shape[1])) > 0
        output = model(data[0].to(self.device), padding_mask.to(self.device))

        mask = torch.ones_like(data[1]).to(self.device)
        mask[:, 0] = (17280 - data[1][:, 2]) / 17280
        mask[:, 1] = (17280 - data[1][:, 3]) / 17280

        loss = self.criterion(output, data[1].type(
            torch.DoubleTensor).to(self.device), mask)
        loss.backward()
        optimiser.step()
        losses.append(loss.detach())
        return loss.detach(), losses

    def val_loss(self, dataloader: torch.utils.data.DataLoader, epoch_no, model: nn.Module, disable_tqdm=False):
        """Run the model on the test set and return validation loss"""
        losses = []
        with tqdm(dataloader, unit="batch", disable=disable_tqdm) as tepoch:
            for idx, data in enumerate(tepoch):
                losses = self._val_one_loop(
                    data=data, losses=losses, model=model)
                tepoch.set_description(f"Epoch {epoch_no}")
                tepoch.set_postfix(loss=sum(losses) / len(losses))
        return losses

    def _val_one_loop(self, data: torch.utils.data.DataLoader, losses: List[float], model: nn.Module):
        with torch.no_grad():
            data[0] = data[0].double()
            padding_mask = torch.ones((data[0].shape[0], data[0].shape[1])) > 0
            output = model(data[0].to(self.device),
                           padding_mask.to(self.device))

            mask = torch.ones_like(data[1]).to(self.device)
            mask[:, 0] = (17280 - data[1][:, 2]) / 17280
            mask[:, 1] = (17280 - data[1][:, 3]) / 17280

            loss = self.criterion(output, data[1].type(
                torch.DoubleTensor).to(self.device), mask)
            losses.append(loss.detach())
        return losses
