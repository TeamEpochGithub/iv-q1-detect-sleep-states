import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from prototyping.data import get_data_loader
from prototyping.model import TimeSeriesSegmentationModel

event_loss_weight = 1e6
label_balance = 0.639847

if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f'Using device: {device}')

    # Define your loss functions
    segmentation_criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    event_classification_criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for softmax

    # Instantiate the model
    model = TimeSeriesSegmentationModel().double().to(device)

    # Define your optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load data
    dataloader = get_data_loader()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader)

        seg_losses = []
        event_losses = []

        for batch in pbar:  # Replace dataloader with your data loading logic
            optimizer.zero_grad()
            data = batch['data'].to(device)
            segmentation_target = batch['segmentation_label'].unsqueeze(1).to(device)
            event_classification_target = batch['event_classification_label'].to(device)

            segment_pred, event_pred = model(data)

            # Compute losses
            weight_rebal = torch.where(segmentation_target == 1.0, 1/(1-label_balance), 1/label_balance)
            segmentation_criterion = nn.BCELoss(weight=weight_rebal)
            segmentation_loss = segmentation_criterion(segment_pred, segmentation_target)
            event_classification_loss = event_classification_criterion(event_pred, event_classification_target)

            # Total loss
            total_loss = segmentation_loss + event_loss_weight*event_classification_loss

            total_loss.backward()
            optimizer.step()

            seg_losses.append(segmentation_loss.item())
            event_losses.append(event_classification_loss.item())

            pbar.set_description(
                f'Epoch {epoch + 1}, Segmentation Loss: {np.mean(seg_losses):.3E}, '
                f'Event Classification Loss: {np.mean(event_classification_loss.item()):.3E}')

        # save the model
        torch.save(model.state_dict(), f'./tm/model_{epoch}.pt')
