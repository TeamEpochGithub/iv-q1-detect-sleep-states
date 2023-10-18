import torch
import torch.nn as nn
import torchsummary


class TimeSeriesSegmentationModel(nn.Module):
    def __init__(self):
        super(TimeSeriesSegmentationModel, self).__init__()

        # Encoding Layers
        self.conv1d = nn.Conv1d(7, 16, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(2)
        self.conv1d2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool1d(2)
        self.conv1d3 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool1d(3)

        # Bidirectional GRU
        self.bgru = nn.GRU(32, hidden_size=16, num_layers=2, bidirectional=True, batch_first=True)

        # Auxiliary State Segmentation Head
        self.segmentation_conv1d = nn.Conv1d(32+32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Final Event Classification Head
        self.final_conv1d = nn.Conv1d(32 + 32+1, 2, kernel_size=1)
        self.final_softmax = nn.Softmax(dim=2)
        torchsummary.summary(self, (7, 17280), device='cpu')

    def forward(self, x):
        # Encoding
        x = self.conv1d(x)
        x = self.maxpool(x)
        x = self.conv1d2(x)
        x = self.maxpool2(x)
        x = self.conv1d3(x)
        x = self.maxpool3(x)

        # Bidirectional GRU
        x_gru_in = x.transpose(1, 2).contiguous()
        x_gru_out, _ = self.bgru(x_gru_in)
        x_gru_out = x_gru_out.transpose(1, 2).contiguous()

        # Concatenate conv with gru output
        x = torch.cat((x, x_gru_out), dim=1)

        # Auxiliary State Segmentation Head
        state_segmentation = self.segmentation_conv1d(x)
        state_segmentation = self.sigmoid(state_segmentation)

        # Concatenate state with conv and gru output
        x = torch.cat((x, state_segmentation.detach()), dim=1)

        # Final Event Classification Head
        event_classification = self.final_conv1d(x)
        event_classification = self.final_softmax(event_classification)

        return state_segmentation, event_classification
