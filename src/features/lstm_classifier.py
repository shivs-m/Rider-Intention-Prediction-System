import torch, torch.nn as nn

class LSTMFeatureClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 6, hidden: int = 256, layers: int = 1, bi: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bi,
        )
        out_dim = hidden * (2 if bi else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):  # x: (B, T, D)
        y, _ = self.lstm(x)
        last = y[:, -1]
        return self.head(last)