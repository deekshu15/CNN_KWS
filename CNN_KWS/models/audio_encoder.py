import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self, n_mels=80, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)
