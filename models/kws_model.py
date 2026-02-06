import torch
import torch.nn as nn


class KWSModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, 32, padding_idx=0)

        self.conv = nn.Sequential(
            nn.Conv1d(80, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Linear(128 + 32, 1)

    def forward(self, mel, kw, kw_len):
        # mel: [B, T, 80]
        x = mel.transpose(1, 2)  # [B, 80, T]
        x = self.conv(x).transpose(1, 2)  # [B, T, 128]

        emb = self.embed(kw)  # [B, K, 32]
        emb = emb.mean(dim=1)  # [B, 32]
        emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)

        out = torch.cat([x, emb], dim=-1)
        logits = self.fc(out).squeeze(-1)

        return logits
