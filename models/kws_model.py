import torch
import torch.nn as nn

class KWSModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, audio_dim=64):
        super().__init__()

        self.audio_enc = nn.Sequential(
            nn.Linear(80, audio_dim),
            nn.ReLU(),
            nn.Linear(audio_dim, audio_dim)
        )

        self.text_emb = nn.Embedding(vocab_size, emb_dim)
        self.text_fc = nn.Linear(emb_dim, audio_dim)

    def forward(self, mels, kw, kw_len):
        a = self.audio_enc(mels)              # [B, T, D]

        k = self.text_emb(kw)
        k = k.mean(dim=1)
        k = self.text_fc(k).unsqueeze(1)      # [B, 1, D]

        return (a * k).sum(dim=-1)             # [B, T]
