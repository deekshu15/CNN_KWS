import torch.nn as nn

class KeywordEncoder(nn.Module):
    def __init__(self, vocab, emb=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)

    def forward(self, k, lengths):
        e = self.emb(k)
        mask = (k != 0).unsqueeze(-1)
        return (e * mask).sum(1) / lengths.unsqueeze(1)
