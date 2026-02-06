import torch
from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    """
    batch: list of (mel [T, F], kw [L], labels [T])
    """

    mels = []
    kws = []
    kw_lens = []
    labels = []
    masks = []

    max_T = max(b[0].shape[0] for b in batch)

    for mel, kw, lbl in batch:
        T = mel.shape[0]

        mels.append(mel)
        kws.append(kw)
        kw_lens.append(len(kw))

        labels.append(lbl)

        mask = torch.zeros(max_T)
        mask[:T] = 1.0
        masks.append(mask)

    mels = pad_sequence(mels, batch_first=True)      # [B, T, F]
    labels = pad_sequence(labels, batch_first=True)  # [B, T]
    masks = torch.stack(masks)                        # [B, T]

    kws = pad_sequence(kws, batch_first=True)         # [B, L]
    kw_lens = torch.tensor(kw_lens)

    return mels, kws, kw_lens, labels, masks
