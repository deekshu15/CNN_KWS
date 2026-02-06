import torch


def collate(batch):
    """
    Enforces the correct contract regardless of dataset mistakes.
    """

    mels, kws, labels = [], [], []

    for mel, a, b in batch:
        # auto-detect which is label vs keyword
        if a.max() <= 1.0 and b.max() > 1:
            lbl, kw = a, b
        elif b.max() <= 1.0 and a.max() > 1:
            lbl, kw = b, a
        else:
            raise RuntimeError("Cannot disambiguate kw vs labels")

        mels.append(mel)
        kws.append(kw)
        labels.append(lbl)

    B = len(mels)
    maxT = max(m.shape[0] for m in mels)
    maxK = max(len(k) for k in kws)

    M = torch.zeros(B, maxT, 80)
    Y = torch.zeros(B, maxT)
    MASK = torch.zeros(B, maxT)

    K = torch.zeros(B, maxK, dtype=torch.long)
    KL = torch.tensor([len(k) for k in kws], dtype=torch.long)

    for i in range(B):
        T = mels[i].shape[0]
        M[i, :T] = mels[i]
        Y[i, :T] = labels[i]
        MASK[i, :T] = 1.0
        K[i, : len(kws[i])] = kws[i]

    # FINAL GUARANTEE
    assert Y.max() <= 1.0
    assert K.max() > 1

    return M, K, KL, Y, MASK
