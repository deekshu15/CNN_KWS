import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from CNN_KWS.datasets.kws_dataset import KWSDataset
from CNN_KWS.utils.collate import collate
from CNN_KWS.models.kws_model import KWSModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_folder(
    folder_id,
    model,
    optimizer,
    char2idx,
    epochs=2,
    batch_size=4,
):
    meta = f"/content/metadata_folder{folder_id}_fixed.csv"
    if not os.path.exists(meta):
        print(f"Skipping folder {folder_id}")
        return model

    ds = KWSDataset(meta, char2idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    model.train()

    for epoch in range(epochs):
        total, count = 0.0, 0

        for m, k, kl, y, mask in tqdm(loader):
            m, k, kl, y, mask = (
                m.to(DEVICE),
                k.to(DEVICE),
                kl.to(DEVICE),
                y.to(DEVICE),
                mask.to(DEVICE),
            )

            logits = model(m, k, kl)

            loss_raw = F.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            )

            valid = mask.sum().item()
            if valid == 0:
                continue

            loss = (loss_raw * mask).sum() / valid

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total += loss.item()
            count += 1

        print(f"Epoch {epoch+1} | Avg Loss: {total / max(count,1):.4f}")

    return model
