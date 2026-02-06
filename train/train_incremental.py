import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from CNN_KWS.datasets.kws_dataset import KWSDataset


def train_one_folder(
    folder_id,
    model,
    optimizer,
    char2idx,
    collate_fn,
    epochs=2,
    batch_size=4,
    device="cpu",
    meta_root="/content",
    ckpt_root="/content/drive/MyDrive/KWS_CHECKPOINTS",
):
    print(f"\n📁 Training folder {folder_id}")

    meta_csv = f"{meta_root}/metadata_folder{folder_id}_fixed.csv"
    if not os.path.exists(meta_csv):
        print(f"⚠️ Metadata missing for folder {folder_id}, skipping")
        return model

    ds = KWSDataset(meta_csv, char2idx)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    model.train()
    os.makedirs(ckpt_root, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0

        for m, k, kl, y, mask in tqdm(loader):
            m = m.to(device)
            k = k.to(device)
            kl = kl.to(device)
            y = y.to(device)
            mask = mask.to(device)

            logits = model(m, k, kl)

            loss_raw = F.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            )

            valid = mask.sum()
            if valid == 0:
                continue

            loss = (loss_raw * mask).sum() / valid

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.6f}")

    ckpt_path = f"{ckpt_root}/kws_folder{folder_id}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"✅ Saved checkpoint: {ckpt_path}")

    return model
