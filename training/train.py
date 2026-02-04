import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.kws_dataset import KWSDataset
from models.kws_model import KWSModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 1
LR = 1e-3

def collate(batch):
    mels, kws, labels = zip(*batch)

    maxT = max(x.shape[0] for x in mels)
    maxL = max(len(k) for k in kws)

    M = torch.zeros(len(batch), maxT, 80)
    Y = torch.zeros(len(batch), maxT)
    mask = torch.zeros(len(batch), maxT)
    K = torch.zeros(len(batch), maxL, dtype=torch.long)
    KL = torch.tensor([len(k) for k in kws])

    for i in range(len(batch)):
        M[i, :mels[i].shape[0]] = mels[i]
        Y[i, :labels[i].shape[0]] = labels[i]
        mask[i, :labels[i].shape[0]] = 1
        K[i, :len(kws[i])] = kws[i]

    return M, K, KL, Y, mask

def train_folder(folder_id):
    print(f"\n🚀 Training folder {folder_id}")

    ds = KWSDataset(
        metadata_csv="data/metadata_fixed.csv",
        folder_id=folder_id
    )

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate
    )

    model = KWSModel(len(ds.char2idx)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    for e in range(EPOCHS):
        total = 0
        for m, k, kl, y, mask in loader:
            m, k, kl, y, mask = (
                m.to(DEVICE),
                k.to(DEVICE),
                kl.to(DEVICE),
                y.to(DEVICE),
                mask.to(DEVICE)
            )

            logits = model(m, k, kl)
            loss = (loss_fn(logits, y) * mask).sum() / mask.sum()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {e+1} loss: {total/len(loader):.4f}")

    os.makedirs(f"checkpoints/folder_{folder_id}", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/folder_{folder_id}/model.pt")
    print(f"✅ Saved checkpoint: checkpoints/folder_{folder_id}/model.pt")

if __name__ == "__main__":
    # Train specific folders by changing the range
    # Examples:
    # - Single folder: train_folder(1)
    # - Specific folders: for folder in [1, 3, 5]: train_folder(folder)
    # - All folders: for folder in range(1, 13): train_folder(folder)
    
    import sys
    
    if len(sys.argv) > 1:
        # Usage: python -m training.train 1 5 10 (trains folders 1, 5, and 10)
        folder_ids = [int(x) for x in sys.argv[1:]]
        for folder in folder_ids:
            train_folder(folder)
    else:
        # Default: train all folders
        for folder in range(1, 13):
            train_folder(folder)
