"""
Colab-Friendly Training Script for Keyword Spotting
=====================================================

This script allows flexible folder-wise training in Google Colab.

Usage in Colab:
---------------
1. Upload your data folder to Colab or mount Google Drive
2. Modify the configuration section below
3. Run specific folders by calling train_folder(folder_id)

Example:
--------
# Train folder 1 with 5 epochs
train_folder(1, epochs=5)

# Train folders 1-3
for folder_id in range(1, 4):
    train_folder(folder_id, epochs=3)
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.kws_dataset import KWSDataset
from models.kws_model import KWSModel

# =====================
# CONFIGURATION
# =====================

METADATA_PATH = "data/metadata_fixed.csv"
CHECKPOINT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # Increase for GPU
LR = 1e-3

print(f"🖥️  Using device: {DEVICE}")

# =====================
# COLLATE FUNCTION
# =====================

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

# =====================
# TRAINING FUNCTION
# =====================

def train_folder(folder_id, epochs=5, resume_checkpoint=None):
    """
    Train a specific folder
    
    Args:
        folder_id (int): Folder number (1-12)
        epochs (int): Number of training epochs
        resume_checkpoint (str): Path to checkpoint to resume from (optional)
    """
    print(f"\n{'='*60}")
    print(f"🚀 Training folder {folder_id}")
    print(f"{'='*60}")

    # Load dataset for this folder
    ds = KWSDataset(
        metadata_csv=METADATA_PATH,
        folder_id=folder_id
    )

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=2 if DEVICE == "cuda" else 0
    )

    # Initialize model
    model = KWSModel(len(ds.char2idx)).to(DEVICE)
    
    # Resume from checkpoint if provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"📂 Resuming from: {resume_checkpoint}")
        model.load_state_dict(torch.load(resume_checkpoint, map_location=DEVICE))
    
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # Training loop
    for e in range(epochs):
        model.train()
        total = 0
        
        for batch_idx, (m, k, kl, y, mask) in enumerate(loader):
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
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {e+1}/{epochs} | Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

        avg_loss = total / len(loader)
        print(f"✅ Epoch {e+1}/{epochs} completed | Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    checkpoint_path = f"{CHECKPOINT_DIR}/folder_{folder_id}/model.pt"
    os.makedirs(f"{CHECKPOINT_DIR}/folder_{folder_id}", exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"💾 Saved checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    return model

# =====================
# EXAMPLE USAGE
# =====================

if __name__ == "__main__":
    # Example 1: Train folder 1 with 5 epochs
    train_folder(1, epochs=5)
    
    # Example 2: Train multiple folders
    # for folder_id in range(1, 4):
    #     train_folder(folder_id, epochs=3)
    
    # Example 3: Resume training from checkpoint
    # train_folder(1, epochs=5, resume_checkpoint="checkpoints/folder_1/model.pt")
