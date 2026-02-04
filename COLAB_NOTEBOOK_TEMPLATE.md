# Keyword Spotting Training - Colab Notebook

This is a template for training your KWS model in Google Colab.

## 📋 Prerequisites

Before running this notebook:
1. Upload your data to Google Drive:
   - `KWS_DATA/audio/` (all 12 folders)
   - `KWS_DATA/metadata_fixed.csv`
2. Enable GPU: Runtime → Change runtime type → GPU (T4 recommended)

---

## 🔧 Setup

### 1. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Clone GitHub Repository
```python
import os
os.chdir('/content')
!git clone https://github.com/deekshu15/CNN_KWS.git
os.chdir('/content/CNN_KWS')
!pwd
```

### 3. Install Dependencies
```python
!pip install -q soundfile torchaudio pandas
```

### 4. Link Data from Google Drive
```python
# Remove placeholder data folder
!rm -rf data

# Create data directory
!mkdir -p data

# Link to your Drive data (update path if needed)
!ln -s /content/drive/MyDrive/KWS_DATA/audio data/audio
!ln -s /content/drive/MyDrive/KWS_DATA/metadata_fixed.csv data/metadata_fixed.csv

# Verify
print("Checking data structure...")
!ls data/
!ls data/audio/ | head -5
```

### 5. Verify GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ No GPU! Enable GPU in Runtime settings")
```

---

## 🚀 Training

### Option 1: Train Single Folder
```python
from train_colab import train_folder

# Train folder 1 with 10 epochs
train_folder(folder_id=1, epochs=10)
```

### Option 2: Train Multiple Folders Sequentially
```python
from train_colab import train_folder

# Train folders 1-12
for folder_id in range(1, 13):
    print(f"\n{'='*60}")
    print(f"📊 Progress: Folder {folder_id}/12")
    print(f"{'='*60}")
    train_folder(folder_id=folder_id, epochs=10)
```

### Option 3: Train Specific Folders
```python
from train_colab import train_folder

# Train only specific folders
for folder_id in [1, 3, 5, 7, 9, 11]:
    train_folder(folder_id=folder_id, epochs=10)
```

---

## 💾 Save Checkpoints to Drive

### Download All Checkpoints
```python
# Zip checkpoints
!zip -r checkpoints.zip checkpoints/

# Copy to Drive for safekeeping
!cp checkpoints.zip /content/drive/MyDrive/KWS_DATA/

print("✅ Checkpoints backed up to Google Drive")
```

### Download to Local Machine
```python
from google.colab import files
files.download('checkpoints.zip')
```

---

## 🔍 Monitor Training

### Check GPU Usage
```python
!nvidia-smi
```

### View Checkpoint Sizes
```python
!ls -lh checkpoints/*/model.pt
!du -sh checkpoints/*
```

### Test Inference (After Training)
```python
# Quick test
!python -m inference.inference
```

---

## 📊 Training Configuration

Edit these in `train_colab.py`:
- `BATCH_SIZE = 16` (increase to 32-64 on GPU)
- `LR = 1e-3` (learning rate)
- `epochs` parameter in `train_folder()`

---

## ⚠️ Troubleshooting

### Path Not Found
```python
# Check if data is linked correctly
!ls -la data/
# Should show symbolic links to Drive
```

### Out of Memory
```python
# Edit train_colab.py and reduce batch size
# Change: BATCH_SIZE = 8
```

### Session Timeout
- Colab free tier: 12 hours max
- Keep tab active to prevent disconnection
- Save checkpoints frequently

---

## 🎯 Expected Timeline

With GPU (T4):
- **Per folder**: 5-15 minutes (10 epochs)
- **All 12 folders**: 1-3 hours total

Without GPU (not recommended):
- **Per folder**: 1-3 hours
- **All 12 folders**: 12-36 hours

---

## ✅ Next Steps

After training:
1. Download `checkpoints.zip`
2. Extract locally
3. Use `inference/inference.py` to test
4. Tune threshold and validate accuracy
