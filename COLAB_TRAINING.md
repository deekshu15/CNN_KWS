# Google Colab Training Guide

This guide explains how to train the Keyword Spotting model in Google Colab with folder-wise training.

## 📋 Prerequisites

1. Google Colab account
2. Your data uploaded to Google Drive or Colab storage
3. GPU runtime enabled in Colab

## 🚀 Quick Start in Colab

### Step 1: Setup Runtime

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Step 2: Mount Google Drive & Setup Data

**Your Drive Structure:**
```
Google Drive/
└── KWS_DATA/
    ├── audio/
    │   ├── folder 1/
    │   ├── folder 2/
    │   ├── ...
    │   └── folder 12/
    └── metadata_fixed.csv
```

**Mount and Setup:**
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone GitHub repo
import os
os.chdir('/content')
!git clone https://github.com/deekshu15/CNN_KWS.git
os.chdir('/content/CNN_KWS')

# Link to your Drive data
!rm -rf data
!mkdir -p data
!ln -s /content/drive/MyDrive/KWS_DATA/audio data/audio
!ln -s /content/drive/MyDrive/KWS_DATA/metadata_fixed.csv data/metadata_fixed.csv

# Verify
!ls data/
!ls data/audio/
```

### Step 3: Install Dependencies

```python
!pip install -r requirements.txt
```

### Step 4: Train Folder-wise

```python
# Import training function
from train_colab import train_folder

# Train folder 1 with 10 epochs
train_folder(folder_id=1, epochs=10)

# Train folder 2 with 10 epochs
train_folder(folder_id=2, epochs=10)

# Continue for other folders...
```

## 📊 Training Strategies

### Strategy 1: Sequential Training (Recommended)

Train one folder at a time, save checkpoint, then move to next:

```python
from train_colab import train_folder

# Train folders 1-12, one by one
for folder_id in range(1, 13):
    print(f"\n{'='*60}")
    print(f"Training Folder {folder_id}/12")
    print(f"{'='*60}")
    train_folder(folder_id, epochs=10)
```

### Strategy 2: Resume Training

If Colab disconnects, resume from last checkpoint:

```python
# Resume folder 5 training
train_folder(
    folder_id=5, 
    epochs=10,
    resume_checkpoint="checkpoints/folder_5/model.pt"
)
```

### Strategy 3: Batch Training

Train specific folders:

```python
# Train only folders 1, 5, and 10
for folder_id in [1, 5, 10]:
    train_folder(folder_id, epochs=10)
```

## 💾 Download Checkpoints

After training, download checkpoints to your local machine:

```python
# Zip all checkpoints
!zip -r checkpoints.zip checkpoints/

# Download (will prompt download dialog)
from google.colab import files
files.download('checkpoints.zip')
```

## ⚙️ Configuration Options

Edit `train_colab.py` to adjust:

- `BATCH_SIZE = 16` - Increase to 32 or 64 for GPU
- `LR = 1e-3` - Learning rate
- `epochs` parameter in `train_folder()` - Training epochs per folder

## 📈 Monitor Training

```python
# Check GPU memory usage
!nvidia-smi

# View training progress in real-time (prints every 100 batches)
```

## 🔍 Verify After Training

```python
# List all saved checkpoints
!ls -lh checkpoints/*/model.pt

# Check checkpoint size
!du -sh checkpoints/*
```

## 💡 Tips

1. **GPU Runtime**: Always enable GPU (Runtime → Change runtime type → GPU)
2. **Session Management**: Colab may disconnect after ~12 hours. Save checkpoints frequently.
3. **Batch Size**: Increase `BATCH_SIZE` to 32 or 64 on GPU for faster training
4. **Monitoring**: Keep the Colab tab active to prevent disconnection
5. **Checkpoints**: Download checkpoints periodically to Google Drive

## 🐛 Troubleshooting

**Out of Memory Error:**
```python
# Reduce batch size in train_colab.py
BATCH_SIZE = 8  # or lower
```

**Module Not Found:**
```python
# Reinstall dependencies
!pip install soundfile torchaudio pandas torch
```

**Path Errors:**
```python
# Verify data structure
!ls -R data/
```

## 📝 Example Complete Workflow

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Navigate to project
import os
os.chdir('/content/drive/MyDrive/CNN Model')

# 3. Install dependencies
!pip install -r requirements.txt

# 4. Train all folders
from train_colab import train_folder

for folder_id in range(1, 13):
    train_folder(folder_id, epochs=10)

# 5. Download checkpoints
!zip -r checkpoints.zip checkpoints/
from google.colab import files
files.download('checkpoints.zip')
```

## 🎯 Expected Results

- **Training time per folder**: ~5-15 minutes (depends on GPU and folder size)
- **Checkpoint size**: ~1-5 MB per folder
- **Total training time (12 folders, 10 epochs each)**: ~1-3 hours on GPU
