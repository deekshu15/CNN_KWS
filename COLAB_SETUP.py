"""
Google Colab Setup Script
=========================

Run this in a Colab notebook cell to set up your training environment.

Your Google Drive structure (ONLY AUDIO FILES NEEDED):
KWS_DATA/
└── audio/
    ├── folder 1/
    ├── folder 2/
    ├── ...
    └── folder 12/

Note: metadata_fixed.csv is already in the GitHub repo!

"""

# =====================
# STEP 1: Mount Google Drive
# =====================
from google.colab import drive
drive.mount('/content/drive')

# =====================
# STEP 2: Clone GitHub Repository
# =====================
import os
os.chdir('/content')
!git clone https://github.com/deekshu15/CNN_KWS.git
os.chdir('/content/CNN_KWS')

# =====================
# STEP 3: Install Dependencies
# =====================
!pip install -q soundfile torchaudio pandas

# =====================
# STEP 4: Create Symbolic Links to Your Drive Data
# =====================

# Remove placeholder data folder if exists
!rm -rf data

# Create data directory structure
!mkdir -p data

# Link to your Drive data
# IMPORTANT: Update the path below if your Drive structure is different
DRIVE_DATA_PATH = '/content/drive/MyDrive/KWS_DATA'

# Create symbolic link to audio folder only
# (metadata_fixed.csv already exists in the GitHub repo)
!ln -s "$DRIVE_DATA_PATH/audio" data/audio

# =====================
# STEP 5: Verify Setup
# =====================
print("\n✅ Checking setup...")

# Check if data is accessible
import os
if os.path.exists('data/audio'):
    print("✅ Audio folder linked successfully")
    # List folders
    folders = !ls data/audio
    print(f"✅ Found {len(folders)} folders: {folders}")
else:
    print("❌ Audio folder not found!")

if os.path.exists('data/metadata_fixed.csv'):
    print("✅ Metadata file found (from GitHub repo)")
    # Check metadata
    import pandas as pd
    df = pd.read_csv('data/metadata_fixed.csv')
    print(f"✅ Loaded {len(df)} samples from metadata")
else:
    print("❌ Metadata file not found!")

# Check GPU
import torch
if torch.cuda.is_available():
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ No GPU detected. Enable GPU: Runtime → Change runtime type → GPU")

print("\n🚀 Setup complete! You can now start training.")
print("\nTo train folder 1:")
print("  from train_colab import train_folder")
print("  train_folder(1, epochs=10)")
