# 🎙️ Keyword Spotting (KWS) Model

A frame-level keyword spotting system that detects keyword occurrences in audio files and provides precise timestamps.

## 📌 Project Overview

**Task**: Detect keyword occurrences in audio and return start/end timestamps

**Input**:
- Audio file (WAV format)
- Keyword (text)

**Output**:
- Start and end timestamps of keyword occurrence

**Constraints**:
- ❌ No ASR (Automatic Speech Recognition)
- ❌ No pretrained models
- ❌ No external datasets
- ✅ Frame-level keyword spotting
- ✅ 46 speakers total

## 📁 Dataset Structure

```
CNN Model/
├── data/
│   ├── metadata_fixed.csv          # Annotations (199,010 samples)
│   └── audio/
│       ├── folder 1/               # ~4 speakers
│       ├── folder 2/
│       ├── ...
│       └── folder 12/
├── models/                         # Model architecture
│   ├── audio_encoder.py            # Frame-level audio encoder
│   ├── keyword_encoder.py          # Character-level keyword encoder
│   └── kws_model.py                # Main KWS model
├── datasets/                       # Dataset loader
│   └── kws_dataset.py
├── training/                       # Training scripts
│   └── train.py
├── inference/                      # Inference scripts
│   └── inference.py
├── checkpoints/                    # Saved models
│   ├── folder_1/model.pt
│   ├── folder_2/model.pt
│   └── ...
├── train_colab.py                  # Colab-friendly training
├── COLAB_TRAINING.md               # Colab guide
└── requirements.txt
```

## 🏗️ Model Architecture

### Audio Encoder
- **Input**: Mel-spectrogram [T, 80]
- **Architecture**: CNN layers
- **Output**: Frame-level embeddings [T, D]

### Keyword Encoder
- **Input**: Character sequence
- **Architecture**: Character-level embedding
- **Output**: Keyword embedding [D]

### Matching & Output
- **Method**: Dot-product / alignment between audio frames and keyword
- **Output**: Frame-level scores [T]
- **Loss**: Masked BCEWithLogitsLoss

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd "CNN Model"

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset

The dataset is already prepared with:
- **199,010 annotated samples**
- **46 speakers** across 12 folders
- **WAV format** audio (16kHz)
- **Frame-level labels** (positive frames ≈ 40 per sample)

Metadata format:
```csv
speaker_id,audio_path,keyword,start_time,end_time
AdityaBDhruva_...,C:\...\00JS.wav,BATH,0.61,1.05
```

### 3. Training

**Local Training (CPU):**
```bash
# Train all folders
python -m training.train

# Train specific folders
python -m training.train 1 5 10
```

**Colab Training (GPU - Recommended):**
See [COLAB_TRAINING.md](COLAB_TRAINING.md) for detailed guide.

### 4. Inference

```bash
python -m inference.inference
```

Edit `inference/inference.py` to customize:
- `CHECKPOINT_PATH` - Model checkpoint to use
- `AUDIO_PATH` - Audio file to test
- `KEYWORD` - Keyword to detect
- `THRESHOLD` - Detection threshold (default: 0.5)

## 📊 Training Strategy

### Folder-wise Training

Due to dataset size, training is done **folder-wise**:

1. **Train folder 1** → Save checkpoint
2. **Train folder 2** → Save checkpoint
3. Continue for all 12 folders

Each folder has **different sample counts** (caused by varying annotation density per speaker).

**Example:**
```python
from train_colab import train_folder

# Train folder 1 with 10 epochs
train_folder(folder_id=1, epochs=10)
```

### Current Status

- ✅ Dataset: 199,010 samples loaded correctly
- ✅ Training: All 12 folder checkpoints saved
- ✅ Inference: Working without errors
- ⚠️ Model accuracy: Needs validation and tuning

## 🔧 Configuration

### Training Parameters

Edit `train_colab.py`:

```python
BATCH_SIZE = 16      # Increase to 32/64 on GPU
EPOCHS = 10          # Epochs per folder
LR = 1e-3           # Learning rate
```

### Inference Parameters

Edit `inference/inference.py`:

```python
THRESHOLD = 0.5      # Detection threshold (0.0 - 1.0)
SAMPLE_RATE = 16000  # Audio sample rate
HOP_LENGTH = 160     # Mel-spectrogram hop length
```

## 📈 Model Checkpoints

Each folder saves its own checkpoint:

```
checkpoints/
├── folder_1/model.pt   (~62,972 samples)
├── folder_2/model.pt   (~17,962 samples)
├── folder_3/model.pt
├── ...
└── folder_12/model.pt  (~17,822 samples)
```

**Note**: Checkpoints store **only model weights**, not `char2idx`. The vocabulary is loaded from `metadata_fixed.csv` during inference.

## 🧪 Testing

### Check GPU
```bash
python check_gpu.py
```

### Test Dataset
```bash
python test_dataset_load.py
```

Output:
```
Loaded samples: 199010
Mel shape: torch.Size([1001, 80])
Keyword length: 8
Positive frames: 40.0
```

### Test Inference
```bash
python -m inference.inference
```

Output:
```
Loaded samples: 199010
✅ Keyword FOUND from 0.62s to 1.08s
```
or
```
❌ Keyword NOT FOUND
```

## 📝 File Descriptions

| File | Description |
|------|-------------|
| `models/kws_model.py` | Main KWS model combining audio & keyword encoders |
| `models/audio_encoder.py` | CNN-based audio encoder |
| `models/keyword_encoder.py` | Character-level keyword encoder |
| `datasets/kws_dataset.py` | Dataset loader with frame-level labels |
| `training/train.py` | Training script for local/server |
| `train_colab.py` | Colab-friendly training with progress monitoring |
| `inference/inference.py` | Inference script for keyword detection |
| `COLAB_TRAINING.md` | Detailed Colab training guide |

## 🎯 Next Steps

1. **Train on GPU (Colab)** - Use folder-wise training strategy
2. **Validate model accuracy** - Test on multiple audio samples
3. **Tune hyperparameters** - Adjust threshold, epochs, learning rate
4. **Model fusion** - Combine folder-wise checkpoints (optional)
5. **Evaluation metrics** - Calculate precision, recall, F1-score

## 🐛 Troubleshooting

### Dataset Loading Issues

```python
# Check metadata path
import pandas as pd
df = pd.read_csv("data/metadata_fixed.csv")
print(len(df))  # Should be 199010
```

### Path Issues

Audio files are in `DATASETS` subdirectories:
```
data/audio/folder 1/AdityaBDhruva_.../DATASETS/00JS.wav
```

### Model Not Detecting Keywords

1. **Lower threshold**: Try 0.2 or 0.3
2. **Check audio file**: Ensure it contains the keyword
3. **Train longer**: Increase epochs per folder
4. **Verify metadata**: Check if annotation exists for that audio

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines]

## 📧 Contact

[Add your contact information]

---

**Status**: ✅ Ready for Colab training
**Last Updated**: February 4, 2026
