import torch
import soundfile as sf
import torchaudio

from models.kws_model import KWSModel
from datasets.kws_dataset import KWSDataset, text_to_char_ids

# =====================
# CONFIG
# =====================

CHECKPOINT_PATH = "checkpoints/folder_1/model.pt"   # change folder if needed
METADATA_PATH   = "data/metadata_fixed.csv"

AUDIO_PATH = (
    "data/audio/folder 1/"
    "AdityaBDhruva_XxYc2jHvSbhNXVeHgBDpDKhK9hU2/DATASETS/00JS.wav"
)

KEYWORD = "FRIGHTEN"

SAMPLE_RATE = 16000
HOP_LENGTH = 160
THRESHOLD = 0.5
DEVICE = "cpu"

# =====================
# Load dataset (for vocab)
# =====================

dataset = KWSDataset(METADATA_PATH)
char2idx = dataset.char2idx

# =====================
# Load model
# =====================

model = KWSModel(vocab_size=len(char2idx))
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =====================
# Load audio
# =====================

wav, sr = sf.read(AUDIO_PATH)
wav = torch.tensor(wav, dtype=torch.float32)

if wav.ndim == 2:
    wav = wav.mean(dim=1)

if sr != SAMPLE_RATE:
    wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=80,
    hop_length=HOP_LENGTH
)(wav)

mel = mel.transpose(0, 1).unsqueeze(0).to(DEVICE)  # [1, T, 80]

# =====================
# Encode keyword
# =====================

kw_ids = torch.tensor(
    text_to_char_ids(KEYWORD, char2idx),
    dtype=torch.long
).unsqueeze(0).to(DEVICE)

kw_len = torch.tensor([kw_ids.shape[1]]).to(DEVICE)

# =====================
# Inference
# =====================

with torch.no_grad():
    logits = model(mel, kw_ids, kw_len)
    scores = torch.sigmoid(logits)[0]

# =====================
# Post-processing
# =====================

active = scores > THRESHOLD

if active.any():
    frames = torch.where(active)[0]
    start = frames[0].item() * HOP_LENGTH / SAMPLE_RATE
    end   = frames[-1].item() * HOP_LENGTH / SAMPLE_RATE
    print(f"✅ Keyword FOUND from {start:.2f}s to {end:.2f}s")
else:
    print("❌ Keyword NOT FOUND")
