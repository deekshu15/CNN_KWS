import pandas as pd
import torch
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset


class KWSDataset(Dataset):
    """
    Returns:
        mel   : [T, 80]
        kw    : [K]   (character IDs)
        label : [T]   (0/1 frame labels)
    """

    def __init__(
        self,
        metadata_csv,
        char2idx,
        sample_rate=16000,
        n_mels=80,
        hop_length=160,
        max_seconds=10.0,
    ):
        self.df = pd.read_csv(metadata_csv)
        self.char2idx = char2idx

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)

        self.mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
        )

        print(f"Loaded samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        # ---------- Audio ----------
        wav, sr = sf.read(r.audio_path)
        wav = torch.tensor(wav, dtype=torch.float32)

        if wav.ndim == 2:
            wav = wav.mean(dim=1)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav, sr, self.sample_rate
            )

        wav = wav[: self.max_len]
        if wav.shape[0] < self.max_len:
            wav = torch.nn.functional.pad(
                wav, (0, self.max_len - wav.shape[0])
            )

        mel = self.mel_fn(wav).transpose(0, 1)  # [T, 80]
        T = mel.shape[0]

        # ---------- Labels (STRICTLY BINARY) ----------
        labels = torch.zeros(T, dtype=torch.float32)

        s = int(float(r.start_time) * self.sample_rate / self.hop_length)
        e = int(float(r.end_time) * self.sample_rate / self.hop_length)

        s = max(0, min(s, T))
        e = max(0, min(e, T))

        if e > s:
            labels[s:e] = 1.0

        # ---------- Keyword ----------
        kw = torch.tensor(
            [self.char2idx[c] for c in str(r.keyword) if c in self.char2idx],
            dtype=torch.long,
        )

        # ---------- HARD ASSERTS ----------
        assert labels.max() <= 1.0
        assert kw.numel() > 0

        return mel, kw, labels
