import pandas as pd
import torch
import soundfile as sf
import torchaudio
import os
from torch.utils.data import Dataset

def text_to_char_ids(text, char2idx):
    """
    Convert keyword string into character IDs.
    Unknown characters are ignored.
    """
    return [char2idx[c] for c in text if c in char2idx]

class KWSDataset(Dataset):
    def __init__(
        self,
        metadata_csv,
        folder_id=None,
        sample_rate=16000,
        n_mels=80,
        hop_length=160,
        max_seconds=10.0
    ):
        self.df = pd.read_csv(metadata_csv)

        if folder_id is not None:
            self.df = self.df[
                self.df["audio_path"].str.contains(f"folder {folder_id}")
            ].reset_index(drop=True)
        
        # Convert absolute paths to relative paths (for cross-platform compatibility)
        # Extracts path after 'data/audio/' and makes it relative
        self.df["audio_path"] = self.df["audio_path"].apply(
            lambda x: self._convert_to_relative_path(x)
        )

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)

        # build character vocab
        chars = set()
        for w in self.df["keyword"]:
            for c in str(w):
                chars.add(c)

        self.char2idx = {c: i + 1 for i, c in enumerate(sorted(chars))}
        self.char2idx["<PAD>"] = 0

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length
        )

        print(f"Loaded samples: {len(self.df)}")

    def _convert_to_relative_path(self, path):
        """
        Convert absolute Windows path to relative path.
        Example: C:\\Users\\...\\data\\audio\\folder 1\\... 
        -> data/audio/folder 1/...
        """
        # Normalize path separators
        path = str(path).replace('\\', '/')
        
        # Find 'data/audio' in the path and extract from there
        if 'data/audio' in path:
            idx = path.find('data/audio')
            return path[idx:]
        
        # If already relative or doesn't contain data/audio, return as is
        return path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        wav, sr = sf.read(r.audio_path)
        wav = torch.tensor(wav, dtype=torch.float32)

        if wav.ndim == 2:
            wav = wav.mean(dim=1)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav, sr, self.sample_rate
            )

        wav = wav[:self.max_len]
        if wav.shape[0] < self.max_len:
            wav = torch.nn.functional.pad(
                wav, (0, self.max_len - wav.shape[0])
            )

        mel = self.mel(wav).transpose(0, 1)   # [T, 80]
        T = mel.shape[0]

        labels = torch.zeros(T)
        s = int(r.start_time * self.sample_rate / self.hop_length)
        e = int(r.end_time * self.sample_rate / self.hop_length)
        labels[s:e] = 1.0

        kw = torch.tensor(
            [self.char2idx[c] for c in r.keyword if c in self.char2idx],
            dtype=torch.long
        )

        return mel, kw, labels
