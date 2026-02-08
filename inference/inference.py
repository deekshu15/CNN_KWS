import torch
import torchaudio
import soundfile as sf
import numpy as np
from scipy.ndimage import gaussian_filter1d

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    CNN-based Keyword Spotting Inference
    Localization-only (no hard detection)
    """

    def __init__(
        self,
        checkpoint_path,
        char2idx,
        keyword_stats,          # dict: keyword -> median_duration_sec
        device="cpu",
        sample_rate=16000,
        n_mels=80,
        hop_length=160,
        max_seconds=10.0,
        smooth_sigma=2.0        # smoothing strength
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)

        self.char2idx = char2idx
        self.keyword_stats = keyword_stats
        self.smooth_sigma = smooth_sigma

        self.model = KWSModel(len(char2idx)).to(device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        self.model.eval()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
        ).to(device)

    @torch.no_grad()
    def infer(self, wav_path, keyword):
        # ---- Load audio ----
        wav, sr = sf.read(wav_path)
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

        mel = self.mel(wav.to(self.device)).transpose(0, 1).unsqueeze(0)

        # ---- Encode keyword ----
        kw_ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        if len(kw_ids) == 0:
            return None

        kw = torch.tensor(kw_ids, device=self.device).unsqueeze(0)
        kl = torch.tensor([len(kw_ids)], device=self.device)

        # ---- Model forward ----
        logits = self.model(mel, kw, kl)[0]
        probs = torch.sigmoid(logits).cpu().numpy()

        # ---- Smooth confidence ----
        probs = gaussian_filter1d(probs, sigma=self.smooth_sigma)

        # ---- Peak localization ----
        center = int(np.argmax(probs))

        # ---- Duration prior ----
        dur_sec = self.keyword_stats.get(keyword, 0.45)
        dur_frames = int(dur_sec * self.sample_rate / self.hop_length)

        # ---- Expand around center ----
        start = center - dur_frames // 2
        end = center + dur_frames // 2

        start = max(start, 0)
        end = min(end, len(probs) - 1)

        start_time = start * self.hop_length / self.sample_rate
        end_time = end * self.hop_length / self.sample_rate

        confidence = float(np.max(probs))

        return (
            round(start_time, 3),
            round(end_time, 3),
            round(confidence, 3),
        )
