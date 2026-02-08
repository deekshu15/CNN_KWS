import torch
import torchaudio
import soundfile as sf
import numpy as np
from scipy.ndimage import gaussian_filter1d

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    CNN-based Keyword Spotting with segment-aware localization
    """

    def __init__(
        self,
        checkpoint_path,
        char2idx,
        keyword_stats,
        device="cpu",
        sample_rate=16000,
        n_mels=80,
        hop_length=160,
        max_seconds=10.0,
        base_threshold=0.18,
        smooth_sigma=2.0,
        min_region_frames=5
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)

        self.base_threshold = base_threshold
        self.smooth_sigma = smooth_sigma
        self.min_region_frames = min_region_frames

        self.char2idx = char2idx
        self.keyword_stats = keyword_stats

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

    # ---------------------------------------------------------
    @torch.no_grad()
    def infer(self, wav_path, keyword):
        # ---------- Audio ----------
        wav, sr = sf.read(wav_path)
        wav = torch.tensor(wav, dtype=torch.float32)

        if wav.ndim == 2:
            wav = wav.mean(dim=1)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        wav = wav[:self.max_len]
        if wav.shape[0] < self.max_len:
            wav = torch.nn.functional.pad(
                wav, (0, self.max_len - wav.shape[0])
            )

        mel = self.mel(wav.to(self.device)).transpose(0, 1).unsqueeze(0)

        # ---------- Keyword ----------
        kw_ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        if len(kw_ids) == 0:
            return None

        kw = torch.tensor(kw_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        kl = torch.tensor([kw.shape[1]], device=self.device)

        # ---------- Forward ----------
        probs = torch.sigmoid(self.model(mel, kw, kl))[0].cpu().numpy()

        # ---------- Smooth ----------
        probs = gaussian_filter1d(probs, sigma=self.smooth_sigma)

        peak = probs.max()
        if peak < self.base_threshold:
            return None

        # ---------- Binary mask ----------
        mask = probs > (0.4 * peak)

        # ---------- Find regions ----------
        regions = []
        start = None

        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= self.min_region_frames:
                    regions.append((start, i))
                start = None

        if start is not None and len(mask) - start >= self.min_region_frames:
            regions.append((start, len(mask)))

        if not regions:
            return None

        # ---------- Select best region ----------
        expected_frames = int(
            self.keyword_stats.get(keyword, 0.45)
            * self.sample_rate / self.hop_length
        )

        best_score = -1
        best_region = None

        for s, e in regions:
            region_probs = probs[s:e]
            mean_p = region_probs.mean()
            dur_penalty = abs((e - s) - expected_frames) / expected_frames
            score = mean_p - 0.3 * dur_penalty

            if score > best_score:
                best_score = score
                best_region = (s, e, region_probs.max())

        s, e, conf = best_region

        start_time = s * self.hop_length / self.sample_rate
        end_time = e * self.hop_length / self.sample_rate

        return (
            round(float(start_time), 3),
            round(float(end_time), 3),
            round(float(conf), 3)
        )
