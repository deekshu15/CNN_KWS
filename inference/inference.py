import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    CNN-based Keyword Spotting Inference
    Robust timestamp localization (competition-ready)
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
        base_threshold=0.22,
        smooth_window=7,          # 🔑 temporal smoothing
        min_duration_ratio=0.6    # 🔑 minimum % of expected duration
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)

        self.base_threshold = base_threshold
        self.smooth_window = smooth_window
        self.min_duration_ratio = min_duration_ratio

        self.keyword_stats = keyword_stats
        self.char2idx = char2idx

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
        # ---------- Load audio ----------
        wav, sr = sf.read(wav_path)
        wav = torch.tensor(wav, dtype=torch.float32)

        if wav.ndim == 2:
            wav = wav.mean(dim=1)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        wav = wav[:self.max_len]
        if wav.shape[0] < self.max_len:
            wav = torch.nn.functional.pad(wav, (0, self.max_len - wav.shape[0]))

        mel = self.mel(wav.to(self.device)).transpose(0, 1).unsqueeze(0)

        # ---------- Encode keyword ----------
        kw = torch.tensor(
            [self.char2idx[c] for c in keyword if c in self.char2idx],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        if kw.shape[1] == 0:
            return None

        kl = torch.tensor([kw.shape[1]], device=self.device)

        # ---------- Forward ----------
        probs = torch.sigmoid(self.model(mel, kw, kl))[0].cpu().numpy()

        # ---------- Temporal smoothing ----------
        if self.smooth_window > 1:
            kernel = np.ones(self.smooth_window) / self.smooth_window
            probs = np.convolve(probs, kernel, mode="same")

        # ---------- Threshold ----------
        active = probs > self.base_threshold
        if not active.any():
            return None

        # ---------- Find contiguous segments ----------
        segments = []
        start = None
        for i, v in enumerate(active):
            if v and start is None:
                start = i
            elif not v and start is not None:
                segments.append((start, i))
                start = None
        if start is not None:
            segments.append((start, len(active) - 1))

        # ---------- Duration-aware selection ----------
        expected_sec = self.keyword_stats.get(keyword, 0.45)
        expected_frames = int(expected_sec * self.sample_rate / self.hop_length)
        min_frames = int(expected_frames * self.min_duration_ratio)

        best = None
        best_score = -1

        for s, e in segments:
            length = e - s
            if length < min_frames:
                continue  # 🔑 remove spikes

            conf = probs[s:e].mean()

            # Early bias (matches annotation)
            time_penalty = 0.15 * (s / len(probs))

            score = conf - time_penalty
            if score > best_score:
                best_score = score
                best = (s, e)

        if best is None:
            return None

        s, e = best

        start_time = s * self.hop_length / self.sample_rate
        end_time = e * self.hop_length / self.sample_rate
        confidence = float(probs[s:e].max())

        return (
            round(start_time, 3),
            round(end_time, 3),
            round(confidence, 3)
        )
