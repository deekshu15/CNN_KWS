import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    Speaker-independent CNN-based Keyword Spotting Inference
    Uses confidence-weighted localization with keyword-level statistics
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
        base_threshold=0.3
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)
        self.base_threshold = base_threshold

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
        # ---- Load audio ----
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

        # ---- Keyword encoding ----
        kw = torch.tensor(
            [self.char2idx[c] for c in keyword if c in self.char2idx],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        kl = torch.tensor([kw.shape[1]], device=self.device)

        # ---- Forward ----
        probs = torch.sigmoid(self.model(mel, kw, kl))[0].cpu().numpy()

        max_p = probs.max()
        if max_p < self.base_threshold:
            return None

        # ---- Confidence-weighted center ----
        idxs = np.arange(len(probs))
        weights = probs * (probs > 0.3 * max_p)

        if weights.sum() == 0:
            return None

        center = int(np.sum(idxs * weights) / np.sum(weights))

        # ---- Keyword-adaptive duration ----
        dur_sec = self.keyword_stats.get(keyword, 0.45)
        dur_frames = int(dur_sec * self.sample_rate / self.hop_length)

        start = center - dur_frames // 2
        end = center + dur_frames // 2

        start = max(start, 0)
        end = min(end, len(probs) - 1)

        start_time = start * self.hop_length / self.sample_rate
        end_time = end * self.hop_length / self.sample_rate

        return (
            float(round(start_time, 3)),
            float(round(end_time, 3)),
            float(round(max_p, 3))
        )
