import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    FINAL CNN-based Keyword Spotting Inference

    - Accurate timestamps when keyword is present
    - Consistent rejection when keyword is absent
    - Stable for multi-word utterances
    - No ASR / No forced alignment
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
        confidence_threshold=0.18,   # absolute confidence gate
        min_frames_ratio=0.25        # duration validation
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)

        self.confidence_threshold = confidence_threshold
        self.min_frames_ratio = min_frames_ratio

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

    @torch.no_grad()
    def infer(self, wav_path, keyword):
        # ---------- Load audio ----------
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

        # ---------- Encode keyword ----------
        kw_ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        if len(kw_ids) == 0:
            return None

        kw = torch.tensor(kw_ids, device=self.device).unsqueeze(0)
        kl = torch.tensor([len(kw_ids)], device=self.device)

        # ---------- Model forward ----------
        probs = torch.sigmoid(self.model(mel, kw, kl))[0].cpu().numpy()

        max_p = probs.max()
        mean_p = probs.mean()

        # 1️⃣ Absolute confidence gate
        if max_p < self.confidence_threshold:
            return None

        # 2️⃣ Relative confidence gate (CRITICAL FIX)
        # Keyword must stand out from background
        if max_p < mean_p * 2.2:
            return None

        # ---------- Duration-based validation ----------
        expected_sec = self.keyword_stats.get(keyword, 0.45)
        expected_frames = int(
            expected_sec * self.sample_rate / self.hop_length
        )

        mask = probs > (0.6 * max_p)
        active_idxs = np.where(mask)[0]

        min_frames = max(3, int(expected_frames * self.min_frames_ratio))
        if len(active_idxs) < min_frames:
            return None

        # ---------- Localization ----------
        center = int(np.mean(active_idxs))

        start = center - expected_frames // 2
        end = center + expected_frames // 2

        start = max(start, 0)
        end = min(end, len(probs) - 1)

        return {
            "start": round(start * self.hop_length / self.sample_rate, 3),
            "end": round(end * self.hop_length / self.sample_rate, 3),
            "confidence": round(float(max_p), 3),
            "start_frame": int(start),
            "end_frame": int(end),
            "num_frames": len(probs),
        }
