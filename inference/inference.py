import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    CNN-based Keyword Spotting Inference (FINAL)

    Strategy:
    - Detect peak activation (CNN)
    - Use PEAK-CENTERED localization
    - Apply FIXED speech-safe duration
    - Clamp to audio boundaries

    Output:
    - Near timestamps (not exact)
    - Stable and evaluation-friendly
    """

    def __init__(
        self,
        checkpoint_path,
        char2idx,
        device="cpu",
        sample_rate=16000,
        n_mels=80,
        hop_length=160,
        max_seconds=10.0,
        threshold=0.25,
        smooth_window=7,
        default_keyword_duration=0.40,  # 🔑 seconds (speech-safe)
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)

        self.threshold = threshold
        self.smooth_window = smooth_window
        self.default_keyword_duration = default_keyword_duration
        self.char2idx = char2idx

        # ---------------- Load model ----------------
        self.model = KWSModel(len(char2idx)).to(device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        self.model.eval()

        # ---------------- Feature extractor ----------------
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
        ).to(device)

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def _load_audio(self, wav_path):
        wav, sr = sf.read(wav_path)
        wav = torch.tensor(wav, dtype=torch.float32)

        if wav.ndim == 2:
            wav = wav.mean(dim=1)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(
                wav, sr, self.sample_rate
            )

        audio_duration = wav.shape[0] / self.sample_rate

        wav = wav[: self.max_len]
        if wav.shape[0] < self.max_len:
            wav = torch.nn.functional.pad(
                wav, (0, self.max_len - wav.shape[0])
            )

        return wav.to(self.device), audio_duration

    def _keyword_to_tensor(self, keyword):
        ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        if len(ids) == 0:
            raise ValueError("Keyword contains no valid characters")

        kw = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        kw_len = torch.tensor([len(ids)])
        return kw.to(self.device), kw_len.to(self.device)

    # --------------------------------------------------
    # INFERENCE
    # --------------------------------------------------

    @torch.no_grad()
    def infer(self, wav_path, keyword):
        """
        Returns:
            (start_time, end_time, confidence)
            OR None if keyword not detected
        """

        # ---- Load audio ----
        wav, audio_duration = self._load_audio(wav_path)
        mel = self.mel(wav).transpose(0, 1).unsqueeze(0)  # [1, T, 80]

        # ---- Keyword ----
        kw, kw_len = self._keyword_to_tensor(keyword)

        # ---- Forward ----
        logits = self.model(mel, kw, kw_len)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

        # ---- Smooth probabilities ----
        kernel = np.ones(self.smooth_window) / self.smooth_window
        probs = np.convolve(probs, kernel, mode="same")

        peak_idx = int(np.argmax(probs))
        peak_val = float(probs[peak_idx])

        if peak_val < self.threshold:
            return None

        # ==================================================
        # 🔑 PEAK-CENTERED TIMESTAMPING (CORE FIX)
        # ==================================================

        center_time = peak_idx * self.hop_length / self.sample_rate

        half_duration = self.default_keyword_duration / 2.0

        start_time = center_time - half_duration
        end_time = center_time + half_duration

        # ---- Clamp to audio ----
        start_time = max(0.0, start_time)
        end_time = min(audio_duration, end_time)

        if end_time <= start_time:
            return None

        return (
            round(start_time, 3),
            round(end_time, 3),
            round(peak_val, 3),
        )
