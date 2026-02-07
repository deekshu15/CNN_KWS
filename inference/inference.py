import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    FINAL ROBUST CNN-BASED KEYWORD SPOTTING INFERENCE

    Input :
        - wav file
        - keyword (text)

    Output:
        - (start_time, end_time, confidence)
        - None if keyword truly absent

    Design:
        - Peak-centered localization
        - CNN latency correction
        - Soft fallback for weak activations
        - Confidence-aware duration scaling
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
        hard_threshold=0.25,
        soft_threshold=0.15,
        smooth_window=7,
        default_keyword_duration=0.40,  # seconds
        center_shift_sec=0.20,          # CNN latency correction
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)

        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.smooth_window = smooth_window
        self.default_keyword_duration = default_keyword_duration
        self.center_shift_sec = center_shift_sec
        self.char2idx = char2idx

        # ---------------- Load model ----------------
        self.model = KWSModel(len(char2idx)).to(device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        self.model.eval()

        # ---------------- Mel Spectrogram ----------------
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
            or None
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

        # ==================================================
        # Robust peak selection
        # ==================================================

        peak_idx = int(np.argmax(probs))
        peak_val = float(probs[peak_idx])

        if peak_val >= self.hard_threshold:
            center_idx = peak_idx
            confidence = peak_val
        else:
            soft_idxs = np.where(probs >= self.soft_threshold)[0]
            if len(soft_idxs) == 0:
                return None
            center_idx = int(np.mean(soft_idxs))
            confidence = float(np.max(probs[soft_idxs]))

        # ==================================================
        # Peak-centered localization + latency correction
        # ==================================================

        center_time = (
            center_idx * self.hop_length / self.sample_rate
        ) - self.center_shift_sec

        # ==================================================
        # 🔑 CONFIDENCE-AWARE DURATION SCALING (FINAL FIX)
        # ==================================================

        scale = max(1.0, 0.35 / confidence)
        half_dur = (self.default_keyword_duration * scale) / 2.0

        start_time = center_time - half_dur
        end_time = center_time + half_dur

        # ---- Clamp to audio boundaries ----
        start_time = max(0.0, start_time)
        end_time = min(audio_duration, end_time)

        if end_time <= start_time:
            return None

        return (
            round(start_time, 3),
            round(end_time, 3),
            round(confidence, 3),
        )
