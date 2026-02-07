import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    Keyword Spotting Inference

    Input:
        - Audio file (.wav)
        - Keyword (string)

    Output:
        - (start_time, end_time, confidence)
        - OR None if keyword not detected
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
        smooth_window=5,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)
        self.smooth_window = smooth_window
        self.char2idx = char2idx

        # ---- Load trained model ----
        self.model = KWSModel(len(char2idx)).to(device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        self.model.eval()

        # ---- Mel spectrogram ----
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
        ).to(device)

    # --------------------------------------------------
    # Utility functions
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

        audio_len_samples = wav.shape[0]
        audio_duration = audio_len_samples / self.sample_rate

        wav = wav[: self.max_len]
        if wav.shape[0] < self.max_len:
            wav = torch.nn.functional.pad(
                wav, (0, self.max_len - wav.shape[0])
            )

        return wav.to(self.device), audio_duration

    def _keyword_to_ids(self, keyword):
        ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        if len(ids) == 0:
            raise ValueError("Keyword contains no valid characters")
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)

    # --------------------------------------------------
    # Inference (COMPETITION-GRADE & SAFE)
    # --------------------------------------------------

    def infer(self, wav_path, keyword):
        """
        Returns:
            (start_time, end_time, confidence)
            OR None
        """
        with torch.no_grad():

            # ---- Load audio ----
            wav, audio_duration = self._load_audio(wav_path)
            mel = self.mel(wav).transpose(0, 1).unsqueeze(0)  # [1, T, 80]

            # ---- Keyword encoding ----
            kw = self._keyword_to_ids(keyword)
            kw_len = torch.tensor([kw.shape[1]]).to(self.device)

            # ---- Model forward ----
            logits = self.model(mel, kw, kw_len)
            probs = torch.sigmoid(logits)[0].cpu().numpy()  # [T]

            # ---- Smooth probabilities ----
            if self.smooth_window > 1:
                kernel = np.ones(self.smooth_window) / self.smooth_window
                probs = np.convolve(probs, kernel, mode="same")

            # ---- Candidate peak selection ----
            threshold = 0.15
            candidates = np.where(probs >= threshold)[0]
            if len(candidates) == 0:
                return None

            peak = int(candidates[np.argmax(probs[candidates])])
            peak_val = float(probs[peak])

            # ---- Gradient-based boundary detection ----
            grad = np.gradient(probs)

            left = peak
            while left > 1 and grad[left] > -0.002:
                left -= 1

            right = peak
            while right < len(probs) - 2 and grad[right] < 0.002:
                right += 1

            # ---- Energy-aware expansion ----
            expand = int(0.15 * (right - left + 1))
            left = max(0, left - expand)
            right = min(len(probs) - 1, right + expand)

            # ---- Convert to time ----
            start_time = left * self.hop_length / self.sample_rate
            end_time = right * self.hop_length / self.sample_rate

            # ---- HARD CLAMP (CRITICAL) ----
            start_time = max(0.0, start_time)
            end_time = min(audio_duration, end_time)

            # ---- Sanity check ----
            if end_time <= start_time:
                return None

            return (
                round(start_time, 3),
                round(end_time, 3),
                round(peak_val, 3),
            )
