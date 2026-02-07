import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    CNN-based Keyword Spotting Inference
    Produces NEAR timestamps (not exact) with guaranteed overlap.
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
        tolerance_ratio=0.05,   # 5% relative tolerance
        min_abs_sec=0.10        # 🔑 100 ms minimum tolerance (tuned)
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)

        self.threshold = threshold
        self.smooth_window = smooth_window
        self.tolerance_ratio = tolerance_ratio
        self.min_abs_frames = int(min_abs_sec * sample_rate / hop_length)

        self.char2idx = char2idx

        # ---- Load CNN model ----
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
    # Utilities
    # --------------------------------------------------

    def _load_audio(self, wav_path):
        wav, sr = sf.read(wav_path)
        wav = torch.tensor(wav, dtype=torch.float32)

        if wav.ndim == 2:
            wav = wav.mean(dim=1)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        audio_duration = wav.shape[0] / self.sample_rate

        wav = wav[: self.max_len]
        if wav.shape[0] < self.max_len:
            wav = torch.nn.functional.pad(
                wav, (0, self.max_len - wav.shape[0])
            )

        return wav.to(self.device), audio_duration

    def _keyword_to_ids(self, keyword):
        ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        if len(ids) == 0:
            raise ValueError("Keyword has no valid characters")
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)

    # --------------------------------------------------
    # INFERENCE
    # --------------------------------------------------

    def infer(self, wav_path, keyword):
        """
        Returns:
            (start_time, end_time, confidence) or None
        """

        with torch.no_grad():

            wav, audio_duration = self._load_audio(wav_path)
            mel = self.mel(wav).transpose(0, 1).unsqueeze(0)  # [1, T, 80]

            kw = self._keyword_to_ids(keyword)
            kw_len = torch.tensor([kw.shape[1]]).to(self.device)

            logits = self.model(mel, kw, kw_len)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

            # ---- Smooth ----
            kernel = np.ones(self.smooth_window) / self.smooth_window
            probs = np.convolve(probs, kernel, mode="same")

            peak_idx = int(np.argmax(probs))
            peak_val = probs[peak_idx]

            if peak_val < self.threshold:
                return None

            # ---- Coarse region around peak ----
            active = probs >= (0.5 * self.threshold)

            start = peak_idx
            while start > 0 and active[start]:
                start -= 1

            end = peak_idx
            while end < len(active) - 1 and active[end]:
                end += 1

            # ---- Hybrid tolerance (FINAL TUNED LOGIC) ----
            duration_frames = max(1, end - start)
            rel_margin = int(self.tolerance_ratio * duration_frames)
            margin = max(rel_margin, self.min_abs_frames)

            # 🔑 Speech-aware expansion (biased)
            start = max(0, start - int(0.4 * margin))
            end = min(len(probs) - 1, end + int(0.6 * margin))

            # ---- Convert to time ----
            start_time = start * self.hop_length / self.sample_rate
            end_time = end * self.hop_length / self.sample_rate

            start_time = max(0.0, start_time)
            end_time = min(audio_duration, end_time)

            if end_time <= start_time:
                return None

            return (
                round(start_time, 3),
                round(end_time, 3),
                round(float(peak_val), 3),
            )
