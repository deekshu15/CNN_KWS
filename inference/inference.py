import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:

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

        wav = wav[: self.max_len]
        if wav.shape[0] < self.max_len:
            wav = torch.nn.functional.pad(
                wav, (0, self.max_len - wav.shape[0])
            )

        return wav.to(self.device)

    def _keyword_to_ids(self, keyword):
        ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        if len(ids) == 0:
            raise ValueError("Keyword contains no valid characters")
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)

    # --------------------------------------------------
    # Inference (Competition-grade)
    # --------------------------------------------------

    def infer(self, wav_path, keyword):
        """
        Returns:
            (start_time, end_time, confidence)
        """
        with torch.no_grad():

            # ---- Load & preprocess audio ----
            wav = self._load_audio(wav_path)
            mel = self.mel(wav).transpose(0, 1).unsqueeze(0)  # [1, T, 80]

            # ---- Encode keyword ----
            kw = self._keyword_to_ids(keyword)
            kw_len = torch.tensor([kw.shape[1]]).to(self.device)

            # ---- Forward pass ----
            logits = self.model(mel, kw, kw_len)
            probs = torch.sigmoid(logits)[0].cpu().numpy()  # [T]

            # ---- Smooth probabilities ----
            if self.smooth_window > 1:
                kernel = np.ones(self.smooth_window) / self.smooth_window
                probs = np.convolve(probs, kernel, mode="same")

            # ---- Peak detection ----
            peak = int(np.argmax(probs))
            peak_val = float(probs[peak])

            # ---- Gradient-based boundary detection ----
            grad = np.gradient(probs)

            left = peak
            while left > 1 and grad[left] > -0.002:
                left -= 1

            right = peak
            while right < len(probs) - 2 and grad[right] < 0.002:
                right += 1

            # ---- Convert to time ----
            start_time = left * self.hop_length / self.sample_rate
            end_time = right * self.hop_length / self.sample_rate

            return (
                round(start_time, 3),
                round(end_time, 3),
                round(peak_val, 3),
            )
