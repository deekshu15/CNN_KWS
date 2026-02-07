import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    CNN-based Keyword Spotting Inference

    Input:
        - Audio file (.wav)
        - Keyword (string)

    Output:
        - List of (start_time, end_time, confidence)
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
        on_threshold=0.30,        # start detection
        off_threshold=0.15,       # stop detection (hysteresis)
        min_duration_sec=0.25,    # 🔑 tail-hold (minimum keyword duration)
        smooth_window=5,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)

        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.smooth_window = smooth_window

        # Minimum duration in frames
        self.min_duration_frames = int(
            min_duration_sec * sample_rate / hop_length
        )

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

    def _keyword_to_ids(self, keyword):
        ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        if len(ids) == 0:
            raise ValueError("Keyword has no valid characters")
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)

    # --------------------------------------------------
    # CNN → FRAME SCORES → TIMESTAMPS
    # --------------------------------------------------

    def infer(self, wav_path, keyword):
        """
        Returns:
            List of (start_time, end_time, confidence)
        """

        with torch.no_grad():

            # ---- Load audio ----
            wav, audio_duration = self._load_audio(wav_path)
            mel = self.mel(wav).transpose(0, 1).unsqueeze(0)  # [1, T, 80]

            # ---- Encode keyword ----
            kw = self._keyword_to_ids(keyword)
            kw_len = torch.tensor([kw.shape[1]]).to(self.device)

            # ---- CNN forward (frame-wise scores) ----
            logits = self.model(mel, kw, kw_len)
            probs = torch.sigmoid(logits)[0].cpu().numpy()  # [T]

            # ---- Smooth probabilities ----
            if self.smooth_window > 1:
                kernel = np.ones(self.smooth_window) / self.smooth_window
                probs = np.convolve(probs, kernel, mode="same")

            # ---- HYSTERESIS THRESHOLDING ----
            active = np.zeros_like(probs, dtype=bool)

            inside = False
            for i, p in enumerate(probs):
                if not inside and p >= self.on_threshold:
                    inside = True
                elif inside and p < self.off_threshold:
                    inside = False
                active[i] = inside

            # ---- SEGMENT EXTRACTION WITH TAIL-HOLD ----
            segments = []
            start = None
            frames_since_start = 0

            for i, val in enumerate(active):
                if val and start is None:
                    start = i
                    frames_since_start = 0

                elif start is not None:
                    frames_since_start += 1

                    # Allow ending only after minimum duration
                    if not val and frames_since_start >= self.min_duration_frames:
                        segments.append((start, i - 1))
                        start = None
                        frames_since_start = 0

            if start is not None:
                segments.append((start, len(active) - 1))

            # ---- Convert segments → timestamps ----
            results = []

            for s, e in segments:
                start_time = s * self.hop_length / self.sample_rate
                end_time = e * self.hop_length / self.sample_rate

                # Clamp to audio duration
                start_time = max(0.0, start_time)
                end_time = min(audio_duration, end_time)

                if end_time <= start_time:
                    continue

                confidence = float(probs[s:e + 1].max())

                results.append((
                    round(start_time, 3),
                    round(end_time, 3),
                    round(confidence, 3)
                ))

            return results
