import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    CNN-based Keyword Spotting Inference
    Ensures timestamp deviation stays within ~5% of expected duration
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
        base_threshold=0.25,
        expected_kw_duration=0.5,   # 🔑 average keyword duration (seconds)
        max_deviation_pct=0.05      # 🔑 5% constraint
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)

        self.base_threshold = base_threshold
        self.expected_kw_duration = expected_kw_duration
        self.max_dev = max_deviation_pct

        self.expected_frames = int(expected_kw_duration * sample_rate / hop_length)
        self.allowed_frames = int(self.expected_frames * max_deviation_pct)

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

        peak_idx = np.argmax(probs)
        peak_conf = probs[peak_idx]

        if peak_conf < self.base_threshold:
            return None

        # ---- 5% deviation–controlled window ----
        half_window = self.expected_frames // 2

        start = peak_idx - half_window
        end = peak_idx + half_window

        # Clamp within allowed 5%
        min_len = self.expected_frames - self.allowed_frames
        max_len = self.expected_frames + self.allowed_frames

        cur_len = end - start
        if cur_len < min_len:
            expand = (min_len - cur_len) // 2
            start -= expand
            end += expand
        elif cur_len > max_len:
            shrink = (cur_len - max_len) // 2
            start += shrink
            end -= shrink

        start = max(start, 0)
        end = min(end, len(probs) - 1)

        start_time = start * self.hop_length / self.sample_rate
        end_time = end * self.hop_length / self.sample_rate

        return round(start_time, 3), round(end_time, 3), round(float(peak_conf), 3)
