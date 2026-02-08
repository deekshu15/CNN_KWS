import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    CNN-based Keyword Spotting Inference
    - Works for single-word and multi-word audio
    - No ASR
    - No forced alignment
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
        base_threshold=0.18,   # 🔑 IMPORTANT: lowered
        min_region_frames=6    # ~60 ms
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)

        self.base_threshold = base_threshold
        self.min_region_frames = min_region_frames

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
        kw = torch.tensor(
            [self.char2idx[c] for c in keyword if c in self.char2idx],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        if kw.shape[1] == 0:
            return "keyword not found"

        kl = torch.tensor([kw.shape[1]], device=self.device)

        # ---------- Forward ----------
        probs = torch.sigmoid(
            self.model(mel, kw, kl)
        )[0].cpu().numpy()

        # ---------- LOCAL REGION DETECTION ----------
        mask = probs >= self.base_threshold

        segments = []
        start = None

        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                if i - start >= self.min_region_frames:
                    segments.append((start, i))
                start = None

        if start is not None and len(mask) - start >= self.min_region_frames:
            segments.append((start, len(mask)))

        # ❌ No valid region → keyword truly absent
        if len(segments) == 0:
            return "keyword not found"

        # ---------- Choose best segment ----------
        best_seg = max(
            segments,
            key=lambda s: probs[s[0]:s[1]].mean()
        )

        s, e = best_seg
        confidence = float(probs[s:e].max())

        start_time = s * self.hop_length / self.sample_rate
        end_time = e * self.hop_length / self.sample_rate

        return {
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "confidence": round(confidence, 3),
            "start_frame": s,
            "end_frame": e,
            "num_frames": len(probs)
        }
