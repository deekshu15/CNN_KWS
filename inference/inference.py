import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    FINAL LOCKED INFERENCE FILE
    ---------------------------
    CNN-based keyword spotting with temporal localization

    Rules:
    - If keyword is present → return timestamps
    - If keyword is NOT present → return "keyword not found"
    - No ASR
    - No forced alignment
    - Stable for single-word & multi-word audio
    """

    def __init__(
        self,
        checkpoint_path,
        char2idx,
        keyword_stats=None,     # optional (kept for API safety)
        device="cpu",
        sample_rate=16000,
        n_mels=80,
        hop_length=160,
        max_seconds=10.0,
        base_threshold=0.18,   # tuned for your trained model
        min_region_frames=6    # ~60 ms (short words safe)
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)

        self.base_threshold = base_threshold
        self.min_region_frames = min_region_frames

        self.char2idx = char2idx
        self.keyword_stats = keyword_stats or {}

        # Load model
        self.model = KWSModel(len(char2idx)).to(device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        self.model.eval()

        # Mel frontend
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
        ).to(device)

    # ======================================================
    @torch.no_grad()
    def infer(self, wav_path, keyword):
        """
        Infer a single keyword in an audio file.
        Returns:
            dict(start, end, confidence, frames) OR "keyword not found"
        """

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
            return "keyword not found"

        kw = torch.tensor(kw_ids, device=self.device).unsqueeze(0)
        kl = torch.tensor([len(kw_ids)], device=self.device)

        # ---------- Forward ----------
        probs = torch.sigmoid(
            self.model(mel, kw, kl)
        )[0].cpu().numpy()

        # ---------- Local region detection ----------
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

        # ❌ Keyword truly absent
        if len(segments) == 0:
            return "keyword not found"

        # ---------- Choose best segment ----------
        best_seg = max(
            segments,
            key=lambda s: probs[s[0]:s[1]].mean()
        )

        s, e = best_seg
        confidence = float(probs[s:e].max())

        # ---------- Convert to time ----------
        start_time = s * self.hop_length / self.sample_rate
        end_time = e * self.hop_length / self.sample_rate

        return {
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "confidence": round(confidence, 3),
            "start_frame": int(s),
            "end_frame": int(e),
            "num_frames": len(probs)
        }

    # ======================================================
    def infer_sequence(self, wav_path, keywords):
        """
        Infer multiple keywords from the same audio file.
        Ensures order stability and prevents overlap confusion.
        """

        results = {}
        used_until = 0

        for kw in keywords:
            out = self.infer(wav_path, kw)

            # Enforce forward-only progression
            if isinstance(out, dict) and out["start_frame"] < used_until:
                out = "keyword not found"

            results[kw] = out

            if isinstance(out, dict):
                used_until = out["end_frame"]

        return results
