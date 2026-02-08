import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    CNN-based Keyword Spotting
    - Single keyword: unchanged logic
    - Multi keyword : monotonic temporal decoding (professional fix)
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
        base_threshold=0.15
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)
        self.base_threshold = base_threshold

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

    # ------------------------------------------------------------
    # 🔹 SINGLE KEYWORD INFERENCE (UNCHANGED)
    # ------------------------------------------------------------
    @torch.no_grad()
    def infer(self, wav_path, keyword, start_frame=0):
        """
        Infer a single keyword starting from start_frame
        """
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

        mel = self.mel(wav.to(self.device)).transpose(0, 1)

        # Restrict search region (THIS is the multi-word fix)
        mel = mel[start_frame:].unsqueeze(0)

        kw = torch.tensor(
            [self.char2idx[c] for c in keyword if c in self.char2idx],
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        if kw.shape[1] == 0:
            return None

        kl = torch.tensor([kw.shape[1]], device=self.device)

        probs = torch.sigmoid(
            self.model(mel, kw, kl)
        )[0].cpu().numpy()

        max_p = probs.max()
        if max_p < self.base_threshold:
            return None

        # Dynamic threshold
        thr = max(self.base_threshold, 0.4 * max_p)
        active = probs >= thr

        if not active.any():
            return None

        # Extract longest active region
        regions = []
        s = None
        for i, v in enumerate(active):
            if v and s is None:
                s = i
            elif not v and s is not None:
                regions.append((s, i))
                s = None
        if s is not None:
            regions.append((s, len(active)))

        start_f, end_f = max(regions, key=lambda x: x[1] - x[0])

        # Convert back to global frame index
        start_f += start_frame
        end_f += start_frame

        start_time = start_f * self.hop_length / self.sample_rate
        end_time = end_f * self.hop_length / self.sample_rate

        return {
            "start": round(start_time, 3),
            "end": round(end_time, 3),
            "confidence": round(float(max_p), 3),
            "start_frame": start_f,
            "end_frame": end_f,
        }

    # ------------------------------------------------------------
    # 🔹 MULTI KEYWORD INFERENCE (FINAL FIX)
    # ------------------------------------------------------------
    def infer_sequence(self, wav_path, keywords):
        """
        Monotonic multi-keyword inference
        Enforces temporal order WITHOUT changing single-word logic
        """
        results = {}
        cursor = 0  # frame index

        for kw in keywords:
            out = self.infer(wav_path, kw, start_frame=cursor)

            if out is None:
                results[kw] = None
                continue

            results[kw] = (
                out["start"],
                out["end"],
                out["confidence"]
            )

            # Move cursor forward (KEY FIX)
            cursor = out["end_frame"]

        return results
