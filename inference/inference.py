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
        base_threshold=0.15,   # soft confidence gate
        peak_ratio=0.4         # relative peak threshold
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)

        self.base_threshold = base_threshold
        self.peak_ratio = peak_ratio

        self.char2idx = char2idx

        # ---- Load model ----
        self.model = KWSModel(len(char2idx)).to(device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        self.model.eval()

        # ---- Mel extractor ----
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
        ).to(device)

    # --------------------------------------------------

    @torch.no_grad()
    def infer(self, wav_path, keyword):
        """
        Returns:
            (start_time_sec, end_time_sec, confidence) or None
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

        # ---------- Mel ----------
        mel = self.mel(wav.to(self.device))          # [80, T]
        mel = mel.transpose(0, 1).unsqueeze(0)       # [1, T, 80]

        # ---------- Keyword encoding ----------
        kw_ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        if len(kw_ids) == 0:
            return None

        kw = torch.tensor(
            kw_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        kl = torch.tensor([len(kw_ids)], device=self.device)

        # ---------- Forward ----------
        logits = self.model(mel, kw, kl)             # [1, T]
        probs = torch.sigmoid(logits)[0].cpu().numpy()

        max_p = float(probs.max())
        if max_p < self.base_threshold:
            return None

        # ---------- Find candidate regions ----------
        thr = self.peak_ratio * max_p
        mask = probs >= thr

        regions = []
        start = None

        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                end = i - 1
                conf = float(probs[start:end + 1].max())
                regions.append((start, end, conf))
                start = None

        if start is not None:
            end = len(mask) - 1
            conf = float(probs[start:end + 1].max())
            regions.append((start, end, conf))

        if len(regions) == 0:
            return None

        # ==========================================================
        # 🔑 CRITICAL FIX FOR MULTI-WORD AUDIO
        # Select the EARLIEST plausible region (not the strongest)
        # ==========================================================
        regions.sort(key=lambda r: r[0])  # sort by start frame

        chosen = None
        for s, e, conf in regions:
            if conf >= self.base_threshold:
                chosen = (s, e, conf)
                break

        if chosen is None:
            return None

        # ---------- Convert frames → time ----------
        s, e, conf = chosen
        start_time = s * self.hop_length / self.sample_rate
        end_time = e * self.hop_length / self.sample_rate

        return (
            round(float(start_time), 3),
            round(float(end_time), 3),
            round(float(conf), 3)
        )
