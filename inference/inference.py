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
        base_threshold=0.25,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)
        self.base_threshold = base_threshold

        # --- Adaptive duration thresholds ---
        self.min_frames_default = int(0.20 * sample_rate / hop_length)  # ~200 ms
        self.min_frames_short   = int(0.06 * sample_rate / hop_length)  # ~60 ms

        self.char2idx = char2idx

        # ---- Load model ----
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

    # ==========================================================
    @torch.no_grad()
    def infer(self, wav_path, keyword, used_mask=None):
        """
        Infer a single keyword.
        Returns dict or None if keyword not present.
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
        T = mel.shape[1]

        # ---------- Encode keyword ----------
        kw_ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        if len(kw_ids) == 0:
            return None

        kw = torch.tensor(kw_ids, device=self.device).unsqueeze(0)
        kl = torch.tensor([len(kw_ids)], device=self.device)

        # ---------- Forward ----------
        probs = torch.sigmoid(self.model(mel, kw, kl))[0].cpu().numpy()

        if used_mask is not None:
            probs = probs * (1.0 - used_mask[:len(probs)])

        # ======================================================
        # 🔑 TEMPORAL PERSISTENCE VALIDATION
        # ======================================================
        above = probs > self.base_threshold

        runs = []
        current = 0
        for v in above:
            if v:
                current += 1
            else:
                if current > 0:
                    runs.append(current)
                current = 0
        if current > 0:
            runs.append(current)

        # Adaptive min duration
        min_req = self.min_frames_default if used_mask is None else self.min_frames_short

        if len(runs) == 0 or max(runs) < min_req:
            return None

        # ======================================================
        # ✔ KEYWORD PRESENT → LOCALIZE
        # ======================================================
        active_idxs = np.where(above)[0]

        start_f = int(active_idxs[0])
        end_f   = int(active_idxs[-1])

        # Hard clip to audio range
        start_f = max(start_f, 0)
        end_f   = min(end_f, T - 1)

        if end_f <= start_f:
            return None

        start_time = start_f * self.hop_length / self.sample_rate
        end_time   = end_f * self.hop_length / self.sample_rate

        return {
            "start": round(float(start_time), 3),
            "end": round(float(end_time), 3),
            "confidence": round(float(probs.max()), 3),
            "start_frame": start_f,
            "end_frame": end_f,
            "num_frames": T,
        }

    # ==========================================================
    def infer_sequence(self, wav_path, keywords):
        """
        Infer multiple keywords in sequence (same audio).
        Prevents overlap and false positives.
        """

        results = {}
        used_mask = None

        for kw in keywords:
            out = self.infer(wav_path, kw, used_mask)
            results[kw] = out

            if out is not None:
                if used_mask is None:
                    used_mask = np.zeros(out["num_frames"], dtype=np.float32)

                used_mask[out["start_frame"]:out["end_frame"] + 1] = 1.0

        return results
