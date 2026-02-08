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
        contrast_k=2.0,          # 🔑 contrast strength
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)
        self.base_threshold = base_threshold
        self.contrast_k = contrast_k

        self.min_frames_default = int(0.20 * sample_rate / hop_length)
        self.min_frames_short   = int(0.06 * sample_rate / hop_length)

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

    # ---------------------------------------------------------
    @torch.no_grad()
    def infer(self, wav_path, keyword, used_mask=None):
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
        T = mel.shape[1]

        # ---- Encode keyword ----
        kw_ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        if len(kw_ids) == 0:
            return "keyword not found"

        kw = torch.tensor(kw_ids, device=self.device).unsqueeze(0)
        kl = torch.tensor([len(kw_ids)], device=self.device)

        probs = torch.sigmoid(self.model(mel, kw, kl))[0].cpu().numpy()

        if used_mask is not None:
            probs = probs * (1.0 - used_mask[:len(probs)])

        # ======================================================
        # 1️⃣ Presence validation (STRICT, NO HALLUCINATION)
        # ======================================================
        max_p = probs.max()
        mean_p = probs.mean()
        std_p = probs.std() + 1e-6

        # Absolute + relative confidence check
        if max_p < self.base_threshold or max_p < mean_p + self.contrast_k * std_p:
            return "keyword not found"

        # Temporal persistence
        above = probs > self.base_threshold
        runs, current = [], 0
        for v in above:
            if v:
                current += 1
            else:
                if current > 0:
                    runs.append(current)
                current = 0
        if current > 0:
            runs.append(current)

        min_req = self.min_frames_default if used_mask is None else self.min_frames_short

        if len(runs) == 0 or max(runs) < min_req:
            return "keyword not found"

        # ======================================================
        # 2️⃣ Localization
        # ======================================================
        active = np.where(above)[0]
        start_f, end_f = int(active[0]), int(active[-1])

        start_f = max(0, start_f)
        end_f = min(T - 1, end_f)

        start_time = start_f * self.hop_length / self.sample_rate
        end_time   = end_f * self.hop_length / self.sample_rate

        return {
            "start": round(float(start_time), 3),
            "end": round(float(end_time), 3),
            "confidence": round(float(max_p), 3),
            "start_frame": start_f,
            "end_frame": end_f,
            "num_frames": T,
        }

    # ---------------------------------------------------------
    def infer_sequence(self, wav_path, keywords):
        results = {}
        used_mask = None

        for kw in keywords:
            out = self.infer(wav_path, kw, used_mask)
            results[kw] = out

            if isinstance(out, dict):
                if used_mask is None:
                    used_mask = np.zeros(out["num_frames"], dtype=np.float32)
                used_mask[out["start_frame"]:out["end_frame"] + 1] = 1.0

        return results
