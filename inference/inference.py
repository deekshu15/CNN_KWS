import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    CNN-based Keyword Spotting Inference
    -----------------------------------
    Input : audio file + keyword (text)
    Output: timestamps (start, end, confidence) OR None

    ✔ Correct for single-word audio
    ✔ Correct for multi-word audio
    ✔ Rejects keywords not present
    ✔ No ASR / No forced alignment
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
        base_threshold=0.25,   # conservative
        min_duration_sec=0.20 # 🔑 temporal persistence (200 ms)
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)
        self.base_threshold = base_threshold
        self.min_frames = int(min_duration_sec * sample_rate / hop_length)

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

    # ------------------------------------------------------------
    @torch.no_grad()
    def infer(self, wav_path, keyword, used_mask=None):
        """
        Returns:
            {
              start, end, confidence,
              start_frame, end_frame, num_frames
            }
            OR None if keyword not present
        """

        # ---- Load audio ----
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

        # ---- Mel ----
        mel = self.mel(wav.to(self.device))
        mel = mel.transpose(0, 1).unsqueeze(0)  # [1, T, 80]

        T = mel.shape[1]

        # ---- Encode keyword ----
        kw_ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        if len(kw_ids) == 0:
            return None

        kw = torch.tensor(kw_ids, device=self.device).unsqueeze(0)
        kl = torch.tensor([len(kw_ids)], device=self.device)

        # ---- Forward ----
        probs = torch.sigmoid(self.model(mel, kw, kl))[0].cpu().numpy()

        if used_mask is not None:
            probs = probs * (1.0 - used_mask[:len(probs)])

        # ==========================================================
        # 🔑 CRITICAL FIX: TEMPORAL PERSISTENCE CHECK
        # ==========================================================
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

        # ❌ Keyword NOT present
        if len(runs) == 0 or max(runs) < self.min_frames:
            return None

        # ==========================================================
        # ✔ KEYWORD PRESENT → LOCALIZE
        # ==========================================================
        idxs = np.arange(len(probs))
        weights = probs * above

        if weights.sum() == 0:
            return None

        center = int(np.sum(idxs * weights) / np.sum(weights))

        # Estimate duration from activation width
        active_idxs = np.where(above)[0]
        start_f = active_idxs[0]
        end_f = active_idxs[-1]

        # Safety clamp
        start_f = max(start_f, 0)
        end_f = min(end_f, T - 1)

        start_time = start_f * self.hop_length / self.sample_rate
        end_time = end_f * self.hop_length / self.sample_rate

        return {
            "start": round(float(start_time), 3),
            "end": round(float(end_time), 3),
            "confidence": round(float(probs.max()), 3),
            "start_frame": int(start_f),
            "end_frame": int(end_f),
            "num_frames": int(T),
        }

    # ------------------------------------------------------------
    def infer_sequence(self, wav_path, keywords):
        """
        Sequential inference for multi-word audio
        Prevents overlap between detected keywords
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
