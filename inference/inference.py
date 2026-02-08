import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    FINAL keyword spotting inference
    - Returns timestamps if keyword exists
    - Returns None if keyword not present
    - Correct for single-word and multi-word audio
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
        base_threshold=0.25,
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
    @torch.no_grad()
    def _infer_single(self, mel, keyword, used_mask=None):
        # ---- Encode keyword ----
        kw_ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        if len(kw_ids) == 0:
            return None

        kw = torch.tensor(kw_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        kl = torch.tensor([len(kw_ids)], device=self.device)

        # ---- Forward ----
        probs = torch.sigmoid(self.model(mel, kw, kl))[0].cpu().numpy()

        # ---- Suppress already-used frames (multi-word case) ----
        if used_mask is not None:
            probs = probs * (1.0 - used_mask)

        max_p = probs.max()
        if max_p < self.base_threshold:
            return None

        # ---- Find contiguous regions ----
        thresh = 0.4 * max_p
        active = probs >= thresh

        segments = []
        start = None
        for i, a in enumerate(active):
            if a and start is None:
                start = i
            elif not a and start is not None:
                segments.append((start, i))
                start = None
        if start is not None:
            segments.append((start, len(probs)))

        if not segments:
            return None

        # ---- Select best segment by integrated confidence ----
        best_seg = None
        best_score = -1

        for s, e in segments:
            seg = probs[s:e]
            score = seg.mean() * (e - s)
            if score > best_score:
                best_score = score
                best_seg = (s, e)

        if best_seg is None:
            return None

        s, e = best_seg

        # ---- Keyword duration constraint ----
        dur_sec = self.keyword_stats.get(keyword, 0.45)
        dur_frames = int(dur_sec * self.sample_rate / self.hop_length)

        center = (s + e) // 2
        s = max(center - dur_frames // 2, 0)
        e = min(center + dur_frames // 2, len(probs))

        # ---- Final confidence check (reject hallucinations) ----
        final_conf = probs[s:e].max()
        if final_conf < self.base_threshold:
            return None

        return {
            "start": round(s * self.hop_length / self.sample_rate, 3),
            "end": round(e * self.hop_length / self.sample_rate, 3),
            "confidence": round(float(final_conf), 3),
            "start_frame": s,
            "end_frame": e,
            "num_frames": len(probs),
        }

    # ------------------------------------------------------------
    def infer(self, wav_path, keyword):
        """
        Single (audio, keyword) inference
        """
        wav, sr = sf.read(wav_path)
        wav = torch.tensor(wav, dtype=torch.float32)

        if wav.ndim == 2:
            wav = wav.mean(dim=1)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        wav = wav[:self.max_len]
        if wav.shape[0] < self.max_len:
            wav = torch.nn.functional.pad(wav, (0, self.max_len - wav.shape[0]))

        mel = self.mel(wav.to(self.device)).transpose(0, 1).unsqueeze(0)

        return self._infer_single(mel, keyword)

    # ------------------------------------------------------------
    def infer_sequence(self, wav_path, keywords):
        """
        Multi-keyword inference on same audio
        Ensures keywords do NOT overlap
        """
        wav, sr = sf.read(wav_path)
        wav = torch.tensor(wav, dtype=torch.float32)

        if wav.ndim == 2:
            wav = wav.mean(dim=1)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        wav = wav[:self.max_len]
        if wav.shape[0] < self.max_len:
            wav = torch.nn.functional.pad(wav, (0, self.max_len - wav.shape[0]))

        mel = self.mel(wav.to(self.device)).transpose(0, 1).unsqueeze(0)

        used_mask = np.zeros(mel.shape[1])
        results = {}

        for kw in keywords:
            res = self._infer_single(mel, kw, used_mask)
            if res is not None:
                used_mask[res["start_frame"]:res["end_frame"]] = 1.0
            results[kw] = res

        return results
