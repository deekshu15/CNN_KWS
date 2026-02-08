import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    CNN-based Keyword Spotting Inference
    Input  : audio file + keyword
    Output : timestamps within audio range
    """

    def __init__(
        self,
        checkpoint_path,
        char2idx,
        keyword_stats,              # dict: keyword -> median duration (sec)
        device="cpu",
        sample_rate=16000,
        n_mels=80,
        hop_length=160,
        max_seconds=10.0,
        base_threshold=0.25
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)
        self.base_threshold = base_threshold

        self.char2idx = char2idx
        self.keyword_stats = keyword_stats

        # ---- Model ----
        self.model = KWSModel(len(char2idx)).to(device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        self.model.eval()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length
        ).to(device)

    # =====================================================
    @torch.no_grad()
    def infer(self, wav_path, keyword, cursor_frame=0):
        """
        Single keyword inference with cursor (for multi-word audio)
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

        mel = self.mel(wav.to(self.device)).transpose(0, 1)
        mel = mel.unsqueeze(0)  # [1, T, 80]

        # ---------- Encode keyword ----------
        kw_ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        if len(kw_ids) == 0:
            return None

        kw = torch.tensor(kw_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        kl = torch.tensor([kw.shape[1]], device=self.device)

        # ---------- Forward ----------
        probs = torch.sigmoid(self.model(mel, kw, kl))[0].cpu().numpy()
        T = len(probs)

        # ---------- Cursor masking (CRITICAL FIX) ----------
        probs[:cursor_frame] = 0.0

        max_p = probs.max()
        if max_p < self.base_threshold:
            return None

        # ---------- Confidence-weighted center ----------
        idxs = np.arange(T)
        weights = probs * (probs > 0.4 * max_p)

        if weights.sum() == 0:
            weights = probs

        center = int(np.sum(idxs * weights) / np.sum(weights))

        # ---------- Duration (keyword-adaptive) ----------
        dur_sec = self.keyword_stats.get(keyword, 0.45)
        dur_frames = int(dur_sec * self.sample_rate / self.hop_length)

        start_frame = center - dur_frames // 2
        end_frame   = center + dur_frames // 2

        # ---------- HARD CLIP TO AUDIO RANGE ----------
        start_frame = max(0, start_frame)
        end_frame   = min(end_frame, T - 1)

        if end_frame <= start_frame:
            return None

        start_time = start_frame * self.hop_length / self.sample_rate
        end_time   = end_frame   * self.hop_length / self.sample_rate

        return {
            "start": round(float(start_time), 3),
            "end": round(float(end_time), 3),
            "confidence": round(float(max_p), 3),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "num_frames": T
        }

    # =====================================================
    @torch.no_grad()
    def infer_sequence(self, wav_path, keywords):
        """
        Multi-keyword inference for the SAME audio
        Keywords are assumed in spoken order
        """

        results = {}
        cursor = 0

        for kw in keywords:
            out = self.infer(wav_path, kw, cursor_frame=cursor)

            if out is None:
                results[kw] = None
                continue

            results[kw] = out

            # --------- Cursor advance (KEY FIX) ---------
            duration = out["end_frame"] - out["start_frame"]
            cursor = out["start_frame"] + int(0.6 * duration)

        return results
