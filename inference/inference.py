import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel


class KWSInferencer:
    """
    Speaker-independent CNN-based Keyword Spotting Inference

    Input  : audio (.wav) + keyword (string)
    Output : (start_time_sec, end_time_sec, confidence) OR None
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
        base_threshold=0.18,        # low threshold (CNN outputs are soft)
        dominance_ratio=2.5         # 🔑 keyword presence rule
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_len = int(sample_rate * max_seconds)

        self.base_threshold = base_threshold
        self.dominance_ratio = dominance_ratio

        self.char2idx = char2idx
        self.keyword_stats = keyword_stats

        # ---- Load trained CNN model ----
        self.model = KWSModel(len(char2idx)).to(device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        self.model.eval()

        # ---- Mel frontend ----
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
        ).to(device)

    # ----------------------------------------------------------
    @torch.no_grad()
    def infer(self, wav_path, keyword):
        """
        Returns:
            (start_time_sec, end_time_sec, confidence) OR None
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

        # ---------- Mel spectrogram ----------
        mel = self.mel(wav.to(self.device))
        mel = mel.transpose(0, 1).unsqueeze(0)   # [1, T, 80]

        # ---------- Keyword encoding ----------
        kw_ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        if len(kw_ids) == 0:
            return None

        kw = torch.tensor(
            kw_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        kl = torch.tensor([kw.shape[1]], device=self.device)

        # ---------- Forward ----------
        probs = torch.sigmoid(
            self.model(mel, kw, kl)
        )[0].cpu().numpy()   # [T]

        # ---------- Presence validation ----------
        peak = probs.max()
        mean = probs.mean()

        # Rule 1: minimum confidence
        if peak < self.base_threshold:
            return None

        # Rule 2: dominance over background (CRITICAL FIX)
        if peak / (mean + 1e-6) < self.dominance_ratio:
            return None

        # ---------- Robust center estimation ----------
        threshold = 0.3 * peak
        active = probs >= threshold

        if active.sum() < 3:
            center = int(np.argmax(probs))
        else:
            idxs = np.arange(len(probs))
            center = int((idxs[active] * probs[active]).sum() /
                         probs[active].sum())

        # ---------- Duration estimation ----------
        dur_sec = self.keyword_stats.get(keyword, 0.45)
        dur_frames = int(dur_sec * self.sample_rate / self.hop_length)

        start = center - dur_frames // 2
        end = center + dur_frames // 2

        start = max(0, start)
        end = min(len(probs) - 1, end)

        start_time = start * self.hop_length / self.sample_rate
        end_time = end * self.hop_length / self.sample_rate

        return (
            round(float(start_time), 3),
            round(float(end_time), 3),
            round(float(peak), 3)
        )
