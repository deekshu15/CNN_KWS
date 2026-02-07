import torch
import torchaudio
import soundfile as sf
import numpy as np

from CNN_KWS.models.kws_model import KWSModel
@torch.no_grad()
def infer(self, wav_path, keyword):
    """
    Returns:
        start_time (sec),
        end_time (sec),
        confidence (0–1)
    """

    wav = self._load_audio(wav_path)
    mel = self.mel(wav).transpose(0, 1).unsqueeze(0)  # [1, T, 80]

    kw = self._keyword_to_ids(keyword)
    kw_len = torch.tensor([kw.shape[1]]).to(self.device)

    logits = self.model(mel, kw, kw_len)
    probs = torch.sigmoid(logits)[0].cpu().numpy()  # [T]

    # ---- Smooth probabilities ----
    if self.smooth_window > 1:
        kernel = np.ones(self.smooth_window) / self.smooth_window
        probs = np.convolve(probs, kernel, mode="same")

    # ---- Find strongest region (ALWAYS) ----
    max_idx = np.argmax(probs)
    max_prob = probs[max_idx]

    # Expand left
    left = max_idx
    while left > 0 and probs[left] > max_prob * 0.5:
        left -= 1

    # Expand right
    right = max_idx
    while right < len(probs) - 1 and probs[right] > max_prob * 0.5:
        right += 1

    start_time = left * self.hop_length / self.sample_rate
    end_time = right * self.hop_length / self.sample_rate

    return round(start_time, 3), round(end_time, 3), round(float(max_prob), 3)
