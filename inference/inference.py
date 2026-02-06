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
        threshold=0.5,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.threshold = threshold
        self.max_len = int(sample_rate * max_seconds)

        # model
        self.model = KWSModel(len(char2idx)).to(device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        self.model.eval()

        self.char2idx = char2idx

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
        ).to(device)

    def _load_audio(self, wav_path):
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

        return wav.to(self.device)

    def _keyword_to_ids(self, keyword):
        ids = [self.char2idx[c] for c in keyword if c in self.char2idx]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def infer(self, wav_path, keyword):
        """
        Returns:
            (start_time, end_time) in seconds
            or None if keyword not detected
        """
        wav = self._load_audio(wav_path)
        mel = self.mel(wav).transpose(0, 1).unsqueeze(0)  # [1, T, 80]

        kw = self._keyword_to_ids(keyword)
        kw_len = torch.tensor([kw.shape[1]]).to(self.device)

        logits = self.model(mel, kw, kw_len)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

        # thresholding
        active = probs > self.threshold
        if not active.any():
            return None

        indices = np.where(active)[0]
        start_frame = indices[0]
        end_frame = indices[-1]

        start_time = start_frame * self.hop_length / self.sample_rate
        end_time = end_frame * self.hop_length / self.sample_rate

        return round(start_time, 3), round(end_time, 3)
