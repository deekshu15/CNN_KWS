# CNN Keyword Spotting (KWS)

## Problem Statement
Build a Keyword Spotting (KWS) system that predicts the start and end
timestamps of a given keyword in an audio file without using ASR,
pretrained models, or external datasets.

## Project Structure
CNN_KWS_PROJECT/
├── CNN_KWS/          # Core implementation (dataset, model, training, inference)
├── checkpoints/      # Trained model checkpoints
├── data/             # (Not included) Large datasets used in Colab
├── evaluate.py       # Evaluation script
├── requirements.txt  # Dependencies

## Dataset Handling
Due to the large size of the audio dataset, all audio files and metadata
are stored in Google Drive and used in Google Colab for training and
evaluation. Only sample outputs and trained checkpoints are included
in this repository.

## How to Run Inference
```python
from CNN_KWS.inference.inference import KWSInferencer
from CNN_KWS.utils.keyword_encoder import keyword_stats
from CNN_KWS.utils.audio_encoder import char2idx

inferencer = KWSInferencer(
    checkpoint_path="checkpoints/kws_folder12.pt",
    char2idx=char2idx,
    keyword_stats=keyword_stats,
    device="cpu"
)

inferencer.infer("sample.wav", "FRIGHTEN")
