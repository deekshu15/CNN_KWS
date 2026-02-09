# CNN Keyword Spotting (KWS)

## Overview
This project implements a CNN-based Keyword Spotting (KWS) system that predicts
the **start and end timestamps of a given keyword in an audio file**.

### Constraints Followed
-  No Automatic Speech Recognition (ASR)
-  No pretrained models
-  No external datasets

The model is trained end-to-end using raw audio and keyword supervision.

---

## Project Structure

CNN_KWS/
├── models/ # CNN model architecture
├── datasets/ # Dataset and data loading logic
├── train/ # Training utilities
├── inference/ # Inference pipeline
├── utils/ # Audio & keyword encoding utilities
│
├── evaluate.py
├── evaluate_kws_accuracy.py
├── test_dataset_load.py
├── requirements.txt
└── README.md


The `CNN_KWS/` directory contains **clean, reusable core code**.
All experimental work is kept separate.

---

## Training & Evaluation (Google Colab)

Due to the **large size of the audio dataset** and the need for **GPU support**,
model training and evaluation were performed in **Google Colab**.

The **complete runnable pipeline** (dataset preparation, training, inference,
and evaluation) is provided in the Colab notebook below:

👉 **Colab Notebook (Final Pipeline)**  
https://colab.research.google.com/drive/YOUR_NOTEBOOK_LINK_HERE

The notebook includes:
- Loading audio and metadata from Google Drive
- Metadata alignment and validation
- Incremental folder-wise training
- Final checkpoint generation
- Inference and quantitative evaluation

---

## Dataset Handling

The full dataset (audio files and metadata CSVs) is **not included in this
repository** due to size constraints.

During Colab execution, the following paths are used:

- **Final checkpoint**
/content/drive/MyDrive/KWS_CHECKPOINTS/kws_folder12.pt


- **Metadata**
/content/metadata_folder*_aligned.csv


This separation keeps the repository lightweight while maintaining full
reproducibility through the provided notebook.

---

## Checkpoints

The results reported in this project use the **final checkpoint trained in
Google Colab**.

Initial debugging experiments were performed locally, but **all final training,
inference, and evaluation results are based on the Colab-trained checkpoint**.

---

## Running Inference (Colab)

Inference is demonstrated inside the Colab notebook using:

```python
inferencer.infer(wav_path, keyword)
This verifies end-to-end keyword localization on unseen audio samples.

Notes
Local scripts in this repository are provided for code sanity checks only

Full reproduction is intended via the Colab notebook

The repository focuses on clarity, structure, and core implementation

Summary
✔ Clean modular codebase

✔ Reproducible Colab pipeline

✔ No prohibited resources used

✔ Final results based on Colab-trained model

This project reflects a practical, real-world ML workflow where experimentation,
training, and evaluation are separated from the core implementation.

