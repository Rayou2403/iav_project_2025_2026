# Speech Emotion Recognition (SER)
Advanced AI – Final Project  
Authors: Rayane Belkalem, Kenza Briber, Falamoudou Koné

---

## 1. Overview
Speech Emotion Recognition (SER) aims to automatically identify human emotions from audio recordings.  
In this project, we compare different audio representations and neural architectures using classical signal processing (Librosa) and deep learning models implemented in PyTorch.

The objective is to evaluate how well different feature types (Mel-spectrograms vs MFCCs) and models (CNN vs CNN+LSTM) perform on emotion classification.

---

## 2. Project Objectives
- Preprocess raw audio (silence removal, resampling, normalization).
- Extract features using Librosa:
  - Mel-spectrograms
  - MFCCs
- Train and compare two architectures:
  - 2D CNN on Mel-spectrograms (baseline model)
  - CNN + LSTM on MFCC sequences
- Evaluate performance using accuracy, F1-score and confusion matrices.
- Analyse which emotions are most often confused.

---

## 3. Dataset
We use processed versions of the CREMA-D dataset (and optionally RAVDESS).  

### Notes
- MFCC sequences have variable length : padding required for batch training.
- Mel-spectrograms behave like images : no padding needed.

