# Sign Language Recognition
### Real-Time ASL Fingerspelling Recognition using MediaPipe + PyTorch

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.x-green.svg)](https://mediapipe.dev)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.74%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

A real-time American Sign Language (ASL) fingerspelling recognition system that converts hand gestures into text and speech. Built as a B.Tech final year project at ABES Engineering College, Ghaziabad.

---

## How It Works

```
Webcam Frame
     │
     ▼
MediaPipe Hand Landmarker
(21 × 3D keypoints)
     │
     ▼
Wrist Normalisation
(63-dim feature vector)
     │
     ▼
MLP Classifier
(256 → 128 → 64 → 26)
     │
     ▼
7-frame Majority Vote
     │
     ▼
Dwell Confirmation (2s hold)
     │
     ▼
Word Builder → Text-to-Speech
```

Rather than classifying raw images, the system extracts 21 hand landmarks from each frame using MediaPipe's pre-trained Hand Landmarker model. These landmarks are wrist-normalised to produce a 63-dimensional feature vector invariant to hand position and scale. A lightweight MLP then classifies this vector into one of 26 ASL letters.

**Why landmarks instead of images?**
Hand geometry — not image texture or colour — is the primary discriminative feature for sign language gestures. This approach achieves 98.74% accuracy with ~800 training images per class, where a CNN would need 5-10x more data and a GPU.

---

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | **98.74%** |
| Macro F1-Score | **0.99** |
| Model Parameters | **60,122** |
| Inference Speed | **~25 FPS (CPU)** |
| Training Samples | **16,796** |
| Classes | **26 (A–Z)** |

Weakest classes: **M** (93% F1) and **N** (89% F1) — expected, as these differ only in finger count over the thumb and are visually similar as static poses.

---

## Features

- Real-time inference at ~25 FPS on laptop CPU — no GPU required
- Full 26-letter ASL alphabet recognition
- Word builder — hold any pose for 2 seconds to confirm a letter
- Text-to-speech — press Enter to speak the sentence aloud
- Dwell confirmation with visual progress bar — prevents accidental triggers
- 7-frame smoothing via majority vote — eliminates prediction flickering
- Professional overlay — hand skeleton, confidence bar, FPS counter

---

## Project Structure

```
sign-language-recognition/
├── src/extract_landmarks.py   # Extract MediaPipe landmarks from dataset
├── src/train.py               # Train MLP classifier on landmark features
├── src/realtime_inference_O.py            # Real-time webcam inference + word builder (.onnx version)
├── src/realtime_inference_P.py            # Real-time webcam inference + word builder (.pth version)
├── model-files/                # Trained model and scaler files
└── README.md
```

---

## Quick Start

### Prerequisites

```bash
pip install mediapipe opencv-python torch scikit-learn joblib numpy pyttsx3
```

### Step 1 — Extract Landmarks (Kaggle)

1. Add dataset: [grassknoted/asl-alphabet](https://www.kaggle.com/grassknoted/asl-alphabet)
2. Run `src/extract_landmarks.py` on Kaggle
3. Output: `landmarks.npz`

### Step 2 — Train Model (Kaggle)

```bash
python src/train.py
```

Output: `model.pth`, `scaler.pkl`, `label_classes.npy`

Download these three files to your local machine.

### Step 3 — Run Real-Time Demo (Local)

```bash
python src/realtime_inference.py
```

The hand landmarker model (~8MB) downloads automatically on first run.

---

## Controls

| Input | Action |
|---|---|
| Hold pose for 2 seconds | Confirm letter into current word |
| `SPACE` | Add current word to sentence |
| `ENTER` | Speak full sentence aloud (TTS) |
| `BACKSPACE` | Delete last letter |
| `C` | Clear everything |
| `S` | Save screenshot |
| `R` | Toggle prediction smoothing |
| `Q` / `ESC` | Quit |

---

## Architecture

### Feature Extraction
- **Model:** Google MediaPipe Hand Landmarker (float16)
- **Output:** 21 landmarks × (x, y, z) = 63 coordinates
- **Normalisation:** Wrist-relative (subtract landmark 0 from all)

### Classifier

```
Input:    63
Hidden 1: 256  → BatchNorm → ReLU → Dropout(0.3)
Hidden 2: 128  → BatchNorm → ReLU → Dropout(0.3)
Hidden 3: 64   → BatchNorm → ReLU → Dropout(0.3)
Output:   26   (softmax)
```

- **Optimiser:** Adam (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** StepLR (step=20, gamma=0.5)
- **Epochs:** 80 | **Batch size:** 64

### Dataset
- **Source:** [Kaggle ASL Alphabet](https://www.kaggle.com/grassknoted/asl-alphabet)
- **Classes:** 26 (A–Z) | **Images per class:** 1,000
- **Train/Test split:** 80/20 stratified
- **Extraction rate:** ~81%

---

## Limitations

- Static gestures only — J and Z require motion
- Single hand processed at a time
- Fingerspelling only — not word-level ASL signs
- Lighting dependent — MediaPipe degrades in very low light

---

## Future Work

- Word-level ASL recognition using LSTM/Transformer on landmark sequences
- ONNX export for faster CPU inference and embedded deployment
- Raspberry Pi deployment as standalone assistive device
- Mobile application (Android/iOS)

---

## Tech Stack

| Component | Technology |
|---|---|
| Hand Detection | Google MediaPipe Hand Landmarker |
| Deep Learning | PyTorch |
| Feature Scaling | scikit-learn |
| Webcam & Display | OpenCV |
| Text-to-Speech | pyttsx3 |
| Training Platform | Kaggle (NVIDIA T4 GPU) |

---

## Authors

**Vineet** — [github.com/vin0san](https://github.com/vin0san) · [linkedin.com/in/vineetkrr](https://linkedin.com/in/vineetkrr)

B.Tech Electronics & Communication Engineering
ABES Engineering College, Ghaziabad (2022–2026)
Under the guidance of *Prof. (Dr.) Ajay Suri*

---

## Acknowledgements

- [Google MediaPipe](https://mediapipe.dev) for the hand landmark detection model
- [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/grassknoted/asl-alphabet) by Akira
- [PyTorch](https://pytorch.org) for the deep learning framework