"""
STEP 1 — Extract MediaPipe Hand Landmarks (mediapipe >= 0.10.13)
================================================================

Run on Kaggle. 
Dataset: grassknoted/asl-alphabet
OUTPUT: landmarks.npz
"""

import os, urllib.request, numpy as np, cv2
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATASET_PATH  = "/kaggle/input/datasets/grassknoted/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
OUTPUT_PATH   = "landmarks.npz"
CLASSES       = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
MAX_PER_CLASS = 500
MODEL_URL     = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_FILE    = "hand_landmarker.task"
# ────────────────────────────────────────────────────────────────────────────

# Download the hand landmarker model file if not present
if not os.path.exists(MODEL_FILE):
    print(f"Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print(f"Downloaded → {MODEL_FILE}")

# Build the landmarker
base_options   = mp_python.BaseOptions(model_asset_path=MODEL_FILE)
options        = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
    running_mode=mp_vision.RunningMode.IMAGE
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)


def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    lm = result.hand_landmarks[0]
    wx, wy, wz = lm[0].x, lm[0].y, lm[0].z

    # Hand-size normalisation: wrist (0) to middle finger MCP (9)
    hand_size = ((lm[9].x - wx)**2 +
                 (lm[9].y - wy)**2 +
                 (lm[9].z - wz)**2) ** 0.5
    hand_size = max(hand_size, 1e-6)

    coords = []
    for point in lm:
        coords.extend([(point.x - wx) / hand_size,
                       (point.y - wy) / hand_size,
                       (point.z - wz) / hand_size])
    return np.array(coords, dtype=np.float32)

def main():
    X, y = [], []
    label_map = {cls: idx for idx, cls in enumerate(CLASSES)}

    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_PATH, cls)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] Not found: {cls_dir}")
            continue

        images  = [f for f in os.listdir(cls_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))][:MAX_PER_CLASS]
        success = 0

        for fname in tqdm(images, desc=f"Class {cls}", leave=False):
            vec = extract_landmarks(os.path.join(cls_dir, fname))
            if vec is not None:
                X.append(vec)
                y.append(label_map[cls])
                success += 1

        print(f"[{cls}] {success}/{len(images)} landmarks extracted")

    X = np.array(X)
    y = np.array(y)
    print(f"\nDataset: X={X.shape}, y={y.shape}")
    np.savez(OUTPUT_PATH, X=X, y=y, classes=np.array(CLASSES))
    print(f"Saved → {OUTPUT_PATH}")

    landmarker.close()


main()