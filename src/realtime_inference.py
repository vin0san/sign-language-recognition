"""
STEP 3 — Real-Time Inference + Word Builder + Text-to-Speech
=============================================================
Run this LOCALLY. Requires model.pth, scaler.pkl, label_classes.npy
in same folder (or model-files/ subfolder).

Install:
    pip install mediapipe opencv-python torch scikit-learn joblib numpy pyttsx3

Controls:
    HOLD pose 2.0s  → confirm letter into word
    SPACEBAR        → confirm current word into sentence
    BACKSPACE key   → delete last letter
    ENTER key       → speak current sentence via TTS
    C key           → clear entire sentence
    S key           → save screenshot
    R key           → toggle smoothing
    Q / ESC         → quit
"""

import os, time, threading, urllib.request
from collections import deque, Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import joblib

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    raise SystemExit("Run: pip install mediapipe")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False
    print("[WARN] pyttsx3 not available. Run: pip install pyttsx3")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "model-files/model.pth"        if os.path.exists("model-files/model.pth")        else "model.pth"
SCALER_PATH  = "model-files/scaler.pkl"       if os.path.exists("model-files/scaler.pkl")       else "scaler.pkl"
CLASSES_PATH = "model-files/label_classes.npy" if os.path.exists("model-files/label_classes.npy") else "label_classes.npy"
MODEL_URL    = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_FILE   = "hand_landmarker.task"

CAMERA_ID         = 0
SMOOTHING_WINDOW  = 7
CONFIDENCE_THRESH = 0.65
DWELL_SECONDS  = 2.0
CONFIRM_COOLDOWN = 1.0   # seconds to ignore input after confirming a letter
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "bg":       (15,  15,  15),
    "green":    (0,   220, 120),
    "blue":     (255, 160,  60),
    "white":    (255, 255, 255),
    "gray":     (160, 160, 160),
    "yellow":   (0,   220, 220),
    "darkgray": (50,  50,  50),
    "red":      (60,  80,  220),
}

# ── MODEL ─────────────────────────────────────────────────────────────────────
class GestureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_model():
    ckpt   = torch.load(MODEL_PATH, map_location="cpu")
    model  = GestureMLP(ckpt["input_dim"], ckpt["hidden_dims"],
                        ckpt["num_classes"], ckpt["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    classes = ckpt.get("classes") or np.load(CLASSES_PATH, allow_pickle=True).tolist()
    scaler  = joblib.load(SCALER_PATH)
    print(f"Model loaded — {ckpt['num_classes']} classes, "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    return model, scaler, classes


# ── FEATURES ──────────────────────────────────────────────────────────────────
def extract_landmarks(lm_list):
    wx, wy, wz = lm_list[0].x, lm_list[0].y, lm_list[0].z
    coords = []
    for p in lm_list:
        coords.extend([p.x - wx, p.y - wy, p.z - wz])
    return np.array(coords, dtype=np.float32)


@torch.no_grad()
def predict(model, scaler, vec, classes):
    x     = torch.tensor(scaler.transform(vec.reshape(1, -1)), dtype=torch.float32)
    probs = torch.softmax(model(x), dim=1).squeeze().numpy()
    idx   = int(probs.argmax())
    return classes[idx], float(probs[idx])


# ── TTS ───────────────────────────────────────────────────────────────────────
def speak(text):
    if not TTS_AVAILABLE or not text.strip():
        return
    def _run():
        try:
            eng = pyttsx3.init()
            eng.setProperty("rate", 150)
            eng.say(text)
            eng.runAndWait()
        except Exception as e:
            print(f"[TTS] {e}")
    threading.Thread(target=_run, daemon=True).start()


# ── DRAWING ───────────────────────────────────────────────────────────────────
def draw_hand(frame, lm_list, w, h):
    CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
            (15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm_list]
    for a, b in CONN:
        cv2.line(frame, pts[a], pts[b], COLORS["green"], 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, COLORS["white"], -1)


def filled_rect(frame, x1, y1, x2, y2, color, alpha=0.75):
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


def draw_letter_panel(frame, label, confidence, dwell_progress):
    """Top-left: big predicted letter + confidence + dwell bar."""
    filled_rect(frame, 15, 30, 340, 175, COLORS["bg"])
    cv2.rectangle(frame, (15, 30), (340, 175), COLORS["green"], 2)

    col = COLORS["yellow"] if dwell_progress > 0 else COLORS["white"]
    cv2.putText(frame, label, (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 3.2, col, 5, cv2.LINE_AA)

    # Confidence bar
    bw = int(285 * min(confidence, 1.0))
    bc = COLORS["green"] if confidence >= CONFIDENCE_THRESH else COLORS["blue"]
    cv2.rectangle(frame, (30, 132), (315, 148), COLORS["darkgray"], -1)
    cv2.rectangle(frame, (30, 132), (30 + bw, 148), bc, -1)
    cv2.putText(frame, f"{confidence*100:.0f}%", (320, 146),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["gray"], 1)

    # Dwell bar
    if dwell_progress > 0:
        dw = int(285 * dwell_progress)
        cv2.rectangle(frame, (30, 152), (315, 166), COLORS["darkgray"], -1)
        cv2.rectangle(frame, (30, 152), (30 + dw, 166), COLORS["yellow"], -1)
        cv2.putText(frame, "HOLD TO CONFIRM", (30, 174),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["yellow"], 1)


def draw_word_panel(frame, current_word, sentence, last_spoken):
    """Bottom strip: word being spelled + full sentence."""
    h, w = frame.shape[:2]
    filled_rect(frame, 0, h - 95, w, h, COLORS["bg"], alpha=0.88)
    cv2.line(frame, (0, h - 95), (w, h - 95), COLORS["green"], 1)

    word_str = "Word: " + (current_word + "_" if current_word is not None else "_")
    cv2.putText(frame, word_str, (15, h - 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS["yellow"], 2, cv2.LINE_AA)

    sent_str = "Sentence: " + (sentence if sentence else "—")
    if len(sent_str) > 65:
        sent_str = "..." + sent_str[-62:]
    cv2.putText(frame, sent_str, (15, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLORS["white"], 1, cv2.LINE_AA)

    if last_spoken:
        cv2.putText(frame, f'🔊 Last: "{last_spoken}"', (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["gray"], 1, cv2.LINE_AA)


def draw_topbar(frame, fps, smoothing_on):
    h, w = frame.shape[:2]
    filled_rect(frame, 0, 0, w, 28, COLORS["bg"], alpha=0.9)
    ctrl = ("HOLD=Confirm Letter  |  SPACE=New Word  |  ENTER=Speak  "
            "| BKSP=Delete  |  C=Clear  |  S=Screenshot  |  Q=Quit")
    cv2.putText(frame, ctrl, (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLORS["gray"], 1)
    sm = "SMOOTH" if smoothing_on else "RAW"
    cv2.putText(frame, f"FPS:{fps:.0f}  [{sm}]", (w - 130, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["green"], 1)


def draw_confirm_flash(frame, letter):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 160, 60), -1)
    cv2.addWeighted(ov, 0.2, frame, 0.8, 0, frame)
    cv2.putText(frame, f"+ {letter}", (w // 2 - 60, h // 2 - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 3.5, COLORS["white"], 6, cv2.LINE_AA)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(MODEL_FILE):
        print("Downloading MediaPipe hand landmarker (~8MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
        print("Downloaded.")

    model, scaler, classes = load_model()

    base_opts  = mp_python.BaseOptions(model_asset_path=MODEL_FILE)
    mp_options = mp_vision.HandLandmarkerOptions(
        base_options=base_opts, num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(mp_options)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {CAMERA_ID}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    WIN = "Sign Language Recognition — ABES ECE"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    # ── State ──────────────────────────────────────────────────────────────────
    pred_buffer   = deque(maxlen=SMOOTHING_WINDOW)
    smoothing_on  = True
    current_word  = ""
    sentence      = ""
    last_spoken   = ""
    screenshot_n  = 0

    dwell_letter   = None
    dwell_start    = 0.0
    dwell_progress = 0.0

    flash_letter = None
    flash_until  = 0.0

    prev_time = time.time()
    print("\nReady. Hold any ASL letter pose for 1 second to type it.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

        curr_time = time.time()
        fps       = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        display_label  = "?"
        confidence     = 0.0
        dwell_progress = 0.0

        if result.hand_landmarks:
            lm_list = result.hand_landmarks[0]
            draw_hand(frame, lm_list, w, h)

            vec = extract_landmarks(lm_list)
            raw_label, confidence = predict(model, scaler, vec, classes)
            pred_buffer.append(raw_label)

            display_label = (Counter(pred_buffer).most_common(1)[0][0]
                             if smoothing_on else raw_label)

            # Dwell confirmation — skip entirely during cooldown
            if curr_time < flash_until:
                dwell_letter   = None
                dwell_start    = 0.0
                dwell_progress = 0.0
            elif confidence >= CONFIDENCE_THRESH:
                if display_label == dwell_letter:
                    elapsed        = curr_time - dwell_start
                    dwell_progress = min(elapsed / DWELL_SECONDS, 1.0)

                    if elapsed >= DWELL_SECONDS:
                        current_word   += display_label
                        flash_letter    = display_label
                        flash_until     = curr_time + CONFIRM_COOLDOWN
                        dwell_letter    = None
                        dwell_start     = 0.0
                        dwell_progress  = 0.0
                        pred_buffer.clear()
                        print(f"✓ {display_label}  |  word: {current_word}")
                else:
                    dwell_letter   = display_label
                    dwell_start    = curr_time
                    dwell_progress = 0.0
            else:
                dwell_letter   = None
                dwell_progress = 0.0

        else:
            pred_buffer.clear()
            dwell_letter   = None
            dwell_progress = 0.0
            cv2.putText(frame, "No hand detected",
                        (w // 2 - 160, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        COLORS["blue"], 2, cv2.LINE_AA)

        # Flash on confirm
        if curr_time < flash_until and flash_letter:
            draw_confirm_flash(frame, flash_letter)

        # Draw UI
        draw_topbar(frame, fps, smoothing_on)
        if result.hand_landmarks:
            draw_letter_panel(frame, display_label, confidence, dwell_progress)
        draw_word_panel(frame, current_word, sentence, last_spoken)

        cv2.imshow(WIN, frame)

        # ── Keys ───────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):          # Quit
            break

        elif key == 13:                    # ENTER — speak
            full = (sentence + " " + current_word).strip()
            if full:
                speak(full)
                last_spoken = full
                print(f"🔊 Speaking: {full}")

        elif key == 32:                    # SPACE — finish word
            if current_word:
                sentence     += (" " if sentence else "") + current_word
                current_word  = ""
                pred_buffer.clear()
                print(f"Word added. Sentence: {sentence}")

        elif key == 8:                     # BACKSPACE — delete last letter
            if current_word:
                current_word = current_word[:-1]
            elif sentence:
                parts        = sentence.rsplit(" ", 1)
                sentence     = parts[0]
                current_word = parts[1] if len(parts) > 1 else ""

        elif key == ord("c"):              # Clear all
            current_word = ""
            sentence     = ""
            last_spoken  = ""
            pred_buffer.clear()
            print("Cleared.")

        elif key == ord("s"):              # Screenshot
            fname = f"screenshot_{screenshot_n:03d}.png"
            cv2.imwrite(fname, frame)
            print(f"Saved: {fname}")
            screenshot_n += 1

        elif key == ord("r"):              # Toggle smoothing
            smoothing_on = not smoothing_on
            print(f"Smoothing: {'ON' if smoothing_on else 'OFF'}")

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    final = (sentence + " " + current_word).strip()
    if final:
        print(f"\nFinal sentence: {final}")
    print("Exited.")


if __name__ == "__main__":
    main()