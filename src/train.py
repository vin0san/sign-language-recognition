"""
STEP 2 — Train MLP Classifier on Landmark Features
====================================================
Ran this on Kaggle after extract_landmarks.py produces landmarks.npz.
GPU is helpful but not required — trains in < 2 minutes on CPU.

INPUT:  landmarks.npz  (output of extract_landmarks.py)
OUTPUT: model.pth      (trained PyTorch MLP)
        label_classes.npy (class names array)

Usage:
    python step2_train.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ── CONFIG ─────────────────────────────────────────────────────────────────
LANDMARKS_PATH = "landmarks.npz"
MODEL_PATH     = "model.pth"
SCALER_PATH    = "scaler.pkl"
CLASSES_PATH   = "label_classes.npy"

EPOCHS      = 50
BATCH_SIZE  = 64
LR          = 1e-3
HIDDEN_DIMS = [256, 128, 64]
DROPOUT     = 0.3
# ───────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── MODEL ───────────────────────────────────────────────────────────────────
class GestureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── DATA ────────────────────────────────────────────────────────────────────
def load_data():
    data    = np.load(LANDMARKS_PATH, allow_pickle=True)
    X       = data["X"].astype(np.float32)
    y       = data["y"].astype(np.int64)
    classes = data["classes"]
    print(f"Loaded: X={X.shape}, y={y.shape}, classes={list(classes)}")
    return X, y, classes


def preprocess(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved → {SCALER_PATH}")

    return (torch.tensor(X_train), torch.tensor(y_train),
            torch.tensor(X_test),  torch.tensor(y_test),
            scaler)


# ── TRAINING ────────────────────────────────────────────────────────────────
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (out.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct    += (out.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    X, y, classes = load_data()
    num_classes   = len(classes)
    np.save(CLASSES_PATH, classes)

    X_train, y_train, X_test, y_test, _ = preprocess(X, y)

    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model     = GestureMLP(63, HIDDEN_DIMS, num_classes, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training for {EPOCHS} epochs...\n")

    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, test_loader,  criterion)
        scheduler.step()

        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "model_state": model.state_dict(),
                "input_dim":   63,
                "hidden_dims": HIDDEN_DIMS,
                "num_classes": num_classes,
                "dropout":     DROPOUT,
                "classes":     classes.tolist(),
            }, MODEL_PATH)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
                  f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}"
                  + (" ← best" if vl_acc == best_val_acc else ""))

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")

    # ── Final evaluation ────────────────────────────────────────────────────
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=classes))

    # Confusion matrix
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes,
                cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Confusion matrix saved → confusion_matrix.png")

    # Training curve
    plt.figure(figsize=(8, 4))
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs,   label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    print("Training curve saved → training_curve.png")

    print(f"\nFiles saved: {MODEL_PATH}, {SCALER_PATH}, {CLASSES_PATH}")


if __name__ == "__main__":
    main()