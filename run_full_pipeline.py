"""
run_full_pipeline.py
====================
Complete end-to-end skin cancer detection pipeline using the
hmnist_28_28_RGB.csv dataset (10,015 samples, all pixel data inline).

Mirrors the proven Kaggle notebook architecture (Custom CNN on 28x28 images)
using the bundled pixel CSV. Oversampling is restricted to the training split;
reported metrics use the original held-out test distribution.

Usage:
    python run_full_pipeline.py
"""

import os, sys, time, json, zipfile, pickle
from src.console import configure_console

configure_console()

from types import SimpleNamespace
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_score, recall_score, f1_score)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, BatchNormalization,
                                     Flatten, Dense, Dropout)
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ── Class labels ────────────────────────────────────────────────────────────
CLASSES = {
    4: ("nv",    "Melanocytic Nevi",           "🟢 GREEN",  "LOW"),
    6: ("mel",   "Melanoma",                   "🔴 RED",    "HIGH"),
    2: ("bkl",   "Benign Keratosis",           "🟡 YELLOW", "MED"),
    1: ("bcc",   "Basal Cell Carcinoma",       "🔴 RED",    "HIGH"),
    5: ("vasc",  "Vascular Lesions",           "🟡 YELLOW", "MED"),
    0: ("akiec", "Actinic Keratoses",          "🔴 RED",    "HIGH"),
    3: ("df",    "Dermatofibroma",             "🟡 YELLOW", "MED"),
}
CLASS_NAMES = [CLASSES[i][0] for i in range(7)]

def banner(title):
    print(f"\n{'='*70}\n  ▶  {title}\n{'='*70}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 0 — Locate / extract CSV
# ════════════════════════════════════════════════════════════════════════════
banner("STEP 0 — DATASET SETUP")
csv_path = os.path.join(DATASET_DIR, "hmnist_28_28_RGB.csv")

# Try to unzip if CSV not yet extracted
if not os.path.exists(csv_path):
    for fname in os.listdir(DATASET_DIR):
        if fname.endswith(".zip"):
            zip_full = os.path.join(DATASET_DIR, fname)
            print(f"Extracting {zip_full} …")
            with zipfile.ZipFile(zip_full, "r") as zf:
                zf.extractall(DATASET_DIR)
            break

if not os.path.exists(csv_path):
    print(f"ERROR: {csv_path} not found.")
    sys.exit(1)

t0 = time.time()
print(f"Loading CSV: {csv_path}")
data = pd.read_csv(csv_path)
print(f"  Shape      : {data.shape}")
print(f"  Load time  : {time.time()-t0:.1f}s")

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — EDA  (class distribution)
# ════════════════════════════════════════════════════════════════════════════
banner("STEP 1 — EXPLORATORY DATA ANALYSIS")
label_counts = data["label"].value_counts().sort_index()
print("\nClass distribution (before oversampling):")
print(f"{'ID':>4}  {'Code':8}  {'Name':35}  {'Count':>6}  {'Risk'}")
print("-" * 72)
for lbl, cnt in label_counts.items():
    code, name, risk, _ = CLASSES[lbl]
    print(f"{lbl:>4}  {code:8}  {name:35}  {cnt:>6}  {risk}")

# Save class distribution plot
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar([CLASSES[i][0] for i in range(7)],
              [label_counts.get(i, 0) for i in range(7)],
              color=["#e74c3c","#2ecc71","#f1c40f","#3498db","#9b59b6","#e67e22","#1abc9c"])
ax.set_title("HAM10000 — Class Distribution (Original)", fontsize=13, fontweight="bold")
ax.set_xlabel("Lesion Type"); ax.set_ylabel("Count")
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+30,
            str(int(b.get_height())), ha="center", va="bottom", fontsize=9)
plt.tight_layout()
dist_path = os.path.join(RESULTS_DIR, "class_distribution.png")
plt.savefig(dist_path, dpi=150); plt.close()
print(f"\nPlot saved → {dist_path}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Preprocessing
# ════════════════════════════════════════════════════════════════════════════
banner("STEP 2 — PREPROCESSING & OVERSAMPLING")
Label = data["label"].to_numpy(dtype=np.int64)
Data = data.drop(columns=["label"]).to_numpy(dtype=np.uint8)
tf.keras.utils.set_random_seed(42)

# Split before oversampling so duplicate samples stay inside training only.
from imblearn.over_sampling import RandomOverSampler
X_train, X_holdout, y_train, y_holdout = train_test_split(
    Data, Label, test_size=0.30, random_state=137, stratify=Label)
X_val, X_test, y_val, y_test = train_test_split(
    X_holdout, y_holdout, test_size=0.50, random_state=137, stratify=y_holdout)
del X_holdout, y_holdout
n_train_original = len(X_train)
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)
print(f"Original training samples: {n_train_original}")
print(f"Training samples after oversampling: {len(X_train)}")
print(f"Validation samples: {len(X_val)} | Test samples (held out): {len(X_test)}")

X_train = X_train.reshape(-1, 28, 28, 3).astype("float32")
X_val = X_val.reshape(-1, 28, 28, 3).astype("float32")
X_test = X_test.reshape(-1, 28, 28, 3).astype("float32")
print(f"\nX_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

# One-hot encode
y_train_cat = to_categorical(y_train, 7)
y_val_cat   = to_categorical(y_val, 7)
y_test_cat  = to_categorical(y_test,  7)

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Model Definition
# ════════════════════════════════════════════════════════════════════════════
banner("STEP 3 — BUILD CUSTOM CNN MODEL")
model = Sequential([
    keras.layers.Input(shape=(28, 28, 3)),
    # Block 1
    Conv2D(32, (3,3), activation="relu", padding="same", kernel_initializer="he_normal"),
    MaxPooling2D(),
    BatchNormalization(),
    # Block 2
    Conv2D(64, (3,3), activation="relu", padding="same", kernel_initializer="he_normal"),
    Conv2D(64, (3,3), activation="relu", padding="same", kernel_initializer="he_normal"),
    MaxPooling2D(),
    BatchNormalization(),
    # Block 3
    Conv2D(128, (3,3), activation="relu", padding="same", kernel_initializer="he_normal"),
    Conv2D(128, (3,3), activation="relu", padding="same", kernel_initializer="he_normal"),
    MaxPooling2D(),
    BatchNormalization(),
    # Block 4
    Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer="he_normal"),
    Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer="he_normal"),
    MaxPooling2D(),
    # Dense head
    Flatten(),
    Dropout(0.2),
    Dense(256, activation="relu", kernel_initializer="he_normal"),
    BatchNormalization(),
    Dense(128, activation="relu", kernel_initializer="he_normal"),
    BatchNormalization(),
    Dense(64, activation="relu", kernel_initializer="he_normal"),
    BatchNormalization(),
    Dense(32, activation="relu", kernel_initializer="he_normal",
          kernel_regularizer=keras.regularizers.L1L2()),
    BatchNormalization(),
    Dense(7, activation="softmax", name="classifier"),
])
model.compile(Adamax(learning_rate=0.001),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
total_params = model.count_params()
print(f"\nTotal parameters: {total_params:,}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Training
# ════════════════════════════════════════════════════════════════════════════
banner("STEP 4 — CLASS-WEIGHTED TRAINING")
datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=10, zoom_range=0.1,
                             width_shift_range=0.1, height_shift_range=0.1)
testgen = ImageDataGenerator(rescale=1./255)

lr_reduce = ReduceLROnPlateau(monitor="val_loss", patience=2,
                               verbose=1, factor=0.5, min_lr=1e-5)
best_model_path = os.path.join(MODELS_DIR, "skin_cancer_best.keras")

class ValidationMacroF1(keras.callbacks.Callback):
    """Add macro F1 to callback logs so rare classes affect model selection."""
    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.images, batch_size=256, verbose=0)
        score = f1_score(self.labels, np.argmax(predictions, axis=1), average="macro")
        if logs is not None:
            logs["val_macro_f1"] = score
        print(f" — val_macro_f1: {score:.4f}")

callbacks = [
    lr_reduce,
    ValidationMacroF1(X_val / 255.0, y_val),
    keras.callbacks.EarlyStopping(
        monitor="val_macro_f1", mode="max", patience=6,
        restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        best_model_path, monitor="val_macro_f1", mode="max",
        save_best_only=True, verbose=1
    ),
]

train_start = time.time()
history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=128),
    epochs=30,
    validation_data=testgen.flow(X_val, y_val_cat, batch_size=128, shuffle=False),
    callbacks=callbacks,
    verbose=1,
)
train_time = time.time() - train_start
print(f"\nTraining complete in {train_time:.0f}s ({train_time/60:.1f} min)")

# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Training curves
# ════════════════════════════════════════════════════════════════════════════
banner("STEP 5 — PLOT TRAINING CURVES")
tr_acc  = history.history["accuracy"]
tr_loss = history.history["loss"]
va_acc  = history.history["val_accuracy"]
va_loss = history.history["val_loss"]
epochs_range = range(1, len(tr_acc)+1)
best_val_acc_epoch = int(np.argmax(va_acc)) + 1
best_val_loss_epoch= int(np.argmin(va_loss)) + 1
best_val_macro_f1_epoch = int(np.argmax(history.history["val_macro_f1"])) + 1

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Custom CNN — Training History (HAM10000)", fontsize=14, fontweight="bold")

axes[0].plot(epochs_range, tr_loss, "r-o", ms=4, label="Train Loss")
axes[0].plot(epochs_range, va_loss, "g-o", ms=4, label="Val Loss")
axes[0].scatter(best_val_loss_epoch, va_loss[best_val_loss_epoch-1],
                s=150, c="blue", zorder=5, label=f"Best epoch={best_val_loss_epoch}")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, tr_acc, "r-o", ms=4, label="Train Accuracy")
axes[1].plot(epochs_range, va_acc, "g-o", ms=4, label="Val Accuracy")
axes[1].scatter(best_val_acc_epoch, va_acc[best_val_acc_epoch-1],
                s=150, c="blue", zorder=5, label=f"Best epoch={best_val_acc_epoch}")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
plt.savefig(curves_path, dpi=150); plt.close()
print(f"Saved → {curves_path}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — Evaluation
# ════════════════════════════════════════════════════════════════════════════
banner("STEP 6 — EVALUATE ON TEST SET")
X_test_norm = X_test / 255.0
y_pred_probs= model.predict(X_test_norm, verbose=0)
y_pred      = np.argmax(y_pred_probs, axis=1)
test_loss, test_acc = model.evaluate(
    testgen.flow(X_test, y_test_cat, batch_size=256, shuffle=False), verbose=0
)

report = classification_report(y_test, y_pred,
                                target_names=CLASS_NAMES, digits=4)
cm      = confusion_matrix(y_test, y_pred)
prec    = precision_score(y_test, y_pred, average="weighted")
rec     = recall_score   (y_test, y_pred, average="weighted")
f1      = f1_score       (y_test, y_pred, average="weighted")
macro_f1 = f1_score      (y_test, y_pred, average="macro")

print(f"\n  Test Accuracy  : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Test Loss      : {test_loss:.4f}")
print(f"  Precision (W)  : {prec:.4f}")
print(f"  Recall    (W)  : {rec:.4f}")
print(f"  F1-Score  (W)  : {f1:.4f}")
print(f"  F1-Score  (M)  : {macro_f1:.4f}")
print(f"\n{report}")

# Confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title("Confusion Matrix — Skin Cancer Detection", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150); plt.close()
print(f"Confusion matrix → {cm_path}")

# Per-class accuracy
print("\nPer-class accuracy:")
print(f"{'Code':8}  {'Name':35}  {'Correct':>8}  {'Total':>6}  {'Acc%':>7}  Risk")
print("-" * 80)
per_class = {}
for idx in range(7):
    mask    = y_test == idx
    correct = int((y_pred[mask] == idx).sum())
    total   = int(mask.sum())
    acc_pct = correct/total*100 if total else 0
    code, name, risk, _ = CLASSES[idx]
    print(f"{code:8}  {name:35}  {correct:>8}  {total:>6}  {acc_pct:>6.2f}%  {risk}")
    per_class[code] = {"correct": correct, "total": total, "accuracy": round(acc_pct,2)}

# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — Sample Predictions
# ════════════════════════════════════════════════════════════════════════════
banner("STEP 7 — SAMPLE PREDICTIONS")
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
fig.suptitle("Sample Test Predictions — Skin Cancer Detection", fontsize=13, fontweight="bold")
sample_indices = np.random.choice(len(X_test), 18, replace=False)
for ax, idx in zip(axes.flat, sample_indices):
    img      = X_test[idx] / 255.0
    true_lbl = y_test[idx]
    pred_lbl = y_pred[idx]
    true_code= CLASSES[true_lbl][0]
    pred_code= CLASSES[pred_lbl][0]
    conf     = y_pred_probs[idx][pred_lbl]
    color    = "green" if true_lbl == pred_lbl else "red"
    ax.imshow(img)
    ax.set_title(f"T:{true_code}\nP:{pred_code} {conf*100:.0f}%",
                 color=color, fontsize=8, fontweight="bold")
    ax.axis("off")
plt.tight_layout()
samples_path = os.path.join(RESULTS_DIR, "sample_predictions.png")
plt.savefig(samples_path, dpi=150); plt.close()
print(f"Sample predictions → {samples_path}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 8 — Save model & artifacts
# ════════════════════════════════════════════════════════════════════════════
banner("STEP 8 — SAVE MODEL & ARTIFACTS")
model_path = os.path.join(MODELS_DIR, "skin_cancer_model.keras")
model.save(model_path)
print(f"Model saved → {model_path}")

encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
label_encoder = SimpleNamespace(classes_=np.asarray(CLASS_NAMES))
with open(encoder_path, "wb") as f:
    pickle.dump(label_encoder, f)
print(f"Label mapping saved → {encoder_path}")

hist_path = os.path.join(RESULTS_DIR, "training_history.pkl")
with open(hist_path, "wb") as f:
    pickle.dump(history.history, f)
print(f"History saved → {hist_path}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 9 — Full JSON results report
# ════════════════════════════════════════════════════════════════════════════
banner("STEP 9 — GENERATE RESULTS REPORT")
total_time = time.time() - t0
epoch_log = []
for ep in range(len(tr_acc)):
    epoch_log.append({
        "epoch":        ep+1,
        "train_loss":   round(tr_loss[ep], 4),
        "train_acc":    round(tr_acc[ep],  4),
        "val_loss":     round(va_loss[ep], 4),
        "val_acc":      round(va_acc[ep],  4),
    })

results = {
    "project":        "AI-Powered Skin Cancer Detection System",
    "dataset":        "HAM10000 (hmnist_28_28_RGB.csv)",
    "model":          f"Custom CNN (7 convolution layers, {total_params:,} params)",
    "image_size":     "28×28×3",
    "total_samples":  int(len(Data)),
    "train_samples":  int(n_train_original),
    "train_samples_after_oversampling": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples":   int(len(X_test)),
    "class_distribution_original": {CLASSES[i][0]: int(label_counts.get(i,0)) for i in range(7)},
    "training": {
        "epochs":              len(tr_acc),
        "best_val_acc_epoch":  best_val_acc_epoch,
        "best_val_acc":        round(max(va_acc), 4),
        "best_val_loss_epoch": best_val_loss_epoch,
        "best_val_loss":       round(min(va_loss), 4),
        "best_val_macro_f1_epoch": best_val_macro_f1_epoch,
        "best_val_macro_f1": round(max(history.history["val_macro_f1"]), 4),
        "final_train_acc":     round(tr_acc[-1], 4),
        "final_val_acc":       round(va_acc[-1], 4),
        "training_time_sec":   round(train_time, 1),
        "training_time_min":   round(train_time/60, 1),
        "epoch_log":           epoch_log,
    },
    "evaluation": {
        "test_accuracy":       round(test_acc, 4),
        "test_accuracy_pct":   round(test_acc*100, 2),
        "test_loss":           round(test_loss, 4),
        "weighted_precision":  round(prec, 4),
        "weighted_recall":     round(rec,  4),
        "weighted_f1":         round(f1,   4),
        "macro_f1":            round(macro_f1, 4),
        "classification_report": report,
        "per_class_accuracy":  per_class,
        "confusion_matrix":    cm.tolist(),
    },
    "artifacts": {
        "model":               model_path,
        "label_encoder":       encoder_path,
        "training_curves":     curves_path,
        "confusion_matrix":    cm_path,
        "sample_predictions":  samples_path,
        "class_distribution":  dist_path,
        "history_pkl":         hist_path,
    },
    "total_pipeline_time_sec": round(total_time, 1),
    "total_pipeline_time_min": round(total_time/60, 1),
}

results_json_path = os.path.join(RESULTS_DIR, "run_results.json")
with open(results_json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Full results saved → {results_json_path}")

# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
banner("PIPELINE COMPLETE — FINAL SUMMARY")
print(f"""
  Dataset       : HAM10000  ({int(len(Data)):,} original samples)
  Model         : Custom CNN — {total_params:,} parameters
  Image size    : 28×28×3
  Train/Val/Test: {n_train_original:,} / {len(X_val):,} / {len(X_test):,}
  Oversampled training count: {len(X_train):,}

  ┌─────────────────────────────────────────────┐
  │            RESULTS SUMMARY                  │
  ├─────────────────────────────────────────────┤
  │  Test Accuracy    : {test_acc*100:>6.2f}%             │
  │  Test Loss        : {test_loss:>7.4f}              │
  │  Weighted Prec.   : {prec*100:>6.2f}%             │
  │  Weighted Recall  : {rec*100:>6.2f}%             │
  │  Weighted F1      : {f1*100:>6.2f}%             │
  │  Macro F1         : {macro_f1*100:>6.2f}%             │
  │  Best Val Acc     : {max(va_acc)*100:>6.2f}% (epoch {best_val_acc_epoch:>2})    │
  │  Best Val Macro F1: {max(history.history['val_macro_f1'])*100:>6.2f}% (epoch {best_val_macro_f1_epoch:>2})    │
  │  Training time    : {train_time/60:>5.1f} min                │
  │  Total time       : {total_time/60:>5.1f} min                │
  └─────────────────────────────────────────────┘
""")
