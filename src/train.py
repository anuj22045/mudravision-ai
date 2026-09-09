"""
MudraVision AI — Model Training Script  (Phase 1 / Phase 2)

This script is the permanent, clean record of exactly how the CNN was trained.
Run this script if you ever need to retrain the model from scratch.

Architecture:
    - 3 × [Conv2D → MaxPooling2D] blocks (feature extraction)
    - Flatten → Dense(128) → Dropout(0.5) → Dense(4, softmax)
    - Input: 128 × 128 × 3 (RGB)
    - Output: 4 classes (10_rupee, 1_rupee, 2_rupee, 5_rupee)

Training choices explained:
    - Class weights: 2_rupee only has 60 images vs 180 for 5_rupee.
      Without class weights the model ignores minority classes.
      class_weight tells the model to penalize errors on rare classes more.
    - Adam optimizer: adaptive learning rate, works well without tuning.
    - categorical_crossentropy: standard loss for multi-class classification.
    - EarlyStopping: stops training if val_loss doesn't improve for 15 epochs.
    - ReduceLROnPlateau: halves learning rate when val_loss plateaus.

Usage:
    python src/train.py
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add src/ to path so preprocessing.py can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import build_dataset, IMG_SIZE


# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent
TRAIN_DIR  = BASE_DIR / "dataset" / "train"
VAL_DIR    = BASE_DIR / "dataset" / "val"
MODEL_DIR  = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "coin_classifier_class_weighted.keras"

MODEL_DIR.mkdir(exist_ok=True)


# ─── Class Configuration ──────────────────────────────────────────────────────
# TensorFlow sorts class folders alphabetically.
# MUST keep this order consistent with INDEX_TO_CLASS in predict.py.

CLASSES = ["10_rupee", "1_rupee", "2_rupee", "5_rupee"]
CLASS_TO_INDEX = {cls: i for i, cls in enumerate(CLASSES)}


# ─── Hyperparameters ─────────────────────────────────────────────────────────

EPOCHS         = 80
LEARNING_RATE  = 0.001


# ─── Data Loading ─────────────────────────────────────────────────────────────

def collect_paths_and_labels(data_dir):
    """
    Walk through the dataset directory and collect image paths + labels.

    Expects structure:
        data_dir/
            10_rupee/  ← class folder
                img1.jpg
                img2.jpg
            1_rupee/
                ...

    Returns:
        image_paths (list of str): absolute paths to each image
        labels (list of int):      class index for each image
    """
    image_paths = []
    labels      = []

    for class_name, class_index in CLASS_TO_INDEX.items():
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            print(f"  Warning: {class_dir} not found, skipping.")
            continue

        image_files = (
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.jpeg")) +
            list(class_dir.glob("*.png"))
        )

        image_paths.extend([str(f) for f in image_files])
        labels.extend([class_index] * len(image_files))

        print(f"  {class_name}: {len(image_files)} images")

    return image_paths, labels


def compute_class_weights(labels):
    """
    Compute per-class weights to handle class imbalance.

    Formula: weight_i = total_samples / (n_classes × count_i)

    This makes the model penalize errors on rare classes (2_rupee)
    more than errors on common classes (5_rupee).

    Returns:
        dict: {class_index: weight}
    """
    labels_array = np.array(labels)
    total         = len(labels_array)
    n_classes     = len(CLASSES)

    weights = {}
    for class_index in range(n_classes):
        count = np.sum(labels_array == class_index)
        if count == 0:
            weights[class_index] = 1.0
        else:
            weights[class_index] = total / (n_classes * count)

    return weights


# ─── Model Architecture ───────────────────────────────────────────────────────

def build_model():
    """
    Build the CNN architecture from scratch.

    Layer-by-layer explanation:
        Conv2D(32, 3×3):  Extract 32 low-level features (edges, textures)
        MaxPooling2D:     Halve spatial dimensions (128→64), keep strongest features
        Conv2D(64, 3×3):  Extract 64 mid-level features (shapes, patterns)
        MaxPooling2D:     64→32
        Conv2D(128, 3×3): Extract 128 high-level features (coin-specific patterns)
        MaxPooling2D:     32→16
        Flatten:          Convert 16×16×128 feature map → 32,768-dim vector
        Dense(128):       Learn combinations of features relevant to coin class
        Dropout(0.5):     Randomly zero 50% of neurons each step → prevents overfitting
        Dense(4, softmax):Output 4 probabilities that sum to 1.0

    Input shape: (128, 128, 3)
    Output shape: (4,) — one probability per coin denomination
    """
    model = tf.keras.Sequential([
        # Block 1 — detect edges and basic shapes
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu",
                               input_shape=(*IMG_SIZE, 3), padding="same"),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Block 2 — detect coin-level patterns (text, rim, design)
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Block 3 — detect class-discriminative features
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2, 2),

        # Classifier head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    print("\n" + "=" * 55)
    print("  MudraVision AI — Training")
    print("=" * 55)

    # ── Load training data ────────────────────────────────────────────────────
    print("\nLoading training data...")
    train_paths, train_labels = collect_paths_and_labels(TRAIN_DIR)
    print(f"  Total training images: {len(train_paths)}")

    print("\nLoading validation data...")
    val_paths, val_labels = collect_paths_and_labels(VAL_DIR)
    print(f"  Total validation images: {len(val_paths)}")

    # ── Build tf.data pipelines ───────────────────────────────────────────────
    train_dataset = build_dataset(train_paths, train_labels, is_training=True)
    val_dataset   = build_dataset(val_paths,   val_labels,   is_training=False)

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = compute_class_weights(train_labels)
    print("\nClass weights (handles imbalance):")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {class_weights[i]:.4f}")

    # ── Build model ───────────────────────────────────────────────────────────
    print("\nModel architecture:")
    model = build_model()
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        # Stop early if val_loss stops improving for 15 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Halve the learning rate when val_loss plateaus for 8 epochs
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        # Save the best model checkpoint automatically
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining for up to {EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # ── Final report ──────────────────────────────────────────────────────────
    best_val_acc = max(history.history["val_accuracy"])
    print()
    print("=" * 55)
    print(f"  Training complete!")
    print(f"  Best val accuracy : {best_val_acc * 100:.2f}%")
    print(f"  Model saved to    : {MODEL_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    train()
