"""
MudraVision AI — Coin Prediction Pipeline  (Phase 2)

This module loads the trained CNN model and predicts the denomination
of a given coin image.

HOW IT WORKS:
    1. Load the image from disk
    2. Apply the SAME preprocessing used during training:
       center crop → resize to 128×128 → normalize to [0-1]
    3. Run model.predict() → get a softmax probability vector
    4. Pick the class with the highest probability (np.argmax)
    5. Check confidence — if < CONFIDENCE_THRESHOLD, flag as uncertain

WHAT IS SOFTMAX OUTPUT?
    The final Dense layer uses softmax activation, which forces all 4
    class probabilities to sum to exactly 1.0.
    Example: [0.05, 0.62, 0.28, 0.05]
              10₹   1₹    2₹    5₹
    The model says it is 62% sure this is ₹1.

WHAT IS CONFIDENCE SCORE?
    The probability of the winning class, expressed as a percentage.
    High confidence (≥ 75%) → model is fairly certain.
    Low confidence (< 75%)  → model is uncertain, result may be wrong.

WHY 75% THRESHOLD?
    The ₹1 vs ₹2 confusion identified in Phase 1:
    both coins look similar (same color/metal, similar size after resize).
    When the model is unsure between them, confidence drops below 75%.
    Flagging these cases prevents silently wrong predictions.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Add src/ to path so preprocessing.py can be imported from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import preprocess_image


# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "coin_classifier_class_weighted.keras"


# ─── Class Labels ─────────────────────────────────────────────────────────────
# Must match the alphabetical order TensorFlow assigned during training.
# TensorFlow sorts class folders alphabetically, so:
#   Index 0 → 10_rupee (comes first alphabetically)
#   Index 1 → 1_rupee
#   Index 2 → 2_rupee
#   Index 3 → 5_rupee

INDEX_TO_CLASS = {
    0: "10_rupee",
    1: "1_rupee",
    2: "2_rupee",
    3: "5_rupee"
}

# Display-friendly names for the UI
DISPLAY_NAME = {
    "10_rupee": "₹10",
    "1_rupee":  "₹1",
    "2_rupee":  "₹2",
    "5_rupee":  "₹5"
}

# ─── Confidence Threshold ─────────────────────────────────────────────────────
# If the top class probability is below this value, the prediction is flagged
# as uncertain. This catches the ₹1 vs ₹2 confusion identified in Phase 1.
CONFIDENCE_THRESHOLD = 75.0  # percent


# ─── Lazy Model Loader ────────────────────────────────────────────────────────
# The model is loaded ONCE the first time predict_coin() is called.
# This avoids the slow TensorFlow startup penalty at import time,
# which matters when Streamlit (Phase 4) imports this module.

_model = None

def _load_model():
    """Load the trained model from disk (only once)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at: {MODEL_PATH}\n"
                "Please train the model first (see notebooks/2_cnn_training.ipynb)."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


# ─── Prediction Function ──────────────────────────────────────────────────────

def predict_coin(image_path):
    """
    Predict the denomination of a coin from an image file.

    This is the main function Phase 4 (Streamlit) will call.

    Args:
        image_path: Path to a coin image (str or Path). Can be JPEG or PNG.

    Returns:
        A dictionary with the following keys:
            predicted_class  (str)   e.g. "5_rupee"
            display_name     (str)   e.g. "₹5"
            confidence       (float) e.g. 91.23   (percentage, 0–100)
            is_confident     (bool)  True if confidence ≥ 75%
            warning          (str)   Empty string if confident, else a message
            all_probabilities (dict) {class_name: probability_%} for all 4 classes

    Example:
        >>> result = predict_coin("dataset/test/5_rupee/img.jpg")
        >>> print(result["display_name"], result["confidence"])
        ₹5  91.23
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ── Step 1: Load image from disk ──────────────────────────────────────────
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    # decode_image handles JPEG, PNG, BMP — more flexible than decode_jpeg

    # ── Step 2: Preprocess (MUST match training pipeline exactly) ─────────────
    # center crop → resize 128×128 → normalize [0-1]
    image, _ = preprocess_image(image, 0)

    # ── Step 3: Add batch dimension ───────────────────────────────────────────
    # model.predict() expects shape (batch_size, 128, 128, 3)
    # We have one image, so batch_size = 1
    image = tf.expand_dims(image, axis=0)

    # ── Step 4: Run inference ─────────────────────────────────────────────────
    model = _load_model()
    probabilities = model.predict(image, verbose=0)[0]
    # probabilities shape: (4,) — one probability per class, sums to 1.0

    # ── Step 5: Find the winning class ────────────────────────────────────────
    # np.argmax returns the index with the highest probability
    predicted_index = int(np.argmax(probabilities))
    predicted_class = INDEX_TO_CLASS[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100.0

    # ── Step 6: Confidence check ──────────────────────────────────────────────
    is_confident = confidence >= CONFIDENCE_THRESHOLD

    if is_confident:
        warning = ""
    else:
        warning = (
            f"Low confidence ({confidence:.1f}%). "
            "The coin may be ₹1 or ₹2 — they look very similar. "
            "Try better lighting or a clearer photo."
        )

    # ── Step 7: Build full probability breakdown ──────────────────────────────
    all_probabilities = {
        INDEX_TO_CLASS[i]: round(float(probabilities[i]) * 100.0, 2)
        for i in range(len(probabilities))
    }

    return {
        "predicted_class":   predicted_class,
        "display_name":      DISPLAY_NAME[predicted_class],
        "confidence":        round(confidence, 2),
        "is_confident":      is_confident,
        "warning":           warning,
        "all_probabilities": all_probabilities
    }


# ─── Manual Test Block ────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run a prediction on a single image from the command line.

    Usage:
        python src/predict.py dataset/test/5_rupee/some_image.jpg

    This lets you manually test the model without writing any extra code.
    """
    if len(sys.argv) < 2:
        print("\nUsage: python src/predict.py <path_to_image>")
        print("Example:")
        print("  python src/predict.py dataset/test/5_rupee/img.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"\nRunning prediction on: {image_path}")
    print("Loading model...")

    result = predict_coin(image_path)

    print()
    print("=" * 45)
    print("  Prediction Result")
    print("=" * 45)
    print(f"  Predicted coin  : {result['predicted_class']}")
    print(f"  Confidence      : {result['confidence']:.2f}%")

    if result["is_confident"]:
        print("  Status          : HIGH CONFIDENCE")
    else:
        print("  Status          : LOW CONFIDENCE - uncertain!")
        # Strip rupee symbol for terminal compatibility
        warning_plain = result['warning'].replace('Rs.1', '1_rupee').replace('Rs.2', '2_rupee')
        print(f"  Warning         : {result['warning'].encode('ascii', errors='replace').decode()}")

    print()
    print("  All class probabilities:")
    print("  " + "-" * 30)
    for class_name, prob in sorted(result["all_probabilities"].items(),
                                    key=lambda x: x[1], reverse=True):
        bar = "|" * int(prob / 5)  # simple ASCII bar: 1 block per 5%
        print(f"  {class_name:<12} : {prob:>6.2f}%  {bar}")
    print("=" * 45)