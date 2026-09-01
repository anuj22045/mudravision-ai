"""
MudraVision AI — Image Preprocessing Pipeline

This module contains all image preprocessing functions used across the project:
- Loading images from disk
- Center cropping to remove excess background
- Resizing to a fixed input size for the CNN
- Normalizing pixel values from [0-255] to [0-1]
- Data augmentation for the training set

WHY NORMALIZE?
    Raw pixel values (0-255) cause large, unstable gradients during training.
    Scaling to [0-1] helps gradient descent converge faster and more reliably.

WHY AUGMENT?
    With only ~510 training images, the CNN would memorize them (overfitting).
    Augmentation creates random variations (rotation, zoom, shift) so the model
    sees different versions each epoch, forcing it to learn general coin features
    rather than memorizing specific photos.
"""

import tensorflow as tf

# ─── Configuration ───────────────────────────────────────────────────────────
IMG_SIZE = (128, 128)
CROP_SIZE = 1600
BATCH_SIZE = 32
SEED = 42


# ─── Data Augmentation Layer ────────────────────────────────────────────────
# Applied ONLY to training data. Not applied to validation or test.
# - RandomRotation: coins can appear at any angle
# - RandomZoom: coins may be closer or farther from camera
# - RandomTranslation: coins may not be perfectly centered
# - NO horizontal flip: flipping would mirror text on coins

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    )
])


# ─── Core Functions ─────────────────────────────────────────────────────────

def load_image(image_path, label):
    """Load a JPEG image from disk and decode it into a tensor.

    Args:
        image_path: String path to the image file.
        label: Integer class label (0-3).

    Returns:
        (image_tensor, label) tuple.
    """
    image = tf.io.read_file(image_path) #read raw bytes from disk
    image = tf.image.decode_jpeg(image, channels=3)
    return image, label


def preprocess_image(image, label):
    """Full preprocessing pipeline for a single image.

    Steps:
        1. Center crop — removes excess background around the coin
        2. Resize to 128x128 — CNN needs fixed input dimensions
        3. Normalize to [0-1] — helps gradient descent converge

    Args:
        image: Raw image tensor (variable size).
        label: Integer class label (0-3).

    Returns:
        (processed_image, label) tuple.
    """
    # Get original dimensions
    height = tf.shape(image)[0]
    width = tf.shape(image)[1]

    # Center crop to square
    left = (width - CROP_SIZE) // 2
    top = (height - CROP_SIZE) // 2
    image = tf.image.crop_to_bounding_box(image, top, left, CROP_SIZE, CROP_SIZE)

    # Resize to fixed CNN input size
    image = tf.image.resize(image, IMG_SIZE)

    # Normalize: [0-255] → [0-1]
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def build_dataset(image_paths, labels, is_training=False):
    """Build a complete tf.data pipeline from image paths.

    Args:
        image_paths: List of file paths to images.
        labels: List of integer class labels.
        is_training: If True, applies shuffle + data augmentation.

    Returns:
        A batched, preprocessed tf.data.Dataset ready for model.fit().
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Load raw images from disk
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply preprocessing (crop → resize → normalize)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        # Shuffle training data each epoch
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=SEED
        )

    # Batch the data
    dataset = dataset.batch(BATCH_SIZE)

    if is_training:
        # Apply augmentation ONLY to training batches
        dataset = dataset.map(
            lambda images, labels: (data_augmentation(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Prefetch next batch while current batch is being processed
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# JPEG image
#     ↓
# load_image()
#     ↓
# 1600 × 1600 center crop
#     ↓
# 128 × 128 resize
#     ↓
# 0–1 normalization
#     ↓
# batch
#     ↓
# augmentation only if training
#     ↓
# prefetch