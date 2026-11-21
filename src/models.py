"""
Model architectures for skin cancer classification.

Three architectures are available:
  1. custom_cnn     — Custom deep CNN from scratch
  2. mobilenet_v2   — MobileNetV2 transfer learning
  3. hybrid         — Dual-branch: Custom CNN + MobileNetV2 (concatenated)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization, Input, Concatenate,
    Rescaling,
)
from tensorflow.keras.applications import MobileNetV2

from src.config import IMAGE_SIZE, NUM_CLASSES


# ------------------------------------------------------------------ #
#  1. Custom CNN                                                      #
# ------------------------------------------------------------------ #
def build_custom_cnn(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES):
    """
    Deep custom CNN with 4 convolutional blocks, batch normalisation,
    global average pooling, and dense classification head.
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 2
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 3
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 4
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Classification head
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    return model


# ------------------------------------------------------------------ #
#  2. MobileNetV2 Transfer Learning                                   #
# ------------------------------------------------------------------ #
def build_mobilenet_v2(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES):
    """
    MobileNetV2 backbone (ImageNet-pretrained, frozen) with a custom
    classification head.  Returns (model, base_mobilenet).
    """
    input_layer = Input(shape=input_shape)
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False  # freeze initially

    x = Rescaling(2.0, offset=-1.0, name="mobilenet_input_preprocessing")(input_layer)
    x = base(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output)
    return model, base


# ------------------------------------------------------------------ #
#  3. Hybrid CNN + MobileNetV2                                        #
# ------------------------------------------------------------------ #
def build_hybrid(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES):
    """
    Dual-branch architecture:
      - Branch A: Custom CNN (learns low-level skin features)
      - Branch B: MobileNetV2 (high-level pre-trained features)
    Both branches are concatenated before the classification head.
    Returns (model, base_mobilenet).
    """
    input_layer = Input(shape=input_shape)

    # --- Branch A: Custom CNN ---
    x1 = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)

    x1 = Conv2D(64, (3, 3), activation="relu", padding="same")(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)

    x1 = Conv2D(128, (3, 3), activation="relu", padding="same")(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)

    x1 = Conv2D(256, (3, 3), activation="relu", padding="same")(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)

    x1 = GlobalAveragePooling2D()(x1)
    x1 = Dense(128, activation="relu")(x1)
    x1 = Dropout(0.3)(x1)

    # --- Branch B: MobileNetV2 ---
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False

    mobilenet_input = Rescaling(
        2.0, offset=-1.0, name="mobilenet_input_preprocessing"
    )(input_layer)
    x2 = base(mobilenet_input)
    x2 = GlobalAveragePooling2D()(x2)
    x2 = Dense(128, activation="relu")(x2)
    x2 = Dropout(0.3)(x2)

    # --- Merge ---
    merged = Concatenate()([x1, x2])
    merged = Dense(256, activation="relu")(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(128, activation="relu")(merged)
    merged = Dropout(0.3)(merged)
    merged = Dense(64, activation="relu")(merged)
    output = Dense(num_classes, activation="softmax")(merged)

    model = Model(inputs=input_layer, outputs=output)
    return model, base
