"""
Training pipeline for skin cancer classification models.

Supports three architectures: custom_cnn, mobilenet_v2, hybrid.
Two-stage training: frozen backbone → fine-tuned top layers (for transfer learning models).
"""

import os
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.config import (
    EPOCHS_STAGE1, EPOCHS_STAGE2,
    LEARNING_RATE_STAGE1, LEARNING_RATE_STAGE2,
    WEIGHT_DECAY, LABEL_SMOOTHING,
    EARLY_STOPPING_PATIENCE, LR_REDUCE_PATIENCE, LR_REDUCE_FACTOR,
    BEST_MODEL_PATH, FINAL_MODEL_PATH, LABEL_ENCODER_PATH,
    TRAINING_HISTORY_PATH, MODELS_DIR, RESULTS_DIR,
    FINETUNED_MODEL_PATH,
    MOBILENET_UNFREEZE_LAYERS, HYBRID_UNFREEZE_LAYERS,
)
from src.models import build_custom_cnn, build_mobilenet_v2, build_hybrid


def _ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _get_callbacks(checkpoint_path=BEST_MODEL_PATH):
    return [
        EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            monitor="val_loss",
            verbose=1,
        ),
        ReduceLROnPlateau(
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            monitor="val_loss",
            verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
    ]


def _compile(model, lr, use_label_smoothing=True):
    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=LABEL_SMOOTHING if use_label_smoothing else 0.0
    )
    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=WEIGHT_DECAY)
    except AttributeError:
        optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])


def train_model(
    model_type,
    train_gen,
    val_gen,
    class_weights,
    encoder,
):
    """
    Train the selected model architecture.

    Parameters
    ----------
    model_type : str
        One of 'custom_cnn', 'mobilenet_v2', 'hybrid'.
    train_gen, val_gen : DirectoryIterator
        Keras data generators.
    class_weights : dict
        Computed class weights.
    encoder : LabelEncoder
        Fitted label encoder.

    Returns
    -------
    model : keras.Model
    history : dict   (combined history from both stages)
    """
    _ensure_dirs()
    base_model = None

    # Build model
    if model_type == "custom_cnn":
        model = build_custom_cnn()
    elif model_type == "mobilenet_v2":
        model, base_model = build_mobilenet_v2()
    elif model_type == "hybrid":
        model, base_model = build_hybrid()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.summary()

    # ----- Stage 1: Frozen backbone -----
    print("\n" + "=" * 60)
    print(f"STAGE 1 — Training ({model_type}) with frozen backbone")
    print("=" * 60)
    use_ls = model_type != "custom_cnn"
    _compile(model, LEARNING_RATE_STAGE1, use_label_smoothing=use_ls)

    callbacks = _get_callbacks(BEST_MODEL_PATH)
    history_s1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_STAGE1,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    combined_history = {k: list(v) for k, v in history_s1.history.items()}

    # ----- Stage 2: Fine-tuning (transfer learning models only) -----
    if base_model is not None:
        unfreeze = (
            MOBILENET_UNFREEZE_LAYERS
            if model_type == "mobilenet_v2"
            else HYBRID_UNFREEZE_LAYERS
        )
        print("\n" + "=" * 60)
        print(f"STAGE 2 — Fine-tuning top {unfreeze} layers of backbone")
        print("=" * 60)
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze]:
            layer.trainable = False

        _compile(model, LEARNING_RATE_STAGE2, use_label_smoothing=use_ls)

        callbacks = _get_callbacks(FINETUNED_MODEL_PATH)
        history_s2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS_STAGE2,
            class_weight=class_weights,
            callbacks=callbacks,
        )

        for k, v in history_s2.history.items():
            combined_history[k].extend(list(v))

        # Fine-tuning can overfit. Keep whichever stage achieved the lower
        # validation loss instead of assuming stage 2 is always an improvement.
        if min(history_s2.history["val_loss"]) < min(history_s1.history["val_loss"]):
            model = tf.keras.models.load_model(FINETUNED_MODEL_PATH)
        else:
            model = tf.keras.models.load_model(BEST_MODEL_PATH)
    else:
        model = tf.keras.models.load_model(BEST_MODEL_PATH)

    # ----- Save artifacts -----
    model.save(FINAL_MODEL_PATH)
    print(f"Model saved → {FINAL_MODEL_PATH}")

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)
    print(f"Label encoder saved → {LABEL_ENCODER_PATH}")

    with open(TRAINING_HISTORY_PATH, "wb") as f:
        pickle.dump(combined_history, f)
    print(f"Training history saved → {TRAINING_HISTORY_PATH}")

    return model, combined_history
