"""
Single-image prediction with Grad-CAM and educational triage guidance.
"""

import os
import pickle
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.console import configure_console
from src.config import (
    IMAGE_SIZE, FINAL_MODEL_PATH, LABEL_ENCODER_PATH,
    CLASS_INFO, RISK_COLORS, RESULTS_DIR,
)
from src.gradcam import find_last_conv_layer, display_gradcam
from src.image_utils import get_model_image_size, preprocess_rgb_image
from src.triage import build_triage, check_image_quality

configure_console()


def load_artifacts(model_path=None, encoder_path=None):
    """Load saved model and label encoder."""
    import tensorflow as tf

    if model_path is None:
        model_path = FINAL_MODEL_PATH
    if encoder_path is None:
        encoder_path = LABEL_ENCODER_PATH

    model = tf.keras.models.load_model(model_path)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    return model, encoder


def predict_single_image(image_path, model, encoder, save_dir=None):
    """
    Classify a single skin lesion image and produce educational triage guidance.

    Parameters
    ----------
    image_path : str
    model : keras.Model
    encoder : LabelEncoder
    save_dir : str or None

    Returns
    -------
    result : dict  with predicted_class, confidence, risk_level, recommendations, etc.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Read and preprocess
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_size = get_model_image_size(model, IMAGE_SIZE)
    img_array = preprocess_rgb_image(img_rgb, image_size)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(predictions))
    pred_class = encoder.classes_[pred_idx]
    confidence = float(predictions[pred_idx])

    info = CLASS_INFO.get(pred_class, {})
    risk = info.get("risk", "UNKNOWN")
    emoji = RISK_COLORS.get(risk, "⚪")

    # Build recommendations
    triage = build_triage(pred_class, confidence)
    recommendations = triage["recommendations"]
    image_warnings = check_image_quality(img_rgb)

    # Print report
    print("=" * 60)
    print("     SKIN CANCER DETECTION & TRIAGE REPORT")
    print("=" * 60)
    print(f"  Image         : {os.path.basename(image_path)}")
    print(f"  Predicted     : {info.get('name', pred_class)} ({pred_class})")
    print(f"  Confidence    : {confidence * 100:.2f}%")
    print(f"  Project Triage: {triage['label']} ({emoji} {risk})")
    print(f"  Description   : {info.get('description', 'N/A')}")
    for warning in image_warnings:
        print(f"  Image quality : {warning}")
    print("-" * 60)
    print("  CLASS PROBABILITIES:")
    for cls, prob in zip(encoder.classes_, predictions):
        bar = "█" * int(prob * 30)
        print(f"    {cls:8s} : {prob * 100:6.2f}%  {bar}")
    print("-" * 60)
    print("  RECOMMENDATIONS:")
    for rec in recommendations:
        print(f"    • {rec}")
    print("=" * 60)

    # Grad-CAM
    try:
        last_conv = find_last_conv_layer(model)
        gradcam_path = os.path.join(save_dir, f"gradcam_{os.path.basename(image_path)}")
        display_gradcam(image_path, model, last_conv,
                        image_size=image_size,
                        save_path=gradcam_path)
    except Exception as e:
        print(f"  [Grad-CAM] Could not generate visualisation: {e}")
        gradcam_path = None

    return {
        "image_path": image_path,
        "predicted_class": pred_class,
        "predicted_name": info.get("name", pred_class),
        "confidence": confidence,
        "risk_level": risk,
        "triage_label": triage["label"],
        "image_quality_warnings": image_warnings,
        "probabilities": {cls: float(p) for cls, p in zip(encoder.classes_, predictions)},
        "recommendations": recommendations,
        "gradcam_path": gradcam_path,
    }


