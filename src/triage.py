"""Educational, uncertainty-aware guidance for the skin-lesion prototype."""

import cv2
import numpy as np

from src.config import CLASS_INFO


LOW_CONFIDENCE_THRESHOLD = 0.60


def build_triage(predicted_class, confidence):
    """Return cautious next-step text; this is not a clinical triage tool."""
    info = CLASS_INFO.get(predicted_class, {})
    risk = info.get("risk", "UNKNOWN")
    recommendations = []

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        label = "Needs clinician review"
        recommendations.append(
            "The model is uncertain. Do not rely on this result; ask a qualified clinician to assess the lesion."
        )
    elif risk == "RED":
        label = "Clinician review recommended"
        recommendations.append(
            "The predicted category includes a cancer-related or precancerous class. Arrange assessment by a qualified clinician."
        )
    elif risk == "YELLOW":
        label = "Consider clinician review"
        recommendations.append(
            "The predicted category may need professional confirmation, especially if the lesion is new or changing."
        )
    else:
        label = "Lower-risk model result"
        recommendations.append(
            "This prediction does not rule out skin cancer. Seek professional advice for a new, changing, bleeding, or non-healing lesion."
        )

    recommendations.extend([
        "If concerned, show the lesion to a qualified healthcare professional; do not use this prototype to make treatment decisions.",
        "For personal monitoring, note changes in asymmetry, border, colour, diameter, or evolution (ABCDE).",
    ])
    return {"label": label, "recommendations": recommendations}


def check_image_quality(image_rgb):
    """Provide non-blocking capture warnings using simple image heuristics."""
    image = np.asarray(image_rgb)
    warnings = []
    if image.ndim < 2:
        return ["Could not assess image dimensions."]

    height, width = image.shape[:2]
    if min(height, width) < 128:
        warnings.append("Image resolution is small; try a larger, sharply focused image.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 40:
        warnings.append("Image may be blurry; try retaking it in focus.")
    brightness = float(gray.mean())
    if brightness < 35 or brightness > 220:
        warnings.append("Image may be too dark or too bright; improve the lighting and retake it.")
    return warnings
