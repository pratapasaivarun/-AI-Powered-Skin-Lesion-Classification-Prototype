"""Shared image preprocessing for training, prediction, and explanations."""

import numpy as np
import cv2

from src.config import IMAGE_SIZE


def preprocess_rgb_image(image, image_size=IMAGE_SIZE):
    """Resize an RGB image and return a normalized model input batch."""
    image_array = np.asarray(image)
    if image_array.ndim != 3 or image_array.shape[2] != 3 or image_array.size == 0:
        raise ValueError("Expected a non-empty RGB image with three color channels.")

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")

    resized = cv2.resize(image_array, (width, height))
    return np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)


def get_model_image_size(model, default=IMAGE_SIZE):
    """Return a model's fixed input dimensions as (width, height)."""
    input_shape = model.input_shape
    if input_shape is None or isinstance(input_shape, list) or len(input_shape) < 3:
        return default

    height, width = input_shape[1:3]
    if height is None or width is None:
        return default
    return int(width), int(height)
