"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for model interpretability.

Highlights which regions of the input image most influenced the classification decision.
"""

import numpy as np
import cv2
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.image_utils import preprocess_rgb_image


def find_last_conv_layer(model):
    """Find a final spatial feature-map layer usable by Grad-CAM."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            if any(
                isinstance(sub_layer, tf.keras.layers.Conv2D)
                for sub_layer in layer.layers
            ):
                return layer.name
    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a given image and model.

    Parameters
    ----------
    img_array : ndarray  shape (1, H, W, 3), preprocessed
    model : keras.Model
    last_conv_layer_name : str
    pred_index : int or None  (default: argmax of predictions)

    Returns
    -------
    heatmap : ndarray  shape (h, w), values in [0, 1]
    """
    target_layer = model.get_layer(last_conv_layer_name)
    if isinstance(model, tf.keras.Sequential):
        # Recreate a symbolic forward pass for Sequential models. In Keras 3,
        # a reloaded Sequential model may predict successfully without exposing
        # model.output/model.inputs for intermediate-output submodels.
        input_shape = model.input_shape
        if isinstance(input_shape, list) or input_shape is None:
            raise ValueError("Grad-CAM supports models with one image input.")
        symbolic_input = tf.keras.Input(
            shape=tuple(input_shape[1:]),
            name=f"gradcam_symbolic_input_{id(model)}",
        )
        value = symbolic_input
        conv_tensor = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            value = layer(value, training=False)
            if layer is target_layer:
                conv_tensor = value
        if conv_tensor is None:
            raise ValueError(f"Layer {last_conv_layer_name} is not in the model graph.")
        grad_model = tf.keras.models.Model(
            inputs=symbolic_input, outputs=[conv_tensor, value]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
    elif isinstance(target_layer, tf.keras.layers.Conv2D):
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )
        with tf.GradientTape() as tape:
            model_input = [img_array] if len(model.inputs) == 1 else img_array
            conv_outputs, predictions = grad_model(model_input, training=False)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
    else:
        nested_conv_layer = next(
            (
                layer
                for layer in reversed(target_layer.layers)
                if isinstance(layer, tf.keras.layers.Conv2D)
            ),
            None,
        )
        if nested_conv_layer is None:
            raise ValueError(f"No Conv2D layer found inside {last_conv_layer_name}.")

        nested_grad_model = tf.keras.models.Model(
            inputs=target_layer.input,
            outputs=[nested_conv_layer.output, target_layer.output],
        )
        nested_input = img_array
        for layer in model.layers:
            if layer is target_layer:
                break
            if not isinstance(layer, tf.keras.layers.InputLayer):
                nested_input = layer(nested_input, training=False)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = nested_grad_model(
                nested_input, training=False
            )
            after_backbone = False
            for layer in model.layers:
                if layer is target_layer:
                    after_backbone = True
                    continue
                if after_backbone and not isinstance(layer, tf.keras.layers.InputLayer):
                    predictions = layer(predictions, training=False)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img, heatmap, alpha=0.4):
    """
    Overlay a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_img : ndarray  (H, W, 3) RGB uint8
    heatmap : ndarray       (h, w) float [0,1]
    alpha : float           blending weight for the heatmap

    Returns
    -------
    superimposed : ndarray (H, W, 3) RGB uint8
    """
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)  # ensure RGB
    superimposed = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed


def display_gradcam(image_path, model, last_conv_layer_name,
                    image_size=(224, 224), save_path=None):
    """
    End-to-end Grad-CAM visualisation: original → heatmap → overlay.

    Parameters
    ----------
    image_path : str
    model : keras.Model
    last_conv_layer_name : str
    image_size : tuple
    save_path : str or None
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = preprocess_rgb_image(img, image_size)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    superimposed = overlay_gradcam(img, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(superimposed)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Grad-CAM saved → {save_path}")
    plt.close()
    return heatmap, superimposed
