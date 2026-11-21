import unittest

import numpy as np
import tensorflow as tf

from src.gradcam import find_last_conv_layer, make_gradcam_heatmap


class GradCamTests(unittest.TestCase):
    def test_builds_heatmap_for_top_level_convolution(self):
        image_input = tf.keras.Input(shape=(8, 8, 3))
        features = tf.keras.layers.Conv2D(4, 3, activation="relu")(image_input)
        x = tf.keras.layers.GlobalAveragePooling2D()(features)
        model_output = tf.keras.layers.Dense(2, activation="softmax")(x)
        model = tf.keras.Model(image_input, model_output)

        feature_layer = find_last_conv_layer(model)
        heatmap = make_gradcam_heatmap(
            np.ones((1, 8, 8, 3), dtype=np.float32), model, feature_layer
        )

        self.assertEqual(heatmap.shape, (6, 6))
        self.assertTrue(np.all(np.isfinite(heatmap)))

    def test_finds_nested_backbone_feature_map(self):
        backbone_input = tf.keras.Input(shape=(8, 8, 3))
        features = tf.keras.layers.Conv2D(4, 3, activation="relu")(backbone_input)
        backbone = tf.keras.Model(backbone_input, features, name="backbone")

        model_input = tf.keras.Input(shape=(8, 8, 3))
        x = backbone(model_input)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        model_output = tf.keras.layers.Dense(2, activation="softmax")(x)
        model = tf.keras.Model(model_input, model_output)

        feature_layer = find_last_conv_layer(model)
        heatmap = make_gradcam_heatmap(
            np.ones((1, 8, 8, 3), dtype=np.float32), model, feature_layer
        )

        self.assertEqual(heatmap.shape, (6, 6))
        self.assertTrue(np.all(np.isfinite(heatmap)))
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
