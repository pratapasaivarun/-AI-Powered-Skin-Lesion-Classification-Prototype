import unittest
from types import SimpleNamespace

import numpy as np

from src.image_utils import get_model_image_size, preprocess_rgb_image


class PreprocessRgbImageTests(unittest.TestCase):
    def test_resizes_and_normalizes_rgb_image(self):
        image = np.full((8, 12, 3), 255, dtype=np.uint8)

        result = preprocess_rgb_image(image, image_size=(6, 4))

        self.assertEqual(result.shape, (1, 4, 6, 3))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, np.ones((1, 4, 6, 3), dtype=np.float32))

    def test_rejects_non_rgb_images(self):
        with self.assertRaisesRegex(ValueError, "RGB image"):
            preprocess_rgb_image(np.zeros((8, 12), dtype=np.uint8))

    def test_rejects_non_positive_dimensions(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            preprocess_rgb_image(np.zeros((8, 12, 3), dtype=np.uint8), (0, 4))

    def test_uses_fixed_model_dimensions_when_available(self):
        model = SimpleNamespace(input_shape=(None, 28, 32, 3))

        self.assertEqual(get_model_image_size(model), (32, 28))

    def test_falls_back_for_dynamic_model_dimensions(self):
        model = SimpleNamespace(input_shape=(None, None, None, 3))

        self.assertEqual(get_model_image_size(model, (224, 224)), (224, 224))


if __name__ == "__main__":
    unittest.main()
