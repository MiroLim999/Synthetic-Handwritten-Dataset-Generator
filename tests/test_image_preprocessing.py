import unittest

import numpy as np
from PIL import Image

from src.image_preprocessing import resize_and_pad


class ImagePreprocessingTests(unittest.TestCase):
    def test_wide_crop_is_scaled_uniformly_and_centered(self):
        image = Image.new("RGB", (1000, 100), "black")
        output = resize_and_pad(image, 384)
        self.assertEqual(output.size, (384, 384))
        pixels = np.asarray(output)
        dark = np.where(pixels[:, :, 0] < 128)
        rendered_height = int(dark[0].max() - dark[0].min() + 1)
        rendered_width = int(dark[1].max() - dark[1].min() + 1)
        self.assertEqual(rendered_width, 384)
        self.assertIn(rendered_height, (38, 39))
        self.assertLessEqual(abs(float(dark[0].mean()) - 191.5), 1.0)

    def test_tall_and_short_crops_preserve_geometry(self):
        tall = resize_and_pad(Image.new("RGB", (20, 100), "black"), 200)
        square = resize_and_pad(Image.new("RGB", (20, 20), "black"), 200)
        self.assertEqual(tall.size, (200, 200))
        self.assertEqual(square.size, (200, 200))

    def test_invalid_size_is_rejected(self):
        with self.assertRaises(ValueError):
            resize_and_pad(Image.new("RGB", (10, 10)), 0)


if __name__ == "__main__":
    unittest.main()
