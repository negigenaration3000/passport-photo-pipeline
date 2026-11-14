import unittest

import numpy as np
from PIL import Image

from pipeline.step06_luminance import LUMINANCE_DELTA_THRESHOLD, check


class Step06LuminanceTests(unittest.TestCase):
    def setUp(self):
        self.skin = {"skin": np.ones((2, 4), dtype=bool)}

    def test_balanced_skin_does_not_need_correction(self):
        image = Image.new("RGB", (4, 2), (100, 100, 100))

        result = check(image, self.skin)

        self.assertEqual(result["delta"], 0.0)
        self.assertFalse(result["needs_correction"])

    def test_unbalanced_skin_needs_correction(self):
        pixels = np.zeros((2, 4, 3), dtype=np.uint8)
        pixels[:, :2] = (0, 0, 0)
        pixels[:, 2:] = (100, 100, 100)

        result = check(Image.fromarray(pixels, "RGB"), self.skin)

        self.assertGreater(result["delta"], LUMINANCE_DELTA_THRESHOLD)
        self.assertTrue(result["needs_correction"])

    def test_no_skin_pixels_returns_zero_without_correction(self):
        image = Image.new("RGB", (4, 2), (255, 255, 255))

        result = check(image, {"skin": np.zeros((2, 4), dtype=bool)})

        self.assertEqual(result["delta"], 0.0)
        self.assertFalse(result["needs_correction"])

    def test_exact_threshold_does_not_need_correction(self):
        # (R, G, B) = (2, 8, 129) gives exactly
        # 0.299*2 + 0.587*8 + 0.114*129 == 20.0.
        pixels = np.zeros((2, 4, 3), dtype=np.uint8)
        pixels[:, 2:] = (2, 8, 129)
        image = Image.fromarray(pixels, "RGB")

        result = check(image, self.skin)

        self.assertAlmostEqual(result["delta"], LUMINANCE_DELTA_THRESHOLD, places=5)
        self.assertFalse(result["needs_correction"])


if __name__ == "__main__":
    unittest.main()
