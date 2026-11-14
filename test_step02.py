import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pipeline.step02_denoise import BLUR_VARIANCE_THRESHOLD, denoise


class Step02DenoiseTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (8, 8), (30, 40, 50))

    def test_clean_image_passes_through_unchanged(self):
        with patch(
            "pipeline.step02_denoise._laplacian_variance",
            return_value=BLUR_VARIANCE_THRESHOLD + 1,
        ), patch("pipeline.step02_denoise._denoise_bgr") as denoise_bgr:
            result = denoise(self.image)

        self.assertIs(result, self.image)
        denoise_bgr.assert_not_called()

    def test_blurry_image_is_denoised(self):
        # The mocked NLM output differs visibly from the input so this test
        # verifies both conditional routing and RGB output conversion.
        mocked_bgr_output = np.full((8, 8, 3), (3, 2, 1), dtype=np.uint8)
        with patch(
            "pipeline.step02_denoise._laplacian_variance",
            return_value=BLUR_VARIANCE_THRESHOLD - 1,
        ), patch(
            "pipeline.step02_denoise._denoise_bgr",
            return_value=mocked_bgr_output,
        ) as denoise_bgr:
            result = denoise(self.image)

        denoise_bgr.assert_called_once()
        self.assertIsNot(result, self.image)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.getpixel((0, 0)), (1, 2, 3))

    def test_threshold_boundary_is_not_denoised(self):
        with patch(
            "pipeline.step02_denoise._laplacian_variance",
            return_value=BLUR_VARIANCE_THRESHOLD,
        ), patch("pipeline.step02_denoise._denoise_bgr") as denoise_bgr:
            result = denoise(self.image)

        self.assertIs(result, self.image)
        denoise_bgr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
