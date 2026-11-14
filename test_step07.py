import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from pipeline.step07_equalise import equalise


class Step07EqualiseTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (2, 2), (10, 20, 30))
        self.masks = {"skin": np.array([[True, False], [False, False]])}

    def test_no_correction_returns_same_object(self):
        result = equalise(self.image, self.masks, {"needs_correction": False})

        self.assertIs(result, self.image)

    @patch("pipeline.step07_equalise.cv2")
    def test_correction_returns_rgb_same_size_and_preserves_non_skin(self, cv2_mock):
        lab = np.zeros((2, 2, 3), dtype=np.uint8)
        enhanced_rgb = np.full((2, 2, 3), (200, 201, 202), dtype=np.uint8)
        cv2_mock.cvtColor.side_effect = [lab, enhanced_rgb]
        cv2_mock.split.return_value = (lab[:, :, 0], lab[:, :, 1], lab[:, :, 2])
        cv2_mock.createCLAHE.return_value = MagicMock(apply=lambda channel: channel)

        result = equalise(self.image, self.masks, {"needs_correction": True})

        output = np.asarray(result)
        original = np.asarray(self.image)
        self.assertIsNot(result, self.image)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, self.image.size)
        np.testing.assert_array_equal(output[~self.masks["skin"]], original[~self.masks["skin"]])
        np.testing.assert_array_equal(output[self.masks["skin"]], enhanced_rgb[self.masks["skin"]])

    def test_wrong_input_types_raise_type_error(self):
        with self.assertRaises(TypeError):
            equalise("not an image", self.masks, {"needs_correction": False})
        with self.assertRaises(TypeError):
            equalise(self.image, "not masks", {"needs_correction": False})
        with self.assertRaises(TypeError):
            equalise(self.image, self.masks, "not a check")


if __name__ == "__main__":
    unittest.main()
