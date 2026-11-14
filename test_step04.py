import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pipeline.step04_refine import refine


class Step04RefineTests(unittest.TestCase):
    def setUp(self):
        self.matte = Image.fromarray(
            np.array(
                [
                    [[10, 20, 30, 20], [40, 50, 60, 40]],
                    [[70, 80, 90, 60], [100, 110, 120, 80]],
                ],
                dtype=np.uint8,
            ),
            mode="RGBA",
        )
        self.guide = Image.new("RGB", (2, 2), (120, 130, 140))

    @patch("pipeline.step04_refine.cv2")
    def test_output_is_rgba_same_size_and_preserves_rgb(self, cv2_mock):
        refined_alpha = np.array([[90, 91], [92, 93]], dtype=np.uint8)
        cv2_mock.ximgproc.guidedFilter.return_value = refined_alpha

        result = refine(self.matte, self.guide)

        original = np.asarray(self.matte)
        output = np.asarray(result)
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.size, self.matte.size)
        np.testing.assert_array_equal(output[:, :, :3], original[:, :, :3])
        np.testing.assert_array_equal(output[:, :, 3], refined_alpha)
        cv2_mock.ximgproc.guidedFilter.assert_called_once()

    def test_wrong_input_types_raise_type_error(self):
        with self.assertRaises(TypeError):
            refine("not an image", self.guide)
        with self.assertRaises(TypeError):
            refine(self.matte, "not an image")


if __name__ == "__main__":
    unittest.main()
