import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pipeline.step05_parse import parse


class Step05ParseTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (3, 2), (30, 40, 50))
        self.labels = np.array(
            [[0, 1, 4], [5, 10, 11]],
            dtype=np.uint8,
        )
        self.expected_keys = {
            "background", "skin", "left_eye", "right_eye", "nose", "mouth",
            "hair", "neck", "clothing", "face",
        }

    @patch("pipeline.step05_parse._infer_labels")
    def test_returns_all_boolean_masks_at_input_shape(self, infer_labels):
        infer_labels.return_value = self.labels

        masks = parse(self.image)

        self.assertEqual(set(masks), self.expected_keys)
        for mask in masks.values():
            self.assertEqual(mask.dtype, np.dtype(bool))
            self.assertEqual(mask.shape, (2, 3))

    @patch("pipeline.step05_parse._infer_labels")
    def test_face_mask_is_union_of_facial_submasks(self, infer_labels):
        infer_labels.return_value = self.labels

        masks = parse(self.image)
        expected_face = masks["skin"] | masks["left_eye"] | masks["right_eye"] | masks["nose"] | masks["mouth"]

        np.testing.assert_array_equal(masks["face"], expected_face)
        self.assertTrue(masks["face"][0, 1])  # skin
        self.assertTrue(masks["face"][1, 2])  # inner mouth
        self.assertFalse(masks["face"][0, 0])  # background


if __name__ == "__main__":
    unittest.main()
