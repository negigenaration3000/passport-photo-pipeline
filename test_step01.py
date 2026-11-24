import math
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from pipeline.step01_align import _face_center_and_eyes, align


class Step01AlignTests(unittest.TestCase):
    def setUp(self):
        # A non-white corner makes it easy to verify white padding after rotation.
        self.image = Image.new("RGB", (100, 100), (30, 40, 50))
        ImageDraw.Draw(self.image).rectangle((40, 40, 60, 60), fill=(255, 0, 0))
        self.face_center = (50.0, 50.0)

    @staticmethod
    def eyes_at(angle_degrees):
        """Two eyes centred at (50, 50), separated by 40 pixels."""
        radians = math.radians(angle_degrees)
        offset_x, offset_y = 20 * math.cos(radians), 20 * math.sin(radians)
        return (50 - offset_x, 50 - offset_y), (50 + offset_x, 50 + offset_y)

    def test_no_tilt_returns_original_without_resampling(self):
        left_eye, right_eye = self.eyes_at(0)
        with patch(
            "pipeline.step01_align._face_center_and_eyes",
            return_value=(self.face_center, left_eye, right_eye),
        ):
            result = align(self.image)

        self.assertIs(result, self.image)
        self.assertEqual(result.tobytes(), self.image.tobytes())

    def test_small_tilt_below_threshold_is_not_rotated(self):
        left_eye, right_eye = self.eyes_at(1.9)
        with patch(
            "pipeline.step01_align._face_center_and_eyes",
            return_value=(self.face_center, left_eye, right_eye),
        ):
            result = align(self.image)

        self.assertIs(result, self.image)

    def test_large_tilt_rotates_on_same_canvas_with_white_padding(self):
        left_eye, right_eye = self.eyes_at(15)
        with patch(
            "pipeline.step01_align._face_center_and_eyes",
            return_value=(self.face_center, left_eye, right_eye),
        ):
            result = align(self.image)

        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, self.image.size)
        self.assertNotEqual(result.tobytes(), self.image.tobytes())
        corners = ((0, 0), (99, 0), (0, 99), (99, 99))
        self.assertIn((255, 255, 255), [result.getpixel(point) for point in corners])

    def test_small_image_upscales_for_detection_only(self):
        small = Image.new("RGB", (160, 200), (30, 40, 50))
        seen_sizes = []

        def fake_detector(img):
            seen_sizes.append(img.size)
            # Upscaled to 400x500 -> return coords in upscaled space (center 200,250)
            return (200.0, 250.0), (150.0, 250.0), (250.0, 250.0)

        with patch(
            "pipeline.step01_align._landmark_face_center_and_eyes",
            side_effect=fake_detector,
        ):
            center, left_eye, right_eye = _face_center_and_eyes(small)

        self.assertEqual(seen_sizes, [(400, 500)])
        self.assertAlmostEqual(center[0], 80.0)
        self.assertAlmostEqual(center[1], 100.0)
        self.assertAlmostEqual(left_eye[0], 60.0)
        self.assertAlmostEqual(right_eye[0], 100.0)


if __name__ == "__main__":
    unittest.main()
