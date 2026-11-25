import unittest
from PIL import Image

from config import settings
from pipeline.face_detect import FaceAnalysis
from pipeline.step11_crop import crop


class Step11CropTests(unittest.TestCase):
    def setUp(self):
        self.img = Image.new("RGB", (600, 800), (200, 200, 200))
        self.matte = Image.new("RGBA", (600, 800), (0, 0, 0, 255))
        self.face = FaceAnalysis(x=200, y=200, w=200, h=250, tilt_deg=0.0)
        self.face_dict = {"x": 200, "y": 200, "w": 200, "h": 250}

    def test_crop_with_matte(self):
        result = crop(self.img, self.matte, self.face)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (settings.PASSPORT_W_PX, settings.PASSPORT_H_PX))

    def test_crop_without_matte(self):
        result = crop(self.img, None, self.face)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (settings.PASSPORT_W_PX, settings.PASSPORT_H_PX))

    def test_crop_with_dict_face(self):
        result = crop(self.img, None, self.face_dict)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (settings.PASSPORT_W_PX, settings.PASSPORT_H_PX))

    def test_crop_invalid_inputs(self):
        with self.assertRaises(TypeError):
            crop("not an image", None, self.face)
        with self.assertRaises(TypeError):
            crop(self.img, Image.new("RGB", (100, 100)), self.face)  # Matte must be RGBA
