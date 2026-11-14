import io
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from pipeline.step03_matte import matte


def _rgba_png_bytes():
    """Build a tiny transparent PNG returned by the mocked fal.ai output URL."""
    buffer = io.BytesIO()
    Image.new("RGBA", (2, 2), (10, 20, 30, 128)).save(buffer, format="PNG")
    return buffer.getvalue()


class Step03MatteTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (2, 2), (10, 20, 30))

    @patch("requests.get")
    @patch("pipeline.step03_matte.fal_client")
    def test_successful_matte_returns_rgba(self, fal_client, mock_get):
        fal_client.subscribe.return_value = {"image": {"url": "https://example.test/matte.png"}}
        response = MagicMock()
        response.content = _rgba_png_bytes()
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        result = matte(self.image)

        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.size, (2, 2))
        self.assertEqual(result.getpixel((0, 0)), (10, 20, 30, 128))
        fal_client.subscribe.assert_called_once()

    @patch("pipeline.step03_matte.fal_client")
    def test_api_failure_raises_runtime_error(self, fal_client):
        fal_client.subscribe.side_effect = ConnectionError("service unavailable")

        with self.assertRaisesRegex(RuntimeError, "BiRefNet alpha matting failed"):
            matte(self.image)


if __name__ == "__main__":
    unittest.main()
