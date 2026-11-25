import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from pipeline.comfy_client import ComfyUIClient, WORKFLOW_TEMPLATE_PATH


class ComfyUIIntegrationTests(unittest.TestCase):
    def test_workflow_json_exists_and_is_valid(self):
        self.assertTrue(WORKFLOW_TEMPLATE_PATH.is_file(), "comfy_passport_workflow_api.json must exist")
        with open(WORKFLOW_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Verify essential node types are present
        class_types = [node.get("class_type") for node in data.values()]
        self.assertIn("LoadImage", class_types)
        self.assertIn("BiRefNet_Loader", class_types)
        self.assertIn("ApplyInstantID", class_types)
        self.assertIn("SolidColorCanvas", class_types)
        self.assertIn("KSampler", class_types)
        self.assertIn("SaveImage", class_types)

    @patch("pipeline.comfy_client.urllib.request.urlopen")
    def test_is_alive_check(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        
        client = ComfyUIClient("http://127.0.0.1:8188")
        self.assertTrue(client.is_alive())

    @patch("pipeline.comfy_client.urllib.request.urlopen")
    def test_generate_flow(self, mock_urlopen):
        # Mock upload response
        mock_upload_resp = MagicMock()
        mock_upload_resp.read.return_value = json.dumps({"name": "uploaded_test.jpg"}).encode("utf-8")
        
        # Mock prompt response
        mock_prompt_resp = MagicMock()
        mock_prompt_resp.read.return_value = json.dumps({"prompt_id": "test-prompt-123"}).encode("utf-8")

        # Mock history response
        mock_hist_resp = MagicMock()
        mock_hist_resp.read.return_value = json.dumps({
            "test-prompt-123": {
                "outputs": {
                    "18": {
                        "images": [{"filename": "out_test.png", "subfolder": "", "type": "output"}]
                    }
                }
            }
        }).encode("utf-8")

        # Mock view image response (returns a dummy 100x100 RGB image)
        import io
        img_bytes = io.BytesIO()
        Image.new("RGB", (100, 100), (24, 104, 183)).save(img_bytes, format="PNG")
        mock_view_resp = MagicMock()
        mock_view_resp.read.return_value = img_bytes.getvalue()

        # Chain responses
        mock_urlopen.return_value.__enter__.side_effect = [
            mock_upload_resp,
            mock_prompt_resp,
            mock_hist_resp,
            mock_view_resp,
        ]

        test_img_path = Path("work/temp_test.jpg")
        Image.new("RGB", (50, 50)).save(test_img_path)

        try:
            client = ComfyUIClient("http://127.0.0.1:8188")
            result_img = client.generate(test_img_path, bg="blue")
            self.assertIsInstance(result_img, Image.Image)
            self.assertEqual(result_img.size, (100, 100))
        finally:
            if test_img_path.is_file():
                test_img_path.unlink()
