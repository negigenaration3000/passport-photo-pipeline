import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from PIL import Image

from orchestrator import process
from pipeline.face_detect import FaceAnalysis


class OrchestratorStudioModeTests(unittest.TestCase):
    def setUp(self):
        self.test_img_path = Path("work/test_input.jpg")
        self.output_path = Path("work/test_output.jpg")
        img = Image.new("RGB", (300, 400), (120, 140, 160))
        img.save(self.test_img_path)

    def tearDown(self):
        if self.test_img_path.is_file():
            self.test_img_path.unlink()
        if self.output_path.is_file():
            self.output_path.unlink()

    @patch("orchestrator.analyse")
    @patch("orchestrator._studio_enhance_qwen")
    def test_studio_mode_flow(self, mock_enhance, mock_analyse):
        # Return a synthetic portrait and face detection
        mock_enhance.return_value = Image.new("RGB", (400, 500), (24, 104, 183))
        mock_analyse.return_value = FaceAnalysis(x=100, y=100, w=200, h=250, tilt_deg=0.0)

        progress_events = []
        def on_progress(pct, msg):
            progress_events.append((pct, msg))

        meta = process(
            image_path=self.test_img_path,
            output_path=self.output_path,
            bg="blue",
            mode="studio",
            on_progress=on_progress,
        )

        self.assertEqual(meta["mode"], "studio")
        self.assertEqual(meta["cost_usd"], 0.03)
        self.assertTrue(self.output_path.is_file())
        self.assertTrue(len(progress_events) >= 4)
        self.assertEqual(progress_events[-1][0], 100)

        # Verify output size is exact passport dimensions
        with Image.open(self.output_path) as out_img:
            self.assertEqual(out_img.size, (413, 531))

    @patch("orchestrator.tile")
    @patch("orchestrator.analyse")
    @patch("orchestrator._studio_enhance_qwen")
    def test_studio_mode_tile(self, mock_enhance, mock_analyse, mock_tile):
        mock_enhance.return_value = Image.new("RGB", (400, 500), (24, 104, 183))
        mock_analyse.return_value = FaceAnalysis(x=100, y=100, w=200, h=250, tilt_deg=0.0)
        mock_tile.return_value = Path("work/tiled.jpg")

        meta = process(
            image_path=self.test_img_path,
            output_path="work/tiled.jpg",
            bg="white",
            mode="studio",
            tile_output=True,
        )

        self.assertEqual(Path(meta["output"]), Path("work/tiled.jpg"))
        mock_tile.assert_called_once()
