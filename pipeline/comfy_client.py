"""
ComfyUI API Client
Communicates with any running ComfyUI instance (Local, RunPod, Modal, or Cloud GPU)
via the native REST API (/prompt, /upload/image, /view) and WebSocket (/ws).
"""

from __future__ import annotations

import io
import json
import os
import urllib.request
import urllib.parse
import uuid
from pathlib import Path
from typing import Callable, Optional
from PIL import Image

DEFAULT_COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
WORKFLOW_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "comfy_passport_workflow_api.json"


class ComfyUIClient:
    def __init__(self, base_url: str = DEFAULT_COMFYUI_URL):
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def is_alive(self) -> bool:
        """Check if ComfyUI instance is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/system_stats", headers={"User-Agent": "PassportPipeline"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def upload_image(self, image_path: str | Path) -> str:
        """Upload an image to ComfyUI's input directory using multipart form-data."""
        path = Path(image_path)
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        
        with open(path, "rb") as f:
            file_data = f.read()

        filename = f"upload_{uuid.uuid4().hex[:8]}_{path.name}"
        
        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode("utf-8"))
        body.extend(b"Content-Type: image/jpeg\r\n\r\n")
        body.extend(file_data)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        req = urllib.request.Request(
            f"{self.base_url}/upload/image",
            data=bytes(body),
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "PassportPipeline",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("name", filename)

    def generate(
        self,
        image_path: str | Path,
        bg: str = "blue",
        on_progress: Optional[Callable[[int, str], None]] = None,
    ) -> Image.Image:
        """
        Execute the ComfyUI passport workflow on the given image.
        Returns the rendered PIL Image before ICAO cropping.
        """
        if on_progress:
            on_progress(10, "[ComfyUI] Uploading portrait to GPU node...")

        uploaded_name = self.upload_image(image_path)

        # Load workflow template
        if not WORKFLOW_TEMPLATE_PATH.is_file():
            raise FileNotFoundError(f"Workflow template missing at {WORKFLOW_TEMPLATE_PATH}")

        with open(WORKFLOW_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        # 1. Inject uploaded filename into LoadImage node (ID: 1)
        workflow["1"]["inputs"]["image"] = uploaded_name

        # 2. Configure background color (Node ID: 10)
        if bg == "white":
            workflow["10"]["inputs"]["red"] = 255
            workflow["10"]["inputs"]["green"] = 255
            workflow["10"]["inputs"]["blue"] = 255
        else:
            workflow["10"]["inputs"]["red"] = 24
            workflow["10"]["inputs"]["green"] = 104
            workflow["10"]["inputs"]["blue"] = 183

        # 3. Post prompt to /prompt
        if on_progress:
            on_progress(25, "[ComfyUI] Queuing workflow (BiRefNet + InstantID + Relight)...")

        payload = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "PassportPipeline"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            prompt_id = res_data.get("prompt_id")

        if not prompt_id:
            raise RuntimeError(f"ComfyUI rejected prompt: {res_data}")

        if on_progress:
            on_progress(45, f"[ComfyUI] Executing job {prompt_id[:8]} on GPU...")

        # 4. Poll history until complete
        import time
        max_wait = 180
        start_time = time.time()
        output_image_data = None

        while time.time() - start_time < max_wait:
            time.sleep(1.0)
            hist_req = urllib.request.Request(f"{self.base_url}/history/{prompt_id}", headers={"User-Agent": "PassportPipeline"})
            try:
                with urllib.request.urlopen(hist_req, timeout=10) as hist_resp:
                    history = json.loads(hist_resp.read().decode("utf-8"))
            except Exception:
                continue

            if prompt_id in history:
                prompt_hist = history[prompt_id]
                outputs = prompt_hist.get("outputs", {})
                
                # Check for SaveImage node (Node ID: 18)
                if "18" in outputs and "images" in outputs["18"]:
                    images = outputs["18"]["images"]
                    if images:
                        img_info = images[0]
                        filename = img_info["filename"]
                        subfolder = img_info.get("subfolder", "")
                        img_type = img_info.get("type", "output")

                        # Fetch image via /view
                        view_params = urllib.parse.urlencode({
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": img_type,
                        })
                        view_url = f"{self.base_url}/view?{view_params}"
                        view_req = urllib.request.Request(view_url, headers={"User-Agent": "PassportPipeline"})
                        with urllib.request.urlopen(view_req, timeout=30) as view_resp:
                            output_image_data = view_resp.read()
                        break

            if on_progress:
                elapsed = int(time.time() - start_time)
                pct = min(85, 45 + elapsed * 2)
                on_progress(pct, f"[ComfyUI] Processing on GPU ({elapsed}s elapsed)...")

        if not output_image_data:
            raise TimeoutError(f"ComfyUI execution timed out after {max_wait}s")

        if on_progress:
            on_progress(90, "[ComfyUI] Image rendered! Passing to ICAO framing...")

        return Image.open(io.BytesIO(output_image_data)).convert("RGB")
