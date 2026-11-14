"""Step 08: cloud face restoration and sharpening via fal-ai/codeformer."""

from __future__ import annotations

import base64
import io
import urllib.request

from PIL import Image

try:
    import fal_client
except ImportError:
    fal_client = None

CODEFORMER_FIDELITY = 0.95


def _png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def restore(img: Image.Image, fidelity: float = CODEFORMER_FIDELITY) -> Image.Image:
    """Return a CodeFormer-restored RGB image at the original input size."""
    if not isinstance(img, Image.Image):
        raise TypeError("img must be a PIL.Image.Image")
    if fal_client is None:
        raise RuntimeError("Step 8 requires fal_client")

    image = img if img.mode == "RGB" else img.convert("RGB")
    original_size = image.size

    try:
        result = fal_client.subscribe(
            "fal-ai/codeformer",
            arguments={
                "image_url":          _png_data_uri(image),
                "fidelity":           fidelity,
                "upscale":            2,
                "face_upsample":      True,
                "background_enhance": False,
            },
            with_logs=False,
        )
        output_url = result["image"]["url"]
        import requests as _requests
        _resp = _requests.get(output_url, timeout=60)
        _resp.raise_for_status()
        restored = Image.open(io.BytesIO(_resp.content)).convert("RGB")

        # CodeFormer upscales 2x — normalise back to original dimensions.
        return restored.resize(original_size, Image.Resampling.LANCZOS)

    except Exception as error:
        raise RuntimeError(f"CodeFormer restore failed: {error}") from error
