"""Step 03: isolate the portrait subject with fal.ai BiRefNet alpha matting."""

from __future__ import annotations

import base64
import io
import urllib.request

from PIL import Image

try:
    import fal_client
except ImportError:  # Lets unit tests import this module without the cloud SDK.
    fal_client = None


def _png_data_uri(image: Image.Image) -> str:
    """Encode an RGB PIL image in the data-URI format expected by fal.ai."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def matte(img: Image.Image) -> Image.Image:
    """Return an RGBA BiRefNet matte, raising clearly if the API fails."""
    if not isinstance(img, Image.Image):
        raise TypeError("img must be a PIL.Image.Image")

    image = img if img.mode == "RGB" else img.convert("RGB")
    if fal_client is None:
        raise RuntimeError("BiRefNet alpha matting failed: fal_client is not installed")

    try:
        result = fal_client.subscribe(
            "fal-ai/birefnet",
            arguments={"image_url": _png_data_uri(image)},
            with_logs=False,
        )
        output_url = result["image"]["url"]
        import requests as _requests
        _resp = _requests.get(output_url, timeout=60)
        _resp.raise_for_status()
        return Image.open(io.BytesIO(_resp.content)).convert("RGBA")
    except Exception as error:
        raise RuntimeError(f"BiRefNet alpha matting failed: {error}") from error
