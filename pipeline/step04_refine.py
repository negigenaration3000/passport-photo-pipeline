"""Step 04: smooth jagged alpha edges using the source portrait as a guide."""

from __future__ import annotations

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # Lets unit tests import this module without OpenCV.
    cv2 = None


GUIDED_FILTER_RADIUS = 8
GUIDED_FILTER_EPS = 1e-3


def refine(matte: Image.Image, guide: Image.Image) -> Image.Image:
    """Return ``matte`` with only its alpha channel guided-filtered by ``guide``."""
    if not isinstance(matte, Image.Image) or not isinstance(guide, Image.Image):
        raise TypeError("matte and guide must be PIL.Image.Image instances")
    if matte.mode != "RGBA":
        raise TypeError("matte must be an RGBA PIL image")
    if guide.mode != "RGB":
        raise TypeError("guide must be an RGB PIL image")
    if matte.size != guide.size:
        raise ValueError("matte and guide must have the same dimensions")

    try:
        guided_filter = cv2.ximgproc.guidedFilter
    except AttributeError as error:
        raise RuntimeError("Step 4 requires opencv-contrib-python (cv2.ximgproc)") from error

    matte_rgba = np.asarray(matte)
    alpha = matte_rgba[:, :, 3]
    # OpenCV accepts BGR guidance. The guide contains real source edges, so the
    # filter smooths matte artifacts without blurring across visible hair edges.
    guide_bgr = cv2.cvtColor(np.asarray(guide), cv2.COLOR_RGB2BGR)
    refined_alpha = guided_filter(
        guide_bgr,
        alpha,
        radius=GUIDED_FILTER_RADIUS,
        eps=GUIDED_FILTER_EPS,
    )

    # Keep RGB byte-for-byte from BiRefNet; only substitute the refined alpha.
    output = matte_rgba.copy()
    output[:, :, 3] = np.clip(refined_alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(output, mode="RGBA")
