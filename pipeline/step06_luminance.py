"""Step 06: detect left/right skin-brightness imbalance in a portrait."""

from __future__ import annotations

import numpy as np
from PIL import Image


LUMINANCE_DELTA_THRESHOLD = 20.0


def check(img: Image.Image, masks: dict) -> dict:
    """Return the masked left/right luminance delta and correction decision."""
    if not isinstance(img, Image.Image):
        raise TypeError("img must be a PIL.Image.Image")
    if not isinstance(masks, dict) or "skin" not in masks:
        raise TypeError("masks must be a dict containing a skin mask")

    image = img if img.mode == "RGB" else img.convert("RGB")
    skin = np.asarray(masks["skin"], dtype=bool)
    width, height = image.size
    if skin.shape != (height, width):
        raise ValueError("skin mask shape must match the image dimensions")

    rgb = np.asarray(image, dtype=np.float32)
    # ITU-R BT.601 luminance weights approximate perceived brightness.
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    midpoint = width // 2
    left_values = luminance[:, :midpoint][skin[:, :midpoint]]
    right_values = luminance[:, midpoint:][skin[:, midpoint:]]

    if left_values.size == 0 or right_values.size == 0:
        delta = 0.0
    else:
        delta = abs(float(left_values.mean()) - float(right_values.mean()))

    return {
        "delta": delta,
        "needs_correction": delta > LUMINANCE_DELTA_THRESHOLD,
    }
