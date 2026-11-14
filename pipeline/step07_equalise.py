"""Step 07: conditionally equalise skin lighting with LAB-space CLAHE."""

from __future__ import annotations

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # Keeps imports/test discovery working without OpenCV installed.
    cv2 = None


CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


def equalise(img: Image.Image, masks: dict, check: dict) -> Image.Image:
    """Apply CLAHE to skin pixels only when Step 6 reports imbalance."""
    if not isinstance(img, Image.Image):
        raise TypeError("img must be a PIL.Image.Image")
    if not isinstance(masks, dict) or "skin" not in masks:
        raise TypeError("masks must be a dict containing a skin mask")
    if not isinstance(check, dict) or "needs_correction" not in check:
        raise TypeError("check must be a dict containing needs_correction")

    image = img if img.mode == "RGB" else img.convert("RGB")
    if not check["needs_correction"]:
        return image

    skin = np.asarray(masks["skin"], dtype=bool)
    width, height = image.size
    if skin.shape != (height, width):
        raise ValueError("skin mask shape must match the image dimensions")
    if cv2 is None:
        raise RuntimeError("Step 7 requires OpenCV")

    rgb = np.asarray(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    equalised_lightness = clahe.apply(lightness)
    equalised_rgb = cv2.cvtColor(
        cv2.merge((equalised_lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2RGB,
    )

    # Preserve every non-skin pixel byte-for-byte; only skin uses the CLAHE result.
    output = rgb.copy()
    output[skin] = equalised_rgb[skin]
    return Image.fromarray(output, mode="RGB")
