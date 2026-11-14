"""
Steps 09 & 10: frequency separation and texture re-blend.

Step 09 — isolate high-frequency skin texture from the post-Step-7
           image (pre-CodeFormer). This preserves authentic pores,
           fine hair, and surface detail before the generative pass
           has a chance to smooth them away.

Step 10 — merge the saved high-frequency layer back onto the
           CodeFormer output so the final image has studio-quality
           lighting with 100% authentic skin texture.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Gaussian blur radius for separating low/high frequency layers.
# Larger = smoother low-freq layer = more texture in high-freq layer.
FREQUENCY_BLUR_RADIUS = 3


def separate(img: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    Split img into (low_freq_image, high_freq_layer).

    low_freq_image  — blurred PIL Image (RGB), contains lighting/colour
    high_freq_layer — numpy float32 array, contains texture detail
                      stored as signed residual: original - blurred
    """
    if not isinstance(img, Image.Image):
        raise TypeError("img must be a PIL.Image.Image")

    image = img if img.mode == "RGB" else img.convert("RGB")
    low_freq = image.filter(
        __import__("PIL.ImageFilter", fromlist=["GaussianBlur"])
        .GaussianBlur(radius=FREQUENCY_BLUR_RADIUS)
    )
    high_freq = (
        np.asarray(image, dtype=np.float32)
        - np.asarray(low_freq, dtype=np.float32)
    )
    return low_freq, high_freq


def blend(restored: Image.Image, high_freq: np.ndarray) -> Image.Image:
    """
    Add the saved high-frequency texture back onto the CodeFormer output.

    restored   — PIL Image (RGB) from Step 8
    high_freq  — numpy float32 residual from Step 9 separate()
    """
    if not isinstance(restored, Image.Image):
        raise TypeError("restored must be a PIL.Image.Image")
    if not isinstance(high_freq, np.ndarray):
        raise TypeError("high_freq must be a numpy ndarray")

    image = restored if restored.mode == "RGB" else restored.convert("RGB")
    if image.size != (high_freq.shape[1], high_freq.shape[0]):
        raise ValueError("restored and high_freq must have the same dimensions")

    merged = np.asarray(image, dtype=np.float32) + high_freq
    return Image.fromarray(np.clip(merged, 0, 255).astype(np.uint8), mode="RGB")
