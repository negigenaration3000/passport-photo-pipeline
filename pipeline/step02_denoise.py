"""Step 02: conditionally remove print/scan noise with conservative FastNLM."""

from __future__ import annotations

from typing import Final

from PIL import Image


BLUR_VARIANCE_THRESHOLD: Final[float] = 100.0


def _laplacian_variance(rgb):
    """Measure high-frequency detail: lower values indicate a degraded image."""
    import cv2

    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())


def _denoise_bgr(bgr):
    """Apply deliberately conservative colour non-local means denoising."""
    import cv2

    return cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h=5,
        hColor=5,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def denoise(img: Image.Image) -> Image.Image:
    """Denoise degraded RGB images; leave clean images entirely untouched.

    Laplacian variance quantifies the amount of high-frequency structure in the
    source.  A value below ``BLUR_VARIANCE_THRESHOLD`` is treated as a soft,
    scanned, or photographed print and is passed through FastNLM.
    """
    if not isinstance(img, Image.Image):
        raise TypeError("img must be a PIL.Image.Image")

    image = img if img.mode == "RGB" else img.convert("RGB")

    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Step 2 requires OpenCV and NumPy") from error

    rgb = np.asarray(image)
    variance = _laplacian_variance(rgb)
    # The threshold is exclusive: an image exactly at 100.0 remains clean.
    if variance >= BLUR_VARIANCE_THRESHOLD:
        return image

    # OpenCV's coloured NLM function operates on BGR arrays. Convert back before
    # creating the output PIL image to preserve the public RGB contract.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    denoised_bgr = _denoise_bgr(bgr)
    denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(denoised_rgb, mode="RGB")
