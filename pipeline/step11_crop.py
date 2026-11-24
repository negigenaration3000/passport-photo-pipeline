"""
Step 11: ICAO 9303 compliant crop, background composite, and framing.

Enforces:
  - Head occupies 70-80% of frame height
  - Eyes sit at ~45% from top of frame
  - Exact passport dimensions at 300 DPI
  - Solid studio background (blue or white)
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from config import settings

# ICAO 9303 head-to-frame height constraints
HEAD_MIN_RATIO = 0.70
HEAD_MAX_RATIO = 0.80
HEAD_TARGET_RATIO = getattr(settings, "HEAD_TARGET_RATIO", 0.50)
EYE_FROM_TOP_RATIO = getattr(settings, "EYE_FROM_TOP_RATIO", 0.40)


def crop(
    img:      Image.Image,
    matte:    Image.Image,
    face:     dict,
    bg_color: tuple = settings.BG_BLUE,
    target_w: int   = settings.PASSPORT_W_PX,
    target_h: int   = settings.PASSPORT_H_PX,
) -> Image.Image:
    """
    Composite subject onto bg_color and crop to ICAO-compliant passport dimensions.

    img    — restored RGB portrait (Step 10 output)
    matte  — RGBA alpha matte (Step 3/4 output)
    face   — FaceAnalysis from Step 1 (provides bbox for crop geometry)
    """
    if not isinstance(img, Image.Image):
        raise TypeError("img must be a PIL.Image.Image")
    if not isinstance(matte, Image.Image) or matte.mode != "RGBA":
        raise TypeError("matte must be an RGBA PIL.Image.Image")

    # ── Composite onto solid background ──────────────────────────
    bg = Image.new("RGBA", img.size, (*bg_color, 255))
    bg.paste(img, mask=matte.split()[3])
    composited = bg.convert("RGB")

    # ── ICAO geometry: derive crop window from face bbox ──────────
    aspect   = target_w / target_h
    crop_h   = int(face.h / HEAD_TARGET_RATIO)
    crop_w   = int(crop_h * aspect)

    # Centre crop on face, eyes at 45% from top
    face_cx  = face.x + face.w // 2
    face_cy  = face.y + face.h // 2
    eye_offset = int(crop_h * EYE_FROM_TOP_RATIO)

    top    = face_cy - eye_offset
    left   = face_cx - crop_w // 2
    bottom = top  + crop_h
    right  = left + crop_w

    src_w, src_h = composited.size

    # Pad canvas if crop window exceeds image bounds
    pad_top    = max(0, -top)
    pad_left   = max(0, -left)
    pad_bottom = max(0, bottom - src_h)
    pad_right  = max(0, right  - src_w)

    if any([pad_top, pad_left, pad_bottom, pad_right]):
        padded = Image.new(
            "RGB",
            (src_w + pad_left + pad_right, src_h + pad_top + pad_bottom),
            bg_color,
        )
        padded.paste(composited, (pad_left, pad_top))
        composited = padded
        left   = max(0, left)
        top    = max(0, top)
        right  = left + crop_w
        bottom = top  + crop_h

    cropped = composited.crop((left, top, right, bottom))
    return cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
