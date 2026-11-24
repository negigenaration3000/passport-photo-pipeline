"""
Passport Photo Pipeline — Master Orchestrator
Chains all 12 steps in the validated production order.
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from pathlib import Path
from typing import Callable, Optional
from PIL import Image

from config import settings
from pipeline.step01_align         import align
from pipeline.step02_denoise       import denoise
from pipeline.step03_matte         import matte
from pipeline.step04_refine        import refine
from pipeline.step05_parse         import parse
from pipeline.step06_luminance     import check as luminance_check
from pipeline.step07_equalise      import equalise
from pipeline.step08_restore       import restore
from pipeline.step09_10_frequency  import separate, blend
from pipeline.step11_crop          import crop
from pipeline.step12_tile          import tile

MAX_INPUT_LONG_SIDE = 1000


def _cap_resolution(
    image: Image.Image,
    max_long_side: int = MAX_INPUT_LONG_SIDE,
) -> Image.Image:
    """Downscale image proportionally so its long side is at most max_long_side."""
    w, h = image.size
    long_side = max(w, h)
    if long_side <= max_long_side:
        return image
    scale = max_long_side / long_side
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    print(f"[orchestrator] Capped input resolution {w}x{h} -> {new_w}x{new_h}")
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def process(
    image_path:  str | Path,
    output_path: str | Path,
    bg:          str  = "blue",     # "blue" | "white"
    enhance:     bool = True,       # set False to skip CodeFormer (dev mode)
    fidelity:    float = 0.95,
    tile_output: bool = False,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> dict:
    """
    Full 12-step passport photo pipeline.
    Returns metadata dict: tier, cost estimate, output path.
    """
    def report(pct: int, msg: str) -> None:
        print(msg)
        if on_progress is not None:
            on_progress(pct, msg)

    bg_color = settings.BG_BLUE if bg == "blue" else settings.BG_WHITE

    # ── Cap input resolution before pipeline starts ──────────────
    raw = _cap_resolution(Image.open(image_path).convert("RGB"))

    # ── Step 1: Tilt correction ───────────────────────────────────
    report(5, "[1/12] Aligning...")
    aligned = align(raw)

    # ── Step 2: Conditional denoise ───────────────────────────────
    report(12, "[2/12] Denoising...")
    denoised = denoise(aligned)

    # ── Step 3: Alpha matte ───────────────────────────────────────
    report(22, "[3/12] Matting (BiRefNet)...")
    rgba_matte = matte(denoised)

    # ── Step 4: Hair edge refinement ─────────────────────────────
    report(36, "[4/12] Refining edges...")
    refined_matte = refine(rgba_matte, denoised)

    # ── Step 5: Semantic face parsing ─────────────────────────────
    report(46, "[5/12] Parsing face regions (BiSeNet)...")
    masks = parse(denoised)

    # ── Step 6: Luminance check ───────────────────────────────────
    report(56, "[6/12] Checking luminance balance...")
    lum = luminance_check(denoised, masks)

    # ── Step 7: Conditional CLAHE ─────────────────────────────────
    report(64, "[7/12] Equalising lighting...")
    equalised = equalise(denoised, masks, lum)

    # ── Steps 9 prep: capture texture before generative step ──────
    report(70, "[9/12] Separating frequencies...")
    _, high_freq = separate(equalised)

    # ── Step 8: CodeFormer restore ────────────────────────────────
    if enhance:
        report(78, "[8/12] Restoring face (CodeFormer)...")
        restored = restore(equalised, fidelity=fidelity)
    else:
        report(78, "[8/12] Skipping CodeFormer (dev mode)...")
        restored = equalised

    # ── Step 10: Texture re-blend ─────────────────────────────────
    report(90, "[10/12] Re-blending texture...")
    textured = blend(restored, high_freq)

    # ── Step 11: ICAO crop + background ──────────────────────────
    report(95, "[11/12] Cropping to ICAO spec...")
    face_analysis = align.__wrapped__(raw) if hasattr(align, "__wrapped__") else None

    # Re-run face analysis for crop geometry
    from pipeline.face_detect import analyse
    import numpy as np
    face = analyse(np.array(textured))
    if face is None:
        raise RuntimeError("Face not detected — cannot crop to ICAO spec.")

    passport = crop(textured, refined_matte, face, bg_color=bg_color)

    # ── Step 12: A4 tiling ────────────────────────────────────────
    if tile_output:
        report(98, "[12/12] Tiling on A4...")
        out = tile(passport, output_path)
    else:
        out = Path(output_path)
        passport.save(out, dpi=(settings.DPI, settings.DPI), quality=97)

    if on_progress is not None:
        on_progress(100, "Complete!")

    cost = 0.001 + (0.004 if enhance else 0.0)   # BiRefNet + CodeFormer
    meta = {
        "output":      str(out),
        "cost_usd":    round(cost, 4),
        "cost_bdt":    round(cost * 110, 2),
        "lum_delta":   lum["delta"],
        "tilt_deg":    getattr(face, "tilt_deg", 0.0),
        "enhanced":    enhance,
    }

    try:
        print(f"\n✅ Done → {out}")
        print(f"   Cost: ${meta['cost_usd']} (~৳{meta['cost_bdt']})")
    except UnicodeEncodeError:
        print(f"\n[DONE] -> {out}")
        print(f"   Cost: ${meta['cost_usd']} (~BDT {meta['cost_bdt']})")
    return meta


if __name__ == "__main__":
    import sys
    process(
        image_path  = sys.argv[1] if len(sys.argv) > 1 else "uploads/test.jpg",
        output_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/passport_a4.jpg",
        bg          = sys.argv[3] if len(sys.argv) > 3 else "blue",
    )
