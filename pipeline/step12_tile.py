"""
Step 12: tile passport photos onto a 300 DPI A4 sheet with cut lines.

Output: JPEG or PNG file ready to send to printer.
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw

from config import settings

# A4 at 300 DPI
A4_W_PX  = int(210 * settings.DPI / 25.4)   # 2480
A4_H_PX  = int(297 * settings.DPI / 25.4)   # 3508
MARGIN   = int(5   * settings.DPI / 25.4)   # 59px
GAP      = int(2   * settings.DPI / 25.4)   # 24px

# Cut line styling
CUT_LINE_COLOR = (180, 180, 180)
CUT_LINE_WIDTH = 1


def tile(
    photo:       Image.Image,
    output_path: str | Path,
    draw_cuts:   bool = True,
) -> Path:
    """
    Tile `photo` across an A4 canvas and save to `output_path`.
    Returns the output path.
    """
    if not isinstance(photo, Image.Image):
        raise TypeError("photo must be a PIL.Image.Image")

    pw, ph = photo.size
    cols   = (A4_W_PX - 2 * MARGIN + GAP) // (pw + GAP)
    rows   = (A4_H_PX - 2 * MARGIN + GAP) // (ph + GAP)

    if cols < 1 or rows < 1:
        raise ValueError("Photo is too large to fit on A4 at 300 DPI")

    sheet = Image.new("RGB", (A4_W_PX, A4_H_PX), (255, 255, 255))
    draw  = ImageDraw.Draw(sheet)

    for row in range(rows):
        for col in range(cols):
            x = MARGIN + col * (pw + GAP)
            y = MARGIN + row * (ph + GAP)
            sheet.paste(photo, (x, y))

            if draw_cuts:
                # Horizontal cut lines above and below each photo
                draw.line([(x, y - GAP // 2), (x + pw, y - GAP // 2)],
                          fill=CUT_LINE_COLOR, width=CUT_LINE_WIDTH)
                draw.line([(x, y + ph + GAP // 2), (x + pw, y + ph + GAP // 2)],
                          fill=CUT_LINE_COLOR, width=CUT_LINE_WIDTH)
                # Vertical cut lines left and right
                draw.line([(x - GAP // 2, y), (x - GAP // 2, y + ph)],
                          fill=CUT_LINE_COLOR, width=CUT_LINE_WIDTH)
                draw.line([(x + pw + GAP // 2, y), (x + pw + GAP // 2, y + ph)],
                          fill=CUT_LINE_COLOR, width=CUT_LINE_WIDTH)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, dpi=(settings.DPI, settings.DPI), quality=97)
    return output_path
