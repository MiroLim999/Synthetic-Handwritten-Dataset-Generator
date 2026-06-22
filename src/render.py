"""
Text rendering: draw a string as "handwriting" using a random font.
"""

import random
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import config


@lru_cache(maxsize=1)
def available_fonts() -> tuple[str, ...]:
    """
    Collect usable font paths.

    Priority: any .ttf/.otf in resources/fonts. If that folder is empty,
    fall back to the Windows handwriting fonts listed in config.
    """
    fonts: list[str] = []

    if config.FONTS_DIR.exists():
        for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
            fonts.extend(str(p) for p in config.FONTS_DIR.glob(ext))

    if not fonts:
        fonts = [f for f in config.WINDOWS_FONT_FALLBACKS if Path(f).exists()]

    if not fonts:
        raise RuntimeError(
            "No fonts found. Add handwriting .ttf files to resources/fonts/ "
            "or ensure the Windows fallback fonts in config.py exist."
        )
    return tuple(fonts)


def render_text(text: str) -> tuple[Image.Image, str]:
    """
    Render `text` in a random handwriting font on a white background.

    Returns (image, font_path_used).
    """
    font_path = random.choice(available_fonts())
    font_size = random.randint(*config.FONT_SIZE_RANGE)
    font = ImageFont.truetype(font_path, font_size)

    # measure the text so the canvas fits it
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    box = measure.textbbox((0, 0), text, font=font)
    text_w, text_h = box[2] - box[0], box[3] - box[1]

    pad = config.IMAGE_PADDING
    img = Image.new("RGB", (text_w + pad * 2, text_h + pad * 2), "white")
    draw = ImageDraw.Draw(img)

    ink = random.randint(*config.INK_DARKNESS_RANGE)
    draw.text((pad - box[0], pad - box[1]), text, font=font, fill=(ink, ink, ink))

    return img, Path(font_path).name
