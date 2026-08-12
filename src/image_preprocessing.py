"""Shared image geometry contract for TrOCR crops."""

from __future__ import annotations

from PIL import Image, ImageOps


TROCR_IMAGE_SIZE = 384
TROCR_PAD_COLOR = (255, 255, 255)
TROCR_RESAMPLE = Image.Resampling.BICUBIC
PREPROCESSING_ID = "aspect-pad-384-v1"


def resize_and_pad(image: Image.Image, size: int = TROCR_IMAGE_SIZE,
                   fill: tuple[int, int, int] = TROCR_PAD_COLOR) -> Image.Image:
    """Fit an image within a square canvas without changing its aspect ratio."""
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")
    image = image.convert("RGB")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("image dimensions must be positive")
    contained = ImageOps.contain(image, (size, size), method=TROCR_RESAMPLE)
    canvas = Image.new("RGB", (size, size), fill)
    left = (size - contained.width) // 2
    top = (size - contained.height) // 2
    canvas.paste(contained, (left, top))
    return canvas
