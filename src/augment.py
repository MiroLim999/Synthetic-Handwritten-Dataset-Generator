"""
Augmentation: make a clean rendered image look like an old, scanned document.

Pure Pillow + NumPy so there are no heavy dependencies. Each effect is applied
randomly based on the probabilities in config.
"""

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

import config


def _rotate(img: Image.Image) -> Image.Image:
    angle = random.uniform(-config.AUG_ROTATE_DEGREES, config.AUG_ROTATE_DEGREES)
    return img.rotate(angle, expand=True, fillcolor="white", resample=Image.BICUBIC)


def _blur(img: Image.Image) -> Image.Image:
    radius = random.uniform(*config.AUG_BLUR_RADIUS)
    return img.filter(ImageFilter.GaussianBlur(radius))


def _brightness(img: Image.Image) -> Image.Image:
    factor = random.uniform(*config.AUG_BRIGHTNESS_RANGE)
    return ImageEnhance.Brightness(img).enhance(factor)


def _fade(img: Image.Image) -> Image.Image:
    """Reduce contrast to mimic faded ink."""
    factor = random.uniform(0.5, 0.85)
    return ImageEnhance.Contrast(img).enhance(factor)


def _paper_tint(img: Image.Image) -> Image.Image:
    """Blend a yellowish overlay to mimic aged paper."""
    tint = random.choice([(255, 250, 225), (250, 244, 220), (245, 238, 210)])
    overlay = Image.new("RGB", img.size, tint)
    alpha = random.uniform(0.08, 0.20)
    return Image.blend(img, overlay, alpha)


def _noise(img: Image.Image) -> Image.Image:
    """Add gaussian noise to mimic scanner grain."""
    std = random.uniform(*config.AUG_NOISE_STD)
    arr = np.asarray(img).astype(np.int16)
    noise = np.random.normal(0, std, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _stains(img: Image.Image) -> Image.Image:
    """Add a few faint brownish blotches to mimic foxing / age spots."""
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    for _ in range(random.randint(2, 6)):
        cx, cy = random.randint(0, w), random.randint(0, h)
        r = random.randint(3, max(4, h // 6))
        shade = random.choice([(120, 90, 40), (150, 120, 70), (90, 70, 30)])
        alpha = random.randint(15, 45)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=shade + (alpha,))
    return img


def _ink_bbox(img: Image.Image) -> tuple[int, int, int, int]:
    """Approximate bounding box around dark ink pixels."""
    gray = np.asarray(img.convert("L"))
    ys, xs = np.where(gray < 220)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, img.width, img.height)
    pad = 2
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(img.width, int(xs.max()) + pad),
        min(img.height, int(ys.max()) + pad),
    )


def _ink_gaps(img: Image.Image) -> Image.Image:
    """Erase tiny local pieces of strokes to mimic broken characters."""
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    gray = np.asarray(img.convert("L"))
    ys, xs = np.where(gray < 210)
    if len(xs) == 0 or len(ys) == 0:
        return img

    min_gap, max_gap = config.BROKEN_INK_GAP_SIZE
    for _ in range(random.randint(*config.BROKEN_INK_GAP_COUNT)):
        w = random.randint(min_gap, max_gap)
        h = random.randint(1, max(2, max_gap // 2))
        idx = random.randrange(len(xs))
        x = int(xs[idx])
        y = int(ys[idx])
        fill = random.choice([(255, 255, 255), (248, 242, 220), (238, 228, 200)])
        alpha = random.randint(175, 255)
        box = [x - w // 2, y - h // 2, x + w // 2, y + h // 2]
        if random.random() < 0.35:
            draw.ellipse(box, fill=fill + (alpha,))
        else:
            draw.rectangle(box, fill=fill + (alpha,))
    return img


def _scratches(img: Image.Image) -> Image.Image:
    """Draw faint paper-colored scratches across parts of the text."""
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    x0, y0, x1, y1 = _ink_bbox(img)
    for _ in range(random.randint(*config.BROKEN_SCRATCH_COUNT)):
        length = random.randint(max(8, w // 5), max(10, w))
        angle = random.uniform(-0.35, 0.35)
        sx = random.randint(0, max(0, w - 1))
        sy = random.randint(max(0, y0 - 3), min(h - 1, max(y0, y1 + 3)))
        ex = sx + int(length)
        ey = sy + int(length * angle)
        fill = random.choice([(250, 246, 226), (238, 230, 205), (225, 214, 185)])
        draw.line((sx, sy, ex, ey), fill=fill + (random.randint(120, 230),),
                  width=random.randint(1, 3))
    return img


def _erode_ink(img: Image.Image) -> Image.Image:
    """Lightly shrink dark strokes so some thin parts disappear."""
    eroded = img.filter(ImageFilter.MaxFilter(3))
    return Image.blend(img.convert("RGB"), eroded.convert("RGB"),
                       random.uniform(*config.BROKEN_ERODE_BLEND))


def _broken_contrast(img: Image.Image) -> Image.Image:
    factor = random.uniform(*config.BROKEN_CONTRAST_RANGE)
    return ImageEnhance.Contrast(img).enhance(factor)


def _semi_broken(img: Image.Image) -> Image.Image:
    """Apply extra localized damage for semi-broken text samples."""
    if random.random() < config.BROKEN_CONTRAST_PROB:
        img = _broken_contrast(img)
    if random.random() < config.BROKEN_ERODE_PROB:
        img = _erode_ink(img)
    if random.random() < config.BROKEN_INK_GAP_PROB:
        img = _ink_gaps(img)
    if random.random() < config.BROKEN_SCRATCH_PROB:
        img = _scratches(img)
    return img


def degrade(img: Image.Image, damage_profile: str = "regular") -> Image.Image:
    """Apply the full random augmentation chain."""
    if random.random() < config.AUG_FADE_PROB:
        img = _fade(img)
    if random.random() < config.AUG_BRIGHTNESS_PROB:
        img = _brightness(img)
    if random.random() < config.AUG_PAPER_TINT_PROB:
        img = _paper_tint(img)
    if random.random() < config.AUG_STAIN_PROB:
        img = _stains(img)
    if damage_profile == "semi_broken":
        img = _semi_broken(img)
    if random.random() < config.AUG_ROTATE_PROB:
        img = _rotate(img)
    if random.random() < config.AUG_BLUR_PROB:
        img = _blur(img)
    if random.random() < config.AUG_NOISE_PROB:
        img = _noise(img)
    return img
