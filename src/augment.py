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


def degrade(img: Image.Image) -> Image.Image:
    """Apply the full random augmentation chain."""
    if random.random() < config.AUG_FADE_PROB:
        img = _fade(img)
    if random.random() < config.AUG_BRIGHTNESS_PROB:
        img = _brightness(img)
    if random.random() < config.AUG_PAPER_TINT_PROB:
        img = _paper_tint(img)
    if random.random() < config.AUG_STAIN_PROB:
        img = _stains(img)
    if random.random() < config.AUG_ROTATE_PROB:
        img = _rotate(img)
    if random.random() < config.AUG_BLUR_PROB:
        img = _blur(img)
    if random.random() < config.AUG_NOISE_PROB:
        img = _noise(img)
    return img
