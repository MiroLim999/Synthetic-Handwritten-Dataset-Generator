"""
Augmentation: make a clean rendered image look like an old, scanned document.

Pure Pillow + NumPy so there are no heavy dependencies. Each effect is applied
randomly based on the probabilities in config.
"""

import io
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

import config
from src import cutoff as cutoff_engine
from src.cutoff import NO_CUT, CutoffResult
from src.generation_profiles import (AugmentationProfile, CutoffProfile,
                                     StrokeDamageProfile, as_cutoff_profile,
                                     as_stroke_damage_profile,
                                     get_augmentation_profile)


def _python_rng(rng):
    """Use an explicit Python RNG, or preserve the legacy global default."""
    return random if rng is None else rng


def _numpy_rng(np_rng):
    """Use an explicit NumPy Generator, or preserve the legacy global default."""
    return np.random if np_rng is None else np_rng


def _draw_rotation_angle(rng=None, degrees=None, min_degrees: float = 0.0) -> float:
    rng = _python_rng(rng)
    degrees = config.AUG_ROTATE_DEGREES if degrees is None else degrees
    if min_degrees > 0:
        # Draw a magnitude in [min, max], then a sign.
        angle = rng.uniform(min_degrees, degrees)
        if rng.random() < 0.5:
            angle = -angle
        return angle
    # The legacy single draw, kept so default output is byte-identical.
    return rng.uniform(-degrees, degrees)


def _rotate_by(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(angle, expand=True, fillcolor="white", resample=Image.BICUBIC)


def _rotate(img: Image.Image, rng=None, degrees=None,
            min_degrees: float = 0.0) -> Image.Image:
    return _rotate_by(img, _draw_rotation_angle(rng, degrees, min_degrees))


def _blur(img: Image.Image, rng=None, radius_range=None) -> Image.Image:
    rng = _python_rng(rng)
    radius_range = config.AUG_BLUR_RADIUS if radius_range is None else radius_range
    radius = rng.uniform(*radius_range)
    return img.filter(ImageFilter.GaussianBlur(radius))


def _brightness(img: Image.Image, rng=None, factor_range=None) -> Image.Image:
    rng = _python_rng(rng)
    factor_range = config.AUG_BRIGHTNESS_RANGE if factor_range is None else factor_range
    factor = rng.uniform(*factor_range)
    return ImageEnhance.Brightness(img).enhance(factor)


def _fade(img: Image.Image, rng=None, factor_range=(0.5, 0.85)) -> Image.Image:
    """Reduce contrast to mimic faded ink."""
    rng = _python_rng(rng)
    factor = rng.uniform(*factor_range)
    return ImageEnhance.Contrast(img).enhance(factor)


def _paper_tint(img: Image.Image, rng=None,
                alpha_range=(0.08, 0.20)) -> Image.Image:
    """Blend a yellowish overlay to mimic aged paper."""
    rng = _python_rng(rng)
    tint = rng.choice([(255, 250, 225), (250, 244, 220), (245, 238, 210)])
    overlay = Image.new("RGB", img.size, tint)
    alpha = rng.uniform(*alpha_range)
    return Image.blend(img, overlay, alpha)


def _noise(img: Image.Image, rng=None, np_rng=None,
           std_range=None) -> Image.Image:
    """Add gaussian noise to mimic scanner grain."""
    rng = _python_rng(rng)
    np_rng = _numpy_rng(np_rng)
    std_range = config.AUG_NOISE_STD if std_range is None else std_range
    std = rng.uniform(*std_range)
    arr = np.asarray(img).astype(np.int16)
    noise = np_rng.normal(0, std, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _paper_texture(img: Image.Image, rng=None, np_rng=None,
                   std_range=(1.5, 6.0)) -> Image.Image:
    """Add low-frequency luminance variation as a paper-fibre proxy."""
    rng = _python_rng(rng)
    np_rng = _numpy_rng(np_rng)
    std = rng.uniform(*std_range)
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    # Generate a small noise field and enlarge it so this is visibly distinct
    # from the pixel-scale scanner grain in _noise.
    low_width = max(2, img.width // 18)
    low_height = max(2, img.height // 18)
    low = np_rng.normal(0.0, std, (low_height, low_width)).astype(np.float32)
    low_min = float(low.min())
    low_peak = float(low.max() - low_min)
    if low_peak > 0:
        normalized = ((low - low_min) / low_peak * 255.0).astype(np.uint8)
        field = Image.fromarray(normalized, mode="L").resize(
            img.size, getattr(Image, "Resampling", Image).BICUBIC
        )
        texture = (np.asarray(field, dtype=np.float32) / 255.0 - 0.5) * (std * 4.0)
        rgb = np.clip(rgb + texture[:, :, None], 0, 255)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def _scan_lines(img: Image.Image, rng=None,
                alpha_range=(4, 16)) -> Image.Image:
    """Overlay subtle horizontal bands as a scanner-line approximation."""
    rng = _python_rng(rng)
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    spacing = rng.randint(5, 14)
    phase = rng.randrange(spacing)
    for y in range(phase, img.height, spacing):
        shade = rng.choice((0, 255))
        alpha = rng.randint(*alpha_range)
        draw.line((0, y, img.width, y), fill=(shade, shade, shade, alpha), width=1)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _jpeg_roundtrip(img: Image.Image, rng=None,
                    quality_range=(58, 88)) -> Image.Image:
    """Approximate a historical lossy scan export without writing to disk."""
    rng = _python_rng(rng)
    buffer = io.BytesIO()
    img.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=rng.randint(*quality_range),
        optimize=False,
        progressive=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        return encoded.convert("RGB").copy()


def _stains(img: Image.Image, rng=None, count_range=(2, 6),
            opacity_range=(15, 45)) -> Image.Image:
    """Add a few faint brownish blotches to mimic foxing / age spots."""
    rng = _python_rng(rng)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    for _ in range(rng.randint(*count_range)):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        r = rng.randint(3, max(4, h // 6))
        shade = rng.choice([(120, 90, 40), (150, 120, 70), (90, 70, 30)])
        alpha = rng.randint(*opacity_range)
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


def _ink_gaps(img: Image.Image, rng=None, count_range=None,
             size_range=None) -> Image.Image:
    """Erase tiny local pieces of strokes to mimic broken characters."""
    rng = _python_rng(rng)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    gray = np.asarray(img.convert("L"))
    ys, xs = np.where(gray < 210)
    if len(xs) == 0 or len(ys) == 0:
        return img

    min_gap, max_gap = config.BROKEN_INK_GAP_SIZE if size_range is None else size_range
    count_range = config.BROKEN_INK_GAP_COUNT if count_range is None else count_range
    count = rng.randint(*count_range)
    for _ in range(count):
        w = rng.randint(min_gap, max_gap)
        h = rng.randint(1, max(2, max_gap // 2))
        idx = rng.randrange(len(xs))
        x = int(xs[idx])
        y = int(ys[idx])
        fill = rng.choice([(255, 255, 255), (248, 242, 220), (238, 228, 200)])
        alpha = rng.randint(175, 255)
        box = [x - w // 2, y - h // 2, x + w // 2, y + h // 2]
        if rng.random() < 0.35:
            draw.ellipse(box, fill=fill + (alpha,))
        else:
            draw.rectangle(box, fill=fill + (alpha,))
    return img


def _scratches(img: Image.Image, rng=None, count_range=None) -> Image.Image:
    """Draw faint paper-colored scratches across parts of the text."""
    rng = _python_rng(rng)
    count_range = config.BROKEN_SCRATCH_COUNT if count_range is None else count_range
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    x0, y0, x1, y1 = _ink_bbox(img)
    for _ in range(rng.randint(*count_range)):
        length = rng.randint(max(8, w // 5), max(10, w))
        angle = rng.uniform(-0.35, 0.35)
        sx = rng.randint(0, max(0, w - 1))
        sy = rng.randint(max(0, y0 - 3), min(h - 1, max(y0, y1 + 3)))
        ex = sx + int(length)
        ey = sy + int(length * angle)
        fill = rng.choice([(250, 246, 226), (238, 230, 205), (225, 214, 185)])
        draw.line((sx, sy, ex, ey), fill=fill + (rng.randint(120, 230),),
                  width=rng.randint(1, 3))
    return img


def _erode_ink(img: Image.Image, rng=None, blend_range=None) -> Image.Image:
    """Lightly shrink dark strokes so some thin parts disappear."""
    rng = _python_rng(rng)
    blend_range = config.BROKEN_ERODE_BLEND if blend_range is None else blend_range
    eroded = img.filter(ImageFilter.MaxFilter(3))
    return Image.blend(img.convert("RGB"), eroded.convert("RGB"),
                       rng.uniform(*blend_range))


def _broken_contrast(img: Image.Image, rng=None, factor_range=None) -> Image.Image:
    rng = _python_rng(rng)
    factor_range = config.BROKEN_CONTRAST_RANGE if factor_range is None else factor_range
    factor = rng.uniform(*factor_range)
    return ImageEnhance.Contrast(img).enhance(factor)


def _semi_broken(img: Image.Image, rng=None,
                 profile: StrokeDamageProfile | dict | None = None) -> Image.Image:
    """Apply extra localized damage for semi-broken text samples.

    ``profile`` may be a :class:`StrokeDamageProfile`, a legacy params dict,
    or None for the config defaults; all three make the same RNG calls in
    the same order.
    """
    rng = _python_rng(rng)
    profile = as_stroke_damage_profile(profile)
    if rng.random() < profile.contrast_probability:
        img = _broken_contrast(img, rng=rng, factor_range=profile.contrast_range)
    if rng.random() < profile.erode_probability:
        img = _erode_ink(img, rng=rng, blend_range=profile.erode_blend_range)
    if rng.random() < profile.gap_probability:
        img = _ink_gaps(img, rng=rng, count_range=profile.gap_count_range,
                        size_range=profile.gap_size_range)
    if rng.random() < profile.scratch_probability:
        img = _scratches(img, rng=rng, count_range=profile.scratch_count_range)
    return img


@dataclass(frozen=True)
class DegradeResult:
    """The degraded image and the character cut applied to it."""

    image: Image.Image
    cutoff: CutoffResult = NO_CUT


def degrade(img: Image.Image, damage_profile: str = "regular",
            rng=None, np_rng=None,
            augmentation_profile: str | AugmentationProfile | None = None,
            edge_clipping: str | CutoffProfile | tuple | None = "none",
            semi_broken_params: StrokeDamageProfile | dict | None = None,
            ) -> Image.Image:
    """Apply the full random augmentation chain and return only the image."""
    return degrade_detailed(
        img, damage_profile, rng=rng, np_rng=np_rng,
        augmentation_profile=augmentation_profile,
        edge_clipping=edge_clipping,
        semi_broken_params=semi_broken_params,
    ).image


def degrade_detailed(img: Image.Image, damage_profile: str = "regular",
                     rng=None, np_rng=None,
                     augmentation_profile: str | AugmentationProfile | None = None,
                     edge_clipping: str | CutoffProfile | tuple | None = "none",
                     semi_broken_params: StrokeDamageProfile | dict | None = None,
                     ) -> DegradeResult:
    """Apply the full random augmentation chain.

    ``augmentation_profile=None`` preserves the mutable configuration-driven
    behavior for backward compatibility.  Passing a registered profile ID or
    :class:`AugmentationProfile` freezes all regular augmentation parameters
    and enables its versioned paper-texture, scan-line, and compression steps.
    Every stochastic operation uses the supplied Python/NumPy RNGs.

    ``edge_clipping`` selects a character cutoff (registered ID, profile, or
    legacy 4-tuple).  The ink mask is built from the clean render only when
    some side can be cut, so cutoff ``none`` makes no extra draws.
    """
    rng = _python_rng(rng)
    np_rng = _numpy_rng(np_rng)
    cutoff_profile = as_cutoff_profile(edge_clipping)
    mask = cutoff_engine.ink_mask(img) if cutoff_profile.active else None
    profile = (
        None if augmentation_profile is None
        else get_augmentation_profile(augmentation_profile)
    )

    fade_probability = (
        config.AUG_FADE_PROB if profile is None else profile.fade_probability
    )
    brightness_probability = (
        config.AUG_BRIGHTNESS_PROB
        if profile is None else profile.brightness_probability
    )
    paper_tint_probability = (
        config.AUG_PAPER_TINT_PROB
        if profile is None else profile.paper_tint_probability
    )
    stain_probability = (
        config.AUG_STAIN_PROB if profile is None else profile.stain_probability
    )
    rotate_probability = (
        config.AUG_ROTATE_PROB if profile is None else profile.rotate_probability
    )
    blur_probability = (
        config.AUG_BLUR_PROB if profile is None else profile.blur_probability
    )
    noise_probability = (
        config.AUG_NOISE_PROB if profile is None else profile.noise_probability
    )

    if rng.random() < fade_probability:
        img = _fade(
            img,
            rng=rng,
            factor_range=(0.5, 0.85) if profile is None else profile.fade_range,
        )
    if rng.random() < brightness_probability:
        img = _brightness(
            img,
            rng=rng,
            factor_range=(
                config.AUG_BRIGHTNESS_RANGE
                if profile is None else profile.brightness_range
            ),
        )
    if rng.random() < paper_tint_probability:
        img = _paper_tint(
            img,
            rng=rng,
            alpha_range=(
                (0.08, 0.20)
                if profile is None else profile.paper_tint_alpha_range
            ),
        )
    if rng.random() < stain_probability:
        if profile is None:
            img = _stains(img, rng=rng)
        else:
            img = _stains(img, rng=rng, count_range=profile.stain_count_range,
                          opacity_range=profile.stain_opacity_range)
    # Params only tune semi-broken damage; they never switch it on.
    if damage_profile.startswith("semi_broken"):
        img = _semi_broken(img, rng=rng, profile=semi_broken_params)
    if rng.random() < rotate_probability:
        angle = _draw_rotation_angle(
            rng,
            degrees=(config.AUG_ROTATE_DEGREES if profile is None
                     else profile.rotate_degrees),
            min_degrees=0.0 if profile is None else profile.rotate_min_degrees,
        )
        img = _rotate_by(img, angle)
        if mask is not None:
            mask = cutoff_engine.rotate_mask(mask, angle)
    if rng.random() < blur_probability:
        img = _blur(
            img,
            rng=rng,
            radius_range=(config.AUG_BLUR_RADIUS if profile is None
                          else profile.blur_radius_range),
        )
    if rng.random() < noise_probability:
        img = _noise(
            img,
            rng=rng,
            np_rng=np_rng,
            std_range=(config.AUG_NOISE_STD if profile is None
                       else profile.noise_std_range),
        )
    if profile is not None:
        if rng.random() < profile.paper_texture_probability:
            img = _paper_texture(
                img,
                rng=rng,
                np_rng=np_rng,
                std_range=profile.paper_texture_std_range,
            )
        if rng.random() < profile.scanline_probability:
            img = _scan_lines(
                img, rng=rng, alpha_range=profile.scanline_alpha_range
            )
        if rng.random() < profile.compression_probability:
            img = _jpeg_roundtrip(
                img, rng=rng, quality_range=profile.jpeg_quality_range
            )
    result = NO_CUT
    if mask is not None:
        img, result = cutoff_engine.apply_cut(img, mask, cutoff_profile, rng)
    return DegradeResult(img, result)
