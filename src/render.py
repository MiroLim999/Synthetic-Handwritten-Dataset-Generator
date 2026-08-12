"""
Text rendering: draw a string as "handwriting" using a random font.
"""

import math
import random
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from src.generation_profiles import WriterStyle


FONT_CACHE_SIZE = 512


@lru_cache(maxsize=FONT_CACHE_SIZE)
def _cached_truetype(font_path: str, font_size: int):
    """Load one immutable Pillow font object per normalized path and size."""
    return ImageFont.truetype(font_path, font_size)


def load_font(font_path: str | Path, font_size: int):
    """Return a cached Pillow font for ``font_path`` and ``font_size``.

    Normalizing the path prevents duplicate cache entries for equivalent
    relative/absolute spellings.  Pillow font objects are only used for reads
    by this project, so sharing them between render calls is safe.
    """
    if isinstance(font_size, bool) or not isinstance(font_size, int) or font_size <= 0:
        raise ValueError("font_size must be a positive integer")
    normalized = str(Path(font_path).expanduser().resolve())
    return _cached_truetype(normalized, font_size)


def clear_font_cache() -> None:
    """Release cached font objects (useful after changing the font folder)."""
    _cached_truetype.cache_clear()


def font_cache_info():
    """Expose cache statistics for diagnostics without exposing its contents."""
    return _cached_truetype.cache_info()

# ---------------------------------------------------------------------------
# Cursive / script font filenames (stem only, lowercase, no extension).
# ---------------------------------------------------------------------------
CURSIVE_FONT_STEMS = {
    # Original cursive fonts
    "satisfy-regular",
    "courgette-regular",
    "cookie-regular",
    "sacramento-regular",
    "greatvibes-regular",
    "pacifico-regular",
    "tangerine-regular",
    "homemadeapple-regular",
    # Added Filipino-style cursive fonts
    "alexbrush-regular",
    "allura-regular",
    "carattere-regular",
    "dancingscript-regular",
    "monsieurladoulaise-regular",
    "mrdehaviland-regular",
    "norican-regular",
    "pinyonscript-regular",
    "ruthie-regular",
    "yesevaone-regular",
    # Palmer/D'Nealian-style — closest to Filipino school cursive
    "licorice-regular",
    "yellowtail-regular",
}

# ---------------------------------------------------------------------------
# Cursive sub-style groups — used by the GUI "Specific cursive style" picker.
# Keys are human-readable labels; values are sets of lowercase font stems.
# ---------------------------------------------------------------------------
CURSIVE_STYLE_GROUPS: dict[str, set[str]] = {
    "Palmer / School cursive": {
        "licorice-regular",
        "yellowtail-regular",
        "dancingscript-regular",
        "carattere-regular",
        "norican-regular",
    },
    "Elegant calligraphy": {
        "greatvibes-regular",
        "sacramento-regular",
        "tangerine-regular",
        "pinyonscript-regular",
        "monsieurladoulaise-regular",
        "mrdehaviland-regular",
        "allura-regular",
        "ruthie-regular",
    },
    "Clean semi-formal": {
        "satisfy-regular",
        "courgette-regular",
        "alexbrush-regular",
        "cookie-regular",
        "yesevaone-regular",
    },
    "Loose informal": {
        "homemadeapple-regular",
        "carattere-regular",
        "dancingscript-regular",
    },
    "Display / decorative": {
        "pacifico-regular",
        "yesevaone-regular",
        "norican-regular",
    },
}

# Font style option keys
FONT_STYLE_ALL = "all"
FONT_STYLE_CURSIVE = "cursive"
FONT_STYLES = (FONT_STYLE_ALL, FONT_STYLE_CURSIVE)


@lru_cache(maxsize=1)
def available_fonts() -> tuple[str, ...]:
    """
    Collect usable font paths.

    Priority: any .ttf/.otf in resources/fonts. If that folder is empty,
    fall back to the Windows handwriting fonts listed in config.
    """
    fonts: list[str] = []

    if config.FONTS_DIR.exists():
        seen: set[str] = set()
        for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
            for p in config.FONTS_DIR.glob(ext):
                key = str(p).lower()          # normalise for Windows
                if key not in seen:
                    seen.add(key)
                    fonts.append(str(p))

    if not fonts:
        fonts = [f for f in config.WINDOWS_FONT_FALLBACKS if Path(f).exists()]

    if not fonts:
        raise RuntimeError(
            "No fonts found. Add handwriting .ttf files to resources/fonts/ "
            "or ensure the Windows fallback fonts in config.py exist."
        )
    return tuple(fonts)


def cursive_fonts() -> tuple[str, ...]:
    """Return only fonts explicitly recognized as cursive/script."""
    return tuple(
        p for p in available_fonts()
        if Path(p).stem.lower() in {s.lower() for s in CURSIVE_FONT_STEMS}
    )


def cursive_fonts_for_group(group_name: str) -> tuple[str, ...]:
    """
    Return fonts belonging to a specific cursive sub-style group.
    An empty/unknown group selects all recognized cursive fonts.  A known group
    with no installed members returns an empty tuple so preflight can fail
    clearly instead of silently broadening the requested domain.
    """
    stems = CURSIVE_STYLE_GROUPS.get(group_name, set())
    result = tuple(
        p for p in available_fonts()
        if Path(p).stem.lower() in {s.lower() for s in stems}
    )
    if not group_name:
        return cursive_fonts()
    return result


def fonts_for_style(font_style: str, cursive_group: str = "") -> tuple[str, ...]:
    """
    Return the appropriate font pool.

    font_style:
        'all'     -> all available fonts
        'cursive' -> cursive/script fonts; optionally filtered by cursive_group
    cursive_group:
        One of the keys in CURSIVE_STYLE_GROUPS, or '' for all cursive fonts.
    """
    if font_style == FONT_STYLE_CURSIVE:
        if cursive_group and cursive_group in CURSIVE_STYLE_GROUPS:
            return cursive_fonts_for_group(cursive_group)
        return cursive_fonts()
    return available_fonts()


def font_display_name(path) -> str:
    """Human-friendly name for a font file, e.g. 'greatvibes-regular' -> 'Great Vibes'."""
    stem = Path(path).stem
    for suffix in ("-regular", "_regular"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.replace("-", " ").replace("_", " ").title().strip()


def font_path_for(name_or_stem: str) -> str | None:
    """Resolve a font filename/stem to its full path within the available pool."""
    if not name_or_stem:
        return None
    target = Path(name_or_stem).stem.lower()
    for p in available_fonts():
        if Path(p).stem.lower() == target:
            return p
    return None


def _clamp_ink(value: float) -> int:
    """Keep styled ink dark enough to remain a useful OCR target."""
    return max(0, min(180, round(value)))


def _styled_render(text: str, font, font_size: int, ink: int,
                   style: WriterStyle, rng) -> Image.Image:
    """Render a connected approximation of one writer's stable hand.

    Pillow does not offer handwriting physics or arbitrary alternate glyphs.
    This renderer therefore uses stable geometric/ink traits: per-character
    baseline motion, spacing/overlap, shear, stroke width, pressure variation,
    and horizontal glyph scaling.  It is deliberately versioned through the
    :class:`~src.generation_profiles.WriterStyle` that produced ``style``.
    """
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    stroke_width = style.stroke_width
    baseline_margin = math.ceil(style.baseline_amplitude) + stroke_width + 3

    metrics = []
    for character in text or " ":
        box = measure.textbbox(
            (0, 0), character, font=font, stroke_width=stroke_width
        )
        width = max(1.0, float(box[2] - box[0]))
        try:
            advance = float(measure.textlength(character, font=font))
        except (AttributeError, TypeError):
            advance = width
        connection = 0.0 if character.isspace() else style.connection_overlap
        advance = max(1.0, advance + style.spacing - connection)
        metrics.append((character, box, width, advance))

    content_width = sum(item[3] for item in metrics[:-1]) + metrics[-1][2]
    top = min(item[1][1] for item in metrics)
    bottom = max(item[1][3] for item in metrics)
    content_height = max(1, bottom - top)
    pad = config.IMAGE_PADDING + baseline_margin
    image = Image.new(
        "RGB",
        (max(1, math.ceil(content_width) + pad * 2), content_height + pad * 2),
        "white",
    )
    draw = ImageDraw.Draw(image)

    x = float(pad)
    base_y = float(pad - top)
    for index, (character, _box, _width, advance) in enumerate(metrics):
        phase = style.baseline_phase + (math.tau * index / style.baseline_period)
        writer_baseline = math.sin(phase) * style.baseline_amplitude
        # Small sample variation is bounded by the stable writer amplitude and
        # comes only from the caller's explicit RNG.
        jitter = rng.uniform(-0.18, 0.18) * style.baseline_amplitude
        pressure = rng.randint(-style.pressure_variation, style.pressure_variation)
        character_ink = _clamp_ink(ink + style.pressure_bias + pressure)
        fill = (character_ink,) * 3
        draw.text(
            (round(x), round(base_y + writer_baseline + jitter)),
            character,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=fill,
        )
        x += advance

    horizontal_scale = max(
        0.70,
        min(1.30, style.glyph_width_scale + (style.spacing / max(1, font_size))),
    )
    if not math.isclose(horizontal_scale, 1.0, abs_tol=0.002):
        image = image.resize(
            (max(1, round(image.width * horizontal_scale)), image.height),
            getattr(Image, "Resampling", Image).BICUBIC,
        )

    if not math.isclose(style.slant, 0.0, abs_tol=0.002):
        shear = style.slant
        extra = math.ceil(abs(shear) * image.height)
        # ``transform`` takes an output-to-input map.  The translation keeps
        # either shear direction inside the expanded white canvas.
        translation = extra if shear > 0 else 0
        image = image.transform(
            (image.width + extra, image.height),
            getattr(Image, "Transform", Image).AFFINE,
            (1.0, -shear, translation, 0.0, 1.0, 0.0),
            resample=getattr(Image, "Resampling", Image).BICUBIC,
            fillcolor="white",
        )
    if style.ink_texture_density > 0:
        pixels = np.asarray(image, dtype=np.uint8).copy()
        ink_positions = np.argwhere(pixels.mean(axis=2) < 190)
        mark_count = round(len(ink_positions) * style.ink_texture_density)
        for _ in range(mark_count):
            y, x = ink_positions[rng.randrange(len(ink_positions))]
            lift = rng.randint(25, 120)
            pixels[y, x] = np.minimum(
                pixels[y, x].astype(np.int16) + lift, 255
            ).astype(np.uint8)
        image = Image.fromarray(pixels, mode="RGB")
    return image


def render_text(text: str, font_style: str = FONT_STYLE_ALL,
                cursive_group: str = "",
                specific_font: str = "", rng=None,
                writer_style: WriterStyle | None = None,
                font_pool: Sequence[str] | None = None) -> tuple[Image.Image, str]:
    """
    Render `text` in a random handwriting font on a white background.

    font_style:   'all' | 'cursive'
    cursive_group: sub-style group name when font_style == 'cursive'
    specific_font: exact font filename/stem to use (overrides the pool)
    rng: optional ``random.Random``-compatible source. ``None`` preserves the
         historical module-global random behavior.

    writer_style: optional stable style returned by
                  :func:`src.generation_profiles.writer_style_for`.  When set,
                  it consistently controls font preference, baseline, spacing,
                  connections, slant, stroke, pressure, and glyph proportions.
    font_pool: optional prevalidated split-specific paths.  This lets the
               evaluation policy enforce held-out fonts while ``writer_style``
               consistently selects one font within the permitted split pool.

    Returns (image, font_path_used).
    """
    rng = random if rng is None else rng
    if font_pool is None:
        pool = fonts_for_style(font_style, cursive_group)
    else:
        if isinstance(font_pool, (str, bytes)):
            raise ValueError("font_pool must be a sequence of paths, not one string")
        pool = tuple(str(path) for path in font_pool if str(path).strip())
        if not pool:
            raise ValueError("font_pool cannot be empty")
    if specific_font:
        target = Path(specific_font).stem.casefold()
        fp = next(
            (path for path in pool if Path(path).stem.casefold() == target),
            None,
        ) or font_path_for(specific_font)
        if fp:
            pool = (fp,)
    if writer_style is not None and not isinstance(writer_style, WriterStyle):
        raise TypeError("writer_style must be a WriterStyle or None")
    if writer_style is None or specific_font:
        font_path = rng.choice(pool)
    else:
        # A writer keeps one font family.  Clamp handles a theoretical 1.0
        # supplied by a custom WriterStyle even though writer_style_for uses
        # random(), whose upper endpoint is excluded.
        font_index = min(len(pool) - 1, int(writer_style.font_preference * len(pool)))
        font_path = pool[font_index]
    font_size = rng.randint(*config.FONT_SIZE_RANGE)
    if writer_style is not None:
        font_size = max(1, round(font_size * writer_style.font_size_scale))
    font = load_font(font_path, font_size)

    ink = rng.randint(*config.INK_DARKNESS_RANGE)
    if writer_style is not None:
        return _styled_render(text, font, font_size, ink, writer_style, rng), Path(font_path).name

    # measure the text so the canvas fits it
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    box = measure.textbbox((0, 0), text, font=font)
    text_w, text_h = box[2] - box[0], box[3] - box[1]

    pad = config.IMAGE_PADDING
    img = Image.new("RGB", (text_w + pad * 2, text_h + pad * 2), "white")
    draw = ImageDraw.Draw(img)

    draw.text((pad - box[0], pad - box[1]), text, font=font, fill=(ink, ink, ink))

    return img, Path(font_path).name
