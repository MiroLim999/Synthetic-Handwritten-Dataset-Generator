"""
Text rendering: draw a string as "handwriting" using a random font.
"""

import random
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import config

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
    """Return only the cursive/script fonts from the available pool."""
    result = tuple(
        p for p in available_fonts()
        if Path(p).stem.lower() in {s.lower() for s in CURSIVE_FONT_STEMS}
    )
    return result if result else available_fonts()


def cursive_fonts_for_group(group_name: str) -> tuple[str, ...]:
    """
    Return fonts belonging to a specific cursive sub-style group.
    Falls back to all cursive fonts if the group is unknown or empty.
    """
    stems = CURSIVE_STYLE_GROUPS.get(group_name, set())
    result = tuple(
        p for p in available_fonts()
        if Path(p).stem.lower() in {s.lower() for s in stems}
    )
    return result if result else cursive_fonts()


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


def render_text(text: str, font_style: str = FONT_STYLE_ALL,
                cursive_group: str = "",
                specific_font: str = "") -> tuple[Image.Image, str]:
    """
    Render `text` in a random handwriting font on a white background.

    font_style:   'all' | 'cursive'
    cursive_group: sub-style group name when font_style == 'cursive'
    specific_font: exact font filename/stem to use (overrides the pool)

    Returns (image, font_path_used).
    """
    pool = fonts_for_style(font_style, cursive_group)
    if specific_font:
        fp = font_path_for(specific_font)
        if fp:
            pool = (fp,)
    font_path = random.choice(pool)
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
