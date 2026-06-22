"""
Central configuration for the dataset toolkit.

Everything tunable lives here: paths, dataset sizes, image settings,
augmentation strength, and how often each field type appears.

Tuned for OLD Maasin City, Southern Leyte civil registry documents
(Birth / Death / Marriage), roughly mid-1900s.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
RESOURCES_DIR = ROOT / "resources"
FONTS_DIR = RESOURCES_DIR / "fonts"
NAMES_DIR = RESOURCES_DIR / "names"
VOCAB_DIR = RESOURCES_DIR / "vocab"
PLACES_FILE = RESOURCES_DIR / "places.txt"

DATASET_DIR = ROOT / "dataset"
SYNTHETIC_DIR = DATASET_DIR / "synthetic"
REAL_DIR = DATASET_DIR / "real"
SPLITS_DIR = DATASET_DIR / "splits"

# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------
# Default number of samples when --count is not passed.
DEFAULT_COUNT = 20_000

# Image rendering
FONT_SIZE_RANGE = (38, 58)        # random font size per sample (px)
IMAGE_PADDING = 22                # white padding around the text (px)
INK_DARKNESS_RANGE = (10, 80)     # 0=black, higher=greyer/faded ink

# Year range for documents (used by date / age generators).
DATE_YEAR_RANGE = (1920, 2000)

# Fonts: cursive / script styles first, to better match period penmanship.
# Drop period-appropriate handwriting .ttf files into resources/fonts to
# override these (strongly recommended for old cursive documents).
WINDOWS_FONT_FALLBACKS = [
    "C:/Windows/Fonts/FRSCRIPT.TTF",   # French Script - elegant cursive
    "C:/Windows/Fonts/SCRIPTBL.TTF",   # Script MT Bold - formal cursive
    "C:/Windows/Fonts/Gabriola.ttf",   # flowing script
    "C:/Windows/Fonts/LHANDW.TTF",     # Lucida Handwriting
    "C:/Windows/Fonts/segoesc.ttf",    # Segoe Script
    "C:/Windows/Fonts/Inkfree.ttf",    # casual handwriting
    "C:/Windows/Fonts/segoepr.ttf",    # Segoe Print
    "C:/Windows/Fonts/BRUSHSCI.TTF",   # Brush Script
    "C:/Windows/Fonts/comici.ttf",     # Comic (italic)
]

# ---------------------------------------------------------------------------
# Augmentation (probability each effect is applied, 0..1)
# Stronger here to mimic genuinely old, faded, stained scans.
# ---------------------------------------------------------------------------
AUG_ROTATE_PROB = 0.85
AUG_ROTATE_DEGREES = 4.5          # max +/- tilt
AUG_BLUR_PROB = 0.65
AUG_BLUR_RADIUS = (0.4, 1.8)
AUG_NOISE_PROB = 0.65
AUG_NOISE_STD = (4, 22)           # gaussian noise std-dev range
AUG_BRIGHTNESS_PROB = 0.65
AUG_BRIGHTNESS_RANGE = (0.65, 1.15)
AUG_PAPER_TINT_PROB = 0.75        # aged paper tint
AUG_FADE_PROB = 0.6               # faded ink (reduce contrast)
AUG_STAIN_PROB = 0.35             # blotches / foxing spots

# ---------------------------------------------------------------------------
# Field mix — relative weights of each field type in synthetic data.
# Higher weight = more samples. Weight the HARD, high-variety fields
# (names, places, written dates, cause of death) more heavily.
# ---------------------------------------------------------------------------
FIELD_WEIGHTS = {
    "full_name":       30,
    "place":           16,
    "date_written":    13,
    "date_numeric":    11,
    "cause_of_death":   8,
    "age":              5,
    "civil_status":     5,
    "sex":              4,
    "religion":         3,
    "citizenship":      3,
    "occupation":       2,
}

# ---------------------------------------------------------------------------
# Split building
# ---------------------------------------------------------------------------
# Fractions for synthetic data (real data is split by writer separately).
SYNTH_TRAIN_FRAC = 0.90
SYNTH_VAL_FRAC = 0.10             # remainder -> val (synthetic has no test set)

# Real (mock) data split by WRITER, never by image.
REAL_TRAIN_FRAC = 0.60
REAL_VAL_FRAC = 0.20
REAL_TEST_FRAC = 0.20

RANDOM_SEED = 42
