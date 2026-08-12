"""
Central configuration for the dataset toolkit.

Everything tunable lives here: paths, dataset sizes, image settings,
augmentation strength, and how often each field type appears.

Tuned for OLD Maasin City, Southern Leyte civil registry documents
(Birth / Death / Marriage), roughly mid-1900s.
"""

import os
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path, PurePosixPath, PureWindowsPath

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
RESOURCES_DIR = ROOT / "resources"
FONTS_DIR = RESOURCES_DIR / "fonts"
# Which name pool to draw from. Each version is its own folder under
# resources/ holding first_names.txt, middle_names.txt, last_names.txt:
#   resources/name1/...   resources/name2/...
# Switch versions by changing this, or pass --names name2 on the CLI.
NAMES_VERSION = "name1"
NAMES_DIR = RESOURCES_DIR / NAMES_VERSION
VOCAB_DIR = RESOURCES_DIR / "vocab"
PLACES_FILE = RESOURCES_DIR / "places.txt"

DATASET_DIR = ROOT / "dataset"
REAL_DIR = DATASET_DIR / "real"
# Each generation run produces its own numbered dataset folder under here:
#   dataset/datasets/dataset_001/{train,val,test}/...
#   dataset/datasets/dataset_002/{train,val,test}/...
DATASETS_DIR = DATASET_DIR / "datasets"
DATASET_PREFIX = "dataset_"     # folder name prefix for each run

# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------
# Default number of samples when --count is not passed.
DEFAULT_COUNT = 20_000
# Safety ceiling for accidental CLI/GUI input. Large, intentional runs should
# be broken into independently validated datasets instead of one huge staging
# transaction. The GUI asks for confirmation at the lower warning threshold.
MAX_GENERATION_COUNT = 1_000_000
LARGE_GENERATION_WARNING_COUNT = 100_000

# What kind of samples to generate. "regular" keeps the existing field mix.
# The semi_broken modes keep clean labels but render images with extra missing
# strokes, scratches, and character gaps so TrOCR learns damaged text.
SAMPLE_MODES = {
    "regular": "Regular mixed fields",
    "semi_broken_mixed": "Semi-broken: mixed",
    "semi_broken_words": "Semi-broken: words",
    "semi_broken_characters": "Semi-broken: characters",
    "semi_broken_numerics": "Semi-broken: numerics",
}
DEFAULT_SAMPLE_MODE = "regular"

# Image rendering
FONT_SIZE_RANGE = (38, 58)        # random font size per sample (px)
IMAGE_PADDING = 22                # white padding around the text (px)
INK_DARKNESS_RANGE = (10, 80)     # 0=black, higher=greyer/faded ink

# Year range for documents (used by date / age generators).
DATE_YEAR_RANGE = (1920, 2000)
# Numeric whole-year ages represented by this generator.  Zero covers a child
# less than one year old; 100..110 provide plausible centenarian records.
AGE_YEAR_RANGE = (0, 110)

# Synthetic name-style probabilities. These are configurable representative
# variants for robustness, not frequencies estimated from a labelled corpus.
NAME_MIDDLE_PROB = 0.60
NAME_MIDDLE_INITIAL_PROB = 0.20   # conditional on including a middle name
NAME_SUFFIX_PROB = 0.08
NAME_SUFFIX_COMMA_PROB = 0.50     # conditional on including a suffix
NAME_SURNAME_FIRST_PROB = 0.05
NAME_UPPERCASE_SURNAME_PROB = 0.05

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

# Extra damage for SAMPLE_MODES starting with "semi_broken". These effects are
# intentionally local: they remove pieces of strokes without changing labels.
BROKEN_INK_GAP_PROB = 0.90
BROKEN_INK_GAP_COUNT = (8, 24)
BROKEN_INK_GAP_SIZE = (3, 16)     # px, used for small white/tinted gaps
BROKEN_SCRATCH_PROB = 0.70
BROKEN_SCRATCH_COUNT = (1, 5)
BROKEN_ERODE_PROB = 0.55
BROKEN_ERODE_BLEND = (0.15, 0.45)
BROKEN_CONTRAST_PROB = 0.70
BROKEN_CONTRAST_RANGE = (0.35, 0.75)

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

# Field names understood by src.fields.  Keeping this declaration in config
# lets configuration be validated before importing the generator or touching
# the output filesystem.
SUPPORTED_FIELD_TYPES = frozenset({
    "full_name", "place", "date_written", "date_numeric", "cause_of_death",
    "age", "civil_status", "sex", "religion", "citizenship", "occupation",
    "character", "numeric",
})

# ---------------------------------------------------------------------------
# Split building
# ---------------------------------------------------------------------------
# Fractions for synthetic data (split into train / val / test).
# These should sum to 1.0; test gets the remainder.
SYNTH_TRAIN_FRAC = 0.80
SYNTH_VAL_FRAC = 0.10
SYNTH_TEST_FRAC = 0.10            # remainder -> test

# Real (mock) data split by WRITER, never by image.
REAL_TRAIN_FRAC = 0.60
REAL_VAL_FRAC = 0.20
REAL_TEST_FRAC = 0.20

RANDOM_SEED = 42

# Names of the splits produced for every dataset, in order.
SPLIT_NAMES = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Configuration and split-allocation validation
# ---------------------------------------------------------------------------
def _require_finite_number(name: str, value) -> float:
    """Return ``value`` as float, rejecting booleans and non-finite numbers."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return numeric


def _validate_fraction_set(name: str, fractions: Sequence[Real]) -> tuple[float, ...]:
    """Validate one train/val/test fraction sequence and return float values."""
    if isinstance(fractions, (str, bytes)):
        raise ValueError(f"{name} must contain one fraction for each of {SPLIT_NAMES}")
    try:
        fractions = tuple(fractions)
    except TypeError:
        raise ValueError(
            f"{name} must contain one fraction for each of {SPLIT_NAMES}"
        ) from None
    if len(fractions) != len(SPLIT_NAMES):
        raise ValueError(f"{name} must contain one fraction for each of {SPLIT_NAMES}")
    values = tuple(
        _require_finite_number(f"{name}[{split}]", value)
        for split, value in zip(SPLIT_NAMES, fractions)
    )
    for split, value in zip(SPLIT_NAMES, values):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}[{split}] must be between 0 and 1; got {value}")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} fractions must sum to 1.0; got {sum(values):.12g}")
    if not any(value > 0.0 for value in values):
        raise ValueError(f"{name} must allocate to at least one split")
    return values


def synthetic_split_fractions() -> tuple[float, float, float]:
    """Return the currently configured synthetic train/val/test fractions."""
    return _validate_fraction_set(
        "synthetic split",
        (SYNTH_TRAIN_FRAC, SYNTH_VAL_FRAC, SYNTH_TEST_FRAC),
    )


def real_split_fractions() -> tuple[float, float, float]:
    """Return the currently configured real-writer train/val/test fractions."""
    return _validate_fraction_set(
        "real split",
        (REAL_TRAIN_FRAC, REAL_VAL_FRAC, REAL_TEST_FRAC),
    )


def allocate_split_counts(total: int, fractions: Sequence[Real]) -> dict[str, int]:
    """Allocate an integer ``total`` across train/val/test deterministically.

    The allocation uses Hamilton's largest-remainder method with split order as
    the stable tie-breaker.  Every split with a positive configured fraction is
    guaranteed at least one item; if ``total`` is too small to do that, a clear
    ``ValueError`` is raised.  This makes the helper suitable for both sample
    counts and writer counts while ensuring no applicable evaluation split is
    silently empty.
    """
    if tuple(SPLIT_NAMES) != ("train", "val", "test"):
        raise ValueError("SPLIT_NAMES must be exactly ('train', 'val', 'test')")
    if isinstance(total, bool) or not isinstance(total, Integral) or total <= 0:
        raise ValueError(f"Split total must be a positive whole number; got {total!r}")
    total = int(total)
    values = _validate_fraction_set("split", fractions)
    required = [index for index, value in enumerate(values) if value > 0.0]
    if total < len(required):
        raise ValueError(
            f"Cannot allocate {total} item(s) across {len(required)} nonempty splits; "
            f"at least {len(required)} are required."
        )

    quotas = [total * value for value in values]
    counts = [math.floor(quota) for quota in quotas]
    remainder = total - sum(counts)
    order = sorted(
        range(len(values)),
        key=lambda index: (-(quotas[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1

    # Largest remainder can still give a low-fraction split zero items for tiny
    # totals. Rebalance from the least harmed eligible donor to honor nonempty
    # split guarantees without changing the total.
    minimums = [1 if value > 0.0 else 0 for value in values]
    for target in required:
        if counts[target] > 0:
            continue
        donors = [
            index for index in range(len(values))
            if counts[index] > minimums[index]
        ]
        if not donors:  # Defensive: the total/required check should prevent it.
            raise ValueError("Could not create nonempty configured splits")
        donor = max(
            donors,
            key=lambda index: (counts[index] - quotas[index], counts[index], -index),
        )
        counts[donor] -= 1
        counts[target] += 1

    return dict(zip(SPLIT_NAMES, counts))


def allocate_synthetic_split_counts(total: int) -> dict[str, int]:
    """Allocate a synthetic sample count using all configured fractions."""
    return allocate_split_counts(total, synthetic_split_fractions())


def allocate_real_split_counts(total: int) -> dict[str, int]:
    """Allocate a real-writer count using all configured fractions."""
    return allocate_split_counts(total, real_split_fractions())


def _validate_pair(name: str, value, *, integral: bool = False,
                   minimum: float | None = None,
                   maximum: float | None = None,
                   minimum_exclusive: bool = False) -> tuple[float, float]:
    """Validate a finite, ordered two-value configuration range."""
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value tuple; got {value!r}")
    if integral and any(isinstance(item, bool) or not isinstance(item, Integral)
                        for item in value):
        raise ValueError(f"{name} values must be whole numbers; got {value!r}")
    low = _require_finite_number(f"{name}[0]", value[0])
    high = _require_finite_number(f"{name}[1]", value[1])
    if low > high:
        raise ValueError(f"{name} must be ordered low-to-high; got {value!r}")
    if minimum is not None:
        invalid = low <= minimum if minimum_exclusive else low < minimum
        if invalid:
            relation = "greater than" if minimum_exclusive else "at least"
            raise ValueError(f"{name} values must be {relation} {minimum}; got {value!r}")
    if maximum is not None and high > maximum:
        raise ValueError(f"{name} values must be at most {maximum}; got {value!r}")
    return low, high


def validate_config(count: int | None = None) -> None:
    """Validate generator configuration without mutating the filesystem.

    Pass the requested run ``count`` to validate it alongside the module
    defaults.  The function raises ``ValueError`` with the setting name on the
    first invalid value and otherwise returns ``None``.
    """
    if tuple(SPLIT_NAMES) != ("train", "val", "test"):
        raise ValueError("SPLIT_NAMES must be exactly ('train', 'val', 'test')")

    for name, value in (
        ("MAX_GENERATION_COUNT", MAX_GENERATION_COUNT),
        ("LARGE_GENERATION_WARNING_COUNT", LARGE_GENERATION_WARNING_COUNT),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive whole number; got {value!r}")
    if LARGE_GENERATION_WARNING_COUNT > MAX_GENERATION_COUNT:
        raise ValueError(
            "LARGE_GENERATION_WARNING_COUNT cannot exceed MAX_GENERATION_COUNT"
        )

    for name, value in (("DEFAULT_COUNT", DEFAULT_COUNT), ("count", count)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive whole number; got {value!r}")
        if int(value) > MAX_GENERATION_COUNT:
            raise ValueError(
                f"{name} exceeds MAX_GENERATION_COUNT ({MAX_GENERATION_COUNT:,}); "
                "generate multiple independently validated datasets instead"
            )

    synthetic_fractions = synthetic_split_fractions()
    real_split_fractions()
    allocate_split_counts(
        int(DEFAULT_COUNT if count is None else count), synthetic_fractions
    )

    for name, value in sorted(globals().items()):
        if name.endswith("_PROB"):
            probability = _require_finite_number(name, value)
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1; got {value!r}")

    rotate = _require_finite_number("AUG_ROTATE_DEGREES", AUG_ROTATE_DEGREES)
    if rotate < 0.0:
        raise ValueError("AUG_ROTATE_DEGREES must be nonnegative")

    _validate_pair("FONT_SIZE_RANGE", FONT_SIZE_RANGE, integral=True, minimum=0,
                   minimum_exclusive=True)
    _validate_pair("INK_DARKNESS_RANGE", INK_DARKNESS_RANGE, integral=True,
                   minimum=0, maximum=255)
    _validate_pair("DATE_YEAR_RANGE", DATE_YEAR_RANGE, integral=True,
                   minimum=1_000, maximum=9_999)
    _validate_pair("AGE_YEAR_RANGE", AGE_YEAR_RANGE, integral=True,
                   minimum=0)
    _validate_pair("AUG_BLUR_RADIUS", AUG_BLUR_RADIUS, minimum=0)
    _validate_pair("AUG_NOISE_STD", AUG_NOISE_STD, minimum=0)
    _validate_pair("AUG_BRIGHTNESS_RANGE", AUG_BRIGHTNESS_RANGE, minimum=0,
                   minimum_exclusive=True)
    _validate_pair("BROKEN_INK_GAP_COUNT", BROKEN_INK_GAP_COUNT, integral=True,
                   minimum=0, minimum_exclusive=True)
    _validate_pair("BROKEN_INK_GAP_SIZE", BROKEN_INK_GAP_SIZE, integral=True,
                   minimum=0, minimum_exclusive=True)
    _validate_pair("BROKEN_SCRATCH_COUNT", BROKEN_SCRATCH_COUNT, integral=True,
                   minimum=0, minimum_exclusive=True)
    _validate_pair("BROKEN_ERODE_BLEND", BROKEN_ERODE_BLEND,
                   minimum=0, maximum=1)
    _validate_pair("BROKEN_CONTRAST_RANGE", BROKEN_CONTRAST_RANGE,
                   minimum=0, maximum=1, minimum_exclusive=True)

    if (isinstance(IMAGE_PADDING, bool) or not isinstance(IMAGE_PADDING, Integral)
            or IMAGE_PADDING <= 0):
        raise ValueError(f"IMAGE_PADDING must be a positive whole number; got {IMAGE_PADDING!r}")

    if not isinstance(FIELD_WEIGHTS, Mapping) or not FIELD_WEIGHTS:
        raise ValueError("FIELD_WEIGHTS must be a nonempty mapping")
    unknown = sorted(set(FIELD_WEIGHTS) - SUPPORTED_FIELD_TYPES)
    if unknown:
        raise ValueError(f"FIELD_WEIGHTS contains unknown field types: {', '.join(unknown)}")
    positive_weight = False
    for field_type, weight in FIELD_WEIGHTS.items():
        number = _require_finite_number(f"FIELD_WEIGHTS[{field_type!r}]", weight)
        if number < 0.0:
            raise ValueError(f"FIELD_WEIGHTS[{field_type!r}] must be nonnegative")
        positive_weight = positive_weight or number > 0.0
    if not positive_weight:
        raise ValueError("FIELD_WEIGHTS must contain at least one positive weight")

    if not isinstance(SAMPLE_MODES, Mapping) or not SAMPLE_MODES:
        raise ValueError("SAMPLE_MODES must be a nonempty mapping")
    if DEFAULT_SAMPLE_MODE not in SAMPLE_MODES:
        raise ValueError("DEFAULT_SAMPLE_MODE must be present in SAMPLE_MODES")


# ---------------------------------------------------------------------------
# Dataset-folder helpers
# ---------------------------------------------------------------------------
_SAFE_DATASET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_MAX_DATASET_NAME_LENGTH = 200
_RESERVATION_SUFFIX = ".reserve"


def _normalise_dataset_name(name) -> str:
    """Return a validated, single-component dataset folder name.

    Positive integer values and ASCII digit strings are normalised to the
    numbered ``dataset_NNN`` form.  Custom names intentionally use a small,
    portable character set so that a name has the same meaning on Windows,
    macOS, and Linux.
    """
    if isinstance(name, bool):
        raise ValueError("Dataset IDs must be positive integers, not booleans.")

    if isinstance(name, int):
        if name <= 0:
            raise ValueError("Dataset IDs must be positive integers.")
        value = f"{DATASET_PREFIX}{name:03d}"
    else:
        value = str(name)
        if not value or not value.strip():
            raise ValueError("Dataset name cannot be empty.")
        if value != value.strip():
            raise ValueError("Dataset name cannot start or end with whitespace.")
        if re.fullmatch(r"[0-9]+", value):
            number = int(value)
            if number <= 0:
                raise ValueError("Dataset IDs must be positive integers.")
            value = f"{DATASET_PREFIX}{number:03d}"

    # Check both path grammars.  PureWindowsPath catches drive-qualified and
    # UNC input even when this module is used on a non-Windows host.
    windows_path = PureWindowsPath(value)
    posix_path = PurePosixPath(value)
    if (windows_path.drive or windows_path.root or windows_path.is_absolute()
            or posix_path.is_absolute()):
        raise ValueError(f"Dataset name must not be an absolute or drive-qualified path: {value!r}")
    if "/" in value or "\\" in value:
        raise ValueError(f"Dataset name must be one folder name, not a path: {value!r}")
    if value in {".", ".."}:
        raise ValueError(f"Dataset name is not allowed: {value!r}")
    if len(value) > _MAX_DATASET_NAME_LENGTH:
        raise ValueError(
            f"Dataset name is too long (maximum {_MAX_DATASET_NAME_LENGTH} characters)."
        )
    if not _SAFE_DATASET_NAME.fullmatch(value):
        raise ValueError(
            "Dataset name may contain only ASCII letters, digits, '.', '-', and '_', "
            "and must start with a letter or digit."
        )
    if value.endswith("."):
        raise ValueError("Dataset name cannot end with a period.")

    # Windows treats reserved device names as reserved even with an extension
    # (for example, CON.txt).  Enforce this everywhere for portability.
    device_name = value.split(".", 1)[0].upper()
    if device_name in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Dataset name uses a reserved Windows device name: {value!r}")
    return value


def assert_safe_dataset_dir(path) -> Path:
    """Validate and return a resolved direct child of :data:`DATASETS_DIR`.

    Call this immediately before any filesystem operation involving a dataset
    directory.  The check resolves symlinks and refuses the datasets container
    itself, nested paths, and paths outside it.  This function does not create
    or modify anything on disk.
    """
    candidate = Path(path)
    _normalise_dataset_name(candidate.name)
    try:
        datasets_root = DATASETS_DIR.resolve(strict=False)
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Could not safely resolve dataset path {candidate!s}: {exc}") from exc

    if resolved == datasets_root or resolved.parent != datasets_root:
        raise ValueError(
            f"Dataset path must be a direct child of {datasets_root}: {candidate!s}"
        )
    return resolved


def _dataset_index(path) -> int:
    """Parse the trailing number from a 'dataset_007' folder name (0 if none)."""
    name = path.name
    if name.startswith(DATASET_PREFIX):
        suffix = name[len(DATASET_PREFIX):]
        if suffix.isdigit():
            return int(suffix)
    return 0


def existing_datasets() -> list:
    """Return existing dataset folders (dataset_001, ...) sorted by number.

    Archives and reservation markers are deliberately excluded so callers
    such as the real-data merge continue to receive actual directories only.
    """
    if not DATASETS_DIR.exists():
        return []
    dirs = [p for p in DATASETS_DIR.iterdir()
            if p.is_dir() and p.name.startswith(DATASET_PREFIX)]
    return sorted(dirs, key=_dataset_index)


def _claimed_dataset_indices() -> set[int]:
    """Return numbered IDs claimed by folders, ZIPs, or active reservations."""
    if not DATASETS_DIR.exists():
        return set()

    prefix = re.escape(DATASET_PREFIX)
    numbered = re.compile(rf"^{prefix}([0-9]+)$", re.IGNORECASE)
    archived = re.compile(
        rf"^{prefix}([0-9]+)\.zip(?:\.sha256)?$", re.IGNORECASE
    )
    reserved = re.compile(
        rf"^\.({prefix})([0-9]+){re.escape(_RESERVATION_SUFFIX)}$",
        re.IGNORECASE,
    )
    claimed = set()
    for entry in DATASETS_DIR.iterdir():
        match = numbered.fullmatch(entry.name) or archived.fullmatch(entry.name)
        if match:
            number = int(match.group(1))
        else:
            match = reserved.fullmatch(entry.name)
            number = int(match.group(2)) if match else 0
        if number > 0:
            claimed.add(number)
    return claimed


def next_dataset_dir() -> "Path":
    """Return the next unclaimed numbered path without reserving it.

    Both output directories and sibling ZIP archives count as claimed.  This
    function is useful for display only; writers must use
    :func:`reserve_dataset_dir` to avoid scan-then-create races.
    """
    claimed = _claimed_dataset_indices()
    next_n = max(claimed, default=0) + 1
    return assert_safe_dataset_dir(DATASETS_DIR / f"{DATASET_PREFIX}{next_n:03d}")


def resolve_dataset_dir(name=None) -> "Path":
    """
    Resolve a dataset output directory.

    - name is None         -> auto-pick the next numbered folder
    - name is an int / digit string -> dataset/datasets/dataset_<n>
    - name is any other string       -> dataset/datasets/<name>
    """
    if name is None:
        return next_dataset_dir()
    safe_name = _normalise_dataset_name(name)
    return assert_safe_dataset_dir(DATASETS_DIR / safe_name)


def _path_lexists(path: Path) -> bool:
    """Like Path.exists(), but also treats broken symlinks as collisions."""
    return os.path.lexists(os.fspath(path))


def _reservation_marker(path: Path) -> Path:
    """Return the hidden direct-child marker used to claim ``path``."""
    return path.parent / f".{path.name}{_RESERVATION_SUFFIX}"


class DatasetReservation:
    """An atomic claim on a final dataset path.

    The final :attr:`path` remains absent, allowing a completed sibling staging
    directory to be atomically renamed into place.  The reservation marker is
    kept until :meth:`release` is called.  Instances are context managers, but
    callers that publish after leaving a context may release explicitly.

    A reservation is advisory: all generator/archive writers must use this API
    and re-run :func:`assert_safe_dataset_dir` immediately before mutation.
    """

    def __init__(self, path: Path, marker_path: Path, token: str):
        self.path = path
        self._marker_path = marker_path
        self._token = token
        self._released = False

    @property
    def released(self) -> bool:
        """Whether this reservation object has already been released."""
        return self._released

    def release(self) -> None:
        """Release this claim; repeated calls on the same object are harmless."""
        if self._released:
            return
        try:
            # Avoid deleting a marker that was externally removed and replaced.
            if self._marker_path.read_text(encoding="ascii") == self._token:
                self._marker_path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._released = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False


def _claim_dataset_path(path: Path) -> DatasetReservation:
    """Atomically create the reservation marker for an already-safe path."""
    path = assert_safe_dataset_dir(path)
    archive_path = path.parent / f"{path.name}.zip"
    checksum_path = archive_path.with_name(f"{archive_path.name}.sha256")
    marker_path = _reservation_marker(path)

    if _path_lexists(path):
        raise FileExistsError(f"Dataset destination already exists: {path}")
    if _path_lexists(archive_path):
        raise FileExistsError(f"Dataset archive already exists: {archive_path}")
    if _path_lexists(checksum_path):
        raise FileExistsError(f"Dataset archive checksum already exists: {checksum_path}")

    token = uuid.uuid4().hex
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(marker_path, flags, 0o600)
    except FileExistsError:
        raise FileExistsError(f"Dataset name is already reserved: {path.name}") from None

    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="") as marker:
            marker.write(token)
            marker.flush()
            os.fsync(marker.fileno())

        # A non-cooperating writer could have created the output between the
        # pre-check and marker creation.  Fail closed instead of overwriting it.
        if _path_lexists(path):
            raise FileExistsError(f"Dataset destination already exists: {path}")
        if _path_lexists(archive_path):
            raise FileExistsError(f"Dataset archive already exists: {archive_path}")
        if _path_lexists(checksum_path):
            raise FileExistsError(
                f"Dataset archive checksum already exists: {checksum_path}"
            )
    except Exception:
        try:
            marker_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return DatasetReservation(path, marker_path, token)


def reserve_dataset_dir(name=None) -> DatasetReservation:
    """Atomically reserve a safe final dataset directory.

    ``name`` follows :func:`resolve_dataset_dir`: ``None`` allocates the next
    numbered run, a positive integer/digit string selects ``dataset_NNN``, and
    a safe custom name selects that direct child.  Existing output directories
    and sibling ZIPs are always refused.

    Automatic allocation retries when another process wins the same marker
    race.  The returned object's ``.path`` is the absent final path; generate
    into a unique sibling staging directory, atomically rename it to ``.path``,
    then call ``.release()`` (normally in ``finally``).
    """
    # Validate explicit input before even creating DATASETS_DIR.  Invalid input
    # must be a side-effect-free error.
    if name is not None:
        path = resolve_dataset_dir(name)
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        return _claim_dataset_path(path)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        path = next_dataset_dir()
        try:
            return _claim_dataset_path(path)
        except FileExistsError:
            # A concurrent allocator may have claimed the result after our
            # scan.  Its marker is now included in the next scan.
            continue


def name_versions() -> list:
    """Return available name-pool folder names (e.g. ['name1', 'name2'])."""
    if not RESOURCES_DIR.exists():
        return []
    required = ("first_names.txt", "middle_names.txt", "last_names.txt")
    versions = []
    for p in sorted(RESOURCES_DIR.iterdir()):
        if p.is_dir() and all((p / f).exists() for f in required):
            versions.append(p.name)
    return versions
