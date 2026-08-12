"""
Field-value generators.

Produces realistic random values for each certificate field type
(names, dates, places, sex, etc.) by drawing from the word lists in
resources/. These string values become the labels for synthetic images.
"""

import calendar
import random
from functools import lru_cache
from pathlib import Path

import config


# Date format domains. The held-out profile is reserved for the test split by
# src.split_policy; the patterns are intentionally disjoint from the default
# forms. They are representative synthetic robustness cases, not an empirical
# estimate of historical format frequencies.
BASE_FORMAT_PROFILE = "base"
HELD_OUT_DATE_FORMAT_PROFILE = "held_out_historical_date_v1"
DATE_FORMAT_FIELD_TYPES = frozenset({"date_written", "date_numeric"})

DATE_FORMAT_PATTERN_IDS = {
    "date_written": {
        BASE_FORMAT_PROFILE: (
            "day_month_full_year",
            "ordinal_day_of_month_numeric_year",
            "ordinal_day_of_month_word_year",
        ),
        HELD_OUT_DATE_FORMAT_PROFILE: (
            "month_day_comma_full_year",
            "abbrev_month_numeric_ordinal_apostrophe_year",
            "the_ordinal_of_month_full_year",
        ),
    },
    "date_numeric": {
        BASE_FORMAT_PROFILE: (
            "padded_month_day_full_year",
            "padded_day_month_full_year",
        ),
        HELD_OUT_DATE_FORMAT_PROFILE: (
            "full_year_month_day",
            "numeric_day_abbrev_month_short_year",
            "unpadded_month_day_apostrophe_year",
        ),
    },
}


def _python_rng(rng):
    """Use an explicit Python RNG, or preserve the legacy global default."""
    return random if rng is None else rng


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _load_list(path: Path) -> tuple[str, ...]:
    """Read a text file into a tuple of non-empty, stripped lines."""
    if not path.exists():
        raise FileNotFoundError(f"Resource list not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        items = [line.strip() for line in f if line.strip()]
    if not items:
        raise ValueError(f"Resource list is empty: {path}")
    return tuple(items)


def _effective_names_dir(names_dir: Path | str | None = None) -> Path:
    return config.NAMES_DIR if names_dir is None else Path(names_dir)


def _first_names(names_dir: Path | str | None = None) -> tuple[str, ...]:
    return _load_list(_effective_names_dir(names_dir) / "first_names.txt")


def _middle_names(names_dir: Path | str | None = None) -> tuple[str, ...]:
    return _load_list(_effective_names_dir(names_dir) / "middle_names.txt")


def _last_names(names_dir: Path | str | None = None) -> tuple[str, ...]:
    return _load_list(_effective_names_dir(names_dir) / "last_names.txt")


def _places() -> tuple[str, ...]:
    return _load_list(config.PLACES_FILE)


def _vocab(name: str) -> tuple[str, ...]:
    return _load_list(config.VOCAB_DIR / f"{name}.txt")


@lru_cache(maxsize=None)
def _all_resource_values(names_dir: Path | str | None = None) -> tuple[str, ...]:
    """All names, places, and vocab entries used for character sampling."""
    values = []
    values.extend(_first_names(names_dir))
    values.extend(_middle_names(names_dir))
    values.extend(_last_names(names_dir))
    values.extend(_places())
    if config.VOCAB_DIR.exists():
        for path in sorted(config.VOCAB_DIR.glob("*.txt")):
            values.extend(_load_list(path))
    return tuple(values)


@lru_cache(maxsize=None)
def _resource_characters(names_dir: Path | str | None = None) -> tuple[str, ...]:
    """Characters seen in the configured resource lists, plus date separators."""
    chars = sorted({
        ch
        for value in _all_resource_values(names_dir)
        for ch in value
        if not ch.isspace()
    })
    chars.extend([
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "/", "-", ".", ",", "'",
    ])
    return tuple(dict.fromkeys(chars))


def clear_resource_cache() -> None:
    """Clear cached resource lists after changing config.NAMES_DIR."""
    _load_list.cache_clear()
    _all_resource_values.cache_clear()
    _resource_characters.cache_clear()


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]

_ONES = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
         "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen",
         "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
_TENS = ["", "", "Twenty", "Thirty", "Forty", "Fifty",
         "Sixty", "Seventy", "Eighty", "Ninety"]


def _num_in_words(n: int) -> str:
    """Words for 0..99, e.g. 42 -> 'Forty-Two'."""
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + ("-" + _ONES[ones] if ones else "")


def _day_in_words(day: int) -> str:
    return _num_in_words(day)


_ORDINAL_ONES = ["", "First", "Second", "Third", "Fourth", "Fifth", "Sixth",
                 "Seventh", "Eighth", "Ninth", "Tenth", "Eleventh", "Twelfth",
                 "Thirteenth", "Fourteenth", "Fifteenth", "Sixteenth",
                 "Seventeenth", "Eighteenth", "Nineteenth"]
_ORDINAL_TENS = {20: "Twentieth", 30: "Thirtieth"}


def _day_ordinal(day: int) -> str:
    """Ordinal words for a day, e.g. 10 -> 'Tenth', 22 -> 'Twenty-Second'."""
    if day < 20:
        return _ORDINAL_ONES[day]
    tens, ones = divmod(day, 10)
    if ones == 0:
        return _ORDINAL_TENS[day]
    return _TENS[tens] + "-" + _ORDINAL_ONES[ones]


def _numeric_ordinal(day: int) -> str:
    """Compact ordinal used in punctuated historical-style variants."""
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def _year_in_words(year: int, rng=None) -> str:
    """A certificate-style year phrase, including correct 2000 wording."""
    rng = _python_rng(rng)
    if year == 2000:
        return "Two Thousand"
    if 2000 < year < 2100:
        joiner = "and " if rng.random() < 0.5 else ""
        return f"Two Thousand {joiner}{_num_in_words(year - 2000)}"
    century, rest = divmod(year, 100)        # 1942 -> (19, 42)
    head = _num_in_words(century)            # "Nineteen"
    if rest == 0:
        return f"{head} Hundred"
    joiner = "and " if rng.random() < 0.5 else ""
    return f"{head} Hundred {joiner}{_num_in_words(rest)}"


# ---------------------------------------------------------------------------
# Field generators
# ---------------------------------------------------------------------------
def _random_date_parts(rng=None) -> tuple[int, int, int]:
    """Return a uniformly selected year/month and a valid day for that month."""
    rng = _python_rng(rng)
    year = rng.randint(*config.DATE_YEAR_RANGE)
    month = rng.randint(1, 12)
    day = rng.randint(1, calendar.monthrange(year, month)[1])
    return day, month, year


def full_name(rng=None, names_dir: Path | str | None = None) -> str:
    """Generate a name with configurable punctuation, casing, and initials.

    These variants are representative synthetic examples for OCR robustness;
    their configured probabilities are not claimed to be corpus-calibrated.
    """
    rng = _python_rng(rng)
    first = rng.choice(_first_names(names_dir))
    last = rng.choice(_last_names(names_dir))
    middle = None
    if rng.random() < config.NAME_MIDDLE_PROB:
        middle = rng.choice(_middle_names(names_dir))
        if rng.random() < config.NAME_MIDDLE_INITIAL_PROB:
            initial = next((char for char in middle if char.isalpha()), middle[0])
            middle = f"{initial.upper()}."

    if rng.random() < config.NAME_UPPERCASE_SURNAME_PROB:
        last = last.upper()
    if rng.random() < config.NAME_SURNAME_FIRST_PROB:
        name = f"{last}, {first}"
        if middle:
            name += f" {middle}"
    else:
        name = " ".join(part for part in (first, middle, last) if part)

    if rng.random() < config.NAME_SUFFIX_PROB:
        separator = ", " if rng.random() < config.NAME_SUFFIX_COMMA_PROB else " "
        name += separator + rng.choice(["Jr.", "Sr.", "III", "II"])
    return name


def place(rng=None) -> str:
    return _python_rng(rng).choice(_places())


def _require_format_profile(field_type: str, format_profile: str | None) -> str:
    profile = BASE_FORMAT_PROFILE if format_profile is None else format_profile
    if profile not in DATE_FORMAT_PATTERN_IDS[field_type]:
        valid = ", ".join(DATE_FORMAT_PATTERN_IDS[field_type])
        raise ValueError(
            f"Unknown format profile {profile!r} for {field_type}; valid: {valid}"
        )
    return profile


def date_format_pattern_ids(field_type: str, format_profile: str) -> tuple[str, ...]:
    """Return stable pattern IDs for provenance and disjointness checks."""
    if field_type not in DATE_FORMAT_PATTERN_IDS:
        raise ValueError(f"No date-format patterns exist for {field_type!r}")
    profile = _require_format_profile(field_type, format_profile)
    return DATE_FORMAT_PATTERN_IDS[field_type][profile]


def _date_written_with_format(rng=None, format_profile: str = BASE_FORMAT_PROFILE
                              ) -> tuple[str, str]:
    rng = _python_rng(rng)
    profile = _require_format_profile("date_written", format_profile)
    day, month_number, year = _random_date_parts(rng=rng)
    month = _MONTHS[month_number - 1]
    roll = rng.random()
    patterns = DATE_FORMAT_PATTERN_IDS["date_written"][profile]
    if profile == BASE_FORMAT_PROFILE:
        if roll < 0.4:
            return f"{day} {month} {year}", patterns[0]
        if roll < 0.7:
            return f"{_day_ordinal(day)} day of {month}, {year}", patterns[1]
        return (
            f"{_day_ordinal(day)} day of {month}, {_year_in_words(year, rng=rng)}",
            patterns[2],
        )

    # Held-out patterns are structurally disjoint: month-first, abbreviated
    # month plus apostrophe year, or leading "the". None occur in base output.
    if roll < 1 / 3:
        return f"{month} {day}, {year}", patterns[0]
    if roll < 2 / 3:
        return f"{month[:3]}. {_numeric_ordinal(day)}, '{year % 100:02d}", patterns[1]
    return f"the {_day_ordinal(day)} of {month}, {year}", patterns[2]


def date_written(rng=None, format_profile: str = BASE_FORMAT_PROFILE) -> str:
    """Generate a written date from the requested format domain."""
    return _date_written_with_format(rng=rng, format_profile=format_profile)[0]


def _date_numeric_with_format(rng=None, format_profile: str = BASE_FORMAT_PROFILE
                              ) -> tuple[str, str]:
    rng = _python_rng(rng)
    profile = _require_format_profile("date_numeric", format_profile)
    day, month, year = _random_date_parts(rng=rng)
    patterns = DATE_FORMAT_PATTERN_IDS["date_numeric"][profile]
    if profile == BASE_FORMAT_PROFILE:
        sep = rng.choice(["/", "-", "."])
        if rng.random() < 0.5:
            return f"{month:02d}{sep}{day:02d}{sep}{year}", patterns[0]
        return f"{day:02d}{sep}{month:02d}{sep}{year}", patterns[1]

    roll = rng.random()
    if roll < 1 / 3:
        return f"{year}/{month:02d}/{day:02d}", patterns[0]
    if roll < 2 / 3:
        return f"{day}-{_MONTHS[month - 1][:3]}-{year % 100:02d}", patterns[1]
    return f"{month}/{day}/'{year % 100:02d}", patterns[2]


def date_numeric(rng=None, format_profile: str = BASE_FORMAT_PROFILE) -> str:
    """Generate a numeric date from the requested format domain."""
    return _date_numeric_with_format(rng=rng, format_profile=format_profile)[0]


def age(rng=None) -> str:
    """Whole years from newborn (0) through age 110, inclusive."""
    return str(_python_rng(rng).randint(*config.AGE_YEAR_RANGE))


def sex(rng=None) -> str:
    return _python_rng(rng).choice(_vocab("sex"))


def civil_status(rng=None) -> str:
    return _python_rng(rng).choice(_vocab("civil_status"))


def religion(rng=None) -> str:
    return _python_rng(rng).choice(_vocab("religion"))


def citizenship(rng=None) -> str:
    return _python_rng(rng).choice(_vocab("citizenship"))


def occupation(rng=None) -> str:
    return _python_rng(rng).choice(_vocab("occupation"))


def cause_of_death(rng=None) -> str:
    return _python_rng(rng).choice(_vocab("cause_of_death"))


def character(rng=None, names_dir: Path | str | None = None) -> str:
    """One to four resource-derived characters for character-level samples."""
    rng = _python_rng(rng)
    chars = _resource_characters(names_dir)
    length = 1 if rng.random() < 0.70 else rng.randint(2, 4)
    return "".join(rng.choice(chars) for _ in range(length))


def numeric(rng=None) -> str:
    """Numeric snippets and dates for damaged digit training."""
    rng = _python_rng(rng)
    roll = rng.random()
    if roll < 0.35:
        return date_numeric(rng=rng)
    if roll < 0.55:
        return age(rng=rng)
    length = rng.randint(1, 6)
    if length >= 4 and rng.random() < 0.35:
        sep = rng.choice(["-", "/", "."])
        left = "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(1, 3)))
        right = "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(1, 3)))
        return f"{left}{sep}{right}"
    return "".join(str(rng.randint(0, 9)) for _ in range(length))


# Maps the field-type names used in config.FIELD_WEIGHTS to their generators.
GENERATORS = {
    "full_name": full_name,
    "place": place,
    "date_written": date_written,
    "date_numeric": date_numeric,
    "age": age,
    "sex": sex,
    "civil_status": civil_status,
    "religion": religion,
    "citizenship": citizenship,
    "occupation": occupation,
    "cause_of_death": cause_of_death,
    "character": character,
    "numeric": numeric,
}


def make_value_with_format(field_type: str, rng=None,
                           names_dir: Path | str | None = None,
                           format_profile: str = BASE_FORMAT_PROFILE
                           ) -> tuple[str, str]:
    """Return ``(label, format_id)`` for row-level evaluation metadata."""
    if field_type not in GENERATORS:
        raise KeyError(f"Unknown field type: {field_type}")
    if field_type == "date_written":
        return _date_written_with_format(rng=rng, format_profile=format_profile)
    if field_type == "date_numeric":
        return _date_numeric_with_format(rng=rng, format_profile=format_profile)
    if format_profile != BASE_FORMAT_PROFILE:
        raise ValueError(
            f"Format profile {format_profile!r} is not supported for {field_type!r}"
        )
    if field_type == "full_name":
        return full_name(rng=rng, names_dir=names_dir), "name_mixed_representative_v1"
    if field_type == "character":
        return character(rng=rng, names_dir=names_dir), "default"
    return GENERATORS[field_type](rng=rng), "default"


def make_value(field_type: str, rng=None,
               names_dir: Path | str | None = None,
               format_profile: str = BASE_FORMAT_PROFILE) -> str:
    """Generate one value; defaults preserve the original in-domain formats."""
    return make_value_with_format(
        field_type,
        rng=rng,
        names_dir=names_dir,
        format_profile=format_profile,
    )[0]
