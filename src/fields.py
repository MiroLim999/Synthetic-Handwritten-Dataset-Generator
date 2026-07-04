"""
Field-value generators.

Produces realistic random values for each certificate field type
(names, dates, places, sex, etc.) by drawing from the word lists in
resources/. These string values become the labels for synthetic images.
"""

import random
from functools import lru_cache
from pathlib import Path

import config


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


def _first_names() -> tuple[str, ...]:
    return _load_list(config.NAMES_DIR / "first_names.txt")


def _middle_names() -> tuple[str, ...]:
    return _load_list(config.NAMES_DIR / "middle_names.txt")


def _last_names() -> tuple[str, ...]:
    return _load_list(config.NAMES_DIR / "last_names.txt")


def _places() -> tuple[str, ...]:
    return _load_list(config.PLACES_FILE)


def _vocab(name: str) -> tuple[str, ...]:
    return _load_list(config.VOCAB_DIR / f"{name}.txt")


@lru_cache(maxsize=1)
def _all_resource_values() -> tuple[str, ...]:
    """All names, places, and vocab entries used for character sampling."""
    values = []
    values.extend(_first_names())
    values.extend(_middle_names())
    values.extend(_last_names())
    values.extend(_places())
    if config.VOCAB_DIR.exists():
        for path in sorted(config.VOCAB_DIR.glob("*.txt")):
            values.extend(_load_list(path))
    return tuple(values)


@lru_cache(maxsize=1)
def _resource_characters() -> tuple[str, ...]:
    """Characters seen in the configured resource lists, plus date separators."""
    chars = sorted({
        ch
        for value in _all_resource_values()
        for ch in value
        if not ch.isspace()
    })
    chars.extend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "/", "-", "."])
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


def _year_in_words(year: int) -> str:
    """1900s years in words, e.g. 1942 -> 'Nineteen Hundred and Forty-Two'."""
    century, rest = divmod(year, 100)        # 1942 -> (19, 42)
    head = _num_in_words(century)            # "Nineteen"
    if rest == 0:
        return f"{head} Hundred"
    joiner = "and " if random.random() < 0.5 else ""
    return f"{head} Hundred {joiner}{_num_in_words(rest)}"


# ---------------------------------------------------------------------------
# Field generators
# ---------------------------------------------------------------------------
def full_name() -> str:
    """First [Middle] Last, occasionally with a suffix."""
    first = random.choice(_first_names())
    last = random.choice(_last_names())
    parts = [first]
    if random.random() < 0.6:                       # include a middle name
        parts.append(random.choice(_middle_names()))
    parts.append(last)
    name = " ".join(parts)
    if random.random() < 0.08:                      # occasional suffix
        name += " " + random.choice(["Jr.", "Sr.", "III", "II"])
    return name


def place() -> str:
    return random.choice(_places())


def date_written() -> str:
    """e.g. '15 March 1967', 'Fifteenth day of March, 1967', or fully spelled."""
    day = random.randint(1, 28)
    month = random.choice(_MONTHS)
    year = random.randint(*config.DATE_YEAR_RANGE)
    roll = random.random()
    if roll < 0.4:
        return f"{day} {month} {year}"
    if roll < 0.7:
        return f"{_day_ordinal(day)} day of {month}, {year}"
    # fully spelled out, common on old certificates
    return f"{_day_ordinal(day)} day of {month}, {_year_in_words(year)}"


def date_numeric() -> str:
    """e.g. '03/15/1967' or '15-03-1967'."""
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(*config.DATE_YEAR_RANGE)
    sep = random.choice(["/", "-", "."])
    if random.random() < 0.5:
        return f"{month:02d}{sep}{day:02d}{sep}{year}"
    return f"{day:02d}{sep}{month:02d}{sep}{year}"


def age() -> str:
    return str(random.randint(1, 99))


def sex() -> str:
    return random.choice(_vocab("sex"))


def civil_status() -> str:
    return random.choice(_vocab("civil_status"))


def religion() -> str:
    return random.choice(_vocab("religion"))


def citizenship() -> str:
    return random.choice(_vocab("citizenship"))


def occupation() -> str:
    return random.choice(_vocab("occupation"))


def cause_of_death() -> str:
    return random.choice(_vocab("cause_of_death"))


def character() -> str:
    """One to four resource-derived characters for character-level samples."""
    chars = _resource_characters()
    length = 1 if random.random() < 0.70 else random.randint(2, 4)
    return "".join(random.choice(chars) for _ in range(length))


def numeric() -> str:
    """Numeric snippets and dates for damaged digit training."""
    roll = random.random()
    if roll < 0.35:
        return date_numeric()
    if roll < 0.55:
        return age()
    length = random.randint(1, 6)
    if length >= 4 and random.random() < 0.35:
        sep = random.choice(["-", "/", "."])
        left = "".join(str(random.randint(0, 9)) for _ in range(random.randint(1, 3)))
        right = "".join(str(random.randint(0, 9)) for _ in range(random.randint(1, 3)))
        return f"{left}{sep}{right}"
    return "".join(str(random.randint(0, 9)) for _ in range(length))


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


def make_value(field_type: str) -> str:
    """Generate one random value for the given field type."""
    if field_type not in GENERATORS:
        raise KeyError(f"Unknown field type: {field_type}")
    return GENERATORS[field_type]()
