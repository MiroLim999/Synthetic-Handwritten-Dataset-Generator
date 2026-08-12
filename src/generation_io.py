"""Streaming and atomic I/O primitives for large generation runs."""

from __future__ import annotations

import csv
import hashlib
import hmac
import math
import os
import re
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path


CHECKSUM_SUFFIX = ".sha256"
_READ_CHUNK_SIZE = 1024 * 1024


class AtomicCsvWriter:
    """Stream dictionary rows to a temporary CSV and publish on success.

    The final path is absent while rows are being written.  A clean context
    exit flushes and fsyncs the temporary file, then atomically replaces the
    final path.  An exception removes only the private temporary file.
    """

    def __init__(self, path: str | Path, columns: Sequence[str], *,
                 overwrite: bool = False):
        self.path = Path(path)
        self.columns = tuple(columns)
        if not self.columns or any(not isinstance(value, str) or not value
                                   for value in self.columns):
            raise ValueError("columns must contain nonempty strings")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("columns cannot contain duplicates")
        self.overwrite = bool(overwrite)
        self.temp_path = self.path.with_name(
            f".{self.path.name}.{uuid.uuid4().hex}.tmp"
        )
        self._file = None
        self._writer = None
        self._closed = False
        self.row_count = 0

    def __enter__(self) -> "AtomicCsvWriter":
        if self._file is not None or self._closed:
            raise RuntimeError("AtomicCsvWriter instances cannot be reused")
        if not self.path.parent.is_dir():
            raise FileNotFoundError(f"CSV parent directory does not exist: {self.path.parent}")
        if self.path.exists() and not self.overwrite:
            raise FileExistsError(f"CSV already exists: {self.path}")
        try:
            self._file = open(self.temp_path, "x", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=list(self.columns),
                extrasaction="raise",
                lineterminator="\n",
            )
            self._writer.writeheader()
        except Exception:
            self._abort()
            raise
        return self

    def write(self, row: Mapping[str, object]) -> None:
        if self._writer is None or self._closed:
            raise RuntimeError("AtomicCsvWriter is not open")
        missing = [column for column in self.columns if column not in row]
        if missing:
            raise ValueError(f"CSV row is missing columns: {', '.join(missing)}")
        self._writer.writerow(dict(row))
        self.row_count += 1

    def writerows(self, rows: Iterable[Mapping[str, object]]) -> None:
        for row in rows:
            self.write(row)

    def _abort(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.close()
        self._file = None
        self._writer = None
        self.temp_path.unlink(missing_ok=True)
        self._closed = True

    def _commit(self) -> None:
        if self._file is None or self._closed:
            raise RuntimeError("AtomicCsvWriter is not open")
        try:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None
            self._writer = None
            if self.overwrite:
                os.replace(self.temp_path, self.path)
            else:
                # A hard-link publish is atomic and fails if another producer
                # created the destination.  It avoids the check/replace race
                # that would silently overwrite an immutable manifest.
                os.link(self.temp_path, self.path)
                self.temp_path.unlink()
            self._closed = True
        except Exception:
            self._abort()
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            self._commit()
        else:
            self._abort()
        return False


@dataclass(frozen=True)
class GenerationPlanItem:
    index: int
    field_type: str
    split: str


def _require_rng(rng):
    if rng is None or not hasattr(rng, "random") or not hasattr(rng, "randrange"):
        raise TypeError("an explicit random.Random-compatible RNG is required")
    return rng


def iter_exact_assignments(
        counts: Mapping[str, int], rng, *, order: Sequence[str] | None = None,
) -> Iterator[str]:
    """Yield a uniformly shuffled multiset using O(number-of-groups) memory."""
    rng = _require_rng(rng)
    if not isinstance(counts, Mapping) or not counts:
        raise ValueError("counts must be a nonempty mapping")
    keys = tuple(sorted(counts) if order is None else order)
    if set(keys) != set(counts) or len(keys) != len(counts):
        raise ValueError("order must contain every count key exactly once")
    remaining = []
    for key in keys:
        value = counts[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"count for {key!r} must be a nonnegative integer")
        remaining.append(int(value))
    total = sum(remaining)
    while total:
        ticket = rng.randrange(total)
        cumulative = 0
        selected = 0
        for index, value in enumerate(remaining):
            cumulative += value
            if ticket < cumulative:
                selected = index
                break
        remaining[selected] -= 1
        total -= 1
        yield keys[selected]


def iter_weighted_choices(
        weights: Mapping[str, Real], count: int, rng,
) -> Iterator[str]:
    """Yield independent weighted choices without allocating ``count`` items."""
    rng = _require_rng(rng)
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("weights must be a nonempty mapping")
    items = []
    cumulative = []
    total = 0.0
    for key in sorted(weights):
        value = weights[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"weight for {key!r} must be finite and nonnegative")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"weight for {key!r} must be finite and nonnegative")
        if numeric == 0:
            continue
        total += numeric
        items.append(key)
        cumulative.append(total)
    if not items:
        raise ValueError("at least one weight must be positive")
    for _ in range(int(count)):
        ticket = rng.random() * total
        for index, boundary in enumerate(cumulative):
            if ticket < boundary:
                yield items[index]
                break
        else:  # Defensive for custom RNGs that incorrectly return exactly 1.0.
            yield items[-1]


def iter_generation_plan(
        count: int,
        field_weights: Mapping[str, Real],
        split_counts: Mapping[str, int],
        rng,
        *,
        split_order: Sequence[str] = ("train", "val", "test"),
) -> Iterator[GenerationPlanItem]:
    """Stream paired field/split choices with exact split totals."""
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if sum(split_counts.values()) != count:
        raise ValueError("split_counts must sum exactly to count")
    fields = iter_weighted_choices(field_weights, int(count), rng)
    splits = iter_exact_assignments(split_counts, rng, order=split_order)
    for index, (field_type, split) in enumerate(zip(fields, splits), start=1):
        yield GenerationPlanItem(index, field_type, split)


@dataclass
class GenerationCounters:
    """Constant-memory counters used for progress and run metadata."""

    total: int = 0
    splits: Counter = field(default_factory=Counter)
    field_types: Counter = field(default_factory=Counter)

    def observe(self, *, split: str, field_type: str) -> None:
        self.total += 1
        self.splits[split] += 1
        self.field_types[field_type] += 1

    def to_metadata(self) -> dict[str, object]:
        return {
            "total": self.total,
            "split_counts": dict(sorted(self.splits.items())),
            "field_type_counts": dict(sorted(self.field_types.items())),
        }


@dataclass
class SyntheticWriterAssigner:
    """Assign split-exclusive pseudo-writers in constant memory.

    Reusing one ID for ``samples_per_writer`` occurrences makes the stable
    :func:`src.generation_profiles.writer_style_for` traits observable across
    multiple samples.  Including the split in every ID prevents a pseudo-writer
    (and therefore its exact style) from leaking across evaluation domains.
    """

    samples_per_writer: int = 32
    prefix: str = "synthetic"
    _seen_by_split: Counter = field(default_factory=Counter, init=False, repr=False)

    def __post_init__(self) -> None:
        self.samples_per_writer = _positive_whole(
            "samples_per_writer", self.samples_per_writer
        )
        if (not isinstance(self.prefix, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.prefix)):
            raise ValueError(
                "prefix must use only letters, digits, underscores, or hyphens"
            )

    def writer_id_for(self, split: str) -> str:
        if not isinstance(split, str) or not split:
            raise ValueError("split must be a nonempty string")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", split):
            raise ValueError("split contains unsupported writer-ID characters")
        occurrence = self._seen_by_split[split]
        self._seen_by_split[split] += 1
        writer_number = occurrence // self.samples_per_writer + 1
        return f"{self.prefix}_{split}_writer_{writer_number:05d}"

    def to_metadata(self) -> dict[str, object]:
        return {
            "assignment_version": 1,
            "samples_per_writer": self.samples_per_writer,
            "prefix": self.prefix,
            "samples_seen_by_split": dict(sorted(self._seen_by_split.items())),
            "writers_by_split": {
                split: math.ceil(count / self.samples_per_writer)
                for split, count in sorted(self._seen_by_split.items())
            },
            "split_exclusive": True,
        }


def _positive_whole(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def sha256_path(path: str | Path) -> str:
    """Hash a regular file in bounded memory."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot hash non-file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_READ_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_sidecar_path(archive_path: str | Path) -> Path:
    """Return ``dataset.zip.sha256`` for ``dataset.zip``."""
    archive_path = Path(archive_path)
    return archive_path.with_name(archive_path.name + CHECKSUM_SUFFIX)


def write_sha256_sidecar(archive_path: str | Path, *, overwrite: bool = False) -> Path:
    """Atomically write a standard SHA-256 sidecar for an archive.

    The sidecar contains ``<hex><two spaces><archive basename>`` and is refused
    when already present unless ``overwrite`` is explicitly requested.
    """
    archive_path = Path(archive_path)
    digest = sha256_path(archive_path)
    sidecar = checksum_sidecar_path(archive_path)
    if sidecar.exists() and not overwrite:
        raise FileExistsError(f"Checksum sidecar already exists: {sidecar}")
    temp = sidecar.with_name(f".{sidecar.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temp, "x", newline="", encoding="ascii") as stream:
            stream.write(f"{digest}  {archive_path.name}\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temp, sidecar)
        else:
            os.link(temp, sidecar)
            temp.unlink()
    finally:
        temp.unlink(missing_ok=True)
    return sidecar


def verify_sha256_sidecar(archive_path: str | Path,
                          sidecar_path: str | Path | None = None) -> bool:
    """Validate the sidecar syntax, basename, and archive digest."""
    archive_path = Path(archive_path)
    sidecar = (
        checksum_sidecar_path(archive_path)
        if sidecar_path is None else Path(sidecar_path)
    )
    line = sidecar.read_text(encoding="ascii").strip()
    parts = line.split("  ", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid SHA-256 sidecar format: {sidecar}")
    expected, filename = parts
    if (len(expected) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected)):
        raise ValueError(f"Invalid SHA-256 digest in sidecar: {sidecar}")
    if filename != archive_path.name:
        raise ValueError(
            f"Checksum sidecar names {filename!r}, expected {archive_path.name!r}"
        )
    return hmac.compare_digest(expected.casefold(), sha256_path(archive_path))
