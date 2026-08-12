"""Dataset schema, image-quality, and artifact-integrity validation."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from src.manifest import MANIFEST_COLUMNS, MANIFEST_SCHEMA_VERSION
from src.provenance import atomic_write_json, hash_dataset_images, sha256_file, utc_now_iso


ALLOWED_SOURCES = frozenset({"synthetic", "real"})
ALLOWED_IMAGE_MODES = frozenset({"1", "L", "LA", "RGB", "RGBA"})
REPORT_FILENAME = "dataset-validation.json"
MONTAGE_FILENAME = "review-montage.jpg"
MAX_REPORTED_DIAGNOSTICS = 1_000
MAX_MONTAGE_SAMPLES = 24


@dataclass(frozen=True)
class _ManifestRecord:
    """One physical manifest record and its source line."""

    line_number: int
    values: dict[str, str] | None


@dataclass
class ValidationReport:
    """Structured validation outcome suitable for JSON serialization."""

    dataset: str
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)
    manifest_sha256: str = ""
    images_sha256: str = ""
    validated_at: str = field(default_factory=utc_now_iso)
    error_count: int = 0
    warning_count: int = 0
    suppressed_error_count: int = 0
    suppressed_warning_count: int = 0

    def add_error(self, message: str) -> None:
        self.valid = False
        self.error_count += 1
        if len(self.errors) < MAX_REPORTED_DIAGNOSTICS:
            self.errors.append(message)
        else:
            self.suppressed_error_count += 1

    def add_warning(self, message: str) -> None:
        self.warning_count += 1
        if len(self.warnings) < MAX_REPORTED_DIAGNOSTICS:
            self.warnings.append(message)
        else:
            self.suppressed_warning_count += 1

    def as_dict(self) -> dict:
        return {
            "schema_version": 1,
            "dataset": self.dataset,
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "statistics": self.statistics,
            "manifest_sha256": self.manifest_sha256,
            "images_sha256": self.images_sha256,
            "validated_at": self.validated_at,
            "diagnostics": {
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "suppressed_error_count": self.suppressed_error_count,
                "suppressed_warning_count": self.suppressed_warning_count,
            },
        }

    def raise_for_errors(self) -> None:
        if not self.valid:
            preview = "; ".join(self.errors[:5])
            extra = self.error_count - 5
            if extra > 0:
                preview += f"; and {extra} more error(s)"
            raise ValueError(f"Dataset validation failed: {preview}")


def _is_leaf_filename(value: str) -> bool:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        return False
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return not (posix.is_absolute() or windows.is_absolute() or windows.drive)


def _iter_manifest(path: Path, report: ValidationReport) -> Iterator[_ManifestRecord]:
    """Yield strict manifest records without retaining the CSV in memory."""
    if not path.is_file():
        report.add_error(f"Missing manifest: {path.name}")
        return

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            try:
                fields = tuple(next(reader))
            except StopIteration:
                report.add_error("Manifest is empty; expected the unified manifest header")
                return
            except csv.Error as exc:
                report.add_error(
                    "Could not parse the manifest.csv header: "
                    f"{exc}. Check CSV quoting and delimiters."
                )
                return

            header_is_exact = fields == MANIFEST_COLUMNS
            if not header_is_exact:
                report.add_error(
                    "Manifest columns must be exactly "
                    f"{MANIFEST_COLUMNS} in this order; found {fields}"
                )

            while True:
                line_number = reader.line_num + 1
                try:
                    raw_values = next(reader)
                except StopIteration:
                    break
                except csv.Error as exc:
                    report.add_error(
                        "Could not parse manifest.csv near row "
                        f"{line_number}: {exc}. Check CSV quoting and delimiters."
                    )
                    break

                if len(raw_values) != len(MANIFEST_COLUMNS):
                    report.add_error(
                        f"Row {line_number}: malformed CSV row has "
                        f"{len(raw_values)} columns; expected exactly "
                        f"{len(MANIFEST_COLUMNS)}. Check for missing or extra "
                        "commas/quotes."
                    )
                    yield _ManifestRecord(line_number, None)
                    continue

                values = (
                    dict(zip(MANIFEST_COLUMNS, raw_values, strict=True))
                    if header_is_exact
                    else None
                )
                yield _ManifestRecord(line_number, values)
    except (OSError, UnicodeError) as exc:
        report.add_error(f"Could not read manifest.csv: {exc}")


def _initialize_validation_index(connection: sqlite3.Connection) -> None:
    """Create the disk-backed exact indexes used by the validator."""
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-16384;
        CREATE TABLE filenames (
            filename_fold TEXT PRIMARY KEY,
            line_number INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE manifest_paths (
            split TEXT NOT NULL,
            filename_fold TEXT NOT NULL,
            line_number INTEGER NOT NULL,
            row_json TEXT NOT NULL,
            PRIMARY KEY (split, filename_fold)
        ) WITHOUT ROWID;
        CREATE TABLE actual_paths (
            split TEXT NOT NULL,
            filename_fold TEXT NOT NULL,
            PRIMARY KEY (split, filename_fold)
        ) WITHOUT ROWID;
        CREATE TABLE real_writer_splits (
            writer_id TEXT NOT NULL,
            split TEXT NOT NULL,
            PRIMARY KEY (writer_id, split)
        ) WITHOUT ROWID;
        CREATE TABLE field_counts (
            field_type TEXT PRIMARY KEY,
            sample_count INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE image_stats (
            width REAL,
            height REAL,
            aspect_ratio REAL,
            contrast REAL,
            ink_coverage REAL,
            mode TEXT NOT NULL
        );
        """
    )


def _increment_field_count(connection: sqlite3.Connection, field_type: str) -> None:
    connection.execute(
        "INSERT INTO field_counts VALUES (?, 1) "
        "ON CONFLICT(field_type) DO UPDATE SET sample_count=sample_count + 1",
        (field_type,),
    )


def _sql_percentile(
    connection: sqlite3.Connection, column: str, percentile: float
) -> float:
    """Return NumPy-compatible linear percentile from a disk-backed column."""
    if column not in {"width", "height", "aspect_ratio", "contrast", "ink_coverage"}:
        raise ValueError(f"Unsupported statistics column: {column}")
    count = int(connection.execute(
        f"SELECT COUNT({column}) FROM image_stats"  # noqa: S608 - allow-listed above
    ).fetchone()[0])
    if count == 0:
        return 0.0
    position = (count - 1) * percentile / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    rows = connection.execute(
        f"SELECT {column} FROM image_stats WHERE {column} IS NOT NULL "
        f"ORDER BY {column} LIMIT 1 OFFSET ?",  # noqa: S608 - allow-listed above
        (lower_index,),
    ).fetchone()
    lower = float(rows[0])
    if upper_index == lower_index:
        return lower
    upper = float(connection.execute(
        f"SELECT {column} FROM image_stats WHERE {column} IS NOT NULL "
        f"ORDER BY {column} LIMIT 1 OFFSET ?",  # noqa: S608 - allow-listed above
        (upper_index,),
    ).fetchone()[0])
    return lower + (upper - lower) * (position - lower_index)


def _sql_scalar(
    connection: sqlite3.Connection, aggregate: str, column: str, default: float = 0
) -> float:
    if aggregate not in {"MIN", "MAX"} or column not in {
        "width", "height", "aspect_ratio", "contrast", "ink_coverage"
    }:
        raise ValueError("Unsupported image statistics query")
    value = connection.execute(
        f"SELECT {aggregate}({column}) FROM image_stats"  # noqa: S608 - allow-listed
    ).fetchone()[0]
    return default if value is None else value


def assess_image_quality(image: Image.Image) -> tuple[dict, list[str], list[str]]:
    """Measure one in-memory crop and return statistics, errors, and warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    original_mode = image.mode
    if original_mode not in ALLOWED_IMAGE_MODES:
        errors.append(
            f"unsupported image mode {original_mode!r}; expected one of "
            f"{', '.join(sorted(ALLOWED_IMAGE_MODES))}"
        )
    image = image.convert("L")
    width, height = image.size
    if width <= 1 or height <= 1:
        errors.append("invalid dimensions")
        return (
            {
                "width": width,
                "height": height,
                "mode": original_mode,
            },
            errors,
            warnings,
        )
    pixels = np.asarray(image, dtype=np.uint8)

    low = float(np.percentile(pixels, 1))
    high = float(np.percentile(pixels, 99))
    contrast = high - low
    # ``dark`` measures all useful foreground contrast, while the stricter
    # ``ink`` mask prevents harmless light paper/noise augmentation from being
    # mistaken for clipped glyphs at a crop boundary.
    dark = pixels < 210
    ink = pixels < 128
    ink_coverage = float(dark.mean())
    edge_ink = int(ink[:2, :].sum() + ink[-2:, :].sum() + ink[:, :2].sum() + ink[:, -2:].sum())
    edge_area = max(1, (2 * width * 2) + (2 * height * 2))
    clipped_threshold = max(3, math.ceil(edge_area * 0.002))

    if contrast < 5:
        errors.append(f"insufficient contrast ({contrast:.2f})")
    if ink_coverage < 0.0002:
        errors.append(f"nearly blank ({ink_coverage:.6f} dark coverage)")
    if edge_ink >= clipped_threshold:
        errors.append(f"dark ink reaches the image edge ({edge_ink} pixels)")

    aspect_ratio = width / height
    if aspect_ratio > 16:
        warnings.append(f"extreme aspect ratio ({aspect_ratio:.2f}:1)")
    if width > 2304:
        warnings.append(f"extreme image width ({width}px)")

    return (
        {
            "width": width,
            "height": height,
            "mode": original_mode,
            "aspect_ratio": aspect_ratio,
            "contrast": contrast,
            "ink_coverage": ink_coverage,
        },
        errors,
        warnings,
    )


def _image_quality(path: Path) -> tuple[dict, list[str], list[str]]:
    with Image.open(path) as opened:
        opened.verify()
    with Image.open(path) as opened:
        return assess_image_quality(opened)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _write_montage(
    dataset_dir: Path, samples: list[tuple[dict, Path]], max_samples: int = 24
) -> Path | None:
    if not samples:
        return None

    selected = []
    seen_groups = set()
    for row, path in samples:
        key = (
            row.get("source", ""),
            row.get("field_type", ""),
            row.get("sample_mode", ""),
            row.get("font", ""),
        )
        if key in seen_groups:
            continue
        seen_groups.add(key)
        selected.append((row, path))
        if len(selected) >= max_samples:
            break

    tile_width, tile_height = 480, 150
    columns = 2
    rows_count = math.ceil(len(selected) / columns)
    canvas = Image.new("RGB", (tile_width * columns, tile_height * rows_count), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (row, path) in enumerate(selected):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        with Image.open(path) as opened:
            image = ImageOps.contain(opened.convert("RGB"), (tile_width - 20, 95))
        image_x = x + (tile_width - image.width) // 2
        canvas.paste(image, (image_x, y + 8))
        caption = (
            f"{row.get('split', '')}/{row.get('source', '')}/"
            f"{row.get('field_type', '')}: {row.get('label', '')}"
        )
        draw.text((x + 8, y + 112), caption[:76], fill="black")

    output = dataset_dir / MONTAGE_FILENAME
    canvas.save(output, "JPEG", quality=90, optimize=True)
    return output


def validate_dataset(
    dataset_dir: Path,
    split_names: Iterable[str] = ("train", "val", "test"),
    *,
    expected_count: int | None = None,
    require_nonempty_splits: bool = True,
    write_artifacts: bool = True,
    create_montage: bool = True,
) -> ValidationReport:
    """Validate a dataset with bounded heap use and optional artifacts.

    Exact row/path membership, writer membership, and percentile samples live
    in a temporary SQLite index.  Only aggregate counters, a capped diagnostic
    preview, and at most ``MAX_MONTAGE_SAMPLES`` review rows remain in memory.
    """
    dataset_dir = Path(dataset_dir)
    split_names = tuple(split_names)
    report = ValidationReport(dataset=dataset_dir.name)
    manifest_path = dataset_dir / "manifest.csv"
    counts_by_split: Counter[str] = Counter()
    counts_by_source: Counter[str] = Counter()
    montage_samples: list[tuple[dict[str, str], Path]] = []
    montage_groups: set[tuple[str, str, str, str]] = set()
    manifest_row_count = 0
    actual_image_count = 0

    with tempfile.TemporaryDirectory(prefix="dataset-validation-") as temporary:
        database_path = Path(temporary) / "validation.sqlite3"
        with sqlite3.connect(database_path) as connection:
            _initialize_validation_index(connection)

            for record in _iter_manifest(manifest_path, report):
                manifest_row_count += 1
                line_number = record.line_number
                row = record.values
                if row is None:
                    continue
                filename = row["filename"]
                split = row["split"]
                label = row["label"]
                source = row["source"]

                if (not _is_leaf_filename(filename)
                        or Path(filename).suffix.lower() != ".png"):
                    report.add_error(
                        f"Row {line_number}: invalid image filename {filename!r}"
                    )
                    continue
                filename_fold = filename.casefold()
                inserted = connection.execute(
                    "INSERT OR IGNORE INTO filenames VALUES (?, ?)",
                    (filename_fold, line_number),
                )
                if inserted.rowcount == 0:
                    report.add_error(
                        f"Row {line_number}: duplicate filename {filename!r}"
                    )
                if split not in split_names:
                    report.add_error(f"Row {line_number}: invalid split {split!r}")
                    continue
                if source not in ALLOWED_SOURCES:
                    report.add_error(f"Row {line_number}: invalid source {source!r}")
                if not label.strip() or label.strip().upper() == "UNREADABLE":
                    report.add_error(
                        f"Row {line_number}: label is empty or UNREADABLE"
                    )
                if row["schema_version"] != MANIFEST_SCHEMA_VERSION:
                    report.add_error(
                        f"Row {line_number}: unsupported schema version "
                        f"{row['schema_version']!r}"
                    )
                writer_id = row["writer_id"]
                if source == "real":
                    if not writer_id.strip():
                        report.add_error(
                            f"Row {line_number}: real sample lacks writer_id"
                        )
                    else:
                        connection.execute(
                            "INSERT OR IGNORE INTO real_writer_splits VALUES (?, ?)",
                            (writer_id, split),
                        )
                elif source == "synthetic":
                    if writer_id.strip():
                        report.add_error(
                            f"Row {line_number}: synthetic sample must have empty "
                            "writer_id"
                        )
                    required = ("field_type", "font", "sample_mode")
                    missing_synthetic = [
                        name for name in required if not row[name].strip()
                    ]
                    if missing_synthetic:
                        report.add_error(
                            f"Row {line_number}: synthetic sample requires non-empty "
                            "field_type, font, and sample_mode; missing "
                            f"{', '.join(missing_synthetic)}"
                        )

                serialized_row = json.dumps(
                    [row[column] for column in MANIFEST_COLUMNS],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                try:
                    connection.execute(
                        "INSERT INTO manifest_paths VALUES (?, ?, ?, ?)",
                        (split, filename_fold, line_number, serialized_row),
                    )
                except sqlite3.IntegrityError:
                    previous_line, previous_serialized = connection.execute(
                        "SELECT line_number, row_json FROM manifest_paths "
                        "WHERE split=? AND filename_fold=?",
                        (split, filename_fold),
                    ).fetchone()
                    previous_values = json.loads(previous_serialized)
                    differing = [
                        column for column, previous_value, current_value in zip(
                            MANIFEST_COLUMNS,
                            previous_values,
                            (row[column] for column in MANIFEST_COLUMNS),
                            strict=True,
                        )
                        if previous_value != current_value
                    ]
                    if differing:
                        report.add_error(
                            f"Row {line_number}: conflicting duplicate manifest rows "
                            f"for {split}/{filename}; first seen at row "
                            f"{previous_line}; differing columns: "
                            f"{', '.join(differing)}"
                        )
                    else:
                        report.add_error(
                            f"Row {line_number}: duplicate manifest row for "
                            f"{split}/{filename}; first seen at row {previous_line}"
                        )
                    report.add_error(
                        f"Row {line_number}: duplicate split path {split}/{filename}"
                    )

                counts_by_split[split] += 1
                counts_by_source[source] += 1
                _increment_field_count(
                    connection, row["field_type"] or "unspecified"
                )

                path = dataset_dir / split / filename
                try:
                    resolved_split = (dataset_dir / split).resolve(strict=False)
                    resolved_path = path.resolve(strict=True)
                    if (resolved_path.parent != resolved_split
                            or not resolved_path.is_file()):
                        raise ValueError("not a direct-child file")
                    stats, errors, warnings = _image_quality(resolved_path)
                    connection.execute(
                        "INSERT INTO image_stats VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            stats.get("width"), stats.get("height"),
                            stats.get("aspect_ratio"), stats.get("contrast"),
                            stats.get("ink_coverage"), stats["mode"],
                        ),
                    )
                    group = (
                        row.get("source", ""), row.get("field_type", ""),
                        row.get("sample_mode", ""), row.get("font", ""),
                    )
                    if (len(montage_samples) < MAX_MONTAGE_SAMPLES
                            and group not in montage_groups):
                        montage_groups.add(group)
                        montage_samples.append((dict(row), resolved_path))
                    for message in errors:
                        report.add_error(f"{split}/{filename}: {message}")
                    for message in warnings:
                        report.add_warning(f"{split}/{filename}: {message}")
                except (OSError, ValueError, UnidentifiedImageError) as exc:
                    report.add_error(
                        f"{split}/{filename}: unreadable or unsafe image ({exc})"
                    )

            if (expected_count is not None
                    and manifest_row_count != expected_count):
                report.add_error(
                    f"Manifest row count is {manifest_row_count}, "
                    f"expected {expected_count}"
                )

            writer_cursor = connection.execute(
                "SELECT writer_id FROM real_writer_splits "
                "GROUP BY writer_id HAVING COUNT(*) > 1 ORDER BY writer_id"
            )
            for (writer_id,) in writer_cursor:
                writer_splits = [
                    split for (split,) in connection.execute(
                        "SELECT split FROM real_writer_splits WHERE writer_id=? "
                        "ORDER BY split", (writer_id,)
                    )
                ]
                report.add_error(
                    f"Real writer {writer_id!r} appears in multiple splits: "
                    f"{', '.join(writer_splits)}"
                )

            for split in split_names:
                split_dir = dataset_dir / split
                if not split_dir.is_dir():
                    report.add_error(f"Missing split directory: {split}")
                    continue
                for path in split_dir.iterdir():
                    if not path.is_file() or path.suffix.lower() != ".png":
                        report.add_error(f"Unexpected entry in {split}: {path.name}")
                        continue
                    connection.execute(
                        "INSERT OR IGNORE INTO actual_paths VALUES (?, ?)",
                        (split, path.name.casefold()),
                    )

            for split in split_names:
                if require_nonempty_splits and counts_by_split[split] == 0:
                    report.add_error(f"Split is empty: {split}")
            missing_cursor = connection.execute(
                "SELECT m.split, m.filename_fold FROM manifest_paths AS m "
                "LEFT JOIN actual_paths AS a USING (split, filename_fold) "
                "WHERE a.filename_fold IS NULL ORDER BY m.split, m.filename_fold"
            )
            for split, filename in missing_cursor:
                report.add_error(
                    f"Manifest image is missing: {split}/{filename}"
                )
            orphan_cursor = connection.execute(
                "SELECT a.split, a.filename_fold FROM actual_paths AS a "
                "LEFT JOIN manifest_paths AS m USING (split, filename_fold) "
                "WHERE m.filename_fold IS NULL ORDER BY a.split, a.filename_fold"
            )
            for split, filename in orphan_cursor:
                report.add_error(
                    f"Orphan image is not in manifest: {split}/{filename}"
                )

            actual_image_count = int(connection.execute(
                "SELECT COUNT(*) FROM actual_paths"
            ).fetchone()[0])
            fields = dict(connection.execute(
                "SELECT field_type, sample_count FROM field_counts "
                "ORDER BY field_type"
            ))
            modes = dict(connection.execute(
                "SELECT mode, COUNT(*) FROM image_stats GROUP BY mode ORDER BY mode"
            ))
            report.statistics = {
                "manifest_rows": manifest_row_count,
                "images": actual_image_count,
                "by_split": dict(sorted(counts_by_split.items())),
                "by_source": dict(sorted(counts_by_source.items())),
                "by_field_type": fields,
                "image_width": {
                    "min": _sql_scalar(connection, "MIN", "width"),
                    "p50": _sql_percentile(connection, "width", 50),
                    "p95": _sql_percentile(connection, "width", 95),
                    "max": _sql_scalar(connection, "MAX", "width"),
                },
                "image_height": {
                    "min": _sql_scalar(connection, "MIN", "height"),
                    "p50": _sql_percentile(connection, "height", 50),
                    "p95": _sql_percentile(connection, "height", 95),
                    "max": _sql_scalar(connection, "MAX", "height"),
                },
                "aspect_ratio": {
                    "p50": _sql_percentile(connection, "aspect_ratio", 50),
                    "p95": _sql_percentile(connection, "aspect_ratio", 95),
                    "max": _sql_scalar(connection, "MAX", "aspect_ratio"),
                },
                "contrast": {
                    "min": _sql_scalar(connection, "MIN", "contrast")
                },
                "ink_coverage": {
                    "min": _sql_scalar(connection, "MIN", "ink_coverage")
                },
                "image_modes": modes,
            }

    if manifest_path.is_file():
        report.manifest_sha256 = sha256_file(manifest_path)
    if actual_image_count:
        report.images_sha256 = hash_dataset_images(dataset_dir, split_names)

    if write_artifacts:
        if create_montage and montage_samples:
            _write_montage(dataset_dir, montage_samples)
        atomic_write_json(dataset_dir / REPORT_FILENAME, report.as_dict())
    return report
