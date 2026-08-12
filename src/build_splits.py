"""Securely reconcile writer-held-out real handwriting into a dataset.

The real input CSV is authoritative and must contain ``filename``, ``label``,
and ``writer_id`` columns.  Real samples and synthetic samples share the
versioned ``manifest.csv`` contract from :mod:`src.manifest`.

Usage::

    python -m src.build_splits                 # merge into the latest dataset
    python -m src.build_splits --dataset 2     # merge into dataset_002
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Iterable, Mapping

import config
from src.dataset_mutation import (assert_no_sibling_archive,
                                  dataset_mutation_lock)
from src.manifest import (
    LABEL_COLUMNS,
    MANIFEST_COLUMNS,
    MANIFEST_SCHEMA_VERSION,
    RUN_METADATA_FILENAME,
    csv_bytes,
    labels_bytes,
    manifest_bytes,
)
from src.dataset_validation import REPORT_FILENAME, validate_dataset
from src.path_safety import is_reparse_point, require_regular_directory
from src.provenance import sha256_file, utc_now_iso


REAL_INPUT_COLUMNS = ("filename", "label", "writer_id")
WRITER_ASSIGNMENTS_FILENAME = "real_writer_splits.csv"
WRITER_ASSIGNMENT_COLUMNS = ("writer_id", "split", "schema_version")
LEGACY_REAL_MANIFEST_FILENAME = "real_manifest.csv"


@dataclass(frozen=True)
class MergeResult:
    """Outcome counts for a successful real-data reconciliation."""

    copied: int
    unchanged: int
    removed: int
    skipped: int
    failed: int
    out_dir: Path


@dataclass(frozen=True)
class _RealInput:
    filename: str
    label: str
    writer_id: str


@dataclass(frozen=True)
class _CopyPlan:
    source: Path
    destination: Path
    source_digest: str
    row: dict[str, str]


def _validate_leaf_filename(filename: str) -> None:
    """Require a portable leaf filename, never a path or drive-qualified name."""
    if not filename or "\x00" in filename:
        raise ValueError("Real-data filename must be a non-empty leaf filename")

    # Check both dialects so Windows paths remain unsafe when this code runs on
    # POSIX, and vice versa. Explicit checks cover mixed-separator paths too.
    posix_path = PurePosixPath(filename)
    windows_path = PureWindowsPath(filename)
    if (
        filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
    ):
        raise ValueError(
            f"Unsafe real-data filename {filename!r}: expected one leaf filename"
        )


def _resolve_direct_child(parent: Path, filename: str, *, must_exist: bool) -> Path:
    """Resolve *filename* and require it to remain directly below *parent*."""
    parent_resolved = parent.resolve(strict=False)
    candidate = parent / filename
    try:
        resolved = candidate.resolve(strict=must_exist)
    except (FileNotFoundError, OSError) as exc:
        if must_exist:
            raise FileNotFoundError(
                f"Required file does not exist: {candidate}"
            ) from exc
        raise ValueError(f"Could not safely resolve path: {candidate}") from exc
    if resolved.parent != parent_resolved:
        raise ValueError(
            f"Unsafe path {candidate}: it does not resolve directly beneath {parent}"
        )
    return resolved


def _validated_split_dirs(out_dir: Path) -> dict[str, Path]:
    """Validate split destinations without creating or changing directories."""
    out_resolved = out_dir.resolve(strict=True)
    split_dirs: dict[str, Path] = {}
    for split in config.SPLIT_NAMES:
        split_dir = out_dir / split
        if is_reparse_point(split_dir):
            raise ValueError(
                f"Split destination must not be a symlink or reparse point: {split_dir}"
            )
        if split_dir.exists() and not split_dir.is_dir():
            raise ValueError(f"Split destination is not a directory: {split_dir}")
        if split_dir.resolve(strict=False).parent != out_resolved:
            raise ValueError(
                f"Split destination is not directly beneath the dataset: {split_dir}"
            )
        split_dirs[split] = split_dir
    return split_dirs


def _read_dict_csv(
    path: Path,
    required_columns: Iterable[str],
    *,
    allow_missing: bool = False,
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Read a CSV strictly enough that malformed rows cannot be skipped."""
    if not path.exists():
        if allow_missing:
            return (), []
        raise FileNotFoundError(f"Required CSV does not exist: {path}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"CSV must be a regular file: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        header = tuple(reader.fieldnames)
        if len(set(header)) != len(header):
            raise ValueError(f"CSV contains duplicate column names: {path}")
        missing = [column for column in required_columns if column not in header]
        if missing:
            raise ValueError(
                f"CSV {path} is missing required columns: {', '.join(missing)}"
            )

        rows: list[dict[str, str]] = []
        for line_number, raw_row in enumerate(reader, start=2):
            if None in raw_row:
                raise ValueError(
                    f"CSV row {line_number} in {path} has extra unnamed values"
                )
            row = {key: (value if value is not None else "") for key, value in raw_row.items()}
            if all(value == "" for value in row.values()):
                continue
            rows.append(row)
    return header, rows


def _read_real_inputs(path: Path) -> list[_RealInput] | None:
    """Read and validate the explicit-writer real input; ``None`` means absent."""
    if not path.exists():
        return None
    _header, rows = _read_dict_csv(path, REAL_INPUT_COLUMNS)
    result: list[_RealInput] = []
    seen: dict[str, _RealInput] = {}
    for line_number, row in enumerate(rows, start=2):
        filename = row["filename"]
        label = row["label"]
        writer_id = row["writer_id"]
        _validate_leaf_filename(filename)
        if not writer_id:
            raise ValueError(
                f"Real-label row {line_number} has an empty writer_id"
            )
        current = _RealInput(filename, label, writer_id)
        key = filename.casefold()
        previous = seen.get(key)
        if previous is not None:
            if previous.label != label:
                raise ValueError(
                    f"Conflicting labels for duplicate real-data filename "
                    f"{previous.filename!r}: {previous.label!r} and {label!r}"
                )
            if previous.writer_id != writer_id:
                raise ValueError(
                    f"Conflicting writer_id values for duplicate real-data "
                    f"filename {previous.filename!r}"
                )
            raise ValueError(f"Duplicate real-data filename: {filename!r}")
        seen[key] = current
        result.append(current)
    return result


def _normal_manifest_row(row: dict[str, str], *, source: str) -> dict[str, str]:
    """Return one row in the current unified column order."""
    return {
        "filename": row.get("filename", ""),
        "label": row.get("label", ""),
        "split": row.get("split", ""),
        "source": source,
        "field_type": row.get("field_type", ""),
        "font": row.get("font", ""),
        "sample_mode": row.get("sample_mode", ""),
        "writer_id": row.get("writer_id", ""),
        "schema_version": str(MANIFEST_SCHEMA_VERSION),
    }


def _validate_manifest_row(row: dict[str, str], origin: Path) -> None:
    filename = row["filename"]
    _validate_leaf_filename(filename)
    if row["split"] not in config.SPLIT_NAMES:
        raise ValueError(
            f"Invalid split {row['split']!r} for {filename!r} in {origin}"
        )
    if row["source"] not in {"synthetic", "real"}:
        raise ValueError(
            f"Invalid source {row['source']!r} for {filename!r} in {origin}"
        )
    if row["schema_version"] != str(MANIFEST_SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported manifest schema version {row['schema_version']!r} "
            f"for {filename!r}"
        )
    if row["source"] == "real" and not row["writer_id"]:
        raise ValueError(f"Real manifest row has no writer_id: {filename!r}")
    if row["source"] == "synthetic" and row["writer_id"]:
        raise ValueError(f"Synthetic manifest row has a writer_id: {filename!r}")


def _load_unified_manifest(path: Path) -> list[dict[str, str]]:
    """Load the current manifest, upgrading legacy synthetic rows in memory."""
    header, rows = _read_dict_csv(path, ("filename", "label", "split"))
    is_unified = tuple(header) == MANIFEST_COLUMNS
    if "source" in header and not is_unified:
        raise ValueError(
            f"Unified manifest must use the exact column contract: {MANIFEST_COLUMNS}"
        )
    if not is_unified:
        allowed_legacy = {
            "filename", "label", "split", "field_type", "font", "sample_mode"
        }
        if not set(header).issubset(allowed_legacy):
            raise ValueError(f"Unsupported legacy manifest columns in {path}: {header}")

    normalized: list[dict[str, str]] = []
    seen: dict[str, dict[str, str]] = {}
    for raw_row in rows:
        if is_unified:
            row = {column: raw_row[column] for column in MANIFEST_COLUMNS}
        else:
            row = _normal_manifest_row(raw_row, source="synthetic")
        _validate_manifest_row(row, path)
        key = row["filename"].casefold()
        if key in seen:
            raise ValueError(f"Duplicate filename in manifest: {row['filename']!r}")
        seen[key] = row
        normalized.append(row)
    return normalized


def _load_legacy_real_manifest(path: Path) -> list[dict[str, str]]:
    """Load the former separate real manifest solely for one-time migration."""
    if not path.exists():
        return []
    header, rows = _read_dict_csv(path, ("filename", "label", "split"))
    writer_column = "writer_id" if "writer_id" in header else "writer"
    if writer_column not in header:
        raise ValueError(f"Legacy real manifest has no writer column: {path}")

    normalized: list[dict[str, str]] = []
    for raw_row in rows:
        raw_row = dict(raw_row)
        raw_row["writer_id"] = raw_row[writer_column]
        row = _normal_manifest_row(raw_row, source="real")
        _validate_manifest_row(row, path)
        normalized.append(row)
    return normalized


def _combine_existing_rows(
    unified: list[dict[str, str]], legacy_real: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Merge legacy real rows into the unified in-memory view without duplicates."""
    combined = list(unified)
    by_filename = {row["filename"].casefold(): row for row in combined}
    for row in legacy_real:
        key = row["filename"].casefold()
        previous = by_filename.get(key)
        if previous is not None:
            comparable = ("filename", "label", "split", "source", "writer_id")
            if any(previous[column] != row[column] for column in comparable):
                raise ValueError(
                    f"Conflicting unified/legacy manifest rows for {row['filename']!r}"
                )
            continue
        by_filename[key] = row
        combined.append(row)
    return combined


def _validate_labels_view(path: Path, manifest_rows: list[dict[str, str]]) -> None:
    """Refuse to discard labels that cannot be represented in the manifest."""
    header, rows = _read_dict_csv(path, LABEL_COLUMNS)
    if tuple(header) != LABEL_COLUMNS:
        raise ValueError(f"labels.csv must use the exact columns {LABEL_COLUMNS}")

    expected = {
        row["filename"].casefold(): (row["filename"], row["label"], row["split"])
        for row in manifest_rows
    }
    actual: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        _validate_leaf_filename(row["filename"])
        key = row["filename"].casefold()
        value = (row["filename"], row["label"], row["split"])
        if key in actual:
            raise ValueError(f"Duplicate filename in labels.csv: {row['filename']!r}")
        actual[key] = value
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            "labels.csv and manifest metadata disagree "
            f"(missing={missing[:3]}, extra={extra[:3]})"
        )


def _record_assignment(
    assignments: dict[str, str], writer_id: str, split: str, origin: str
) -> None:
    if not writer_id:
        raise ValueError(f"Empty writer_id in {origin}")
    if split not in config.SPLIT_NAMES:
        raise ValueError(f"Invalid split {split!r} for writer {writer_id!r} in {origin}")
    previous = assignments.get(writer_id)
    if previous is not None and previous != split:
        raise ValueError(
            f"Writer {writer_id!r} appears in both {previous!r} and {split!r}"
        )
    assignments[writer_id] = split

#nice
def _load_writer_assignments(
    path: Path, existing_real_rows: list[dict[str, str]]
) -> dict[str, str]:
    assignments: dict[str, str] = {}
    if path.exists():
        header, rows = _read_dict_csv(path, WRITER_ASSIGNMENT_COLUMNS)
        if tuple(header) != WRITER_ASSIGNMENT_COLUMNS:
            raise ValueError(
                f"Writer assignment file must use {WRITER_ASSIGNMENT_COLUMNS}"
            )
        for row in rows:
            if row["schema_version"] != str(MANIFEST_SCHEMA_VERSION):
                raise ValueError(
                    f"Unsupported writer assignment schema: {row['schema_version']!r}"
                )
            _record_assignment(
                assignments, row["writer_id"], row["split"], str(path)
            )
    for row in existing_real_rows:
        _record_assignment(
            assignments, row["writer_id"], row["split"], "existing real manifest"
        )
    return assignments


def _assign_new_writers(
    assignments: dict[str, str], active_writers: set[str]
) -> None:
    """Assign only unseen writers; existing and historical assignments never move."""
    new_writers = sorted(active_writers - set(assignments))
    if not new_writers:
        return
    random.Random(config.RANDOM_SEED).shuffle(new_writers)
    targets = config.allocate_real_split_counts(len(active_writers))
    active_counts = {split: 0 for split in config.SPLIT_NAMES}
    for writer_id in active_writers & set(assignments):
        active_counts[assignments[writer_id]] += 1

    for writer_id in new_writers:
        split = max(
            config.SPLIT_NAMES,
            key=lambda candidate: (
                targets[candidate] - active_counts[candidate],
                -config.SPLIT_NAMES.index(candidate),
            ),
        )
        assignments[writer_id] = split
        active_counts[split] += 1


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _destination_index(split_dirs: dict[str, Path]) -> dict[str, list[Path]]:
    existing: dict[str, list[Path]] = {}
    for split_dir in split_dirs.values():
        if not split_dir.exists():
            continue
        for child in split_dir.iterdir():
            existing.setdefault(child.name.casefold(), []).append(child)
    return existing


def _real_row(item: _RealInput, split: str) -> dict[str, str]:
    return {
        "filename": item.filename,
        "label": item.label,
        "split": split,
        "source": "real",
        "field_type": "",
        "font": "",
        "sample_mode": "",
        "writer_id": item.writer_id,
        "schema_version": str(MANIFEST_SCHEMA_VERSION),
    }


def _preflight_reconciliation(
    inputs: list[_RealInput],
    existing_rows: list[dict[str, str]],
    assignments: dict[str, str],
    real_img_dir: Path,
    out_dir: Path,
) -> tuple[list[dict[str, str]], list[_CopyPlan], set[Path], int]:
    """Produce a complete desired manifest and file-operation plan."""
    split_dirs = _validated_split_dirs(out_dir)
    destinations = _destination_index(split_dirs)
    existing_real = {
        row["filename"].casefold(): row
        for row in existing_rows
        if row["source"] == "real"
    }
    existing_synthetic = {
        row["filename"].casefold(): row
        for row in existing_rows
        if row["source"] == "synthetic"
    }

    # Validate all existing real files before staging or deleting anything.
    old_paths: dict[str, Path] = {}
    for key, row in existing_real.items():
        split_dir = split_dirs[row["split"]]
        raw_path = split_dir / row["filename"]
        if raw_path.is_symlink():
            raise ValueError(f"Existing real image must not be a symlink: {raw_path}")
        resolved = _resolve_direct_child(split_dir, row["filename"], must_exist=True)
        if not resolved.is_file():
            raise FileNotFoundError(f"Existing real image is not a file: {resolved}")
        old_paths[key] = raw_path

    desired_real: list[dict[str, str]] = []
    copy_plan: list[_CopyPlan] = []
    unchanged_paths: set[Path] = set()
    for item in inputs:
        key = item.filename.casefold()
        if key in existing_synthetic:
            raise FileExistsError(
                f"Real filename collides with a synthetic manifest row: {item.filename}"
            )

        source = _resolve_direct_child(real_img_dir, item.filename, must_exist=True)
        if not source.is_file():
            raise FileNotFoundError(f"Real source image is not a file: {source}")
        source_digest = _file_digest(source)

        old_row = existing_real.get(key)
        if old_row is not None and old_row["filename"] != item.filename:
            raise ValueError(
                f"Case-only filename change is not allowed: "
                f"{old_row['filename']!r} -> {item.filename!r}"
            )
        allowed_collision = old_paths.get(key)
        for collision in destinations.get(key, []):
            if allowed_collision is None or collision != allowed_collision:
                raise FileExistsError(
                    f"Destination filename already exists in the dataset: {collision}"
                )

        split = assignments[item.writer_id]
        destination = split_dirs[split] / item.filename
        _resolve_direct_child(split_dirs[split], item.filename, must_exist=False)
        row = _real_row(item, split)
        desired_real.append(row)

        unchanged = (
            old_row == row
            and allowed_collision == destination
            and _file_digest(allowed_collision) == source_digest
        )
        if unchanged:
            unchanged_paths.add(destination)
        else:
            copy_plan.append(_CopyPlan(source, destination, source_digest, row))

    desired_keys = {row["filename"].casefold() for row in desired_real}
    removed_count = len(set(existing_real) - desired_keys)
    removal_paths = {
        path for path in old_paths.values() if path not in unchanged_paths
    }

    split_order = {split: index for index, split in enumerate(config.SPLIT_NAMES)}
    desired_real.sort(
        key=lambda row: (
            split_order[row["split"]],
            row["filename"].casefold(),
            row["filename"],
        )
    )
    synthetic_rows = [row for row in existing_rows if row["source"] == "synthetic"]
    return synthetic_rows + desired_real, copy_plan, removal_paths, removed_count


def _same_bytes(path: Path, desired: bytes) -> bool:
    return path.is_file() and not path.is_symlink() and path.read_bytes() == desired


def _json_bytes(value: Mapping) -> bytes:
    """Serialize metadata deterministically for atomic byte comparison."""
    return (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _load_json_mapping(path: Path, *, allow_missing: bool = True) -> dict:
    """Load one regular JSON mapping without accepting links or arrays."""
    if not path.exists():
        if allow_missing:
            return {}
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON metadata must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON metadata must contain an object: {path}")
    return value


def _write_staged(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _commit_files(
    transaction_dir: Path,
    installs: dict[Path, Path],
    removals: set[Path],
    after_install: Callable[[], dict[Path, bytes]] | None = None,
) -> None:
    """Commit replacements, validate, then publish derived bytes or roll back."""
    targets = sorted(set(installs) | removals, key=lambda path: str(path))
    backup_dir = transaction_dir / "backups"
    backup_dir.mkdir()
    backups: list[tuple[Path, Path | None]] = []
    installed: list[Path] = []
    created_dirs: list[Path] = []

    def prepare_target(target: Path) -> None:
        if any(previous == target for previous, _backup in backups):
            return
        if not target.parent.exists():
            target.parent.mkdir(parents=True)
            created_dirs.append(target.parent)
        backup: Path | None = None
        if _path_exists(target):
            if target.is_symlink() or not target.is_file():
                raise ValueError(f"Transaction target is not a regular file: {target}")
            backup = backup_dir / f"{len(backups):04d}.bak"
            os.replace(target, backup)
        backups.append((target, backup))

    try:
        for target in targets:
            prepare_target(target)

        for target in targets:
            staged = installs.get(target)
            if staged is None:
                continue
            os.replace(staged, target)
            installed.append(target)

        derived = after_install() if after_install is not None else {}
        derived_stage = transaction_dir / "derived"
        for index, (target, content) in enumerate(
            sorted(derived.items(), key=lambda pair: str(pair[0]))
        ):
            if _same_bytes(target, content):
                continue
            staged = derived_stage / f"{index:04d}.artifact"
            _write_staged(staged, content)
            prepare_target(target)
            os.replace(staged, target)
            installed.append(target)
    except Exception:
        for target in reversed(installed):
            if _path_exists(target):
                target.unlink()
        for target, backup in reversed(backups):
            if backup is not None and backup.exists():
                os.replace(backup, target)
        for directory in reversed(created_dirs):
            try:
                directory.rmdir()
            except OSError:
                pass
        raise


def _execute_transaction(
    out_dir: Path,
    copy_plan: list[_CopyPlan],
    removal_paths: set[Path],
    metadata: dict[Path, bytes],
    legacy_manifest: Path,
    after_install: Callable[[], dict[Path, bytes]],
) -> None:
    """Stage new bytes, validate the installed view, and publish atomically."""
    metadata_changes = {
        path: content for path, content in metadata.items() if not _same_bytes(path, content)
    }
    remove_targets = set(removal_paths)
    if legacy_manifest.exists():
        if legacy_manifest.is_symlink() or not legacy_manifest.is_file():
            raise ValueError(f"Legacy manifest must be a regular file: {legacy_manifest}")
        remove_targets.add(legacy_manifest)
    transaction_dir = Path(
        tempfile.mkdtemp(prefix=".real-merge-", dir=str(out_dir))
    )
    try:
        installs: dict[Path, Path] = {}
        image_stage = transaction_dir / "images"
        for index, item in enumerate(copy_plan):
            # Recheck path containment at the mutation boundary.
            _validate_leaf_filename(item.row["filename"])
            source = _resolve_direct_child(
                config.REAL_DIR / "images", item.row["filename"], must_exist=True
            )
            if source != item.source:
                raise ValueError(f"Real source path changed during merge: {source}")
            staged = image_stage / f"{index:08d}.img"
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
            if _file_digest(staged) != item.source_digest:
                raise OSError(f"Real source changed while being copied: {source}")
            installs[item.destination] = staged

        metadata_stage = transaction_dir / "metadata"
        for index, (target, content) in enumerate(
            sorted(metadata_changes.items(), key=lambda pair: str(pair[0]))
        ):
            staged = metadata_stage / f"{index:04d}.csv"
            _write_staged(staged, content)
            installs[target] = staged

        # All final image targets/removals originated from validated manifest
        # rows or input leaf filenames. Metadata targets are fixed direct-child
        # names. Recheck the dataset path immediately before commit as well.
        config.assert_safe_dataset_dir(out_dir)
        _commit_files(
            transaction_dir,
            installs,
            remove_targets,
            after_install=after_install,
        )
    finally:
        shutil.rmtree(transaction_dir, ignore_errors=True)


def _validation_report_bytes(path: Path, validation_report) -> bytes | None:
    """Return new report bytes only when current artifact hashes are stale."""
    current: dict = {}
    if path.exists():
        try:
            current = _load_json_mapping(path, allow_missing=False)
        except ValueError:
            current = {}
    if (
        current.get("valid") is True
        and current.get("manifest_sha256") == validation_report.manifest_sha256
        and current.get("images_sha256") == validation_report.images_sha256
    ):
        return None
    return _json_bytes(validation_report.as_dict())


def _real_merge_state(
    *,
    real_csv: Path,
    source_csv_sha256: str,
    assignments: dict[str, str],
    active_writers: set[str],
    result: MergeResult,
    validation_report,
) -> dict:
    active_counts = {split: 0 for split in config.SPLIT_NAMES}
    for writer_id in active_writers:
        active_counts[assignments[writer_id]] += 1
    return {
        "source_csv": real_csv.name,
        "source_csv_sha256": source_csv_sha256,
        "writer_count": len(active_writers),
        "writer_counts_by_split": active_counts,
        # Keep historical assignments too. A writer removed from the current
        # CSV must retain a test holdout assignment if later reintroduced.
        "writer_split_assignments": {
            writer_id: assignments[writer_id]
            for writer_id in sorted(assignments, key=lambda value: (value.casefold(), value))
        },
        "manifest_sha256": validation_report.manifest_sha256,
        "images_sha256": validation_report.images_sha256,
        "result": {
            "copied": result.copied,
            "unchanged": result.unchanged,
            "removed": result.removed,
            "skipped": result.skipped,
            "failed": result.failed,
        },
    }


def _updated_run_metadata(
    existing: dict,
    *,
    real_csv: Path,
    source_csv_sha256: str,
    assignments: dict[str, str],
    active_writers: set[str],
    result: MergeResult,
    validation_report,
) -> dict:
    """Preserve generator provenance while recording the latest material merge."""
    updated = copy.deepcopy(existing)
    candidate = _real_merge_state(
        real_csv=real_csv,
        source_csv_sha256=source_csv_sha256,
        assignments=assignments,
        active_writers=active_writers,
        result=result,
        validation_report=validation_report,
    )
    previous = updated.get("real_merge")
    comparison_keys = tuple(key for key in candidate if key != "result")
    unchanged_state = isinstance(previous, dict) and all(
        previous.get(key) == candidate[key] for key in comparison_keys
    )
    if unchanged_state:
        # Preserve both timestamp and last material merge result on a no-op.
        candidate = previous
    else:
        candidate["merged_at"] = utc_now_iso()

    statistics = validation_report.statistics
    updated.update({
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "image_count": statistics.get("images", 0),
        "row_count": statistics.get("manifest_rows", 0),
        "split_counts": statistics.get("by_split", {}),
        "manifest_sha256": validation_report.manifest_sha256,
        "images_sha256": validation_report.images_sha256,
        "validation_report": REPORT_FILENAME,
        "real_merge": candidate,
    })
    return updated


def _resolve_output(dataset) -> Path:
    if dataset is None:
        existing = config.existing_datasets()
        if not existing:
            raise FileNotFoundError(
                "No dataset folder found. Run generate_synthetic first."
            )
        out_dir = existing[-1]
    else:
        out_dir = config.resolve_dataset_dir(dataset)
    out_dir = config.assert_safe_dataset_dir(out_dir)
    if is_reparse_point(out_dir):
        raise ValueError(
            f"Dataset folder must not be a symlink or reparse point: {out_dir}"
        )
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {out_dir}")
    return out_dir


def _build_locked(out_dir: Path) -> MergeResult:
    """Perform one reconciliation while the caller owns the mutation lock."""
    real_csv = config.REAL_DIR / "labels.csv"
    if is_reparse_point(config.REAL_DIR):
        raise ValueError(
            f"Real-data root must not be a symlink or reparse point: {config.REAL_DIR}"
        )
    if real_csv.exists():
        require_regular_directory(config.REAL_DIR, description="Real-data root")
        require_regular_directory(
            config.REAL_DIR / "images", description="Real image source root"
        )
    real_inputs = _read_real_inputs(real_csv)
    if real_inputs is None:
        print("Real: labels.csv not found. Nothing to reconcile.")
        return MergeResult(0, 0, 0, 0, 0, out_dir)

    unified_path = out_dir / "manifest.csv"
    labels_path = out_dir / "labels.csv"
    legacy_path = out_dir / LEGACY_REAL_MANIFEST_FILENAME
    assignment_path = out_dir / WRITER_ASSIGNMENTS_FILENAME
    report_path = out_dir / REPORT_FILENAME
    run_metadata_path = out_dir / RUN_METADATA_FILENAME

    unified_rows = _load_unified_manifest(unified_path)
    legacy_real = _load_legacy_real_manifest(legacy_path)
    existing_rows = _combine_existing_rows(unified_rows, legacy_real)
    _validate_labels_view(labels_path, existing_rows)

    existing_real = [row for row in existing_rows if row["source"] == "real"]
    existing_run_metadata = _load_json_mapping(run_metadata_path)
    source_csv_sha256 = sha256_file(real_csv)
    assignments = _load_writer_assignments(assignment_path, existing_real)
    active_writers = {item.writer_id for item in real_inputs}
    _assign_new_writers(assignments, active_writers)

    desired_rows, copy_plan, removal_paths, removed_count = _preflight_reconciliation(
        real_inputs,
        existing_rows,
        assignments,
        config.REAL_DIR / "images",
        out_dir,
    )
    assignment_rows = [
        {
            "writer_id": writer_id,
            "split": split,
            "schema_version": str(MANIFEST_SCHEMA_VERSION),
        }
        for writer_id, split in sorted(
            assignments.items(), key=lambda item: (item[0].casefold(), item[0])
        )
    ]
    metadata = {
        unified_path: manifest_bytes(desired_rows),
        labels_path: labels_bytes(desired_rows),
        assignment_path: csv_bytes(WRITER_ASSIGNMENT_COLUMNS, assignment_rows),
    }
    unchanged_count = len(real_inputs) - len(copy_plan)
    result = MergeResult(
        copied=len(copy_plan),
        unchanged=unchanged_count,
        removed=removed_count,
        skipped=0,
        failed=0,
        out_dir=out_dir,
    )

    def validate_and_build_artifacts() -> dict[Path, bytes]:
        if sha256_file(real_csv) != source_csv_sha256:
            raise OSError(f"Real source CSV changed during merge: {real_csv}")
        validation_report = validate_dataset(
            out_dir,
            split_names=config.SPLIT_NAMES,
            expected_count=len(desired_rows),
            require_nonempty_splits=True,
            write_artifacts=False,
            create_montage=False,
        )
        validation_report.raise_for_errors()
        if sha256_file(real_csv) != source_csv_sha256:
            raise OSError(f"Real source CSV changed during validation: {real_csv}")

        derived: dict[Path, bytes] = {}
        report_content = _validation_report_bytes(report_path, validation_report)
        if report_content is not None:
            derived[report_path] = report_content
        updated_metadata = _updated_run_metadata(
            existing_run_metadata,
            real_csv=real_csv,
            source_csv_sha256=source_csv_sha256,
            assignments=assignments,
            active_writers=active_writers,
            result=result,
            validation_report=validation_report,
        )
        derived[run_metadata_path] = _json_bytes(updated_metadata)
        return derived

    _execute_transaction(
        out_dir,
        copy_plan,
        removal_paths,
        metadata,
        legacy_path,
        validate_and_build_artifacts,
    )
    print(
        f"Real: reconciled {len(real_inputs)} samples into {out_dir} "
        f"(copied={result.copied}, unchanged={result.unchanged}, "
        f"removed={result.removed})"
    )
    return result


def build(dataset=None) -> MergeResult:
    """Reconcile real data by writer and return structured outcome counts."""
    out_dir = _resolve_output(dataset)
    with dataset_mutation_lock(out_dir, purpose="merge real data"):
        # A published sibling archive is immutable handoff state. Mutating its
        # source folder would silently make that archive stale.
        assert_no_sibling_archive(out_dir)
        return _build_locked(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge real data into a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="dataset name or number (default: latest)",
    )
    args = parser.parse_args()
    build(args.dataset)


if __name__ == "__main__":
    main()
