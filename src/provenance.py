"""Reproducible dataset metadata and artifact hashing helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def utc_now_iso() -> str:
    """Return an RFC 3339-compatible UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_named_files(paths: Iterable[tuple[str, Path]]) -> str:
    """Hash a deterministic sequence of logical names plus file contents."""
    digest = hashlib.sha256()
    for logical_name, path in sorted(paths, key=lambda item: item[0]):
        encoded_name = logical_name.replace("\\", "/").encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        with Path(path).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def hash_dataset_images(dataset_dir: Path, split_names: Iterable[str]) -> str:
    """Hash direct-child split files in deterministic split/name order.

    File names are indexed in a temporary on-disk SQLite database instead of
    accumulated in a Python list.  The ordered cursor and each file's chunked
    reader are both streaming, so heap use does not scale with dataset size.
    """
    dataset_dir = Path(dataset_dir)
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="dataset-image-hash-") as temporary:
        database_path = Path(temporary) / "files.sqlite3"
        with sqlite3.connect(database_path) as connection:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-8192")
            connection.execute(
                "CREATE TABLE files ("
                "split_order INTEGER NOT NULL, "
                "logical_name TEXT NOT NULL COLLATE BINARY, "
                "physical_path TEXT NOT NULL)"
            )
            for split_order, split in enumerate(split_names):
                split_dir = dataset_dir / split
                if not split_dir.is_dir():
                    continue
                for path in split_dir.iterdir():
                    if path.is_file():
                        connection.execute(
                            "INSERT INTO files VALUES (?, ?, ?)",
                            (split_order, f"{split}/{path.name}", str(path)),
                        )

            cursor = connection.execute(
                "SELECT logical_name, physical_path FROM files "
                "ORDER BY split_order, logical_name COLLATE BINARY, rowid"
            )
            for logical_name, physical_path in cursor:
                encoded_name = logical_name.replace("\\", "/").encode("utf-8")
                digest.update(len(encoded_name).to_bytes(8, "big"))
                digest.update(encoded_name)
                with Path(physical_path).open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
    return digest.hexdigest()


def resource_hashes(resource_dir: Path) -> dict[str, str]:
    """Return relative-path hashes for generator text/font resources."""
    resource_dir = Path(resource_dir)
    result = {}
    if not resource_dir.is_dir():
        return result
    for path in sorted(resource_dir.rglob("*")):
        if path.is_file() and path.name != ".gitkeep":
            logical_name = path.relative_to(resource_dir).as_posix()
            result[logical_name] = sha256_file(path)
    return result


def source_hashes(root: Path, relative_paths: Iterable[str]) -> dict[str, str]:
    """Hash an explicit set of source files relative to a project root."""
    root = Path(root).resolve()
    result = {}
    for relative in sorted(set(relative_paths)):
        logical = Path(relative).as_posix()
        candidate = (root / relative).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError:
            raise ValueError(f"Source path escapes project root: {relative}") from None
        if not candidate.is_file():
            raise ValueError(f"Source path is not a regular file: {relative}")
        result[logical] = sha256_file(candidate)
    return result


def json_safe(value: Any) -> Any:
    """Convert common configuration values to stable JSON-compatible data."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_safe(item) for item in value), key=str)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def configuration_snapshot(module: Any, *, names: Iterable[str] | None = None) -> dict:
    """Capture selected uppercase settings from a configuration module.

    Callers may provide an explicit allow-list.  By default, all public
    uppercase values are captured except output/input paths, whose content is
    represented independently by resource hashes and effective run options.
    """
    excluded = {
        "ROOT", "DATASET_DIR", "DATASETS_DIR", "REAL_DIR", "RESOURCES_DIR",
        "NAMES_DIR", "VOCAB_DIR", "FONTS_DIR", "PLACES_FILE",
    }
    selected = names if names is not None else (
        name for name in vars(module)
        if name.isupper() and not name.startswith("_") and name not in excluded
    )
    return {
        name: json_safe(getattr(module, name))
        for name in sorted(selected)
        if hasattr(module, name)
    }


def installed_versions(distributions: Iterable[str]) -> dict[str, str]:
    """Return installed package versions, using ``not-installed`` if absent."""
    versions = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    versions["python"] = sys.version.split()[0]
    return versions


def git_revision(root: Path) -> str:
    """Return the repository commit or ``unknown`` outside a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def git_is_dirty(root: Path) -> bool | None:
    """Return tracked/untracked worktree status, or ``None`` outside Git."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=Path(root), capture_output=True, text=True, check=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def atomic_write_json(path: Path, value: Mapping) -> None:
    """Durably write JSON using a same-directory atomic replacement."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
