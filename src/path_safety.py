"""Filesystem safety checks shared by dataset mutation operations."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from src.provenance import sha256_file


def is_reparse_point(path: Path) -> bool:
    """Return true for symbolic links and Windows reparse points/junctions."""
    path = Path(path)
    if path.is_symlink():
        return True
    try:
        attrs = getattr(os.lstat(path), "st_file_attributes", 0)
    except OSError:
        return False
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(flag and attrs & flag)


def require_regular_directory(path: Path, *, description: str) -> Path:
    """Require an existing, non-reparse directory and return its resolved path."""
    path = Path(path)
    if is_reparse_point(path):
        raise ValueError(f"{description} must not be a symlink or reparse point: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{description} is not a directory: {path}")
    return path.resolve(strict=True)


def snapshot_regular_tree(root: Path) -> dict[str, str]:
    """Hash every file in *root* without following reparse points.

    Logical names use forward slashes.  Unsupported filesystem objects are
    rejected so an archive cannot silently include pipes, devices, or links.
    """
    root = Path(root)
    root_resolved = require_regular_directory(root, description="Dataset root")
    snapshot: dict[str, str] = {}

    def visit(directory: Path, relative: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name.casefold())
        except OSError as exc:
            raise ValueError(f"Could not safely enumerate dataset directory {directory}: {exc}") from exc
        for entry in entries:
            entry_path = Path(entry.path)
            logical_path = relative / entry.name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ValueError(f"Could not inspect dataset entry {entry_path}: {exc}") from exc
            attrs = getattr(metadata, "st_file_attributes", 0)
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if entry.is_symlink() or bool(reparse_flag and attrs & reparse_flag):
                raise ValueError(
                    f"Dataset contains a symlink or reparse point: {logical_path.as_posix()}"
                )
            mode = metadata.st_mode
            if stat.S_ISDIR(mode):
                visit(entry_path, logical_path)
            elif stat.S_ISREG(mode):
                resolved = entry_path.resolve(strict=True)
                try:
                    resolved.relative_to(root_resolved)
                except ValueError:
                    raise ValueError(
                        f"Dataset entry escapes its root: {logical_path.as_posix()}"
                    ) from None
                snapshot[logical_path.as_posix()] = sha256_file(entry_path)
            else:
                raise ValueError(
                    f"Dataset contains an unsupported filesystem entry: "
                    f"{logical_path.as_posix()}"
                )

    visit(root, Path())
    return dict(sorted(snapshot.items()))
