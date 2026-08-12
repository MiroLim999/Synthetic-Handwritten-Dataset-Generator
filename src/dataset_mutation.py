"""Dataset-scoped coordination for operations that publish or mutate a run.

The marker is deliberately stored beside the dataset directory so it survives
temporary replacement/removal of the directory itself.  It coordinates this
toolkit's merge and packaging operations; semantic post-archive validation
still protects against non-cooperating writers.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def mutation_lock_path(dataset_dir: Path) -> Path:
    """Return the single lock path shared by all mutations of *dataset_dir*."""
    dataset_dir = Path(dataset_dir)
    return dataset_dir.parent / f".{dataset_dir.name}.mutation.lock"


@contextmanager
def dataset_mutation_lock(dataset_dir: Path, *, purpose: str) -> Iterator[Path]:
    """Acquire an atomic, process-visible mutation marker for one dataset.

    Cleanup removes only the marker whose random ownership token this context
    wrote.  That prevents an old context from unlinking a replacement marker.
    A marker left by a terminated process is intentionally not guessed stale;
    deleting one is an explicit operator decision because PID reuse makes
    automatic reclamation unsafe.
    """
    dataset_dir = Path(dataset_dir)
    lock_path = mutation_lock_path(dataset_dir)
    token = uuid.uuid4().hex
    payload = json.dumps(
        {"pid": os.getpid(), "purpose": str(purpose), "token": token},
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    descriptor: int | None = None
    acquired = False
    try:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(
                f"Dataset is busy with another mutation: {lock_path}"
            ) from exc
        acquired = True
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        yield lock_path
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if acquired:
            try:
                if lock_path.read_bytes() == payload:
                    lock_path.unlink()
            except FileNotFoundError:
                pass


def sibling_archive_paths(dataset_dir: Path) -> tuple[Path, Path]:
    """Return the published ZIP and checksum paths for one dataset folder."""
    dataset_dir = Path(dataset_dir)
    archive = dataset_dir.parent / f"{dataset_dir.name}.zip"
    return archive, archive.with_name(f"{archive.name}.sha256")


def assert_no_sibling_archive(dataset_dir: Path) -> tuple[Path, Path]:
    """Refuse mutation or packaging when either published artifact exists."""
    archive, checksum = sibling_archive_paths(dataset_dir)
    for artifact in (archive, checksum):
        if os.path.lexists(artifact):
            raise FileExistsError(
                f"Dataset archive artifact already exists: {artifact}"
            )
    return archive, checksum
