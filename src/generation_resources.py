"""Preflight resource estimates for synthetic generation and packaging.

The defaults are conservative engineering assumptions, not measurements of a
particular corpus.  Callers may replace ``average_png_bytes`` with an observed
pilot-run value for a tighter estimate; the assumptions are retained in the
returned metadata so warnings remain explainable.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path


ESTIMATE_SCHEMA_VERSION = 1
DEFAULT_AVERAGE_PNG_BYTES = 96 * 1024
DEFAULT_METADATA_BYTES_PER_ROW = 768
DEFAULT_FIXED_ARTIFACT_BYTES = 4 * 1024 * 1024
DEFAULT_ARCHIVE_RATIO = 1.05
DEFAULT_SAFETY_FACTOR = 1.20
DEFAULT_FREE_SPACE_RESERVE_BYTES = 64 * 1024 * 1024
DEFAULT_LARGE_JOB_COUNT = 100_000
DEFAULT_MAX_WORKING_IMAGE_PIXELS = 4_000_000
DEFAULT_FONT_CACHE_BYTES = 128 * 1024 * 1024


def _positive_int(name: str, value: int, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or int(value) < minimum):
        relation = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {relation} integer; got {value!r}")
    return int(value)


def _finite(name: str, value: Real, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.expanduser().absolute()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(f"No existing parent for destination: {path}")
        candidate = parent
    return candidate


@dataclass(frozen=True)
class GenerationResourceEstimate:
    estimate_schema_version: int
    sample_count: int
    archive_requested: bool
    estimated_dataset_bytes: int
    estimated_archive_bytes: int
    estimated_peak_disk_bytes: int
    required_free_space_bytes: int
    available_free_space_bytes: int
    estimated_peak_memory_bytes: int
    confirmation_recommended: bool
    enough_free_space: bool
    warnings: tuple[str, ...]
    assumptions: dict[str, object]

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


class InsufficientDiskSpaceError(OSError):
    """Raised before generation when the conservative estimate cannot fit."""

    def __init__(self, estimate: GenerationResourceEstimate):
        self.estimate = estimate
        super().__init__(
            "Insufficient free space for generation: need approximately "
            f"{format_bytes(estimate.required_free_space_bytes)}, have "
            f"{format_bytes(estimate.available_free_space_bytes)}"
        )


def format_bytes(value: int) -> str:
    """Format a byte count for a CLI/GUI warning."""
    value = _positive_int("value", value, allow_zero=True)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units[:-1]:
        if amount < 1024.0:
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} {units[-1]}"


def estimate_generation_resources(
        destination: str | Path,
        sample_count: int,
        *,
        create_archive: bool = False,
        average_png_bytes: int = DEFAULT_AVERAGE_PNG_BYTES,
        metadata_bytes_per_row: int = DEFAULT_METADATA_BYTES_PER_ROW,
        fixed_artifact_bytes: int = DEFAULT_FIXED_ARTIFACT_BYTES,
        archive_ratio: float = DEFAULT_ARCHIVE_RATIO,
        safety_factor: float = DEFAULT_SAFETY_FACTOR,
        free_space_reserve_bytes: int = DEFAULT_FREE_SPACE_RESERVE_BYTES,
        large_job_count: int = DEFAULT_LARGE_JOB_COUNT,
        max_working_image_pixels: int = DEFAULT_MAX_WORKING_IMAGE_PIXELS,
        font_cache_bytes: int = DEFAULT_FONT_CACHE_BYTES,
) -> GenerationResourceEstimate:
    """Estimate peak disk/memory requirements and inspect destination space.

    Archive creation temporarily needs both the source dataset and archive, so
    their estimates are added for peak disk usage even for ``zip-only`` jobs.
    Memory is constant with sample count when the streaming APIs are used; the
    estimate covers one RGB image, NumPy working buffers, encoding overhead,
    and the bounded font cache.
    """
    count = _positive_int("sample_count", sample_count)
    average_png_bytes = _positive_int("average_png_bytes", average_png_bytes)
    metadata_bytes_per_row = _positive_int(
        "metadata_bytes_per_row", metadata_bytes_per_row, allow_zero=True
    )
    fixed_artifact_bytes = _positive_int(
        "fixed_artifact_bytes", fixed_artifact_bytes, allow_zero=True
    )
    reserve = _positive_int(
        "free_space_reserve_bytes", free_space_reserve_bytes, allow_zero=True
    )
    large_job_count = _positive_int("large_job_count", large_job_count)
    pixels = _positive_int("max_working_image_pixels", max_working_image_pixels)
    font_cache_bytes = _positive_int(
        "font_cache_bytes", font_cache_bytes, allow_zero=True
    )
    archive_ratio = _finite("archive_ratio", archive_ratio)
    safety_factor = _finite("safety_factor", safety_factor, minimum=1.0)

    per_row = average_png_bytes + metadata_bytes_per_row
    dataset_bytes = fixed_artifact_bytes + count * per_row
    archive_bytes = (
        math.ceil(dataset_bytes * archive_ratio) if create_archive else 0
    )
    peak_disk = dataset_bytes + archive_bytes
    required = math.ceil(peak_disk * safety_factor) + reserve
    available = shutil.disk_usage(_nearest_existing_path(Path(destination))).free

    # RGB source + float32/int16/uint8 augmentation arrays can coexist.  The
    # 24-bytes-per-pixel allowance is conservative for those intermediates.
    working_image_bytes = pixels * 24
    encoding_buffer_bytes = max(average_png_bytes * 4, 4 * 1024 * 1024)
    peak_memory = font_cache_bytes + working_image_bytes + encoding_buffer_bytes

    warnings = []
    confirmation_recommended = count >= large_job_count
    if confirmation_recommended:
        warnings.append(
            f"Large job ({count:,} samples): request user confirmation before starting."
        )
    enough = available >= required
    if not enough:
        warnings.append(
            "Estimated peak disk usage plus reserve exceeds available free space."
        )
    warnings.append(
        "Disk estimates use configured per-image assumptions; calibrate with a pilot run."
    )

    return GenerationResourceEstimate(
        estimate_schema_version=ESTIMATE_SCHEMA_VERSION,
        sample_count=count,
        archive_requested=bool(create_archive),
        estimated_dataset_bytes=dataset_bytes,
        estimated_archive_bytes=archive_bytes,
        estimated_peak_disk_bytes=peak_disk,
        required_free_space_bytes=required,
        available_free_space_bytes=available,
        estimated_peak_memory_bytes=peak_memory,
        confirmation_recommended=confirmation_recommended,
        enough_free_space=enough,
        warnings=tuple(warnings),
        assumptions={
            "average_png_bytes": average_png_bytes,
            "metadata_bytes_per_row": metadata_bytes_per_row,
            "fixed_artifact_bytes": fixed_artifact_bytes,
            "archive_ratio": archive_ratio,
            "safety_factor": safety_factor,
            "free_space_reserve_bytes": reserve,
            "large_job_count": large_job_count,
            "max_working_image_pixels": pixels,
            "font_cache_bytes": font_cache_bytes,
            "archive_peak_includes_source_and_archive": bool(create_archive),
        },
    )


def preflight_generation_resources(
        destination: str | Path,
        sample_count: int,
        *,
        create_archive: bool = False,
        require_free_space: bool = True,
        **estimate_options,
) -> GenerationResourceEstimate:
    """Return an estimate, raising before writes when disk space is inadequate."""
    estimate = estimate_generation_resources(
        destination,
        sample_count,
        create_archive=create_archive,
        **estimate_options,
    )
    if require_free_space and not estimate.enough_free_space:
        raise InsufficientDiskSpaceError(estimate)
    return estimate
