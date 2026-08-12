"""Streaming image-statistic summaries for synthetic/real comparison.

The metrics are descriptive diagnostics, not a realism score.  They expose
measurable gaps (brightness, contrast, ink coverage, geometry, and edge
density) so augmentation profiles can be calibrated against a representative
real subset without asserting that matching marginal statistics proves domain
equivalence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


STATISTICS_SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
METRIC_NAMES = (
    "width",
    "height",
    "aspect_ratio",
    "mean_luminance",
    "luminance_std",
    "robust_contrast",
    "ink_coverage",
    "edge_density",
)


@dataclass(frozen=True)
class ImageStatistics:
    width: float
    height: float
    aspect_ratio: float
    mean_luminance: float
    luminance_std: float
    robust_contrast: float
    ink_coverage: float
    edge_density: float


@dataclass(frozen=True)
class MetricSummary:
    count: int
    mean: float
    stddev: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class ImageGroupSummary:
    statistics_schema_version: int
    image_count: int
    metrics: dict[str, MetricSummary]

    def to_metadata(self) -> dict[str, object]:
        return {
            "statistics_schema_version": self.statistics_schema_version,
            "image_count": self.image_count,
            "metrics": {
                name: asdict(summary) for name, summary in self.metrics.items()
            },
        }


@dataclass(frozen=True)
class MetricComparison:
    generated_mean: float
    real_mean: float
    mean_difference: float
    standardized_difference: float | None


@dataclass(frozen=True)
class ImageStatisticsComparison:
    statistics_schema_version: int
    generated: ImageGroupSummary
    real: ImageGroupSummary
    metrics: dict[str, MetricComparison]
    interpretation: str

    def to_metadata(self) -> dict[str, object]:
        return {
            "statistics_schema_version": self.statistics_schema_version,
            "generated": self.generated.to_metadata(),
            "real": self.real.to_metadata(),
            "metrics": {
                name: asdict(comparison)
                for name, comparison in self.metrics.items()
            },
            "interpretation": self.interpretation,
        }


@dataclass
class _RunningMetric:
    count: int = 0
    mean: float = 0.0
    sum_squared_delta: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.sum_squared_delta += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def finish(self) -> MetricSummary:
        if self.count == 0:
            raise ValueError("Cannot summarize an empty image group")
        variance = self.sum_squared_delta / self.count
        return MetricSummary(
            count=self.count,
            mean=self.mean,
            stddev=math.sqrt(max(0.0, variance)),
            minimum=self.minimum,
            maximum=self.maximum,
        )


def calculate_image_statistics(image: Image.Image) -> ImageStatistics:
    """Calculate bounded, deterministic metrics for one decoded image."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("image dimensions must be positive")
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    low, high = np.percentile(gray, (5.0, 95.0))
    horizontal = np.abs(np.diff(gray, axis=1)) if image.width > 1 else np.empty((0,))
    vertical = np.abs(np.diff(gray, axis=0)) if image.height > 1 else np.empty((0,))
    edge_pixels = int(np.count_nonzero(horizontal > 20.0)) + int(
        np.count_nonzero(vertical > 20.0)
    )
    edge_denominator = horizontal.size + vertical.size
    return ImageStatistics(
        width=float(image.width),
        height=float(image.height),
        aspect_ratio=float(image.width / image.height),
        mean_luminance=float(gray.mean()),
        luminance_std=float(gray.std()),
        robust_contrast=float(high - low),
        ink_coverage=float(np.mean(gray < 200.0)),
        edge_density=(float(edge_pixels / edge_denominator)
                      if edge_denominator else 0.0),
    )


def iter_image_paths(root: str | Path) -> Iterator[Path]:
    """Yield supported image files under ``root`` in portable stable order."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    paths = (
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )
    yield from sorted(paths, key=lambda path: path.as_posix().casefold())


def summarize_image_statistics(
        images: Iterable[Image.Image | str | Path],
) -> ImageGroupSummary:
    """Summarize an iterable in constant memory (apart from one decoded image)."""
    running = {name: _RunningMetric() for name in METRIC_NAMES}
    count = 0
    for item in images:
        if isinstance(item, Image.Image):
            statistics = calculate_image_statistics(item)
        else:
            with Image.open(Path(item)) as image:
                statistics = calculate_image_statistics(image)
        count += 1
        for name in METRIC_NAMES:
            running[name].add(float(getattr(statistics, name)))
    if count == 0:
        raise ValueError("At least one image is required for a statistics summary")
    return ImageGroupSummary(
        statistics_schema_version=STATISTICS_SCHEMA_VERSION,
        image_count=count,
        metrics={name: running[name].finish() for name in METRIC_NAMES},
    )


def compare_image_statistics(
        generated_images: Iterable[Image.Image | str | Path],
        real_images: Iterable[Image.Image | str | Path],
) -> ImageStatisticsComparison:
    """Compare generated and real group means using pooled marginal spread."""
    generated = summarize_image_statistics(generated_images)
    real = summarize_image_statistics(real_images)
    comparisons = {}
    for name in METRIC_NAMES:
        generated_metric = generated.metrics[name]
        real_metric = real.metrics[name]
        difference = generated_metric.mean - real_metric.mean
        pooled = math.sqrt(
            (generated_metric.stddev ** 2 + real_metric.stddev ** 2) / 2.0
        )
        standardized = difference / pooled if pooled > 1e-12 else None
        comparisons[name] = MetricComparison(
            generated_mean=generated_metric.mean,
            real_mean=real_metric.mean,
            mean_difference=difference,
            standardized_difference=standardized,
        )
    return ImageStatisticsComparison(
        statistics_schema_version=STATISTICS_SCHEMA_VERSION,
        generated=generated,
        real=real,
        metrics=comparisons,
        interpretation=(
            "Descriptive marginal differences only; they help tune profiles but "
            "do not by themselves establish visual or OCR-domain equivalence."
        ),
    )

