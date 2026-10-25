"""Measure what each cutoff profile does to a real generation run.

For every profile this prints the mean attempts per kept sample, the share of
samples with a real cut (ink loss > 0), the mean and max ink loss, and how
many attempts were rejected for ink at a side that was not cut.

Usage:
  python -m tools.cutoff_stats
  python -m tools.cutoff_stats --count 500 --profiles heavy --mode semi_broken_mixed
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
import tempfile
from pathlib import Path
from unittest import mock

import config
from src.cutoff import CUTOFF_ANNOTATIONS_FILENAME
from src.generation_profiles import CUTOFF_PROFILES


def measure(profile_id: str, *, count: int, seed: int, mode: str,
            augmentation: str) -> dict[str, float]:
    """Generate ``count`` samples with one cutoff profile and summarize them."""
    from src import generate_synthetic

    calls = {"attempts": 0, "edge_rejections": 0}
    original = generate_synthetic.assess_image_quality

    def counting(image, **kwargs):
        result = original(image, **kwargs)
        calls["attempts"] += 1
        if any("image edge" in error for error in result[1]):
            calls["edge_rejections"] += 1
        return result

    with tempfile.TemporaryDirectory(prefix="cutoff-stats-") as temporary, \
            mock.patch.object(config, "DATASETS_DIR", Path(temporary) / "datasets"), \
            mock.patch.object(generate_synthetic, "assess_image_quality", counting), \
            contextlib.redirect_stdout(io.StringIO()):
        out_dir = generate_synthetic.generate(
            count, dataset=f"stats_{profile_id}", seed=seed, sample_mode=mode,
            augmentation_profile=augmentation, edge_clipping=profile_id,
            show_bar=False, workers=1,
        )
        sidecar = out_dir / CUTOFF_ANNOTATIONS_FILENAME
        losses = []
        if sidecar.is_file():
            with sidecar.open(encoding="utf-8", newline="") as handle:
                losses = [float(row["ink_loss"]) for row in csv.DictReader(handle)]
    real = [loss for loss in losses if loss > 0]
    return {
        "attempts_per_sample": calls["attempts"] / count,
        "cut_share": len(real) / count,
        "mean_ink_loss": sum(real) / len(real) if real else 0.0,
        "max_ink_loss": max(real, default=0.0),
        "edge_rejections": calls["edge_rejections"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", default="regular", choices=list(config.SAMPLE_MODES))
    parser.add_argument("--augmentation", default="historical_scan_v1")
    parser.add_argument("--profiles", nargs="+", default=list(CUTOFF_PROFILES),
                        choices=list(CUTOFF_PROFILES))
    args = parser.parse_args(argv)

    print(f"{args.count} samples, seed {args.seed}, mode {args.mode}, "
          f"augmentation {args.augmentation}")
    print(f"{'profile':<10} {'attempts':>9} {'cut %':>7} {'mean loss':>10} "
          f"{'max loss':>9} {'limit':>6} {'edge rej':>9}")
    for profile_id in args.profiles:
        stats = measure(profile_id, count=args.count, seed=args.seed,
                        mode=args.mode, augmentation=args.augmentation)
        limit = CUTOFF_PROFILES[profile_id].max_ink_loss
        print(f"{profile_id:<10} {stats['attempts_per_sample']:>9.3f} "
              f"{stats['cut_share'] * 100:>6.1f}% {stats['mean_ink_loss']:>10.4f} "
              f"{stats['max_ink_loss']:>9.4f} {limit:>6.2f} "
              f"{stats['edge_rejections']:>9d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
