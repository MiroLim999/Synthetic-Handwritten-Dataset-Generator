"""Record and check output hashes for a fixed matrix of generator configs.

Every config generates ``count=48, seed=42`` into a temporary datasets
directory and records the ``manifest_sha256`` and ``images_sha256`` from its
run metadata.  The hashes depend on the locally installed fonts, so the
baseline file lives under the gitignored ``dataset/`` folder.

Usage:
  python -m tools.golden_hashes --update            # write the baseline
  python -m tools.golden_hashes --check             # compare with the baseline
  python -m tools.golden_hashes --update --only "semi_broken_*"
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import config
from src.manifest import RUN_METADATA_FILENAME

GOLDEN_PATH = config.DATASET_DIR / ".golden-hashes.json"
GOLDEN_COUNT = 48
GOLDEN_SEED = 42
MODES = ("regular", "semi_broken_mixed", "semi_broken_characters")
AUGMENTATIONS = ("historical_scan_v1", "clean", "scan_noise")
CUTOFFS = ("none", "heavy")


def golden_configs() -> dict[str, dict]:
    """Return the config matrix keyed by a stable, human-readable name."""
    configs: dict[str, dict] = {}
    for mode in MODES:
        for augmentation in AUGMENTATIONS:
            for cutoff in CUTOFFS:
                configs[f"{mode}/{augmentation}/{cutoff}"] = {
                    "sample_mode": mode,
                    "augmentation_profile": augmentation,
                    "edge_clipping": cutoff,
                }
    configs["regular/historical_scan_v1/none/workers=1"] = {
        "sample_mode": "regular",
        "augmentation_profile": "historical_scan_v1",
        "edge_clipping": "none",
        "workers": 1,
    }
    return configs


def compute_hashes(names: list[str]) -> dict[str, dict[str, str]]:
    """Generate each named config and return its output hashes."""
    from src.generate_synthetic import generate

    configs = golden_configs()
    results: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory(prefix="golden-hashes-") as temporary:
        root = Path(temporary) / "datasets"
        with mock.patch.object(config, "DATASETS_DIR", root):
            for index, name in enumerate(names, start=1):
                print(f"[{index}/{len(names)}] {name}", flush=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    out_dir = generate(
                        GOLDEN_COUNT,
                        dataset=f"golden_{index:03d}",
                        seed=GOLDEN_SEED,
                        show_bar=False,
                        **configs[name],
                    )
                metadata = json.loads(
                    (out_dir / RUN_METADATA_FILENAME).read_text(encoding="utf-8")
                )
                results[name] = {
                    "manifest_sha256": metadata["manifest_sha256"],
                    "images_sha256": metadata["images_sha256"],
                }
    return results


def _selected(only: str | None) -> list[str]:
    names = list(golden_configs())
    if only:
        names = [name for name in names if fnmatch.fnmatchcase(name, only)]
        if not names:
            raise SystemExit(f"--only {only!r} matches no golden config")
    return names


def _load_baseline() -> dict[str, dict[str, str]]:
    if not GOLDEN_PATH.is_file():
        raise SystemExit(
            f"No golden baseline at {GOLDEN_PATH}; run with --update first"
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["configs"]


def update(only: str | None) -> int:
    names = _selected(only)
    computed = compute_hashes(names)
    baseline = {}
    if only and GOLDEN_PATH.is_file():
        baseline = _load_baseline()
    baseline.update(computed)
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(
        json.dumps({
            "count": GOLDEN_COUNT,
            "seed": GOLDEN_SEED,
            "configs": dict(sorted(baseline.items())),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Updated {len(computed)} config(s) in {GOLDEN_PATH}")
    return 0


def check(only: str | None) -> int:
    baseline = _load_baseline()
    names = _selected(only)
    computed = compute_hashes(names)
    changed = []
    for name in names:
        expected = baseline.get(name)
        if expected != computed[name]:
            changed.append(name)
            if expected is None:
                print(f"CHANGED {name}: not in the baseline")
                continue
            for key in ("manifest_sha256", "images_sha256"):
                if expected.get(key) != computed[name][key]:
                    print(
                        f"CHANGED {name}: {key} "
                        f"{expected.get(key, '')[:12]} -> {computed[name][key][:12]}"
                    )
    if changed:
        print(f"{len(changed)} of {len(names)} golden config(s) changed")
        return 1
    print(f"All {len(names)} golden config(s) match")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--update", action="store_true",
                        help="regenerate and write the baseline")
    action.add_argument("--check", action="store_true",
                        help="regenerate and compare with the baseline")
    parser.add_argument("--only", metavar="GLOB",
                        help="limit to configs whose name matches this pattern")
    args = parser.parse_args(argv)
    return update(args.only) if args.update else check(args.only)


if __name__ == "__main__":
    sys.exit(main())
