# Developer Studio Plan

Execution plan for fixing and extending the Developer Studio in `gui.py` and the
generator code behind it. Every task is a checkbox: tick it only when its
acceptance criteria and tests pass.

## How to use this plan

- Work through phases in order: **S0 -> S7**. Later phases build on earlier ones.
- Do one work package at a time (for example S1.2), one commit per package.
- Read every file a package lists under "Affected files" before editing it.
- After each package, run the [gate commands](#gate-commands). After each phase,
  complete its verification gate.
- Never change generated output by accident. A change that alters output on
  purpose must be listed in [Intentional output changes](#intentional-output-changes)
  and re-baselined with `python -m tools.golden_hashes --update` in the same commit.
- If a task turns out wrong or unnecessary, strike it through (`~~task~~`) with a
  one-line reason instead of deleting it.

## Decisions

| Topic | Decision |
|---|---|
| Character Cutoff | Real cuts. Cut lines are measured from the ink bounding box, so letters are actually clipped. The quality check allows ink only at sides cut on purpose, ink loss is capped, and every cut sample is recorded. |
| Presets | One preset format with optional sections: `augmentation`, `stroke_damage`, `cutoff`, `run`. Loading applies only the sections a preset contains. |
| Range controls | Every effect is a chance (probability) plus a min/max range, mapping 1:1 to the profile dataclasses. |

## Assumptions (change these before starting if you disagree)

- Built-in presets stay in `resources/presets/` and are read-only in the GUI. User
  presets go to `resources/presets/user/` (gitignored).
- The studio moves to its own **Studio** tab; the Generate tab keeps a one-line
  recipe summary.
- The Kaggle contract is frozen: no changes to `MANIFEST_COLUMNS`,
  `EVALUATION_ANNOTATION_COLUMNS`, `metadata_schema_version` (stays `1`; new
  metadata keys are additive), or the notebooks.
- The cutoff numbers in S3.1 are starting values, tuned in S3.9 and S6.
- `profile_1.json` stays a tracked built-in, migrated mechanically.

## Baseline (2026-10-25, `main` at `006470a`)

- `python -m unittest discover -s tests`: 175 tests, OK (4 skipped).
- `python -m ruff check .`: 1 pre-existing error (F841, unused `np_rng` in
  `generate()`).
- `python -m mypy`: clean (2 files).

## Ground rules

- **Determinism.** Default paths (no recipe, cutoff `none`, built-in profiles) keep
  their exact RNG draw order. New random draws happen only behind options that are
  off by default.
- **Never bump `PROFILE_SCHEMA_VERSION`.** It is part of the writer-style seed
  (`_style_seed`), so bumping it changes every dataset. If a profile needs a shape
  version, add a separate key to its `to_metadata()`.
- **Picklable profiles.** New profile types are top-level frozen dataclasses;
  generation sends them to `ProcessPoolExecutor` workers.
- **Tk only on the UI thread.** Background work reports through `self.q`, the way
  `_refresh_datasets` does today.
- **Testable without a display.** Logic lives in pure modules under `src/`; Tk code
  stays thin. GUI tests use the existing bare-`App` pattern
  (`object.__new__(gui.App)`).
- **Validate before writing.** Invalid recipes, presets, or flags fail before any
  dataset folder is reserved.

## Gate commands

```powershell
python -m unittest discover -s tests -v
python -m ruff check .
python -m mypy
python -m tools.golden_hashes --check
```

## Intentional output changes

| Phase | Output that changes on purpose | Golden configs to re-baseline |
|---|---|---|
| S1 | GUI runs only: Regular mode stops applying stroke damage, and the field mix follows the sample mode. | none (GUI runs are not in the matrix) |
| S2 | CLI semi-broken runs: labels of 3 or fewer characters get capped gaps and erosion, because the short-label safeguard now covers the CLI too. Custom profile IDs become content-addressed (metadata only). | `semi_broken_*` |
| S3 | Any run with cutoff other than `none`: cuts become real, ink-relative cuts, and `cutoff-annotations.csv` is added. | `cutoff=heavy` |
| S4 | Every new dataset gains `studio-recipe.json` and new metadata keys. Images do not change. | none |

Everything else stays byte-identical.

## Architecture at a glance

| Module | Responsibility | Phase |
|---|---|---|
| `src/studio_spec.py` | One parameter table (path, label, bounds, step, unit, default, tooltip, profile field). Drives the UI, validation, and recipes. | S2 |
| `src/generation_profiles.py` | Extended `AugmentationProfile`; new `StrokeDamageProfile` and `CutoffProfile`; spec-to-profile builders; content-addressed IDs. | S2-S3 |
| `src/cutoff.py` | Ink mask, rotation tracking, ink-relative cut, ink-loss bound. | S3 |
| `src/studio_recipe.py` | Recipe v2 with optional sections: load, save, validate, v1 migration, canonical hash, apply rules, studio state. | S4-S5 |
| `src/sample_pipeline.py` | Single-attempt sample function shared by the generator and the preview. | S6 |
| `src/preview.py` | First N samples of the planned run, plus retry statistics. | S6 |
| `gui_widgets.py` | `RangeModel` (pure), `RangeSlider`, chance slider. | S5 |
| `gui_studio.py` | `StudioPanel` built from `studio_spec`, preset bar, preview grid. | S5-S6 |

---

# S0 - Baseline and safety net

## S0.1 Branch and clean gates

Affected files: `src/generate_synthetic.py`, `.gitignore`

- [x] Create branch `feature/developer-studio` from `main`.
- [x] Run the gate commands (without the golden check) and confirm the baseline above.
- [x] Remove the unused `np_rng` in `generate()` so ruff is clean. It is never
      read, so output is unaffected.
- [x] Stop tracking logs: add `logs/` to `.gitignore`, then run
      `git rm --cached logs/gui.log logs/gui-stdout.log`.

## S0.2 Golden-hash tool

Affected files: `tools/golden_hashes.py` (new)

- [x] Generate count=48, seed=42 datasets into a temporary `DATASETS_DIR`, patching
      `config.DATASETS_DIR` the way the tests do. Record `manifest_sha256` and
      `images_sha256` for each config.
- [x] Matrix: mode {`regular`, `semi_broken_mixed`, `semi_broken_characters`} x
      augmentation {`historical_scan_v1`, `clean`, `scan_noise`} x cutoff {`none`,
      `heavy`}, plus `regular/historical_scan_v1/none` at `workers=1`.
- [x] `--update` writes `dataset/.golden-hashes.json`. The file is already
      gitignored, and its hashes depend on the locally installed fonts. `--check`
      prints every config whose hashes changed and exits non-zero.
- [x] Put the entry point under `if __name__ == "__main__":`, because process pools
      on Windows re-import the main module.
- [x] Run `--update` once on the untouched baseline.

Acceptance criteria:

- `--check` passes immediately after `--update`.
- The `workers=1` config hashes equal the default-workers config.

### S0 verification gate

- [x] All gate commands pass, ruff included.

---

# S1 - Correctness fixes in the current studio

Fix the confirmed bugs before redesigning anything.

## S1.1 Testable run snapshot (enabler)

Affected files: `gui.py`

- [x] Move everything `App.start()` collects into
      `App._snapshot_run_options() -> RunOptions`, a frozen dataclass of plain
      values. `start()` validates and launches from that snapshot.
- [x] Keep the existing validation messages and confirmation dialogs.

## S1.2 Stroke damage only in semi-broken modes

Affected files: `src/augment.py`, `gui.py`

- [x] In `degrade()`, run `_semi_broken` only when
      `damage_profile.startswith("semi_broken")`. Params tune the damage; they never
      turn it on.
- [x] `RunOptions` and `_update_preview()` pass `semi_broken_params=None` outside
      semi-broken modes.
- [x] Disable the Stroke Damage sliders while the mode is not semi-broken. This must
      keep working with the `_run_controls` lock/restore in `_set_job_active()`.

Acceptance criteria:

- With `damage_profile="regular"`, `degrade()` returns pixel-identical output with
  and without stroke params (fixed seed).

## S1.3 Field weights follow the sample mode

Affected files: `src/generate_synthetic.py`, `gui.py`

- [x] Rename `_field_weights_for_mode` to public `field_weights_for_mode`; keep the
      private name as an alias.
- [x] Fill the weight grid from `field_weights_for_mode(mode)` at startup and on
      every mode change, with 0 for types the mode doesn't use. If that replaces
      edited values, say so in the status line.
- [x] "Reset Weights" resets to the current mode's weights.

Acceptance criteria:

- For every mode, the generation plan built from the GUI grid equals the plan built
  from `field_weights_for_mode(mode)`. This is exact because
  `iter_weighted_choices` sorts keys and skips zero weights.
- GUI Regular runs contain no `character` or `numeric` rows unless the user gives
  those fields a weight.

## S1.4 Honest labels and Clean Minimal

Affected files: `gui.py`, `resources/presets/clean_minimal.json`

- [x] Rename the sliders, widening the label column (currently `width=13`) so the
      new names fit:
      - "Contrast Drop" -> "Contrast-drop chance"
      - "Stroke Thin" -> "Stroke-thin chance"
      - "Gap Prob" -> "Gap chance"
      - "Gap Count" -> "Max gaps"
      - "Scratch Prob" -> "Scratch chance"
      - "Ink Fading" -> "Ink contrast", with the direction "lower = more faded" in
        the section subtitle
- [x] Set `"contrast": 0.0` in `clean_minimal.json`. Today the value 1.0 means broken
      contrast on 100% of images.

## S1.5 Safe preset loading

Affected files: `gui.py`

- [x] Validate every value first (number, finite, clamped to the slider range;
      unknown keys reported), then apply them all. Never apply half a preset.
- [x] Reset `_slider_updating` in a `finally` block.
- [x] Set Char cutoff to Custom when any `crop_*` value is above 0; otherwise set it
      to None.
- [x] The status line lists what was applied and which keys were ignored.

## S1.6 Safe preset saving

Affected files: `gui.py`

- [x] Slug the name to `[a-z0-9_-]`, 1-64 characters. Reject empty results and
      Windows reserved names (reuse `config._WINDOWS_RESERVED_NAMES`). The saved
      file must stay inside `PRESET_DIR`.
- [x] Ask before overwriting an existing preset.
- [x] Write atomically (temp file, then `os.replace`), with values snapped to each
      slider's step.

## S1.7 Slider steps that mean something

Affected files: `gui.py`

- [x] Add pure helpers `snap_value(value, step, low, high)` and
      `format_value(value, step, unit)`.
- [x] The slider callback snaps its variable to the step. Every label (move, preset
      load, profile change, reset) uses `format_value`, so there is no more `int()`
      truncation.

## S1.8 Reset scope

Affected files: `gui.py`

- [x] The studio header's "Reset" resets only studio sliders. Degradation sliders
      take the selected profile's values (as `_on_degradation_preset_change` does);
      stroke and crop sliders return to their defaults.
- [x] "Reset All Controls" also resets field weights and split ratios, which it
      skips today.

## S1.9 Provenance for GUI runs (additive)

Affected files: `src/generate_synthetic.py`

- [x] Add three keys to `run-metadata.json` under `effective_selection`:
      `stroke_damage_params` (null outside semi-broken modes), `field_weights`
      (effective weights with zeros dropped), and `split_fractions`.
- [x] `metadata_schema_version` stays `1`.

## S1.10 Tests

Affected files: `tests/test_studio_bugfixes.py` (new)

- [x] Regular-mode `degrade` ignores stroke params (pixel equality).
- [x] The snapshot passes `semi_broken_params=None` in Regular mode and a dict in
      semi-broken modes.
- [x] For each mode, the grid weights give the same plan as `field_weights_for_mode`.
- [x] A malformed preset leaves sliders unchanged and responsive
      (`_slider_updating` is False).
- [x] Saving rejects `a/b`, `..\x`, `con`, `:x`, and blank names. An existing name
      with a declined overwrite leaves the file unchanged (use a temporary
      `PRESET_DIR`).
- [x] Edge cases for `snap_value` and `format_value`.
- [x] Metadata contains the S1.9 keys.
- [x] Parity: GUI options (Regular, built-in profile, default weights and splits) and
      CLI defaults with the same seed give identical `images_sha256` and manifest
      hash.

### S1 verification gate

- [x] Gate commands pass, and the golden check reports no changes (the CLI is
      untouched).
- [x] Manual check: load every bundled preset, then generate 200 samples in Regular
      and in "Semi-broken: characters". Check the field mix and the metadata.

---

# S2 - Range-based profile models (backend)

## S2.1 Parameter spec table

Affected files: `src/studio_spec.py` (new)

- [ ] One entry per parameter: path (for example `augmentation.blur.range`),
      section, label, kind (`chance` | `range` | `value`), bounds, step, unit,
      default, tooltip, and the target profile field(s).
- [ ] Augmentation parameters:
      - chance + range: fade, brightness, paper tint, rotation (min/max absolute
        tilt), blur, noise, paper texture, scanlines, JPEG
      - stains: chance + count range + opacity range
- [ ] Stroke damage: contrast, erode (blend range), gaps (count and size ranges),
      scratches (count range).
- [ ] Cutoff: chance + range for each side, plus max ink loss.
- [ ] Integer parameters (scanline alpha, JPEG quality, counts, opacity) use step 1
      and are stored as ints.
- [ ] Bounds are at least the clamps of `create_custom_augmentation_profile`:
      tint <= 0.50, rotation <= 15 degrees, blur <= 5, noise <= 50, texture <= 20,
      scanlines <= 60, JPEG 10-100. They must also cover every registered profile.

## S2.2 Extend `AugmentationProfile` with byte-identical defaults

Affected files: `src/generation_profiles.py`, `src/augment.py`

- [ ] Add `rotate_min_degrees = 0.0`. At 0, `_rotate` keeps its single legacy
      `uniform(-max, max)` draw. Above 0, it draws a magnitude, then a sign.
- [ ] Add `stain_count_range = (2, 6)` and `stain_opacity_range = (15, 45)`.
      `_stains` uses them with the same calls as today.
- [ ] Validate the new fields in `__post_init__`; `to_metadata()` adds
      `augmentation_schema_version: 2`. Do not touch `PROFILE_SCHEMA_VERSION`.

## S2.3 Spec <-> profile builders

Affected files: `src/generation_profiles.py`

- [ ] Add `augmentation_profile_from_spec(spec)` and
      `augmentation_spec_from_profile(profile)`.
- [ ] Content-addressed IDs: `custom_<first 12 hex of sha256(canonical JSON)>`. When
      the values equal a registered profile, reuse that profile's ID.
- [ ] Keep `create_custom_augmentation_profile` working unchanged and mark it
      deprecated.

## S2.4 `StrokeDamageProfile`

Affected files: `src/generation_profiles.py`

- [ ] Fields: `contrast_probability`, `contrast_range`, `erode_probability`,
      `erode_blend_range`, `gap_probability`, `gap_count_range`, `gap_size_range`,
      `scratch_probability`, `scratch_count_range`.
- [ ] `stroke_damage_from_config()` reads the `BROKEN_*` values at call time, so
      `mock.patch` on config keeps working.
- [ ] `from_legacy_params(dict)` reproduces today's GUI dict exactly:
      - `gap_count: N` -> `gap_count_range = (1, max(1, int(N)))`
      - missing keys fall back to config, as `_semi_broken` does now
- [ ] `constrained_for_label(label, attempt)` takes over the safeguard logic from
      `_generate_valid_sample` and reproduces it exactly for legacy params:
      - labels of 3 or fewer characters: cap both ends of the gap-count range at
        3 x length; cap erode chance at 0.35
      - attempts above 8: scale gap chance, erode chance, and the gap-count range by
        `max(0.1, 1 - 0.1 * (attempt - 8))`, with a minimum of 1 gap
- [ ] Add `to_metadata()` and validation; keep the class frozen and picklable.

## S2.5 Augment and generator wiring

Affected files: `src/augment.py`, `src/generate_synthetic.py`

- [ ] `_semi_broken` takes a `StrokeDamageProfile` and makes the same RNG calls in
      the same order for the config defaults.
- [ ] `degrade(semi_broken_params=...)` accepts a dict, a profile, or None.
- [ ] `generate(stroke_damage_profile=...)`, keeping `semi_broken_params` as an
      alias. The safeguard now applies to every semi-broken run.
- [ ] Metadata: replace S1.9's raw `stroke_damage_params` with a top-level
      `stroke_damage_profile` (`to_metadata()`).

## S2.6 GUI interim wiring

Affected files: `gui.py`

- [ ] `_build_custom_developer_profile()` uses `augmentation_profile_from_spec`
      with chance 1.0 and min = max. Images are unchanged; the ID becomes
      content-addressed.
- [ ] `_build_semi_broken_params()` returns `StrokeDamageProfile.from_legacy_params(...)`.

## S2.7 Tests

Affected files: `tests/test_augmentation_spec.py` (new),
`tests/test_stroke_damage_profiles.py` (new)

- [ ] Every registered profile survives spec -> profile -> spec without loss and
      keeps its ID.
- [ ] Content IDs are stable across processes and change whenever any value
      changes.
- [ ] Every registered profile lies inside the spec bounds.
- [ ] The interim GUI builder equals `create_custom_augmentation_profile`
      field-for-field, ID aside.
- [ ] Config-default stroke damage is byte-identical to the previous path;
      `tests/test_rng_reproducibility.py` passes unmodified.
- [ ] `from_legacy_params` and `constrained_for_label` match today's behavior.

### S2 verification gate

- [ ] Gate commands pass. Only `semi_broken_*` golden configs changed; re-baseline
      exactly those.

---

# S3 - Real character cutoff

## S3.1 `CutoffProfile`

Affected files: `src/generation_profiles.py`, `config.py`

- [ ] Add `CutoffSide(probability, fraction_range)` and `CutoffProfile` with
      `profile_id`, `top`, `bottom`, `left`, `right`, `max_ink_loss`,
      `short_label_chars=3`, `short_label_max_fraction=0.10`, and method
      `ink_bbox_v1`.
- [ ] Fractions are of the ink box: its height for top/bottom, its width for
      left/right.
- [ ] Registry, with starting values tuned in S3.9:

  | Profile | Top / bottom (each) | Left / right (each) | Max ink loss |
  |---|---|---|---|
  | `none` | 0 | 0 | - |
  | `light` | 50% chance, 0.03-0.08 | 0 | 0.10 |
  | `moderate` | 60%, 0.06-0.14 | 15%, 0.02-0.05 | 0.15 |
  | `heavy` | 75%, 0.10-0.22 | 30%, 0.03-0.08 | 0.20 |

- [ ] `constrained_for_label(label)`: labels of 3 or fewer characters get no
      left/right cuts, and their vertical cuts are capped at
      `short_label_max_fraction`.
- [ ] Content-addressed IDs for custom profiles. `to_metadata()` includes
      `cutoff_schema_version: 1` and the method. Frozen and picklable.
- [ ] Relabel `config.EDGE_CLIPPING_OPTIONS`, keeping the keys:
      - "Light - tips of tall letters"
      - "Moderate - loops and some word ends"
      - "Heavy - ascenders, descenders and word ends"
      - "Custom"

## S3.2 Cutoff engine

Affected files: `src/cutoff.py` (new), `src/augment.py`

- [ ] At the start of `degrade`, build an ink mask from the clean render
      (gray < 200), but only when some side has chance > 0.
- [ ] Split `_rotate` into `_draw_rotation_angle` (same draws) and `_rotate_by`.
      Apply the same angle to the image (bicubic, white fill) and the mask
      (nearest, 0 fill), both with `expand=True`.
- [ ] Do the cut where `_clip_edges` runs today. For each side, in the order top,
      bottom, left, right, and only if its chance is above 0: draw
      `rng.random() < p`, then `rng.uniform(*range)`.
- [ ] Measure cut lines inward from the mask's bounding box. Sides that are not cut
      keep their full extent and padding. Keep at least 10 px in each dimension.
- [ ] Ink loss = removed mask pixels / all mask pixels. If it exceeds
      `max_ink_loss`, halve all fractions (up to 3 times, no new draws). If it still
      exceeds the limit, make no cut.
- [ ] Add `degrade_detailed() -> DegradeResult(image, cutoff: CutoffResult)`.
      `degrade()` becomes a wrapper that returns only the image.
- [ ] No ink means no cut. Cutoff `none` makes zero extra draws, so its output stays
      byte-identical.

## S3.3 Quality check aware of cuts

Affected files: `src/dataset_validation.py`, `src/generate_synthetic.py`

- [ ] `assess_image_quality(image, *, cut_sides=frozenset())` counts edge ink per
      side and skips the sides that were cut. The threshold uses only the remaining
      sides' area.
- [ ] With empty `cut_sides`, results are identical to today (same slices, same
      threshold).
- [ ] `_generate_valid_sample` applies `constrained_for_label`, passes the cut sides
      to the check, and keeps the fallback that drops cutting after 15 attempts.

## S3.4 Per-sample record: `cutoff-annotations.csv`

Affected files: `src/generate_synthetic.py`

- [ ] Worker results on both the thread and process paths carry the
      `CutoffResult` of the accepted attempt.
- [ ] Write rows with `AtomicCsvWriter`, in manifest order, for cut samples only:
      `filename, split, cut_top, cut_bottom, cut_left, cut_right, ink_loss`
      (4 decimal places).
- [ ] Leave `evaluation-annotations.csv` alone; the notebooks pin its columns.

## S3.5 Validator, archive, and merge

Affected files: `src/dataset_validation.py`, `src/handoff_validation.py`,
`src/generate_synthetic.py`

- [ ] `validate_dataset` loads the sidecar, when present, into its SQLite index so
      memory stays bounded.
- [ ] It rejects malformed rows, duplicates, values outside [0, 1), and rows without
      a matching synthetic manifest row. Valid rows pass their cut sides to the
      image check.
- [ ] Report statistics gain `cutoff: {samples, by_side, max_ink_loss}`.
- [ ] Additive metadata keys: `cutoff_profile` (`to_metadata()`),
      `cutoff_annotations` (filename), and `cutoff_annotations_sha256`.
      `effective_selection.edge_clipping` holds the cutoff profile ID.
- [ ] `validate_handoff_bytes` and `validate_handoff_files` get an optional sidecar
      argument and check its hash whenever metadata names the file. This is
      generator-side only; the notebooks are untouched.
- [ ] `_write_montage` includes up to 4 cut samples, captioned `cut:top` and so on.

## S3.6 GUI interim and old API

Affected files: `gui.py`, `src/augment.py`, `tests/test_degradation_and_clipping.py`

- [ ] The Char cutoff dropdown maps to `CUTOFF_PROFILES`.
- [ ] Custom crop sliders build a `CutoffProfile` with chance 1.0 and min = max per
      side. Slider ranges come from the spec. Labels: "Top cut - % of letter height",
      "Left cut - % of word width".
- [ ] `_on_clipping_preset_change` fills the custom sliders with range midpoints
      (interim; replaced in S5).
- [ ] Remove `EDGE_CLIPPING_RANGES` and the canvas-relative `_clip_edges`, and
      update their imports.
- [ ] Rewrite `test_edge_clipping_dimensions` for the new behavior: a blank image is
      not cut, and an image with ink is cut relative to its ink.

## S3.7 Tests

Affected files: `tests/test_character_cutoff.py` (new), `tests/test_dataset_validation.py`

- [ ] A top cut removes ink at the top: ink touches the top edge and no other edge.
- [ ] Ink loss stays at or below the maximum, and the shrink steps add no random
      draws.
- [ ] Short labels get no side cuts and capped vertical cuts.
- [ ] Rotation plus cut stays aligned (mask and image dimensions match).
- [ ] Cutoff `none` is byte-identical to before.
- [ ] The quality check still rejects ink at sides that were not cut.
- [ ] The validator accepts a correct sidecar and rejects orphan, duplicate, and
      out-of-range rows, plus hash mismatches.
- [ ] Zipping a dataset that has a sidecar works, and a real-data merge keeps it
      valid.
- [ ] Thread and process worker paths write identical sidecars.

## S3.8 Measurement tool

Affected files: `tools/cutoff_stats.py` (new)

- [ ] For each cutoff profile, print:
      - mean attempts per kept sample
      - % of samples with a real cut (ink loss > 0)
      - mean and max ink loss
      - edge-ink rejections on sides that were not cut

## S3.9 Tune the defaults

- [ ] Adjust the S3.1 table with `python -m tools.cutoff_stats` and the review
      montage until the targets below hold and cut labels still read correctly in
      the montage.

Acceptance criteria:

- `heavy`: at least 50% of samples carry a real cut, at most 1.2 attempts per kept
  sample (1.53 before), and max ink loss within the profile limit.
- No dataset fails validation because of intended cuts.

### S3 verification gate

- [ ] Gate commands pass. Only `cutoff=heavy` golden configs changed; re-baseline
      exactly those.
- [ ] Manual check: generate 500 samples with Heavy cutoff, then review the montage
      and `cutoff-annotations.csv`.

---

# S4 - Studio Recipe v2, presets, CLI parity, provenance

## S4.1 Recipe format

Affected files: `src/studio_recipe.py` (new), `resources/presets/README.md` (new)

- [ ] `StudioRecipe` has `name`, `description`, and optional sections
      `augmentation`, `stroke_damage`, `cutoff`, and `run`. Any section may carry an
      informational `based_on`.
- [ ] Strict validation against `studio_spec`:
      - values within bounds, and min <= max
      - unknown keys reported with their JSON path
      - `schema_version` above 2 rejected with a clear message
- [ ] Canonical JSON (sorted keys) and `recipe_sha256()`.
- [ ] Document the format in `resources/presets/README.md`.

Example (rotation `range` is the min/max absolute tilt in degrees):

```json
{
  "format": "synthetic-generator.studio-recipe",
  "schema_version": 2,
  "name": "Historical faded scan",
  "description": "Aged ledger look with light top cutoffs",
  "augmentation": {
    "based_on": "historical_scan_v1",
    "fade":          {"probability": 0.60, "range": [0.50, 0.85]},
    "brightness":    {"probability": 0.65, "range": [0.65, 1.15]},
    "paper_tint":    {"probability": 0.75, "range": [0.08, 0.20]},
    "stains":        {"probability": 0.35, "count_range": [2, 6], "opacity_range": [15, 45]},
    "rotation":      {"probability": 0.85, "range": [0.0, 4.5]},
    "blur":          {"probability": 0.65, "range": [0.4, 1.8]},
    "noise":         {"probability": 0.65, "range": [4.0, 22.0]},
    "paper_texture": {"probability": 0.50, "range": [1.5, 6.0]},
    "scanlines":     {"probability": 0.20, "range": [4, 16]},
    "jpeg":          {"probability": 0.30, "range": [58, 88]}
  },
  "stroke_damage": {
    "contrast":  {"probability": 0.70, "range": [0.35, 0.75]},
    "erode":     {"probability": 0.55, "range": [0.15, 0.45]},
    "gaps":      {"probability": 0.90, "count_range": [8, 24], "size_range": [3, 16]},
    "scratches": {"probability": 0.70, "count_range": [1, 5]}
  },
  "cutoff": {
    "top":    {"probability": 0.5, "range": [0.03, 0.08]},
    "bottom": {"probability": 0.5, "range": [0.03, 0.08]},
    "left":   {"probability": 0.0, "range": [0.0, 0.0]},
    "right":  {"probability": 0.0, "range": [0.0, 0.0]},
    "max_ink_loss": 0.10
  },
  "run": {
    "sample_mode": "semi_broken_mixed",
    "field_weights": {"full_name": 30, "place": 16, "character": 10, "numeric": 10},
    "split_fractions": [0.8, 0.1, 0.1],
    "font_style": "cursive", "cursive_group": "", "specific_font": "",
    "names_version": "name1", "writer_profile": "writer_style_v1",
    "samples_per_writer": 32
  }
}
```

## S4.2 Migrating today's presets

Affected files: `src/studio_recipe.py`, `resources/presets/*.json`

- [ ] Detect old flat files (no `schema_version`) and convert them in memory, using
      the same clamps as `create_custom_augmentation_profile`:

  | Old key | New location | Conversion |
  |---|---|---|
  | `fade` | `augmentation.fade` | chance 1 if v < 0.98, else 0; range [v, v] |
  | `tint`, `blur`, `noise`, `texture` | matching augmentation entry | chance 1 if v > 0; range [v, v] |
  | `stain` | `augmentation.stains` | chance v; default count and opacity ranges |
  | `rotate` | `augmentation.rotation` | chance 1 if v > 0; range [0, v] |
  | `scanline` | `augmentation.scanlines` | round; chance 1 if > 0; range [v, v] |
  | `jpeg` | `augmentation.jpeg` | round; chance 1 if < 100; range [v, v] |
  | (none) | `augmentation.brightness` | chance 0 (old files had no brightness) |
  | `contrast`, `erode`, `scratch` | `stroke_damage.*` | chance v; ranges from config `BROKEN_*` |
  | `gap_prob`, `gap_count` | `stroke_damage.gaps` | chance; count [1, max(1, int(v))]; size from config |
  | `crop_*` (percent) | `cutoff.<side>` | if v > 0: chance 1, range [v/100, v/100], now ink-relative |

- [ ] Saving always writes v2.
- [ ] Re-author the bundled presets as v2 with real ranges and a `description`:
      `clean_minimal`, `heavy_damage`, `historical_faded`, `historical_1914_ledger`.
- [ ] Migrate `profile_1.json` mechanically, rounding values to the spec steps.

## S4.3 Preset storage

Affected files: `src/studio_recipe.py`, `.gitignore`

- [ ] Built-ins in `resources/presets/*.json` are read-only.
- [ ] User presets live in `resources/presets/user/*.json`, which is created on
      demand and gitignored.
- [ ] Listing returns both groups, each entry flagged built-in or user.
- [ ] Filenames are slugs. A name clash with a built-in is refused; a clash with a
      user preset needs confirmation.
- [ ] Duplicate, rename, delete (user presets only), import (validate, then copy
      into `user/`), and export.

## S4.4 Apply and precedence rules

Affected files: `src/studio_recipe.py`

- [ ] `apply_recipe(settings, recipe)` changes only the sections the recipe
      contains.
- [ ] Precedence: defaults < recipe < explicit CLI flags or later GUI edits.
      Document the order.

## S4.5 Generator and provenance

Affected files: `src/generate_synthetic.py`, `src/handoff_validation.py`

- [ ] `generate(recipe=...)` takes the resolved sections. The existing keyword
      arguments stay as overrides.
- [ ] Every dataset gets `studio-recipe.json`: the effective, fully resolved recipe
      with all sections.
- [ ] Additive metadata key `studio_recipe: {file, sha256, name, source_preset,
      modified}`. Its hash is checked the same way as the cutoff sidecar in S3.5.

## S4.6 CLI parity

Affected files: `src/generate_synthetic.py`

- [ ] New flags:
      - `--recipe PATH`
      - `--custom-split-fractions TR VAL TE` and `--custom-field-weights JSON`
        (the README already documents both)
      - `--cutoff {none,light,moderate,heavy}`
      - `--workers N`
      - `--print-recipe`: print the effective recipe and exit without writing
- [ ] Every flag is validated before the output folder is reserved.

## S4.7 Tests

Affected files: `tests/test_studio_recipe.py` (new), `tests/test_cli_recipe_parity.py` (new)

- [ ] Each section round-trips, and validation errors name the JSON path.
- [ ] Every bundled preset loads. Migrated old values reproduce
      `create_custom_augmentation_profile` exactly.
- [ ] Only present sections are applied, precedence holds, and built-ins cannot be
      overwritten.
- [ ] Parity: the same recipe and seed through the GUI snapshot and through the CLI
      give identical image and manifest hashes.
- [ ] Re-running a dataset's own `studio-recipe.json` with the same seed and count
      reproduces it exactly.

## S4.8 Docs

Affected files: `README.md`

- [ ] Correct the CLI section and the output structure (add `studio-recipe.json`
      and `cutoff-annotations.csv`).
- [ ] Document preset sections and import/export, and remove the claims about preset
      export/delete that are not true today.

### S4 verification gate

- [ ] Gate commands pass, with the golden hashes unchanged.
- [ ] Every CLI example in the README runs.

---

# S5 - Studio UI rebuild

## S5.1 Pure studio state

Affected files: `src/studio_recipe.py`

- [ ] Conversion between `StudioState` and `StudioRecipe`.
- [ ] Dirty tracking against the loaded source (a preset or a base profile), named
      "custom (based on X)".

## S5.2 Range widgets

Affected files: `gui_widgets.py` (new)

- [ ] `RangeModel` (pure): clamp, snap, keep min <= max, drag either handle, nudge
      from the keyboard.
- [ ] `RangeSlider`: a two-handle `tk.Canvas` with:
      - keyboard focus per handle
      - arrow keys, with Shift for larger steps
      - double-click to type a value
      - a disabled state
- [ ] A chance slider that shows a percentage.

## S5.3 `StudioPanel`

Affected files: `gui_studio.py` (new)

- [ ] Built from `studio_spec` and grouped as Ink and paper / Geometry / Scanner /
      Stroke damage / Character cutoff.
- [ ] `get_state()` and `set_state()`, a `<<StudioChanged>>` virtual event, and a
      reset per section.
- [ ] A base-profile picker that loads exact values; any edit marks the recipe as
      custom.
- [ ] Stroke damage is enabled only in semi-broken modes, with a hint explaining
      why.
- [ ] Cutoff offers quick picks (None/Light/Moderate/Heavy), max ink loss, and a
      note about short labels.

## S5.4 Studio tab and Generate tab

Affected files: `gui.py`

- [ ] Add a Studio tab. On the Generate tab, a single summary row ("Recipe:
      Historical faded* - based on historical_scan_v1 - [Open Studio]") replaces
      the slider card and the Degradation and Char cutoff dropdowns.
- [ ] Replace the index-based tab checks (`_on_tab_change`, and the scroll
      handler's `nb.index(...) != 0`) with checks on the tab widget itself. Add
      wheel scrolling to the Studio tab.
- [ ] Register every studio widget in `_run_controls` so it locks during generation.

## S5.5 Preset bar

Affected files: `gui_studio.py`

- [ ] A list showing built-ins (marked with a lock) and user presets, with chips for
      the sections each one contains.
- [ ] Actions:
      - Load, which warns about unsaved changes
      - Save (user presets only)
      - Save as, with section checkboxes; augmentation, stroke damage, and cutoff
        are checked by default
      - Duplicate, Rename, Delete (with confirmation), Import, Export
- [ ] A `*` marker whenever the current state differs from the loaded preset.
- [ ] "Copy CLI command" for a saved preset. With unsaved changes, it asks you to
      save first.

## S5.6 Reset and cleanup

Affected files: `gui.py`, `config.py`

- [ ] Three resets: per section, "Reset studio", and a separate "Reset all" on the
      Generate tab.
- [ ] Once nothing uses them, remove `_make_slider`, the midpoint logic in
      `_on_degradation_preset_change`, the defaults dict in
      `_reset_developer_defaults`, and `custom_dev_v1` from
      `config.DEGRADATION_PROFILES`.

## S5.7 Tests

Affected files: `tests/test_range_model.py` (new), `tests/test_studio_state.py` (new)

- [ ] `RangeModel` keeps its invariants.
- [ ] State <-> recipe round trip, dirty tracking, read-only built-ins, and
      rejection of bad imports.
- [ ] Tab detection works by widget identity.
- [ ] Optional: a smoke test with real widgets, skipped when Tk has no display.

### S5 verification gate

- [ ] Gate commands pass, with the golden hashes unchanged.
- [ ] Manual check:
      - reach and operate every control with the keyboard alone
      - load, edit, save, export, and import a preset
      - generate a dataset from the Studio tab

---

# S6 - Faithful preview and feedback

## S6.1 Shared sample pipeline

Affected files: `src/sample_pipeline.py` (new), `src/generate_synthetic.py`

- [ ] Move one attempt (value, render, degrade, test-split holdout, quality check)
      into `attempt_sample(...) -> SampleAttempt`. `_generate_valid_sample` loops
      over it.
- [ ] RNG use stays exactly the same (confirmed by the golden check).

## S6.2 Preview of the actual run

Affected files: `src/preview.py` (new)

- [ ] `preview_samples(settings, recipe, *, count, seed, n, offset=0, split=None)`
      reproduces the first N items of the run plan: the same field and split
      choices, writer IDs, item seeds, font pools, and retries. Each tile is
      identical to the image generation would write.
- [ ] The optional split filter walks the plan to show the test-split look.
- [ ] When the Seed field is blank, the preview uses a temporary seed and labels the
      grid "example; the run will pick its own seed".

## S6.3 Preview grid

Affected files: `gui_studio.py`

- [ ] 6, 9, or 12 tiles, shown at true scale (only ever scaled down).
- [ ] Badges: field, font, cut sides, "kept on attempt k", and the file the tile will
      become (for example `train/syn_000003.png`).
- [ ] Controls: split filter, "Next N" (offset), and a seed shared with the Generate
      tab.

## S6.4 Background rendering

Affected files: `gui_studio.py`

- [ ] Wait 150 ms after the last change, then render on a worker thread from a
      plain-data snapshot.
- [ ] Results come back through the queue with a generation token; stale results
      are dropped.
- [ ] Failures show an error tile instead of being silently swallowed.

## S6.5 Retry estimate

Affected files: `src/preview.py`, `gui.py`

- [ ] Run 48 single attempts in the background and show "about x% would be retried
      (edge ink y%, blank z%, low contrast w%)" plus mean attempts per sample.
- [ ] Multiply the duration estimate by the mean attempts.

## S6.6 Compare with real scans

Affected files: `gui_studio.py`

- [ ] When `dataset/real/images` has images, compare 64 real images against 64
      preview samples using `compare_image_statistics`.
- [ ] Show standardized differences with arrows and plain hints (for example
      "synthetic is darker: -1.2 sd").
- [ ] Without real images, the panel is disabled and explains why.

## S6.7 Tests

Affected files: `tests/test_preview_pipeline.py` (new)

- [ ] Preview tiles match the generated images pixel for pixel, for the same seed,
      count, and settings, both for the first N and at an offset.
- [ ] The split filter is deterministic.
- [ ] Results with a stale token are discarded (fake `after` and queue).
- [ ] Retry statistics are deterministic.
- [ ] Comparison logic works against a temporary real-image folder.

### S6 verification gate

- [ ] Gate commands pass, with the golden hashes unchanged.
- [ ] Manual check: dragging controls stays smooth, and after generating with the
      same seed and count, the files named on the tiles match the tiles.

---

# S7 - Polish, documentation, final gate

Affected files: `gui_studio.py`, `README.md`, `pyproject.toml`

- [ ] Tooltips from `studio_spec` on every control.
- [ ] Per-effect bypass toggles for A/B comparison (preview only, never saved).
- [ ] Before/after view of a selected tile (clean render vs final image).
- [ ] A "Developer Studio" guide in the README: workflow, ranges, cutoff, presets,
      the reproduce command, and calibration.
- [ ] Add `studio_spec`, `studio_recipe`, `cutoff`, `preview`, and `sample_pipeline`
      to `[tool.mypy] files`, and make them pass.
- [ ] Final manual run:
      - fresh clone, install, launch the GUI
      - load every bundled preset
      - generate in each mode with Heavy cutoff, and merge real data
      - zip, then verify the SHA-256
      - check the montage and the metadata

### S7 verification gate

- [ ] Gate commands pass, with no unexpected golden changes.
- [ ] The Kaggle notebook contract tests pass without changes.

---

# Definition of done

- [ ] GUI Regular mode never applies stroke damage, and the GUI field mix follows
      the sample mode unless the user edits it.
- [ ] Every studio effect is chance plus min/max, and built-in profiles round-trip
      without loss.
- [ ] Character Cutoff:
      - clips real letters relative to the ink, with bounded ink loss
      - records every cut sample in `cutoff-annotations.csv`, hash-bound in metadata
      - passes dataset validation
- [ ] Presets:
      - use v2 with optional sections, and old files migrate
      - built-ins are read-only
      - user presets support duplicate, rename, delete, import, and export
- [ ] Every dataset contains `studio-recipe.json`. Re-running it with the same seed
      and count, through the GUI or the CLI, reproduces the dataset byte for byte.
- [ ] The preview shows the real first N samples of the run, renders in the
      background, and reports the retry rate.
- [ ] The README matches the code. Tests, ruff, mypy, and the golden check pass, and
      the Kaggle contract is unchanged.

# Out of scope (later)

- Writer-style controls (slant, baseline, spacing, pressure) in the studio.
- Automatic recipe tuning against real-scan statistics.
- Separate recipes per field type.
- Cut versus uncut metrics in the Kaggle notebooks.
