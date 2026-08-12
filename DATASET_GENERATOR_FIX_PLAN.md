# Dataset Generator Fix Plan

This is the prioritized implementation backlog for the synthetic dataset
generator and its Kaggle training handoff.

## How to use this plan

- Complete priorities in order: **P0 -> P1 -> P2 -> P3**.
- Do not mark a task complete until its acceptance criteria and tests pass.
- Prefer small commits aligned with the work packages below.
- Preserve a real, writer-held-out test set throughout the work.
- Do not use custom dataset paths or `--zip-only` until P0 is complete.

## Completion gates

| Priority | Gate |
|---|---|
| P0 | No user input can escape the dataset root, overwrite an existing run, or delete an unintended path. Interrupted runs cannot appear complete. |
| P1 | Synthetic and real samples share one validated manifest; runs are reproducible; evaluation splits measure meaningful generalization. |
| P2 | Generation and training are robust at scale, image geometry is preserved, and the GUI/Kaggle workflows recover cleanly from failures. |
| P3 | Installation, documentation, dependencies, licensing, tests, and CI are complete and portable. |

---

# P0 - Critical safety and data-loss blockers

Complete every P0 work package before generating irreplaceable data.

## P0.1 Constrain every dataset path

Affected files: `config.py`, `src/generate_synthetic.py`, `gui.py`

- [x] Accept only a single safe dataset folder name or a positive numeric ID.
- [x] Reject absolute paths, drive-qualified paths, `.`, `..`, separators,
      empty normalized names, and Windows reserved device names.
- [x] Resolve the candidate path and require its parent to be exactly
      `config.DATASETS_DIR.resolve()`.
- [x] Recheck containment immediately before creating, archiving, copying, or
      deleting anything.
- [x] Centralize containment validation so GUI and CLI use the same code.
- [x] Produce a clear error without creating any directory when validation
      fails.

Acceptance criteria:

- Inputs such as `.`, `..`, `../x`, `..\\..`, `C:\\Temp\\x`, `/tmp/x`,
  `foo/bar`, `foo\\bar`, `CON`, and `NUL` are rejected.
- Numeric IDs and documented safe names such as `2` and `my_test_run` resolve
  beneath `dataset/datasets/`.
- No invalid-path test changes the filesystem.

## P0.2 Make recursive deletion independently safe

Affected files: `src/generate_synthetic.py`, `gui.py`

- [x] Introduce one guarded deletion helper.
- [x] Require the deletion target to be a direct child of `DATASETS_DIR`.
- [x] Explicitly reject the repository root, dataset root, datasets container,
      home directory, filesystem root, and unresolved paths.
- [x] Keep `--zip-only` disabled or fail closed until these checks exist.
- [x] Confirm the ZIP is complete and readable before deleting its source.
- [x] Never follow symlinks or junctions into a broader tree during deletion.

Acceptance criteria:

- Tests prove `--dataset ..\\.. --zip-only` and equivalent inputs cannot delete
  the repository or dataset container.
- A valid direct-child dataset can be archived and removed safely.

## P0.3 Refuse existing output and archive collisions

Affected files: `config.py`, `src/generate_synthetic.py`

- [x] Refuse a nonempty destination by default.
- [x] Refuse an existing sibling ZIP by default.
- [x] Remove implicit overwrite behavior caused by `exist_ok=True` and fixed
      `syn_*.png` names.
- [x] If overwrite is ever added, require an explicit option and use the same
      guarded deletion path.
- [x] Ensure a smaller rerun cannot leave stale images from a larger run.
- [x] Ensure a sample cannot remain in an old split after being reassigned.

Acceptance criteria:

- Reusing a dataset name fails before writing anything.
- Existing images, manifests, and archives remain byte-for-byte unchanged after
  a rejected run.

## P0.4 Reserve automatic dataset numbers atomically

Affected files: `config.py`, `src/generate_synthetic.py`

- [x] Include both dataset directories and dataset ZIPs when finding the next
      number.
- [x] Reserve the chosen number atomically rather than using a separate
      scan-then-create sequence.
- [x] Retry allocation if another process wins the race.
- [x] Ensure custom named datasets cannot interfere with numbered allocation.

Acceptance criteria:

- Two simultaneous generators receive different output directories.
- A zip-only `dataset_013.zip` causes the next automatic run to select
  `dataset_014` or higher.

## P0.5 Make generation transactional

Affected files: `src/generate_synthetic.py`, `gui.py`

- [x] Generate into a uniquely named temporary sibling directory.
- [x] Validate all resources and fonts before beginning image writes.
- [x] Write manifests safely and flush/close them before publication.
- [x] Validate expected row counts, image counts, split counts, and image
      readability before publishing.
- [x] Atomically rename the staging directory to the final dataset name.
- [x] On error or cancellation, remove or clearly mark the staging directory.
- [x] Prevent partial output from appearing in the GUI dataset list as complete.

Acceptance criteria:

- Injected font, callback, image-save, disk-full, and manifest-write failures do
  not create a final dataset directory.
- A successful run appears at its final path only after all validation passes.

## P0.6 Secure real-data filenames and copies

Affected files: `src/build_splits.py`

- [x] Reject absolute filenames and filenames containing path traversal or
      directory separators.
- [x] Require source images to resolve directly beneath `REAL_DIR/images`.
- [x] Require destination images to resolve directly beneath the chosen split.
- [x] Reject duplicate filenames and conflicting labels.
- [x] Detect collisions with existing synthetic or real images.
- [x] Validate every real source row and source image before copying any file.

Acceptance criteria:

- Malicious CSV filenames cannot read or overwrite files outside their intended
  roots.
- Any invalid row aborts the merge without partially modifying the dataset.

## P0.7 Protect active GUI work

Affected file: `gui.py`

- [x] Add a controlled close/cancel handler for an active generation job.
- [x] Disable deletion of the active output.
- [x] Disable run-affecting controls while a job is active.
- [x] Ensure cancellation waits for a safe boundary and cleans staging output.
- [x] Prevent a background worker from being silently killed as a daemon while
      writing data.

Acceptance criteria:

- Closing or cancelling during generation never leaves a published partial
  dataset.
- The active dataset cannot be deleted from the Datasets tab.

### P0 verification gate

- [x] Path-containment and guarded-deletion tests pass.
- [x] Existing-output and existing-ZIP tests pass.
- [x] Concurrent allocation test passes.
- [x] Failure-injection and cancellation tests pass.
- [x] Workflows using `(next)` and a valid custom name complete successfully.

---

# P1 - High-priority integrity, reproducibility, and evaluation fixes

## P1.1 Define one unified manifest contract

Affected files: `src/generate_synthetic.py`, `src/build_splits.py`,
`kaggle-code/trocr-finetuning-code.ipynb`

Required core columns:

```text
filename,label,split,source,field_type,font,sample_mode,writer_id
```

- [x] Define which columns are required and which may be empty.
- [x] Write synthetic rows with `source=synthetic` and an empty `writer_id`.
- [x] Write real rows with `source=real`, an explicit `writer_id`, and suitable
      values for non-applicable synthetic fields.
- [x] Make the Kaggle notebook consume this unified manifest.
- [x] Stop relying on separate files whose contents disagree.
- [x] Add a manifest schema version.

Acceptance criteria:

- Every image used by Kaggle appears in the unified manifest.
- Selecting “Merge real data” measurably increases the relevant Kaggle split
  counts.

## P1.2 Make real merging idempotent and writer-safe

Affected file: `src/build_splits.py`

- [x] Require an explicit `writer_id` column instead of parsing filenames.
- [x] Preserve a stable writer-to-split assignment.
- [x] Reconcile or remove the prior real merge before rebuilding it.
- [x] Never append duplicate manifest or label rows.
- [x] Reject missing images, malformed writers, duplicate rows, and conflicting
      labels instead of silently skipping them.
- [x] Return structured merge results: copied, unchanged, skipped, and failed.
- [x] Keep the real test writers permanently held out.

Acceptance criteria:

- Running the same merge twice produces byte-identical manifests and no extra
  files.
- Adding writers cannot leave an existing writer in more than one split.

## P1.3 Add complete dataset-integrity validation

Affected files: new validation module, generator, real merge, Kaggle notebook

- [x] Validate required manifest columns and schema version.
- [x] Validate allowed split and source values.
- [x] Require nonempty train, validation, and test splits when applicable.
- [x] Require exactly one existing readable image per manifest row.
- [x] Reject orphan images, duplicate paths, duplicate rows, and conflicts.
- [x] Validate image dimensions, modes, blankness, ink coverage, and extreme
      aspect ratios.
- [x] Produce a machine-readable validation report.
- [x] Run validation before ZIP creation and before Kaggle training.

Acceptance criteria:

- A deliberately corrupted dataset fails validation with actionable messages.
- Every published dataset contains a passing validation report.

## P1.4 Validate configuration before filesystem mutation

Affected files: `config.py`, `src/generate_synthetic.py`, `src/build_splits.py`

- [x] Validate positive sample counts and set a documented upper bound or
      confirmation threshold.
- [x] Validate split fractions are finite, between zero and one, and sum to one.
- [x] Use `SYNTH_TEST_FRAC` and `REAL_TEST_FRAC`, or remove them from config.
- [x] Handle rounding and tiny sample/writer counts explicitly.
- [x] Validate field weights are recognized, nonnegative, and not all zero.
- [x] Validate every probability and numeric range.
- [x] Validate name pool, resources, font style/group, and specific font.
- [x] Fail on unknown selections instead of silently falling back.

Acceptance criteria:

- Invalid configuration fails before any output directory is created.
- Small-count behavior is documented and tested.

## P1.5 Make runs independently reproducible

Affected files: `src/generate_synthetic.py`, `src/augment.py`, `src/fields.py`,
`src/render.py`

- [x] Seed Python and NumPy from the same recorded run seed.
- [x] Use explicit run-local RNG objects rather than global state.
- [x] Stop mutating global name-pool configuration.
- [x] Define whether automatic runs use a random seed or a user-visible fixed
      seed.
- [x] Add a GUI seed option and show the effective seed after generation.
- [x] Test that identical inputs and seeds produce identical outputs.
- [x] Test that different automatic seeds do not duplicate whole manifests.

Acceptance criteria:

- Two isolated runs with the same seed have identical manifests and image
  hashes.
- Different seeds produce meaningfully different samples.

## P1.6 Persist complete run provenance

Affected file: `src/generate_synthetic.py`

- [x] Write a run-level metadata JSON containing the seed, generator revision,
      schema version, configuration snapshot, name pool, font selection,
      augmentation settings, resource hashes, dependency versions, timestamps,
      and image/row counts.
- [x] Record the effective selection rather than only requested values.
- [x] Record whether and when real data were merged.
- [x] Include a deterministic manifest hash and optional image-set hash.

Acceptance criteria:

- A dataset can be traced to its exact inputs and effective configuration.

## P1.7 Preserve image aspect ratio end to end

Affected files: generator validation and
`kaggle-code/trocr-finetuning-code.ipynb`

- [x] Define a canonical aspect-preserving resize-and-pad transform for TrOCR.
- [x] Apply the same transform during training, validation, testing, and
      inference.
- [x] Decide whether to cap or flag extreme generated widths.
- [x] Save padding/background parameters in model provenance.
- [x] Add visual tests for short, normal, and very long labels.

Acceptance criteria:

- No crop is non-uniformly stretched to 384x384.
- Long-date samples remain visually legible after preprocessing.

## P1.8 Add post-generation label/image quality checks

Affected files: `src/augment.py`, generator validation

- [x] Detect blank and nearly blank images.
- [x] Detect clipped ink and insufficient padding.
- [x] Measure minimum contrast and ink coverage.
- [x] Reject or regenerate semi-broken samples that no longer support the clean
      label.
- [x] Create a review montage stratified by field, font, and damage profile.

Acceptance criteria:

- Every run produces a quality report and review montage.
- Invalid samples are regenerated or excluded before publication.

## P1.9 Build meaningful evaluation splits

Affected files: generator split logic, manifest schema, Kaggle notebook

- [x] Label synthetic metrics explicitly as synthetic in-distribution metrics.
- [x] Hold out complete fonts for at least one evaluation split.
- [x] Hold out selected names, places, vocabulary, or formatting patterns.
- [x] Hold out selected degradation profiles or strength ranges.
- [x] Maintain a writer-held-out real test set.
- [x] Report metrics by source, field type, seen/unseen label, and held-out
      condition.
- [x] Prevent real test writers from entering training after later merges.

Acceptance criteria:

- Reports clearly separate synthetic in-distribution performance from real and
  held-out-domain performance.
- No model-selection decision uses the final real test set.

## P1.10 Correct field-generation coverage

Affected file: `src/fields.py`

- [x] Generate calendar-valid dates including the 29th, 30th, 31st, and leap
      days.
- [x] Correct wording for the year 2000 and test year boundaries.
- [x] Add newborn and centenarian formats where supported by the real domain.
- [x] Review uniform age/year distributions against actual registry data.
- [x] Add representative abbreviations, punctuation, initials, casing, and
      historical formats.
- [x] Review name suffix and middle-name probabilities.
- [x] Remove duplicated resource entries.

Acceptance criteria:

- Unit tests cover all calendar boundaries and supported wording formats.
- Field distributions are documented and justified by domain evidence.

### P1 verification gate

- [x] Unified-manifest integration test passes for synthetic plus real data.
- [x] Repeated real merge is idempotent.
- [x] Same-seed output hashes match exactly.
- [x] Dataset-integrity validator passes on a clean generated dataset.
- [x] Aspect-preserving preprocessing visual tests pass.
- [x] Evaluation report separates synthetic and real/held-out results.

---

# P2 - Medium-priority scalability and workflow hardening

## P2.1 Improve synthetic writer realism

- [x] Add writer-consistent variations in baseline, spacing, slant, stroke
      width, pressure, glyph form, and character connections.
- [x] Model relevant form lines, paper texture, neighboring marks, compression,
      and scan artifacts.
- [x] Calibrate augmentation probabilities and severity against real scans.
- [x] Version augmentation profiles.
- [x] Compare generated and real image statistics.

## P2.2 Improve generator performance and resilience

Affected files: `src/generate_synthetic.py`, `src/render.py`

- [x] Stream manifest output or checkpoint it safely instead of retaining every
      row in memory.
- [x] Avoid materializing all field and split assignments at once.
- [x] Cache fonts by path and size where beneficial.
- [x] Estimate disk, memory, and archive size before starting.
- [x] Warn or require confirmation for very large jobs.
- [x] Check available disk space.
- [x] Support safe cancellation at sample and packaging boundaries.
- [x] Verify archives and write checksums.

## P2.3 Harden GUI threading and reporting

Affected file: `gui.py`

- [x] Snapshot every run option on the UI thread.
- [x] Never access Tk variables or call Tk methods from worker threads.
- [x] Route all background results through the UI queue.
- [x] Report real merge copied/skipped/error counts.
- [x] Treat a no-op merge as an explicit warning.
- [x] Save full tracebacks to a persistent log.
- [x] Limit dataset-size scans to one active refresh generation.
- [x] Handle files disappearing during refresh.
- [x] Move large deletion work off the UI thread and display progress.
- [x] Use `config.DEFAULT_COUNT` rather than a duplicated literal.
- [x] Make the window responsive to scaling and smaller displays.
- [x] Handle folder-opening failures and non-Windows platforms.

## P2.4 Make Kaggle training reproducible and resumable

Affected file: `kaggle-code/trocr-finetuning-code.ipynb`

- [x] Require explicit `DATA_ROOT` when multiple manifests are present.
- [x] Run full dataset validation before loading the model.
- [x] Seed Python, NumPy, Torch, CUDA, and DataLoader workers.
- [x] Save model, processor, optimizer, scheduler, scaler, epoch, best metric,
      configuration, and RNG states.
- [x] Resume only when checkpoint provenance matches the current dataset and
      configuration.
- [x] Use a unique or clean output directory for each run.
- [x] Process the last partial gradient-accumulation group.
- [x] Use ceiling division for optimizer/scheduler step counts.
- [x] Remove `drop_last=True` unless deliberately required.
- [x] Assert CUDA availability when CUDA AMP is required.
- [x] Reconsider `DataParallel`; document or replace it with a more robust
      distributed approach.
- [x] Select a safe evaluation batch size dynamically.
- [x] Assert equal prediction/reference counts and nonempty splits.
- [x] Audit token lengths before training and reject silent truncation.
- [x] Use consistent generation length semantics.
- [x] Load both the saved processor and saved model for final evaluation.
- [x] Use validation—not test—for baseline/model-development comparisons.
- [x] Pin or record Torch, Transformers, Pillow, CUDA, and model revisions.
- [x] Stop globally suppressing all warnings.
- [x] Resolve the recorded DataLoader worker teardown warnings.

## P2.5 Strengthen model evaluation and artifact reporting

Affected file: `kaggle-code/trocr-finetuning-code.ipynb`

- [x] Add per-field, per-source, seen/unseen, and held-out-domain metrics.
- [x] Touch the locked test set once after model choices are frozen.
- [x] Require `evaluation-report.json` before packaging.
- [x] Ensure the report cell runs before the ZIP cell.
- [x] Hash image content or the versioned dataset artifact, not only the
      manifest.
- [x] Include profile, hyperparameters, seeds, dependency versions, processor
      configuration, dataset version, and split definition in the report.
- [x] Assert that packaged weights match the evaluated weights.

### P2 verification gate

- [x] A Kaggle run can resume after a simulated interruption.
- [x] No training examples are discarded by batching or accumulation.
- [x] Repeated seeded runs yield reproducible metrics within the documented
      determinism guarantee.
- [x] The packaged ZIP contains the verified model, processor, configuration,
      and evaluation report.

---

# P3 - Lower-priority portability, documentation, and maintenance

## P3.1 Resolve fonts and licensing

- [x] Either distribute the advertised fonts legally or stop claiming they ship
      with the repository.
- [x] Document font sources, licenses, installation, filenames, and checksums.
- [x] Make clean non-Windows installations fail with actionable instructions or
      provide a portable licensed fallback.
- [x] Add required font/resource attribution and license files.

## P3.2 Repair packaging and dependencies

- [x] Declare the minimum supported Python version.
- [x] Add Faker to development/tool dependencies.
- [x] Remove OpenCV until scan processing exists, or restore the missing scan
      processor.
- [x] Add a pinned constraints or lock file.
- [x] Add dependency and security auditing.
- [x] Replace the hardcoded personal Conda launcher with portable environment
      discovery and a documented fallback.

## P3.3 Correct documentation and privacy guidance

- [x] Restore or remove references to `src/make_sheets.py` and
      `src/process_scans.py`.
- [x] Correct outdated GUI theme and workflow descriptions.
- [x] Document the unified manifest schema and dataset validation process.
- [x] Document the difference between synthetic, held-out synthetic, and real
      test metrics.
- [x] Document consent, access, retention, sharing, and deletion requirements
      for real handwriting data.
- [x] Add a troubleshooting section for fonts, partial runs, Kaggle paths, and
      checkpoint recovery.

## P3.4 Add automated quality controls

- [x] Add unit tests for path handling, field values, split calculation,
      rendering, augmentation, RNG behavior, and manifest validation.
- [x] Add integration tests for generation, reruns, real merges, cancellation,
      concurrent allocation, ZIP creation, and zip-only behavior.
- [x] Add malicious-path and guarded-delete tests.
- [x] Add same-seed reproducibility tests using image hashes.
- [x] Add dataset-integrity validation to CI.
- [x] Add formatting, linting, and type checking.
- [x] Add dependency vulnerability checks.

### P3 verification gate

- [x] A clean checkout can be installed and run using only documented steps.
- [x] All automated checks pass in CI.
- [x] Documentation matches the actual generator and Kaggle workflows.

---

# Suggested implementation sequence

1. `P0.1-P0.4`: safe paths, guarded deletion, collision prevention, allocation.
2. `P0.5-P0.7`: transactional output and safe GUI/real-file operations.
3. `P1.1-P1.4`: unified manifest, idempotent merge, validation, config checks.
4. `P1.5-P1.6`: deterministic RNG and complete provenance.
5. `P1.7-P1.10`: image contract, quality checks, evaluation splits, field coverage.
6. `P2.1-P2.3`: realism, performance, and GUI hardening.
7. `P2.4-P2.5`: reproducible/resumable Kaggle training and artifact reports.
8. `P3`: portability, documentation, dependencies, tests, and CI.

# Definition of done for the project

- [x] No input can write, copy, archive, or delete outside the intended dataset
      directory.
- [x] Published runs are atomic, immutable by default, and internally valid.
- [x] Synthetic and real data use one versioned manifest.
- [x] Real writer isolation is explicit and tested.
- [x] Same-seed runs are reproducible across Python and NumPy operations.
- [x] TrOCR preprocessing preserves crop aspect ratios.
- [x] Reports distinguish synthetic in-distribution results from held-out and
      real-world results.
- [x] Kaggle training is resumable and its packaged report is bound to the exact
      model and dataset artifact.
- [x] A clean checkout works using documented, licensed dependencies.
- [x] Unit, integration, integrity, and safety tests pass in CI.
