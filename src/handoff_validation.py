"""Validate the exact dataset artifacts consumed by the Kaggle notebook."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path

from src.cutoff import CUTOFF_ANNOTATIONS_FILENAME
from src.fields import BASE_FORMAT_PROFILE
from src.manifest import MANIFEST_COLUMNS, MANIFEST_SCHEMA_VERSION
from src.split_policy import EVALUATION_ANNOTATION_COLUMNS


# Format IDs src.fields emits for base-profile, non-date fields.  Mirrors
# LEGACY_NON_DATE_FORMAT_ID_ALLOWLIST in kaggle-code/trocr-finetuning-code2.ipynb.
BASE_NON_DATE_FORMAT_IDS: dict[str, frozenset[str]] = {
    "full_name": frozenset({"name_mixed_representative_v1"}),
    **{
        field_type: frozenset({"default"})
        for field_type in ("age", "cause_of_death", "character", "citizenship",
                           "civil_status", "numeric", "occupation", "place",
                           "religion", "sex")
    },
}


def _json_object(content: bytes, name: str) -> dict:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _csv_rows(content: bytes, name: str, columns: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        text = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        actual = tuple(reader.fieldnames or ())
        if actual != columns:
            raise ValueError(
                f"{name} columns do not match the required contract: {actual!r}"
            )
        rows = list(reader)
    except UnicodeError as exc:
        raise ValueError(f"Unreadable {name}: {exc}") from exc
    if any(None in row for row in rows):
        raise ValueError(f"{name} contains extra unnamed values")
    return rows


def validate_handoff_bytes(
    *,
    manifest_content: bytes,
    metadata_content: bytes,
    annotations_content: bytes,
    validation_content: bytes,
    manifest_sha256: str,
    images_sha256: str,
    cutoff_annotations_content: bytes | None = None,
) -> None:
    """Require provenance and synthetic annotations to match current bytes.

    ``cutoff_annotations_content`` is the optional cutoff sidecar.  Whenever
    run metadata names it, it must be present and match the recorded hash.
    """
    if hashlib.sha256(manifest_content).hexdigest() != manifest_sha256:
        raise ValueError("Internal manifest hash calculation mismatch")
    manifest_rows = _csv_rows(
        manifest_content, "manifest.csv", tuple(MANIFEST_COLUMNS)
    )
    annotations = _csv_rows(
        annotations_content,
        "evaluation-annotations.csv",
        tuple(EVALUATION_ANNOTATION_COLUMNS),
    )
    metadata = _json_object(metadata_content, "run-metadata.json")
    report = _json_object(validation_content, "dataset-validation.json")

    if report.get("valid") is not True or report.get("errors"):
        raise ValueError("dataset-validation.json does not report a valid dataset")
    if report.get("manifest_sha256") != manifest_sha256:
        raise ValueError("dataset-validation.json is stale: manifest hash mismatch")
    if report.get("images_sha256") != images_sha256:
        raise ValueError("dataset-validation.json is stale: image-set hash mismatch")

    if metadata.get("metadata_schema_version") != 1:
        raise ValueError("run-metadata.json has an unsupported metadata schema")
    if str(metadata.get("manifest_schema_version")) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("run-metadata.json has an unsupported manifest schema")
    if metadata.get("manifest_sha256") != manifest_sha256:
        raise ValueError("run-metadata.json is stale: manifest hash mismatch")
    if metadata.get("images_sha256") != images_sha256:
        raise ValueError("run-metadata.json is stale: image-set hash mismatch")
    if metadata.get("row_count") != len(manifest_rows):
        raise ValueError("run-metadata.json row count does not match manifest.csv")
    if metadata.get("image_count") != len(manifest_rows):
        raise ValueError("run-metadata.json image count does not match manifest.csv")
    if metadata.get("validation_report") != "dataset-validation.json":
        raise ValueError("run-metadata.json does not identify dataset-validation.json")
    if metadata.get("evaluation_annotations") != "evaluation-annotations.csv":
        raise ValueError("run-metadata.json does not identify evaluation-annotations.csv")
    annotation_digest = hashlib.sha256(annotations_content).hexdigest()
    if metadata.get("evaluation_annotations_sha256") != annotation_digest:
        raise ValueError(
            "run-metadata.json is stale: evaluation annotation hash mismatch"
        )

    named_cutoff = metadata.get("cutoff_annotations")
    if named_cutoff is not None:
        if named_cutoff != CUTOFF_ANNOTATIONS_FILENAME:
            raise ValueError("run-metadata.json names an unexpected cutoff sidecar")
        if cutoff_annotations_content is None:
            raise ValueError(
                f"Missing {CUTOFF_ANNOTATIONS_FILENAME} named by run-metadata.json")
        if (metadata.get("cutoff_annotations_sha256")
                != hashlib.sha256(cutoff_annotations_content).hexdigest()):
            raise ValueError("run-metadata.json is stale: cutoff annotation hash mismatch")
    elif cutoff_annotations_content is not None:
        raise ValueError(
            f"run-metadata.json does not identify {CUTOFF_ANNOTATIONS_FILENAME}")

    policy = metadata.get("evaluation_policy")
    if not isinstance(policy, Mapping) or policy.get("policy_version") != "1":
        raise ValueError("run-metadata.json lacks a supported evaluation policy")
    conditions = policy.get("evaluation_conditions")
    test_by_field = policy.get("test_evaluation_conditions_by_field", {})
    format_policy = policy.get("format_holdout")
    if not isinstance(conditions, Mapping) or not isinstance(test_by_field, Mapping):
        raise ValueError("run-metadata.json evaluation conditions are malformed")
    if not isinstance(format_policy, Mapping):
        raise ValueError("run-metadata.json format holdout policy is malformed")
    format_fields = set(format_policy.get("fields", []))
    format_profiles = format_policy.get("profiles", {})
    pattern_ids = format_policy.get("pattern_ids", {})
    if not isinstance(format_profiles, Mapping) or not isinstance(pattern_ids, Mapping):
        raise ValueError("run-metadata.json format holdout details are malformed")

    manifest_lookup: dict[tuple[str, str], dict[str, str]] = {}
    folded_manifest_keys: set[tuple[str, str]] = set()
    for row in manifest_rows:
        key = (row["split"], row["filename"])
        folded = (row["split"], row["filename"].casefold())
        if folded in folded_manifest_keys:
            raise ValueError(f"manifest.csv contains a duplicate key: {key!r}")
        folded_manifest_keys.add(folded)
        manifest_lookup[key] = row
    synthetic_rows = [row for row in manifest_rows if row["source"] == "synthetic"]
    train_labels = {
        row["label"] for row in synthetic_rows if row["split"] == "train"
    }
    train_fonts = {
        row["font"].casefold()
        for row in synthetic_rows
        if row["split"] == "train" and row["font"]
    }

    annotation_keys: set[tuple[str, str]] = set()
    folded_annotation_keys: set[tuple[str, str]] = set()
    for row in annotations:
        key = (row["split"], row["filename"])
        folded = (row["split"], row["filename"].casefold())
        if folded in folded_annotation_keys:
            raise ValueError(
                f"evaluation-annotations.csv contains a duplicate key: {key!r}"
            )
        folded_annotation_keys.add(folded)
        annotation_keys.add(key)
        manifest_row = manifest_lookup.get(key)
        if manifest_row is None or manifest_row["source"] != "synthetic":
            raise ValueError(
                f"Evaluation annotation has no matching synthetic row: {key!r}"
            )
        for column in (
            "label_seen_in_train", "font_seen_in_train", "format_seen_in_train"
        ):
            if row[column].casefold() not in {"true", "false"}:
                raise ValueError(
                    f"Evaluation annotation {key!r} has invalid {column}"
                )
        split = row["split"]
        field_type = manifest_row["field_type"]
        expected_condition = (
            test_by_field.get(field_type) if split == "test" else None
        ) or conditions.get(split)
        if not expected_condition or row["evaluation_condition"] != expected_condition:
            raise ValueError(
                f"Evaluation annotation {key!r} violates the condition policy"
            )
        split_profiles = format_profiles.get(split, {})
        if not isinstance(split_profiles, Mapping):
            raise ValueError("run-metadata.json format profiles are malformed")
        expected_profile = (
            split_profiles.get(field_type, BASE_FORMAT_PROFILE)
            if field_type in format_fields
            else BASE_FORMAT_PROFILE
        )
        if row["format_profile"] != expected_profile:
            raise ValueError(
                f"Evaluation annotation {key!r} violates the format policy"
            )
        field_patterns = pattern_ids.get(field_type, {})
        if not isinstance(field_patterns, Mapping):
            raise ValueError("run-metadata.json pattern IDs are malformed")
        allowed_ids = set(field_patterns.get(expected_profile, []))
        if not allowed_ids and expected_profile == BASE_FORMAT_PROFILE:
            allowed_ids = set(BASE_NON_DATE_FORMAT_IDS.get(field_type, ()))
        if allowed_ids and row["format_id"] not in allowed_ids:
            raise ValueError(
                f"Evaluation annotation {key!r} has an unsupported format ID"
            )
        if not allowed_ids and row["format_id"]:
            raise ValueError(
                f"Evaluation annotation {key!r} has an unexpected format ID"
            )
        expected_flags = {
            "label_seen_in_train": manifest_row["label"] in train_labels,
            "font_seen_in_train": bool(
                manifest_row["font"]
                and manifest_row["font"].casefold() in train_fonts
            ),
            "format_seen_in_train": expected_profile == BASE_FORMAT_PROFILE,
        }
        for column, expected in expected_flags.items():
            actual = row[column].casefold() == "true"
            if actual != expected:
                raise ValueError(
                    f"Evaluation annotation {key!r} has stale {column}"
                )

    expected_keys = {(row["split"], row["filename"]) for row in synthetic_rows}
    missing = expected_keys - annotation_keys
    orphaned = annotation_keys - expected_keys
    if missing:
        raise ValueError(f"Missing synthetic evaluation annotation: {sorted(missing)[0]!r}")
    if orphaned:
        raise ValueError(f"Orphan synthetic evaluation annotation: {sorted(orphaned)[0]!r}")


def validate_handoff_files(
    dataset_dir: Path, *, manifest_sha256: str, images_sha256: str
) -> None:
    """Read and validate every required Kaggle handoff artifact."""
    dataset_dir = Path(dataset_dir)
    required = {
        "manifest_content": dataset_dir / "manifest.csv",
        "metadata_content": dataset_dir / "run-metadata.json",
        "annotations_content": dataset_dir / "evaluation-annotations.csv",
        "validation_content": dataset_dir / "dataset-validation.json",
    }
    content: dict[str, bytes] = {}
    for argument, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing required Kaggle handoff artifact: {path}")
        content[argument] = path.read_bytes()
    cutoff_path = dataset_dir / CUTOFF_ANNOTATIONS_FILENAME
    validate_handoff_bytes(
        **content,
        cutoff_annotations_content=(
            cutoff_path.read_bytes() if cutoff_path.is_file() else None),
        manifest_sha256=manifest_sha256,
        images_sha256=images_sha256,
    )
