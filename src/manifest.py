"""Shared, versioned dataset-manifest contract.

Both synthetic generation and real-data merging write the same row shape.  CSV
serialization lives here so independent producers create byte-stable metadata.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = "1"
MANIFEST_COLUMNS = (
    "filename",
    "label",
    "split",
    "source",
    "field_type",
    "font",
    "sample_mode",
    "writer_id",
    "schema_version",
)
LABEL_COLUMNS = ("filename", "label", "split")
RUN_METADATA_FILENAME = "run-metadata.json"


def csv_bytes(
    columns: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> bytes:
    """Serialize dictionary rows as deterministic UTF-8 CSV bytes."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(columns),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return output.getvalue().encode("utf-8")


def manifest_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    """Serialize rows using :data:`MANIFEST_COLUMNS`."""
    return csv_bytes(MANIFEST_COLUMNS, rows)


def labels_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    """Serialize the compatibility labels view of manifest rows."""
    return csv_bytes(LABEL_COLUMNS, rows)
