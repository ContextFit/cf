"""Structure-aware extractors for common data interchange files.

These extractors keep source records as the unit of meaning: JSON objects,
JSONL lines, and CSV/TSV rows. The tokenizer still produces final token IDs;
this module only chooses better chunk boundaries and attaches row/object
metadata for explainable retrieval.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from contextfit.extractors import document


Scalar = str | int | float | bool | None


def _stringify(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _record_title(record: Any, fallback: str) -> str:
    if isinstance(record, dict):
        for key in ("id", "uuid", "name", "title", "subject", "date", "timestamp"):
            value = record.get(key)
            if value not in (None, ""):
                return f"{key}={_stringify(value)}"
    return fallback


def _record_text(record: Any, *, label: str, source: str) -> str:
    lines = [f"Source: {source}", f"Record: {label}"]
    if isinstance(record, dict):
        for key in sorted(record):
            lines.append(f"{key}: {_stringify(record[key])}")
    else:
        lines.append(_stringify(record))
    return "\n".join(lines)


def _emit_group(
    chunks: list[dict],
    *,
    base_meta: dict[str, str],
    records: list[tuple[str, Any, dict[str, str]]],
    ordinal: int,
    chunk_type: str,
) -> int:
    if not records:
        return ordinal
    text = "\n\n---\n\n".join(_record_text(record, label=label, source=base_meta["source"]) for label, record, _meta in records)
    first_label = records[0][0]
    last_label = records[-1][0]
    metadata = {
        **base_meta,
        "chunk_type": chunk_type,
        "chunk_ordinal": str(ordinal),
        "record_count": str(len(records)),
        "record_start": first_label,
        "record_end": last_label,
    }
    # Preserve stable fields from single-record chunks directly in metadata.
    if len(records) == 1:
        metadata.update(records[0][2])
    chunks.append({"text": text, "metadata": metadata})
    return ordinal + 1


def _chunk_records(
    path: Path,
    records: list[tuple[str, Any, dict[str, str]]],
    *,
    chunk_size: int,
    overlap: int,
    chunk_type: str,
) -> list[dict]:
    base_meta = document.extract(path, "")
    if not records:
        return [{"text": f"Source: {base_meta['source']}\n(empty file)", "metadata": {**base_meta, "chunk_type": chunk_type, "record_count": "0"}}]

    chunks: list[dict] = []
    acc: list[tuple[str, Any, dict[str, str]]] = []
    acc_est = 0.0
    budget = max(64, chunk_size * 0.8)
    ordinal = 0
    previous_tail: list[tuple[str, Any, dict[str, str]]] = []

    for label, record, meta in records:
        est = document._token_estimate(_record_text(record, label=label, source=base_meta["source"]))
        if acc and acc_est + est > budget:
            ordinal = _emit_group(chunks, base_meta=base_meta, records=acc, ordinal=ordinal, chunk_type=chunk_type)
            previous_tail = acc[-1:] if overlap > 0 else []
            acc = previous_tail.copy()
            acc_est = sum(document._token_estimate(_record_text(r, label=l, source=base_meta["source"])) for l, r, _m in acc)
        acc.append((label, record, meta))
        acc_est += est

    if acc and acc != previous_tail:
        ordinal = _emit_group(chunks, base_meta=base_meta, records=acc, ordinal=ordinal, chunk_type=chunk_type)
    return chunks


def _flatten_json_records(obj: Any) -> list[tuple[str, Any, dict[str, str]]]:
    if isinstance(obj, list):
        return [(f"item[{i}] {_record_title(item, '')}".strip(), item, {"json_index": str(i)}) for i, item in enumerate(obj)]
    if isinstance(obj, dict):
        # Common API export shape: top-level keys each contain record arrays.
        array_keys = [k for k, v in obj.items() if isinstance(v, list)]
        if array_keys:
            records: list[tuple[str, Any, dict[str, str]]] = []
            for key in array_keys:
                for i, item in enumerate(obj[key]):
                    label = f"{key}[{i}] {_record_title(item, '')}".strip()
                    records.append((label, item, {"json_path": key, "json_index": str(i)}))
            return records
        return [(_record_title(obj, "object"), obj, {"json_path": "$"})]
    return [("value", obj, {"json_path": "$"})]


def chunk_json(path: Path, text: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """Chunk a JSON file by top-level record/object boundaries."""
    obj = json.loads(text)
    return _chunk_records(path, _flatten_json_records(obj), chunk_size=chunk_size, overlap=overlap, chunk_type="json_records")


def chunk_jsonl(path: Path, text: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """Chunk newline-delimited JSON by source line records."""
    records: list[tuple[str, Any, dict[str, str]]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        records.append((f"line {line_no} {_record_title(obj, '')}".strip(), obj, {"line": str(line_no)}))
    return _chunk_records(path, records, chunk_size=chunk_size, overlap=overlap, chunk_type="jsonl_records")


def chunk_delimited(path: Path, text: str, chunk_size: int = 512, overlap: int = 64, delimiter: str = ",") -> list[dict]:
    """Chunk CSV/TSV by rows while preserving header fields."""
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    records: list[tuple[str, Any, dict[str, str]]] = []
    for row_no, row in enumerate(reader, 1):
        clean = {str(k): (v if v is not None else "") for k, v in row.items() if k is not None}
        records.append((f"row {row_no} {_record_title(clean, '')}".strip(), clean, {"row": str(row_no)}))
    chunk_type = "tsv_rows" if delimiter == "\t" else "csv_rows"
    return _chunk_records(path, records, chunk_size=chunk_size, overlap=overlap, chunk_type=chunk_type)
