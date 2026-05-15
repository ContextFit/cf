"""Structure-aware iCalendar (.ics) extraction.

Each VEVENT is treated as a source-verifiable record with stable event
metadata. The implementation is dependency-free and intentionally conservative:
it unfolds folded lines, parses common event fields, and leaves recurrence rules
and attendee lists visible in the chunk text.
"""

from __future__ import annotations

from pathlib import Path

from contextfit.extractors import document
from contextfit.extractors.structured import _chunk_records


def _unfold_ics_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw.startswith((" ", "\t")) and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def _split_name_params(line: str) -> tuple[str, str, str]:
    if ":" not in line:
        return line.upper(), "", ""
    left, value = line.split(":", 1)
    parts = left.split(";")
    return parts[0].upper(), ";".join(parts[1:]), value


def _clean_ics_value(value: str) -> str:
    return (
        value.replace("\\n", "\n")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\\\", "\\")
        .strip()
    )


def _parse_events(text: str) -> list[dict[str, str | list[str]]]:
    events: list[dict[str, str | list[str]]] = []
    current: dict[str, str | list[str]] | None = None
    for line in _unfold_ics_lines(text):
        name, params, value = _split_name_params(line)
        value = _clean_ics_value(value)
        if name == "BEGIN" and value.upper() == "VEVENT":
            current = {}
            continue
        if name == "END" and value.upper() == "VEVENT":
            if current is not None:
                events.append(current)
            current = None
            continue
        if current is None:
            continue
        if name == "ATTENDEE":
            attendee = value
            if params:
                attendee = f"{attendee} ({params})"
            current.setdefault("ATTENDEE", [])
            assert isinstance(current["ATTENDEE"], list)
            current["ATTENDEE"].append(attendee)
        elif name in {"UID", "SUMMARY", "DESCRIPTION", "LOCATION", "DTSTART", "DTEND", "DUE", "CREATED", "LAST-MODIFIED", "ORGANIZER", "RRULE", "STATUS"}:
            current[name] = value
    return events


def _event_record(event: dict[str, str | list[str]]) -> dict[str, str]:
    mapping = {
        "UID": "uid",
        "SUMMARY": "summary",
        "DESCRIPTION": "description",
        "LOCATION": "location",
        "DTSTART": "start",
        "DTEND": "end",
        "DUE": "due",
        "ORGANIZER": "organizer",
        "RRULE": "rrule",
        "STATUS": "status",
        "CREATED": "created",
        "LAST-MODIFIED": "last_modified",
    }
    record: dict[str, str] = {}
    for key, out_key in mapping.items():
        value = event.get(key)
        if isinstance(value, str) and value:
            record[out_key] = value
    attendees = event.get("ATTENDEE")
    if isinstance(attendees, list) and attendees:
        record["attendees"] = "; ".join(str(v) for v in attendees)
    return record


def chunk_ics(path: Path, text: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """Chunk an iCalendar file by VEVENT records."""
    events = _parse_events(text)
    records = []
    for i, event in enumerate(events):
        record = _event_record(event)
        label = record.get("uid") or record.get("summary") or f"event {i}"
        meta = {
            "event_index": str(i),
            "uid": record.get("uid", ""),
            "summary": record.get("summary", ""),
            "start": record.get("start", ""),
            "end": record.get("end", ""),
        }
        records.append((f"event[{i}] {label}", record, meta))
    if not records:
        base_meta = document.extract(path, text)
        return [{"text": text, "metadata": {**base_meta, "chunk_type": "ics_events", "record_count": "0"}}]
    return _chunk_records(path, records, chunk_size=chunk_size, overlap=overlap, chunk_type="ics_events")
