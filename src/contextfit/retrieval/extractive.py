"""Deterministic query-focused evidence extraction.

This module compresses retrieved chunks without using an LLM.  It keeps source
text auditable while sending less irrelevant context downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

ExtractiveMode = Literal["none", "spans", "rows", "bullets", "auto"]
CitationMode = Literal["inline", "handles"]

_WORD_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]*")
_TMD_ROW_RE = re.compile(r"^(?P<table>\w+)\[(?P<id>[^\]]*)\]:\s*(?P<body>.+)$")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)?(.+?)\s*$")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "give",
    "how", "in", "include", "into", "is", "it", "its", "of", "on", "or", "the",
    "their", "this", "to", "what", "when", "where", "which", "who", "with", "without",
}

_ANSWER_FIELD_KEYS = {
    "audience",
    "cta",
    "depends_on",
    "format",
    "headline",
    "hook",
    "notes",
    "owner",
    "password",
    "phase",
    "priority",
    "status",
    "success_metric",
    "target_date",
    "username",
    "version",
}
_FIELD_ALIASES = {
    "blocker": "notes",
    "blocking": "notes",
    "dependency": "depends_on",
    "dependencies": "depends_on",
    "metric": "success_metric",
    "metrics": "success_metric",
    "test": "tests",
    "tests": "tests",
}


@dataclass(frozen=True)
class EvidenceItem:
    """A compact source-backed evidence unit extracted from a chunk."""

    kind: str
    text: str
    score: float
    line_start: int | None = None
    line_end: int | None = None
    row_id: str | None = None
    table: str | None = None
    fields: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "text": self.text,
            "score": round(float(self.score), 4),
        }
        if self.line_start is not None:
            data["line_start"] = self.line_start
        if self.line_end is not None:
            data["line_end"] = self.line_end
        if self.row_id is not None:
            data["row_id"] = self.row_id
        if self.table is not None:
            data["table"] = self.table
        if self.fields:
            data["fields"] = self.fields
        return data


def format_evidence_compact(
    items: list[EvidenceItem],
    *,
    metadata: dict[str, Any] | None = None,
    chunk_id: int | None = None,
    chunk_score: float | None = None,
    start_index: int = 1,
    citation_mode: CitationMode = "inline",
    expires_at: str | None = None,
) -> str | tuple[str, dict[str, Any]]:
    """Render evidence as prompt-efficient TMD-like lines.

    JSON remains the best API interchange format. This format is for prompt
    injection: it preserves source/line/row provenance while avoiding repeated
    JSON keys, braces, quotes, and nested metadata maps.
    """

    if not items:
        return ""

    if citation_mode == "handles":
        text, references = format_evidence_handles(
            items,
            metadata=metadata,
            chunk_id=chunk_id,
            chunk_score=chunk_score,
            start_index=start_index,
            expires_at=expires_at,
        )
        return text, references

    lines: list[str] = []
    source = _compact_source(metadata or {})
    chunk_attrs: list[str] = []
    if source:
        chunk_attrs.append(f"src={_quote_atom(source)}")
    if chunk_score is not None:
        chunk_attrs.append(f"rs={chunk_score:.2f}")
    if chunk_id is not None or chunk_attrs:
        label = str(chunk_id) if chunk_id is not None else str(start_index)
        lines.append(f"c[{label}] " + " ".join(chunk_attrs))

    for offset, item in enumerate(items, start=start_index):
        attrs = [f"k={_compact_kind(item.kind)}"]
        line_ref = _line_ref(item)
        if line_ref:
            attrs.append(f"l={line_ref}")
        if item.row_id:
            attrs.append(f"r={_quote_atom(item.row_id)}")
        if item.table:
            attrs.append(f"t={_quote_atom(item.table)}")
        attrs.append(f"s={item.score:.1f}")
        lines.append(f"e[{offset}] " + " ".join(attrs))
        lines.append(_compact_text(item.text))
    return "\n".join(lines)


def format_evidence_handles(
    items: list[EvidenceItem],
    *,
    metadata: dict[str, Any] | None = None,
    chunk_id: int | None = None,
    chunk_score: float | None = None,
    start_index: int = 1,
    expires_at: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Render prompt text with tiny handles plus a provenance sidecar.

    The prompt gets only `@rN` handles.  Full source/line/row/score metadata
    stays in the sidecar and can expire independently of the prompt text.
    """

    if not items:
        return "", {}
    source = _compact_source(metadata or {})
    expires_at = expires_at or default_reference_expiry()
    lines: list[str] = []
    references: dict[str, Any] = {}
    for offset, item in enumerate(items, start=start_index):
        handle = f"@r{offset}"
        lines.append(f"{handle} {_compact_text(item.text)}")
        ref: dict[str, Any] = {
            "kind": item.kind,
            "expires_at": expires_at,
        }
        if source:
            ref["source"] = source
        if chunk_id is not None:
            ref["chunk_id"] = chunk_id
        if chunk_score is not None:
            ref["rank_score"] = round(float(chunk_score), 4)
        ref["evidence_score"] = round(float(item.score), 4)
        if item.line_start is not None:
            ref["line_start"] = item.line_start
        if item.line_end is not None:
            ref["line_end"] = item.line_end
        if item.row_id:
            ref["row_id"] = item.row_id
        if item.table:
            ref["table"] = item.table
        references[handle] = ref
    return "\n".join(lines), references


def default_reference_expiry(ttl_seconds: int = 3600) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()


def extract_evidence(
    text: str,
    query: str,
    *,
    mode: ExtractiveMode = "auto",
    max_items: int = 5,
    max_chars: int = 1200,
    metadata: dict[str, Any] | None = None,
) -> list[EvidenceItem]:
    """Extract query-focused evidence from decoded chunk text.

    The extractor is deterministic and intentionally simple:
    - `.tmd` chunks prefer whole matching rows with parsed fields.
    - Markdown-like chunks prefer matching bullets/sentences.
    - Span mode returns tight character windows around query-term hits.
    """

    if not text or not query or max_items <= 0 or max_chars <= 0:
        return []

    terms = _query_terms(query)
    if not terms:
        return []

    source = str((metadata or {}).get("source", ""))
    is_tmd = (metadata or {}).get("domain") == "tmd" or source.endswith(".tmd")

    selected_mode = mode
    if mode == "auto":
        selected_mode = "rows" if is_tmd or _has_tmd_rows(text) else "bullets"

    items: list[EvidenceItem]
    if selected_mode == "none":
        return []
    if selected_mode == "rows":
        items = _extract_tmd_rows(text, terms, max_items=max_items)
        if items:
            return _budget_items(_offset_lines(items, metadata), max_chars)
        # Fall back for chunks from structured sources without visible rows.
        items = _extract_bullets(text, terms, max_items=max_items)
    elif selected_mode == "bullets":
        items = _extract_bullets(text, terms, max_items=max_items)
    elif selected_mode == "spans":
        items = _extract_spans(text, terms, max_items=max_items)
    else:
        raise ValueError("extractive mode must be one of none, spans, rows, bullets, auto")

    if not items and selected_mode != "spans":
        items = _extract_spans(text, terms, max_items=max_items)
    return _budget_items(_offset_lines(items, metadata), max_chars)


def _compact_kind(kind: str) -> str:
    return {"tmd_row": "row", "bullet": "bul", "span": "span"}.get(kind, kind)


def _compact_source(metadata: dict[str, Any]) -> str:
    for key in ("source", "path", "file", "file_path", "email_path", "uri"):
        value = metadata.get(key)
        if value:
            source = str(value)
            break
    else:
        return ""
    if source.startswith("file://"):
        source = source[7:]
    try:
        path = Path(source)
        if path.is_absolute():
            # Keep enough provenance for humans while avoiding long home prefixes.
            parts = path.parts
            for marker in ("workspace", "cf", "tmd"):
                if marker in parts:
                    idx = parts.index(marker)
                    return "/".join(parts[idx:])
            return path.name
    except Exception:
        pass
    return source


def _line_ref(item: EvidenceItem) -> str:
    if item.line_start is None:
        return ""
    if item.line_end is not None and item.line_end != item.line_start:
        return f"{item.line_start}-{item.line_end}"
    return str(item.line_start)


def _quote_atom(value: str) -> str:
    if not value:
        return '""'
    if re.fullmatch(r"[A-Za-z0-9_./:@+-]+", value):
        return value
    return '"' + value.replace('"', '\\"') + '"'


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in _WORD_RE.findall(query):
        term = raw.strip().lower()
        if len(term) < 2 or term in _STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def _has_tmd_rows(text: str) -> bool:
    return any(_TMD_ROW_RE.match(line.strip()) for line in text.splitlines())


def _score(candidate: str, terms: list[str]) -> float:
    low = candidate.lower()
    score = 0.0
    for term in terms:
        if term in low:
            score += 1.0
            # Reward exact-looking identifiers and field values.
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", low):
                score += 0.5
    # Reward structured facts that are usually answer-bearing.
    answer_keys = (
        "status|depends_on|target_date|version|commit|pid|kb|kbs|priority|"
        "audience|format|success_metric|notes|username|password|tests|passed"
    )
    score += 0.3 * len(re.findall(rf"\b(?:{answer_keys})=", low))
    score += 0.4 * len(re.findall(r"\b(?:username|password|credential|pid|tests?|passed|version)\b", low))
    if any(term in {"auth", "basic", "credential", "credentials"} for term in terms):
        if re.search(r"\b(?:username|password)\b", low):
            score += 3.0
    score += 0.2 * len(re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b[0-9a-f]{7,40}\b", low))
    return score


def _parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    # Split key=value pairs on comma+space before a plausible next key.
    parts = re.split(r",\s+(?=[A-Za-z_][A-Za-z0-9_]*=)", body)
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip().strip('"')
    return fields


def _project_tmd_row_text(
    table: str,
    row_id: str,
    fields: dict[str, str],
    terms: list[str],
    *,
    fallback: str,
) -> str:
    if not fields:
        return fallback

    requested = {
        "phase",
        "target_date",
        "headline",
        "status",
        "priority",
        "audience",
        "success_metric",
        "depends_on",
        "username",
        "password",
        "version",
    }
    for term in terms:
        alias = _FIELD_ALIASES.get(term)
        if alias:
            requested.add(alias)
        if term in fields:
            requested.add(term)
        for key, value in fields.items():
            if term in key.lower() or term in value.lower():
                requested.add(key)

    # Keep rows answer-dense. Long narrative fields are included only when
    # explicitly requested or directly matched by the query.
    ordered = [
        "phase",
        "target_date",
        "asset_type",
        "headline",
        "status",
        "priority",
        "owner",
        "audience",
        "format",
        "success_metric",
        "depends_on",
        "username",
        "password",
        "version",
        "notes",
        "hook",
        "cta",
        "core_message",
    ]
    parts: list[str] = []
    for key in ordered:
        if key not in fields or key not in requested:
            continue
        value = _compact_text(fields[key])
        if len(value) > 180:
            value = value[:177].rstrip() + "…"
        parts.append(f"{key}={_format_field_value(value)}")
    for key in sorted(fields):
        if key in ordered or key not in requested:
            continue
        value = _compact_text(fields[key])
        if len(value) > 140:
            value = value[:137].rstrip() + "…"
        parts.append(f"{key}={_format_field_value(value)}")
    if not parts:
        return fallback
    return f"{table}[{row_id}]: " + ", ".join(parts)


def _format_field_value(value: str) -> str:
    if not value:
        return '""'
    if value.startswith("[") and value.endswith("]"):
        return value
    if re.fullmatch(r"[A-Za-z0-9_./:@+-]+", value):
        return value
    return '"' + value.replace('"', '\"') + '"'


def _extract_tmd_rows(text: str, terms: list[str], *, max_items: int) -> list[EvidenceItem]:
    rows: list[EvidenceItem] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        match = _TMD_ROW_RE.match(stripped)
        if not match:
            continue
        score = _score(stripped, terms)
        if score <= 0:
            continue
        fields = _parse_fields(match.group("body"))
        table = match.group("table")
        row_id = match.group("id")
        rows.append(
            EvidenceItem(
                kind="tmd_row",
                text=_project_tmd_row_text(table, row_id, fields, terms, fallback=stripped),
                score=score,
                line_start=line_no,
                line_end=line_no,
                row_id=row_id,
                table=table,
                fields=fields,
            )
        )
    return sorted(rows, key=lambda item: item.score, reverse=True)[:max_items]


def _candidate_units(text: str) -> list[tuple[int, int, str]]:
    units: list[tuple[int, int, str]] = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        # Preserve markdown headings and bullets as units.
        if stripped.startswith("#") or stripped.startswith("-") or stripped.startswith("*"):
            units.append((i, i, stripped))
            continue
        match = _BULLET_RE.match(stripped)
        if match and len(stripped) <= 360:
            units.append((i, i, stripped))
            continue
        # Otherwise split longer prose into sentences.
        for sentence in re.split(r"(?<=[.!?])\s+", stripped):
            sentence = sentence.strip()
            if sentence:
                units.append((i, i, sentence))
    return units


def _focused_unit_text(unit: str, terms: list[str], max_len: int = 320) -> str:
    compact = _compact_text(unit)
    if len(compact) <= max_len:
        return compact
    low = compact.lower()
    hits = [low.find(term) for term in terms if low.find(term) >= 0]
    if not hits:
        return compact[: max_len - 1].rstrip() + "…"
    start = max(0, min(hits) - 80)
    end = min(len(compact), max(hits) + 180)
    excerpt = compact[start:end].strip()
    if start > 0:
        excerpt = "…" + excerpt
    if end < len(compact):
        excerpt = excerpt.rstrip() + "…"
    if len(excerpt) > max_len:
        excerpt = excerpt[: max_len - 1].rstrip() + "…"
    return excerpt


def _extract_bullets(text: str, terms: list[str], *, max_items: int) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for start, end, unit in _candidate_units(text):
        score = _score(unit, terms)
        stripped = unit.strip()
        low = stripped.lower()
        if stripped.startswith("-"):
            score += 0.7
            label_match = re.match(r"^-\s*([^:]{2,80}):", stripped)
            if label_match:
                label = label_match.group(1).lower()
                if any(term in label for term in terms):
                    score += 3.0
                if "measurement" in terms or "plan" in terms:
                    score += 1.2
        if low.startswith(("heading path:", "purpose:", "table:", "created:", "owner:", "schema:")):
            score *= 0.25
        elif stripped.startswith("#"):
            score *= 0.45
        if score <= 0:
            continue
        items.append(
            EvidenceItem(
                kind="bullet",
                text=_focused_unit_text(unit, terms),
                score=score,
                line_start=start,
                line_end=end,
            )
        )
    return sorted(items, key=lambda item: item.score, reverse=True)[:max_items]


def _extract_spans(text: str, terms: list[str], *, max_items: int) -> list[EvidenceItem]:
    low = text.lower()
    spans: list[EvidenceItem] = []
    used: list[tuple[int, int]] = []
    for term in terms:
        idx = low.find(term)
        if idx < 0:
            continue
        start = max(0, idx - 60)
        end = min(len(text), idx + len(term) + 140)
        if any(not (end < a or start > b) for a, b in used):
            continue
        used.append((start, end))
        excerpt = text[start:end].strip()
        spans.append(
            EvidenceItem(
                kind="span",
                text=excerpt,
                score=_score(excerpt, terms),
                line_start=text.count("\n", 0, start) + 1,
                line_end=text.count("\n", 0, end) + 1,
            )
        )
        if len(spans) >= max_items:
            break
    return sorted(spans, key=lambda item: item.score, reverse=True)[:max_items]


def _offset_lines(
    items: list[EvidenceItem], metadata: dict[str, Any] | None
) -> list[EvidenceItem]:
    if not metadata:
        return items
    base = metadata.get("line_start") or metadata.get("start_line") or metadata.get("line")
    try:
        offset = int(base) - 1
    except (TypeError, ValueError):
        return items
    try:
        context_line_count = int(metadata.get("chunk_context_line_count") or 0)
    except (TypeError, ValueError):
        context_line_count = 0
    if offset <= 0 and context_line_count <= 0:
        return items
    base_line = offset + 1

    def adjust(line_no: int | None) -> int | None:
        if line_no is None:
            return None
        adjusted = line_no - context_line_count + offset
        return max(base_line, adjusted)

    return [
        EvidenceItem(
            kind=item.kind,
            text=item.text,
            score=item.score,
            line_start=adjust(item.line_start),
            line_end=adjust(item.line_end),
            row_id=item.row_id,
            table=item.table,
            fields=item.fields,
        )
        for item in items
    ]


def _budget_items(items: list[EvidenceItem], max_chars: int) -> list[EvidenceItem]:
    kept: list[EvidenceItem] = []
    used = 0
    for item in items:
        text = item.text
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[: max(0, remaining - 1)].rstrip() + "…"
        kept.append(
            EvidenceItem(
                kind=item.kind,
                text=text,
                score=item.score,
                line_start=item.line_start,
                line_end=item.line_end,
                row_id=item.row_id,
                table=item.table,
                fields=item.fields,
            )
        )
        used += len(text) + 1
    return kept
