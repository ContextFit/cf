"""
MetadataIndex — per-chunk structured metadata for filtering, boosting,
and aggregation.  Completely data-type agnostic.

Stores arbitrary field:value pairs per chunk and supports:
  - Pre-filter by any field before BM25 (exact / contains / domain)
  - Score boosting when query terms appear in specified fields
  - Generic aggregate() for "who/what/where did I …" queries
    e.g. group_by="from_email", group_by="author", group_by="tag"

What goes where:
  MetadataIndex   — generic storage, filtering, aggregation
  extractors/     — data-type-specific field extraction (email, document, …)
  engine          — wires extractor → metadata → query

File layout (under kb/metadata/):
  meta.jsonl      — one JSON object per line: {"id": chunk_id, "fields": {...}}
  senders.json    — {email: [chunk_id, …]} shortcut for from_email lookups
"""

from __future__ import annotations

import json
import os
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

from contextfit.retrieval.query_spec import MetadataPredicate

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)
_MIN_CHUNK_ID = -1
_MAX_CHUNK_ID = 2**63 - 1


def _coerce_compare_value(value: Any) -> tuple[int, float | str]:
    """Coerce common metadata scalar values for range comparisons."""
    if isinstance(value, (int, float)):
        return (0, float(value))
    if isinstance(value, datetime):
        return (1, value.isoformat())
    if isinstance(value, date):
        return (1, value.isoformat())

    text = str(value).strip()
    if not text:
        return (2, "")

    # Common benchmark/log date shape: "2023/04/11 (Tue) 23:18".
    m = re.match(r"^(\d{4})[-/](\d{2})[-/](\d{2})", text)
    if m:
        return (1, f"{m.group(1)}-{m.group(2)}-{m.group(3)}")

    for parser in (datetime.fromisoformat, date.fromisoformat):
        try:
            return (1, parser(text.replace("Z", "+00:00")).isoformat())
        except ValueError:
            pass

    try:
        return (0, float(int(text)))
    except ValueError:
        pass
    try:
        return (0, float(text))
    except ValueError:
        return (2, text.lower())


class MetadataIndex:
    """
    Generic per-chunk metadata store.
    Knows nothing about emails, documents, or any specific data type.
    """

    def __init__(self) -> None:
        # chunk_id → {field: value}
        self._meta: dict[int, dict[str, str]] = {}
        # field → token_lower → [chunk_ids]  (for filter pushdown)
        self._field_idx: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # field → sorted [(comparable_value, chunk_id), ...] for range filters
        self._range_idx: dict[str, list[tuple[tuple[int, float | str], int]]] = (
            defaultdict(list)
        )
        self._range_dirty: set[str] = set()
        # email shortcut: from_email_lower → [chunk_ids]
        self._senders: dict[str, list[int]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, chunk_id: int, fields: dict[str, Any]) -> None:
        """Register metadata for one chunk."""
        str_fields = {k: str(v) for k, v in fields.items()}
        self._meta[chunk_id] = str_fields

        for field, value in str_fields.items():
            self._index_field(chunk_id, field, value)

        # Special shortcut for from_email
        sender = str_fields.get("from_email", "")
        if sender:
            self._senders[sender.lower()].append(chunk_id)

    def _index_field(self, chunk_id: int, field: str, value: str) -> None:
        low = value.lower()
        self._field_idx[field][low].append(chunk_id)
        for word in re.split(r"[\s,;|@<>\"'()\[\]]+", low):
            if len(word) > 2:
                self._field_idx[field][word].append(chunk_id)
        self._range_idx[field].append((_coerce_compare_value(value), chunk_id))
        self._range_dirty.add(field)

    def add_file_chunks(
        self,
        path: Path,
        chunk_ids: list[int],
        extractor: Callable[[Path, str], dict[str, str]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Extract metadata from a file and register it for all its chunks.

        Args:
            path:       Source file path.
            chunk_ids:  Chunk IDs produced from this file.
            extractor:  Callable (path, text) → dict.  If None, only
                        records source path.
            extra:      Additional fields to merge in (override extractor).
        """
        try:
            text = path.read_text(errors="replace")
        except Exception:
            text = ""

        if extractor is not None:
            fields = extractor(path, text)
        else:
            fields = {"source": str(path), "filename": path.name}

        if extra:
            fields.update(extra)

        for cid in chunk_ids:
            self.add(cid, fields)

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def filter(
        self,
        field: str,
        value: str,
        mode: str = "contains",
    ) -> set[int]:
        """
        Return chunk IDs where ``field`` matches ``value``.

        mode:
          "contains"  — value appears anywhere in the field (case-insensitive)
          "exact"     — full field value equals value
        """
        low = value.lower()
        fidx = self._field_idx.get(field, {})

        if mode == "exact":
            return set(fidx.get(low, []))

        matches: set[int] = set()
        for token, ids in fidx.items():
            if low in token:
                matches.update(ids)
        return matches

    def filter_predicate(
        self,
        predicate: MetadataPredicate | dict[str, Any],
    ) -> set[int]:
        """Return chunk IDs matching one structured metadata predicate."""
        pred = MetadataPredicate.from_obj(predicate)
        op = pred.op.lower()

        if op in {"contains", "exact"}:
            if pred.value is None:
                return set()
            return self.filter(pred.field, str(pred.value), mode=op)

        if op == "in":
            values = pred.values or ([] if pred.value is None else [pred.value])
            matches: set[int] = set()
            for value in values:
                matches.update(self.filter(pred.field, str(value), mode="exact"))
            return matches

        if op == "exists":
            return {
                cid
                for cid, meta in self._meta.items()
                if str(meta.get(pred.field, "")).strip()
            }

        aliases = {
            "after": "gt",
            "before": "lt",
            "on_or_after": "gte",
            "on_or_before": "lte",
        }
        compare_op = aliases.get(op, op)
        if compare_op in {"gt", "gte", "lt", "lte"}:
            if pred.value is None:
                return set()
            return self._filter_compare(pred.field, compare_op, pred.value)

        raise ValueError(f"unsupported metadata filter operator: {pred.op}")

    def filter_predicates(
        self,
        predicates: Iterable[MetadataPredicate | dict[str, Any]],
        mode: str = "and",
    ) -> set[int]:
        """Combine multiple metadata predicates with AND or OR."""
        parts = [self.filter_predicate(predicate) for predicate in predicates]
        if not parts:
            return set(self._meta)

        if mode.lower() == "or":
            result: set[int] = set()
            for part in parts:
                result.update(part)
            return result
        if mode.lower() != "and":
            raise ValueError("metadata filter mode must be 'and' or 'or'")

        result = parts[0]
        for part in parts[1:]:
            result &= part
        return result

    def _filter_compare(self, field: str, op: str, target: Any) -> set[int]:
        target_value = _coerce_compare_value(target)
        rows = self._sorted_range_rows(field)
        if not rows:
            return set()

        left = 0
        right = len(rows)
        if op == "gt":
            left = bisect_right(rows, (target_value, _MAX_CHUNK_ID))
        elif op == "gte":
            left = bisect_left(rows, (target_value, _MIN_CHUNK_ID))
        elif op == "lt":
            right = bisect_left(rows, (target_value, _MIN_CHUNK_ID))
        elif op == "lte":
            right = bisect_right(rows, (target_value, _MAX_CHUNK_ID))
        return {cid for _value, cid in rows[left:right]}

    def _sorted_range_rows(
        self,
        field: str,
    ) -> list[tuple[tuple[int, float | str], int]]:
        rows = self._range_idx.get(field, [])
        if field in self._range_dirty:
            rows.sort()
            self._range_dirty.discard(field)
        return rows

    def filter_domain(self, domain: str) -> set[int]:
        """
        Return all chunk IDs where from_email ends with @domain.
        Convenience shortcut; equivalent to filter("from_domain", domain).
        """
        low = domain.lower().lstrip("@")
        result: set[int] = set()
        for sender, ids in self._senders.items():
            d = sender.split("@")[-1]
            if d == low or d.endswith(f".{low}"):
                result.update(ids)
        return result

    # Backwards-compat alias
    filter_sender_domain = filter_domain

    # ------------------------------------------------------------------
    # Hybrid boost
    # ------------------------------------------------------------------

    def boost_scores(
        self,
        scores: list[tuple[int, float]],
        query_terms: list[str],
        fields: list[str] | None = None,
        multiplier: float = 2.0,
    ) -> list[tuple[int, float]]:
        """
        Multiply scores for chunks where query terms appear in metadata fields.

        Args:
            scores:      BM25 results — [(chunk_id, score), …]
            query_terms: Raw query words (lowercased internally)
            fields:      Which fields to check. Defaults to all fields.
            multiplier:  Score multiplier when a term is found (default 2×).
        """
        boosted = []
        for cid, score in scores:
            meta = self._meta.get(cid, {})
            if fields:
                field_text = " ".join(meta.get(f, "") for f in fields).lower()
            else:
                field_text = " ".join(meta.values()).lower()
            boost = 1.0
            for term in query_terms:
                if term.lower() in field_text:
                    boost *= multiplier
            boosted.append((cid, score * boost))
        boosted.sort(key=lambda x: -x[1])
        return boosted

    # ------------------------------------------------------------------
    # Generic aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        chunk_ids: Iterable[int],
        group_by: str,
        top_k: int | None = None,
        min_count: int = 1,
        exclude_values: list[str] | None = None,
        include_meta_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Group matching chunks by a metadata field and return sorted counts.

        This is the generic replacement for email-specific search_contacts().
        Works for any field on any data type:

            # Emails  — who sent what
            meta.aggregate(ids, group_by="from_email")

            # Documents — which authors / sections
            meta.aggregate(ids, group_by="author")
            meta.aggregate(ids, group_by="section")

            # Any data  — tag / category / project clouds
            meta.aggregate(ids, group_by="tags")
            meta.aggregate(ids, group_by="project")

        Args:
            chunk_ids:           Chunk IDs to aggregate over.
            group_by:            Field name to group on.
            top_k:               Return only top-k groups (None = all).
            min_count:           Skip groups with fewer than this many chunks.
            exclude_values:      Skip these group values (case-insensitive).
            include_meta_fields: Additional fields to include in each result
                                 (sampled from the first chunk in the group).

        Returns:
            List of dicts sorted by count descending:
              {"value": str, "count": int,
               "chunk_ids": [int, …],
               "<extra_field>": str, …}
        """
        exclude_set = {v.lower() for v in (exclude_values or [])}
        groups: dict[str, dict] = {}

        for cid in chunk_ids:
            meta = self._meta.get(cid, {})
            value = meta.get(group_by, "")
            if not value:
                continue
            if value.lower() in exclude_set:
                continue
            if value not in groups:
                groups[value] = {
                    "value": value,
                    "count": 0,
                    "chunk_ids": [],
                }
                # Attach extra meta fields from first occurrence
                for field in (include_meta_fields or []):
                    groups[value][field] = meta.get(field, "")
            groups[value]["count"] += 1
            groups[value]["chunk_ids"].append(cid)

        results = [g for g in groups.values() if g["count"] >= min_count]
        results.sort(key=lambda x: -x["count"])
        if top_k is not None:
            results = results[:top_k]
        return results

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, chunk_id: int) -> dict[str, str]:
        """Return metadata dict for a chunk (empty dict if unknown)."""
        return self._meta.get(chunk_id, {})

    def fields(self) -> set[str]:
        """Return all indexed field names."""
        return set(self._field_idx.keys())

    def __len__(self) -> int:
        return len(self._meta)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        tmp = path / "meta.jsonl.tmp"
        with open(tmp, "w") as f:
            for cid, fields in self._meta.items():
                f.write(json.dumps({"id": cid, "fields": fields}) + "\n")
        os.replace(tmp, path / "meta.jsonl")

        # Senders shortcut
        tmp = path / "senders.json.tmp"
        with open(tmp, "w") as f:
            json.dump(dict(self._senders), f, separators=(",", ":"))
        os.replace(tmp, path / "senders.json")

    @classmethod
    def load(cls, path: Path | str) -> MetadataIndex:
        path = Path(path)
        idx = cls()

        meta_path = path / "meta.jsonl"
        if meta_path.exists():
            with open(meta_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    cid = obj["id"]
                    fields: dict[str, str] = obj["fields"]
                    idx._meta[cid] = fields
                    for field, value in fields.items():
                        idx._index_field(cid, field, str(value))

        senders_path = path / "senders.json"
        if senders_path.exists():
            raw = json.loads(senders_path.read_text())
            for sender, ids in raw.items():
                idx._senders[sender] = ids

        return idx

    @classmethod
    def exists(cls, path: Path | str) -> bool:
        return (Path(path) / "meta.jsonl").exists()
