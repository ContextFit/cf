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
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)


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
            low = value.lower()
            self._field_idx[field][low].append(chunk_id)
            for word in re.split(r"[\s,;|@<>\"'()\[\]]+", low):
                if len(word) > 2:
                    self._field_idx[field][word].append(chunk_id)

        # Special shortcut for from_email
        sender = str_fields.get("from_email", "")
        if sender:
            self._senders[sender.lower()].append(chunk_id)

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
    def load(cls, path: Path | str) -> "MetadataIndex":
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
                        low = str(value).lower()
                        idx._field_idx[field][low].append(cid)
                        for word in re.split(r"[\s,;|@<>\"'()\[\]]+", low):
                            if len(word) > 2:
                                idx._field_idx[field][word].append(cid)

        senders_path = path / "senders.json"
        if senders_path.exists():
            raw = json.loads(senders_path.read_text())
            for sender, ids in raw.items():
                idx._senders[sender] = ids

        return idx

    @classmethod
    def exists(cls, path: Path | str) -> bool:
        return (Path(path) / "meta.jsonl").exists()
