"""Lightweight, local relationship/backlink index.

The index is intentionally derived data: canonical truth remains in source
chunks, while extracted entities/edges point back to chunk IDs for evidence.
No LLM calls, embeddings, or benchmark-specific labels are used.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][\w&.+'-]*(?:\s+(?:of|and|&|the|[A-Z][\w&.+'-]*))*|[A-Z]{2,}(?:\.[A-Z]{2,})?)\b"
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP_ENTITIES = {
    "A", "An", "And", "As", "At", "By", "For", "From", "I", "In", "It",
    "Of", "On", "Or", "The", "This", "That", "To", "We", "What", "When",
    "Where", "Who", "Why", "How", "Tell", "Question", "Answer", "Turn",
    "User", "Assistant", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
}
_QUERY_STOP_TERMS = {
    "question", "date", "answer", "what", "when", "where", "who", "why", "how",
    "which", "did", "was", "were", "is", "are", "my", "me", "i", "you", "the",
    "a", "an", "this", "that", "last", "past", "current", "previous", "ago",
}
_RELATION_QUERY_TERMS = {
    "who", "works", "worked", "work", "employer", "company", "founded", "founder",
    "started", "created", "launched", "invested", "backed", "funded", "owns",
    "owned", "bought", "purchased", "uses", "used", "prefers", "likes", "loves",
    "enjoys", "met", "meet", "lunch", "spoke",
}
_REL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("works_at", re.compile(r"(?P<s>[A-Z][\w&.+'-]*(?:\s+[A-Z][\w&.+'-]*){0,3})\s+(?:works|worked)\s+(?:at|for|with)\s+(?P<o>[A-Z][\w&.+'-]*(?:\s+(?:of|and|&|the|[A-Z][\w&.+'-]*)){0,5})", re.I)),
    ("founded", re.compile(r"(?P<s>[A-Z][\w&.+'-]*(?:\s+[A-Z][\w&.+'-]*){0,3})\s+(?:founded|started|created|launched)\s+(?P<o>[A-Z][\w&.+'-]*(?:\s+(?:of|and|&|the|[A-Z][\w&.+'-]*)){0,5})", re.I)),
    ("invested_in", re.compile(r"(?P<s>[A-Z][\w&.+'-]*(?:\s+[A-Z][\w&.+'-]*){0,3})\s+(?:invested\s+in|backed|funded)\s+(?P<o>[A-Z][\w&.+'-]*(?:\s+(?:of|and|&|the|[A-Z][\w&.+'-]*)){0,5})", re.I)),
    ("owns", re.compile(r"(?P<s>[A-Z][\w&.+'-]*(?:\s+[A-Z][\w&.+'-]*){0,3})\s+(?:owns|owned|bought|purchased|uses|used)\s+(?P<o>[A-Z][\w&.+'-]*(?:\s+(?:of|and|&|the|[A-Z][\w&.+'-]*)){0,5})", re.I)),
    ("prefers", re.compile(r"(?P<s>[A-Z][\w&.+'-]*(?:\s+[A-Z][\w&.+'-]*){0,3})\s+(?:prefers|likes|loves|enjoys)\s+(?P<o>[A-Z][\w&.+'-]*(?:\s+(?:of|and|&|the|[A-Z][\w&.+'-]*)){0,5})", re.I)),
    ("met_with", re.compile(r"(?P<s>[A-Z][\w&.+'-]*(?:\s+[A-Z][\w&.+'-]*){0,3})\s+(?:met\s+with|had\s+lunch\s+with|spoke\s+with)\s+(?P<o>[A-Z][\w&.+'-]*(?:\s+(?:of|and|&|the|[A-Z][\w&.+'-]*)){0,5})", re.I)),
]
_RELATION_TRIGGER_RE = re.compile(
    r"\b(?:works?|worked|founded|started|created|launched|invested|backed|funded|owns?|owned|bought|purchased|uses?|used|prefers|likes|loves|enjoys|met\s+with|had\s+lunch\s+with|spoke\s+with)\b",
    re.I,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class RelationshipEdge:
    subject: str
    predicate: str
    object: str
    chunk_id: int
    confidence: float = 0.75


@dataclass(frozen=True)
class RelationshipMatch:
    chunk_id: int
    score: float
    reason: str


def _normalize_entity(entity: str) -> str:
    words = _WORD_RE.findall(entity.lower())
    return " ".join(words)


def _is_specific_entity(norm: str, display: str | None = None) -> bool:
    words = norm.split()
    if not words or any(w in _QUERY_STOP_TERMS for w in words):
        return False
    if len(words) >= 2:
        return True
    token = words[0]
    shown = display or token
    return len(token) >= 4 and (any(ch.isdigit() for ch in token) or shown.isupper())


def _contains_entity_phrase(query_norm: str, entity_norm: str) -> bool:
    # Both strings are normalized to lowercase word tokens joined by spaces, so
    # padded substring matching gives word-boundary behavior without compiling a
    # regex for every indexed entity during query-time gating.
    return f" {entity_norm} " in f" {query_norm} "


def _query_has_relation_intent(query: str) -> bool:
    terms = set(_WORD_RE.findall(query.lower()))
    return bool(terms & _RELATION_QUERY_TERMS)


def _display_entity(entity: str) -> str:
    cleaned = " ".join(entity.strip(" .,:;!?()[]{}\n\t").split())
    cleaned = re.sub(
        r"\s+and\s+(?:works?|worked|founded|started|created|launched|invested|backed|funded|owns?|owned|bought|purchased|uses?|used|prefers|likes|loves|enjoys|met|had|spoke)\b.*$",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"^(?:at|for|with|in|on|by)\s+", "", cleaned, flags=re.I)
    return cleaned


def extract_entities(text: str) -> list[str]:
    """Extract title/acronym-like entity mentions from text."""
    entities: list[str] = []
    seen: set[str] = set()
    for match in _ENTITY_RE.finditer(text):
        display = _display_entity(match.group(0))
        if not display or display in _STOP_ENTITIES:
            continue
        norm = _normalize_entity(display)
        if len(norm) < 2 or norm in seen:
            continue
        # Avoid treating sentence-leading common verbs/questions as entities.
        if display.split()[0] in _STOP_ENTITIES:
            continue
        seen.add(norm)
        entities.append(display)
    return entities


def extract_relationship_edges(text: str, chunk_id: int) -> list[RelationshipEdge]:
    """Extract a small set of high-signal relationship edges."""
    edges: list[RelationshipEdge] = []
    seen: set[tuple[str, str, str]] = set()
    sentences = _SENTENCE_SPLIT_RE.split(text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or not _RELATION_TRIGGER_RE.search(sentence):
            continue
        # Long conversation chunks can contain paragraph-sized pseudo-sentences.
        # Relationship regexes are intentionally heuristic; cap the window so a
        # missing terminator cannot create expensive, low-precision scans.
        if len(sentence) > 800:
            sentence = sentence[:800]
        for predicate, pattern in _REL_PATTERNS:
            for match in pattern.finditer(sentence):
                subject = _display_entity(match.group("s"))
                obj = _display_entity(match.group("o"))
                if not subject or not obj:
                    continue
                s_norm = _normalize_entity(subject)
                o_norm = _normalize_entity(obj)
                if not s_norm or not o_norm or s_norm == o_norm:
                    continue
                key = (s_norm, predicate, o_norm)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(RelationshipEdge(subject=subject, predicate=predicate, object=obj, chunk_id=chunk_id))
    return edges


class RelationshipIndex:
    """Derived entity/relationship index with evidence-preserving chunk links."""

    def __init__(self) -> None:
        self._entity_chunks: dict[str, set[int]] = defaultdict(set)
        self._entity_display: dict[str, str] = {}
        self._edges: list[RelationshipEdge] = []
        self._neighbors: dict[str, set[str]] = defaultdict(set)
        self._chunk_ids: set[int] = set()

    def add_text(self, chunk_id: int, text: str, metadata: dict[str, Any] | None = None) -> None:
        self._chunk_ids.add(chunk_id)
        for entity in extract_entities(text):
            norm = _normalize_entity(entity)
            if not norm:
                continue
            self._entity_chunks[norm].add(chunk_id)
            self._entity_display.setdefault(norm, entity)

        for edge in extract_relationship_edges(text, chunk_id):
            self.add_edge(edge)

        # Metadata often contains clean entity-like handles: source titles,
        # authors, senders, projects, rows. Index short textual values as weak
        # backlink anchors without inventing edges.
        for value in (metadata or {}).values():
            if not isinstance(value, str) or len(value) > 120:
                continue
            for entity in extract_entities(value):
                norm = _normalize_entity(entity)
                if norm:
                    self._entity_chunks[norm].add(chunk_id)
                    self._entity_display.setdefault(norm, entity)

    def add_edge(self, edge: RelationshipEdge) -> None:
        self._edges.append(edge)
        self._chunk_ids.add(edge.chunk_id)
        s = _normalize_entity(edge.subject)
        o = _normalize_entity(edge.object)
        if not s or not o:
            return
        self._entity_chunks[s].add(edge.chunk_id)
        self._entity_chunks[o].add(edge.chunk_id)
        self._entity_display.setdefault(s, edge.subject)
        self._entity_display.setdefault(o, edge.object)
        self._neighbors[s].add(o)
        self._neighbors[o].add(s)

    def _is_selective(self, entity: str, max_fraction: float = 0.20, max_chunks: int = 50) -> bool:
        ids = self._entity_chunks.get(entity, set())
        if not ids:
            return False
        corpus = max(1, len(self._chunk_ids))
        if corpus < 20:
            return len(ids) <= max_chunks
        return len(ids) <= max_chunks and (len(ids) / corpus) <= max_fraction

    def query_entities(self, query: str) -> list[str]:
        """Find specific, selective indexed entities mentioned in a query."""
        q_norm = _normalize_entity(query)
        explicit = {_normalize_entity(e) for e in extract_entities(query)}
        matches: set[str] = set()
        for entity in explicit:
            display = self._entity_display.get(entity, entity)
            if entity in self._entity_chunks and _is_specific_entity(entity, display) and self._is_selective(entity):
                matches.add(entity)
        for entity, display in self._entity_display.items():
            if (
                _is_specific_entity(entity, display)
                and self._is_selective(entity)
                and _contains_entity_phrase(q_norm, entity)
            ):
                matches.add(entity)
        return sorted(matches, key=lambda e: (-len(e), e))

    def related_matches(self, query: str, direct_score: float = 1.0, neighbor_score: float = 0.35) -> list[RelationshipMatch]:
        """Return chunks linked to entities mentioned in query and neighbors."""
        scores: dict[int, tuple[float, list[str]]] = {}
        relation_intent = _query_has_relation_intent(query)
        for entity in self.query_entities(query):
            label = self._entity_display.get(entity, entity)
            for cid in self._entity_chunks.get(entity, set()):
                score, reasons = scores.get(cid, (0.0, []))
                scores[cid] = (score + direct_score, reasons + [f"entity:{label}"])
            if not relation_intent:
                continue
            for neighbor in self._neighbors.get(entity, set()):
                if not self._is_selective(neighbor, max_fraction=0.15, max_chunks=25):
                    continue
                n_label = self._entity_display.get(neighbor, neighbor)
                for cid in self._entity_chunks.get(neighbor, set()):
                    score, reasons = scores.get(cid, (0.0, []))
                    scores[cid] = (score + neighbor_score, reasons + [f"neighbor:{label}->{n_label}"])
        matches = [RelationshipMatch(cid, score, ",".join(reasons[:3])) for cid, (score, reasons) in scores.items()]
        matches.sort(key=lambda m: (-m.score, m.chunk_id))
        return matches

    def boost_scores(
        self,
        scores: list[tuple[int, float]],
        query: str,
        multiplier: float = 1.15,
        append: bool = True,
        append_limit: int = 3,
    ) -> list[tuple[int, float]]:
        """Boost and conservatively append candidates linked by backlinks."""
        if not scores or multiplier <= 1.0:
            return scores
        matches = self.related_matches(query)
        if not matches:
            return scores
        related = {m.chunk_id: m.score for m in matches}
        existing_ids = {cid for cid, _score in scores}
        has_direct_anchor = any(m.chunk_id in existing_ids and "entity:" in m.reason for m in matches)

        base = max(score for _cid, score in scores)
        seen: set[int] = set()
        boosted: list[tuple[int, float]] = []
        for cid, score in scores:
            seen.add(cid)
            rel = min(1.0, related.get(cid, 0.0))
            if rel:
                # Keep this a precision-oriented tie-breaker. Direct entity hits
                # should help order close candidates, not swamp lexical evidence.
                score = score * (1.0 + min(0.20, (multiplier - 1.0) * 0.35 * rel))
            boosted.append((cid, score))
        if append and has_direct_anchor and _query_has_relation_intent(query):
            append_base = base * min(0.02, max(0.005, (multiplier - 1.0) * 0.02))
            added = 0
            for match in matches:
                if added >= append_limit:
                    break
                if match.chunk_id not in seen and "neighbor:" in match.reason:
                    boosted.append((match.chunk_id, append_base * min(1.0, match.score)))
                    added += 1
        boosted.sort(key=lambda x: (-x[1], x[0]))
        return boosted

    def evidence_for_chunk(self, chunk_id: int) -> dict[str, Any]:
        entities = [self._entity_display.get(e, e) for e, ids in self._entity_chunks.items() if chunk_id in ids]
        edges = [asdict(e) for e in self._edges if e.chunk_id == chunk_id]
        return {"entities": sorted(entities), "edges": edges}

    def __len__(self) -> int:
        return sum(len(ids) for ids in self._entity_chunks.values())

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "entity_chunks": {e: sorted(ids) for e, ids in self._entity_chunks.items()},
            "entity_display": self._entity_display,
            "edges": [asdict(e) for e in self._edges],
            "chunk_ids": sorted(self._chunk_ids),
        }
        tmp = path / "relationships.json.tmp"
        tmp.write_text(json.dumps(payload, separators=(",", ":")))
        os.replace(tmp, path / "relationships.json")

    @classmethod
    def load(cls, path: Path | str) -> "RelationshipIndex":
        path = Path(path)
        idx = cls()
        raw = json.loads((path / "relationships.json").read_text())
        for entity, ids in raw.get("entity_chunks", {}).items():
            idx._entity_chunks[entity] = set(int(cid) for cid in ids)
        idx._entity_display = {str(k): str(v) for k, v in raw.get("entity_display", {}).items()}
        idx._chunk_ids = {int(cid) for cid in raw.get("chunk_ids", [])}
        if not idx._chunk_ids:
            for ids in idx._entity_chunks.values():
                idx._chunk_ids.update(ids)
        for obj in raw.get("edges", []):
            edge = RelationshipEdge(**obj)
            idx._edges.append(edge)
            s = _normalize_entity(edge.subject)
            o = _normalize_entity(edge.object)
            if s and o:
                idx._neighbors[s].add(o)
                idx._neighbors[o].add(s)
        return idx

    @classmethod
    def exists(cls, path: Path | str) -> bool:
        return (Path(path) / "relationships.json").exists()
