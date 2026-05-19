"""Token-native evidence-atom selection for multi-session retrieval.

This module builds small, source-backed lexical/facet atoms from retrieved
sessions, then greedily selects sessions that add useful evidence coverage.
It is deterministic and label-free: no embeddings, no LLM calls, and no
benchmark-answer access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any


_TURN_RE = re.compile(r"Turn\s+\d+\s+\(([^)]+)\):\s*", re.I)
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*")
_PROPER_RE = re.compile(r"\b[A-Z][A-Za-z0-9_'-]{2,}\b")
_DATE_RE = re.compile(
    r"\b(?:20\d{2}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}|"
    r"\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
    r"today|yesterday|tomorrow|last\s+\w+|next\s+\w+|this\s+\w+)\b",
    re.I,
)
_NUMBER_RE = re.compile(r"\b(?:\$?\d+(?:[,.]\d+)*(?:\.\d+)?%?|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "give", "had", "has", "have", "help", "how",
    "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "our", "please", "so",
    "that", "the", "their", "them", "they", "this", "to", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
}
_PROPER_STOPS = {
    "What", "When", "Where", "Which", "Who", "Why", "How", "Can", "Could", "Should",
    "Would", "Will", "The", "This", "That", "These", "Those", "Question", "Date",
}

_FACETS: dict[str, re.Pattern[str]] = {
    "preference": re.compile(r"\b(love|like|enjoy|prefer|favorite|favourite|go-?to|hate|avoid|fan of|into)\b", re.I),
    "constraint": re.compile(r"\b(can'?t|cannot|must|need to|have to|budget|deadline|allergy|limit|constraint|only have|under\s+\$?\d+)\b", re.I),
    "goal": re.compile(r"\b(want to|trying to|goal|aim|improve|progress|working on|planning to|hope to|looking to)\b", re.I),
    "temporal": re.compile(r"\b(current|currently|latest|recent|recently|upcoming|next|this week|this weekend|changed|switched|started|stopped|no longer|used to)\b", re.I),
    "decision": re.compile(r"\b(decided|chose|picked|selected|went with|will use|agreed|committed|settled on)\b", re.I),
    "open_loop": re.compile(r"\b(todo|to-do|follow up|remind|pending|outstanding|still need|supposed to|schedule|book|send|finish)\b", re.I),
    "entity_context": re.compile(r"\b(my|our)\s+[a-z][a-z0-9_'-]{2,}\b", re.I),
}


@dataclass(frozen=True)
class EvidenceAtom:
    """Small deterministic evidence unit used for coverage selection."""

    facet: str
    text: str
    source_id: str
    score: float
    tokens: frozenset[str] = field(default_factory=frozenset)
    entities: frozenset[str] = field(default_factory=frozenset)


def query_evidence_facets(query: str) -> set[str]:
    """Infer broad evidence facets from query wording."""
    q = str(query)
    facets = {name for name, pat in _FACETS.items() if pat.search(q)}
    if re.search(r"\b(recommend|suggest|should i|what should|advice|plan|prepare|consider)\b", q, re.I):
        facets.update({"preference", "constraint", "goal", "entity_context"})
    if re.search(r"\b(count|how many|list|which ones|all the|number of)\b", q, re.I):
        facets.update({"number", "entity_context"})
    if _DATE_RE.search(q):
        facets.add("temporal")
    if _NUMBER_RE.search(q):
        facets.add("number")
    return facets


def extract_evidence_atoms(text: str, *, source_id: str) -> list[EvidenceAtom]:
    """Extract deterministic evidence atoms from a session-like text blob."""
    user_text = _user_text(text)
    sentences = _split_sentences(user_text)
    atoms: list[EvidenceAtom] = []
    for sent in sentences:
        tokens = frozenset(_tokens(sent))
        if not tokens:
            continue
        entities = frozenset(_entities(sent))
        for facet, pat in _FACETS.items():
            if pat.search(sent):
                atoms.append(
                    EvidenceAtom(
                        facet=facet,
                        text=_compact(sent),
                        source_id=source_id,
                        score=1.0 + min(len(tokens), 18) / 36.0,
                        tokens=tokens,
                        entities=entities,
                    )
                )
        if _DATE_RE.search(sent):
            atoms.append(EvidenceAtom("temporal", _compact(sent), source_id, 1.15, tokens, entities))
        if _NUMBER_RE.search(sent):
            atoms.append(EvidenceAtom("number", _compact(sent), source_id, 1.10, tokens, entities))
    if not atoms and user_text:
        tokens = frozenset(_tokens(user_text))
        if tokens:
            atoms.append(
                EvidenceAtom(
                    facet="lexical_context",
                    text=_compact(user_text),
                    source_id=source_id,
                    score=0.55,
                    tokens=tokens,
                    entities=frozenset(_entities(user_text)),
                )
            )
    return atoms


def rerank_sessions_by_evidence_atoms(
    query: str,
    bm25_session_order: list[str],
    session_texts: dict[str, str],
    *,
    top_k: int = 10,
    candidate_k: int = 40,
) -> list[str]:
    """Select sessions by marginal evidence-atom coverage.

    The base rank supplies recall. Evidence atoms add a second signal: choose
    sessions that cover distinct facets/entities/query terms instead of many
    near-duplicate high-BM25 sessions.
    """
    if top_k <= 0:
        return []
    base_pool = [sid for sid in bm25_session_order if sid in session_texts]
    if not base_pool:
        base_pool = list(session_texts)
    if len(base_pool) <= 1:
        return base_pool[:top_k]

    pool = base_pool[: max(top_k, candidate_k)]
    query_tokens = set(_tokens(query))
    query_entities = set(_entities(query))
    query_facets = query_evidence_facets(query)
    base_rank = {sid: i for i, sid in enumerate(base_pool, start=1)}

    atoms_by_session = {sid: extract_evidence_atoms(session_texts.get(sid, ""), source_id=sid) for sid in pool}
    session_tokens = {sid: set().union(*(set(atom.tokens) for atom in atoms_by_session[sid])) for sid in pool}
    session_entities = {sid: set().union(*(set(atom.entities) for atom in atoms_by_session[sid])) for sid in pool}
    session_facets = {sid: {atom.facet for atom in atoms_by_session[sid]} for sid in pool}

    selected: list[str] = [pool[0]]
    remaining = pool[1:]
    covered_tokens: set[str] = session_tokens[selected[0]] & query_tokens
    covered_entities: set[str] = session_entities[selected[0]] & query_entities
    covered_facets: set[str] = set(session_facets[selected[0]])

    while remaining and len(selected) < top_k:
        best_sid = remaining[0]
        best_score = -1e9
        for sid in remaining:
            rank = base_rank.get(sid, len(base_pool) + 1)
            tokens = session_tokens[sid]
            entities = session_entities[sid]
            facets = session_facets[sid]
            q_hits = tokens & query_tokens
            new_q_hits = q_hits - covered_tokens
            e_hits = entities & query_entities
            new_e_hits = e_hits - covered_entities
            aligned_facets = facets & query_facets
            new_facets = facets - covered_facets
            atom_strength = sum(atom.score for atom in atoms_by_session[sid]) / max(1.0, len(atoms_by_session[sid]) ** 0.5)
            redundancy = max((_overlap(tokens, session_tokens[sel]) for sel in selected), default=0.0)
            score = (
                1.20 / (20 + rank)
                + 0.70 * _overlap(query_tokens, tokens)
                + 0.16 * len(new_q_hits)
                + 0.32 * len(e_hits)
                + 0.45 * len(new_e_hits)
                + 0.78 * len(aligned_facets)
                + 0.55 * len(new_facets)
                + 0.08 * atom_strength
                - 0.18 * redundancy
            )
            if score > best_score:
                best_score = score
                best_sid = sid
        selected.append(best_sid)
        remaining.remove(best_sid)
        covered_tokens |= session_tokens[best_sid] & query_tokens
        covered_entities |= session_entities[best_sid] & query_entities
        covered_facets |= session_facets[best_sid]
    return selected


def _user_text(text: str) -> str:
    parts = _TURN_RE.split(str(text))
    if len(parts) <= 1:
        return str(text)
    out: list[str] = []
    i = 1
    while i + 1 < len(parts):
        role = parts[i].strip().lower()
        content = parts[i + 1].strip()
        if role in {"user", "human", "client", "customer"} and content:
            out.append(content)
        i += 2
    return "\n".join(out) or str(text)


def _split_sentences(text: str) -> list[str]:
    compact = " ".join(str(text).split())
    if not compact:
        return []
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", compact) if p.strip()]


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for raw in _WORD_RE.findall(str(text).lower()):
        if len(raw) <= 2 or raw in _STOPWORDS:
            continue
        out.append(_stem(raw))
    return out


def _entities(text: str) -> list[str]:
    return [m.group(0).lower() for m in _PROPER_RE.finditer(str(text)) if m.group(0) not in _PROPER_STOPS]


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def _compact(text: str, max_chars: int = 420) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
