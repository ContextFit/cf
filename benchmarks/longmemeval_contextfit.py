#!/usr/bin/env python3
"""Evaluate ContextFit retrieval on LongMemEval_S.

This is a retrieval benchmark, not a full QA generation benchmark.
For each LongMemEval item, we build a per-question ContextFit KB from the
provided haystack sessions, query with the question, and measure whether the
retrieved chunks come from the ground-truth answer_session_ids.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from contextfit.retrieval.engine import RetrievalEngine
from contextfit.retrieval.memory_atoms import augment_query_for_memory_atoms, atom_type_priors, episode_relevance_score, extract_memory_atoms, query_memory_intents
from contextfit.retrieval.evidence_atoms import rerank_sessions_by_evidence_atoms


PREFERENCE_RE = re.compile(
    r"\b("
    r"i\s+(?:really\s+|generally\s+|usually\s+|always\s+|don'?t\s+)?"
    r"(?:like|love|prefer|enjoy|hate|dislike|want|need|choose|use|watch|listen|eat|drink|wear|play|read)"
    r"|my\s+(?:favorite|favourite|go-to|preference|preferred)"
    r"|works\s+(?:best|better)\s+for\s+me"
    r")\b",
    re.I,
)


def session_to_text(
    session_id: str,
    date: str,
    turns: list[dict[str, Any]],
    include_answer_marker: bool = False,
) -> str:
    lines = [f"Session ID: {session_id}", f"Date: {date}", ""]
    for i, turn in enumerate(turns, 1):
        role = turn.get("role", "unknown")
        content = str(turn.get("content", ""))
        has_answer = " [HAS_ANSWER]" if include_answer_marker and turn.get("has_answer") else ""
        lines.append(f"Turn {i} ({role}){has_answer}: {content}")
    return "\n".join(lines)


def extract_preference_facts(session_id: str, date: str, turns: list[dict[str, Any]]) -> str:
    """Lightweight, non-LLM preference memory extraction for benchmark runs."""
    facts = []
    for i, turn in enumerate(turns, 1):
        content = " ".join(str(turn.get("content", "")).split())
        if not content or not PREFERENCE_RE.search(content):
            continue
        # Keep the text compact and token-near the words preference queries use.
        facts.append(f"Preference memory from {date}, session {session_id}, turn {i}: {content[:1200]}")
    return "\n".join(facts)


def _embedding_cache_path(text: str, model: str, cache_dir: Path) -> Path:
    h = hashlib.sha256((model + "\0" + text).encode("utf-8")).hexdigest()
    return cache_dir / f"{h}.json"


def embed_texts(texts: list[str], model: str = "text-embedding-3-small", cache_dir: Path = Path("benchmarks/cache/openai_embeddings")) -> list[list[float]]:
    """Embed texts with OpenAI, using a local content-addressed cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[list[float] | None] = [None] * len(texts)
    missing: list[tuple[int, str, Path]] = []
    for i, text in enumerate(texts):
        path = _embedding_cache_path(text, model, cache_dir)
        if path.exists():
            out[i] = json.loads(path.read_text())
        else:
            missing.append((i, text, path))

    api_key = os.environ.get("OPENAI_API_KEY")
    if missing and not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --openai-vector-rerank")

    for start in range(0, len(missing), 64):
        batch = missing[start : start + 64]
        payload = json.dumps({"model": model, "input": [text for _, text, _ in batch]}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for (i, _text, path), item in zip(batch, data["data"], strict=True):
            emb = item["embedding"]
            path.write_text(json.dumps(emb, separators=(",", ":")))
            out[i] = emb

    return [x for x in out if x is not None]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for rank, value in enumerate(ranking, start=1):
            scores[value] = scores.get(value, 0.0) + 1.0 / (k + rank)
            best_rank[value] = min(best_rank.get(value, rank), rank)
    return sorted(scores, key=lambda v: (-scores[v], best_rank[v]))


def vector_rank_sessions(query: str, session_texts: list[tuple[str, str]]) -> list[str]:
    texts = [query] + [text[:24000] for _, text in session_texts]
    embeddings = embed_texts(texts)
    q = np.array(embeddings[0], dtype=np.float32)
    docs = np.array(embeddings[1:], dtype=np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-9)
    docs = docs / np.maximum(np.linalg.norm(docs, axis=1, keepdims=True), 1e-9)
    sims = docs @ q
    order = np.argsort(-sims)
    return [session_texts[int(i)][0] for i in order]


def token_chain_expand_sessions(chunks, base_scores: list[float], top_k: int, anchor_k: int = 3) -> list[str]:
    """Session ranking with token-native evidence-chain expansion.

    Start from ContextFit's query ranking, then promote sessions whose rare token
    signatures overlap with the strongest anchors. This is designed for
    multi-session/update cases where one evidence session is easy to find and
    the companion evidence shares entity/topic tokens but not the query wording.
    """
    if not chunks:
        return []
    # Keep one representative chunk per session for rank output, but use all
    # candidate chunks for document frequency.
    token_sets = [set(map(int, c.tokens.tolist())) for c in chunks]
    df: dict[int, int] = defaultdict(int)
    for toks in token_sets:
        for tok in toks:
            df[tok] += 1
    max_df = max(2, int(len(chunks) * 0.25))
    rare_sets = [{tok for tok in toks if df[tok] <= max_df} for toks in token_sets]
    anchors = rare_sets[: min(anchor_k, len(rare_sets))]

    seen: set[str] = set()
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for rank, (chunk, rare) in enumerate(zip(chunks, rare_sets, strict=False), start=1):
        sid = chunk.metadata.get("session_id") or chunk.metadata.get("source")
        if sid is None:
            continue
        sid = str(sid)
        seen.add(sid)
        base_rr = 1.0 / (20 + rank)
        related = 0.0
        if rare:
            for anchor in anchors:
                if anchor is rare or not anchor:
                    continue
                related = max(related, len(rare & anchor) / max(8.0, (len(rare) * len(anchor)) ** 0.5))
        # Conservative: keep base retrieval dominant; relatedness breaks ties
        # and pulls companion evidence upward.
        score = 0.82 * base_rr + 0.18 * related
        if sid not in scores or score > scores[sid]:
            scores[sid] = score
            best_rank[sid] = rank
    return sorted(seen, key=lambda sid: (-scores.get(sid, 0.0), best_rank.get(sid, 10**9)))[:top_k]


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "i", "in", "is", "it", "me", "my", "of", "on",
    "or", "our", "that", "the", "their", "there", "they", "this", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "with", "you", "your", "question", "date",
}


def _word_tokens(text: str) -> set[str]:
    return {
        m.group(0).lower()
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9_'-]{2,}", text)
        if m.group(0).lower() not in _STOPWORDS
    }


def _entity_tokens(text: str) -> set[str]:
    return {
        m.group(0).lower()
        for m in re.finditer(r"\b[A-Z][A-Za-z0-9_'-]{2,}\b", text)
        if m.group(0).lower() not in _STOPWORDS
    }


def coverage_rerank_sessions(
    query: str,
    session_texts: list[tuple[str, str]],
    base_order: list[str],
    top_k: int,
) -> list[str]:
    """Token-native companion-evidence reranker for multi-session questions.

    Full-evidence LongMemEval misses usually are not solved by generic diversity:
    the missing evidence sessions are companion episodes near the same entity/topic
    cluster as the strongest anchor. Preserve the top anchor, then rank the rest
    by a mix of original relevance, query overlap, and anchor/selected overlap.
    """
    text_by_sid = {str(sid): text for sid, text in session_texts}
    pool = [str(sid) for sid in base_order if str(sid) in text_by_sid]
    if len(pool) <= top_k:
        return pool[:top_k]

    query_terms = _word_tokens(query)
    query_entities = _entity_tokens(query)
    session_terms = {sid: _word_tokens(text_by_sid[sid]) for sid in pool}
    session_entities = {sid: _entity_tokens(text_by_sid[sid]) for sid in pool}
    base_rr = {sid: 1.0 / (20 + i) for i, sid in enumerate(pool, 1)}

    selected: list[str] = [pool.pop(0)]
    anchor = selected[0]
    covered_terms: set[str] = session_terms[anchor] & query_terms
    covered_entities: set[str] = session_entities[anchor] & query_entities

    while pool and len(selected) < top_k:
        best_sid = None
        best_score = -1e9
        for sid in pool:
            terms = session_terms[sid]
            entities = session_entities[sid]
            q_hits = terms & query_terms
            e_hits = entities & query_entities
            new_terms = q_hits - covered_terms
            new_entities = e_hits - covered_entities
            anchor_overlap = len(terms & session_terms[anchor]) / max(24.0, (len(terms) * len(session_terms[anchor])) ** 0.5)
            selected_overlap = max(
                [len(terms & session_terms[c]) / max(24.0, (len(terms) * len(session_terms[c])) ** 0.5) for c in selected]
                or [0.0]
            )
            # Companion evidence tends to share the anchor's topical signature,
            # but should also add at least some query/entity coverage.
            score = (
                1.00 * base_rr[sid]
                + 0.090 * anchor_overlap
                + 0.035 * selected_overlap
                + 0.018 * len(q_hits)
                + 0.030 * len(new_terms)
                + 0.030 * len(e_hits)
                + 0.045 * len(new_entities)
            )
            if score > best_score:
                best_score = score
                best_sid = sid
        assert best_sid is not None
        selected.append(best_sid)
        covered_terms |= session_terms[best_sid] & query_terms
        covered_entities |= session_entities[best_sid] & query_entities
        pool.remove(best_sid)
    return selected


def targeted_expansion_sessions(
    query: str,
    session_texts: list[tuple[str, str]],
    base_order: list[str],
    top_k: int,
    protected_k: int = 8,
) -> list[str]:
    """Source-set-preserving companion expansion for multi-session queries.

    Preserve the strongest baseline anchors, then fill only the tail slots with
    sessions from the broader retrieved pool that share anchor/entity/query
    signatures. This is intentionally more conservative than replacement
    reranking: it explores down the likely evidence path without discarding the
    source set that baseline retrieval already trusted.
    """
    if len(base_order) <= top_k:
        return base_order[:top_k]

    text_by_sid = {str(sid): text for sid, text in session_texts}
    pool = [str(sid) for sid in base_order if str(sid) in text_by_sid]
    if len(pool) <= top_k:
        return pool[:top_k]

    protected_n = max(1, min(protected_k, top_k, len(pool)))
    protected = pool[:protected_n]
    baseline_tail = pool[protected_n:top_k]
    candidates = [sid for sid in pool[protected_n:] if sid not in protected]

    query_terms = _word_tokens(query)
    query_entities = _entity_tokens(query)
    session_terms = {sid: _word_tokens(text_by_sid[sid]) for sid in pool}
    session_entities = {sid: _entity_tokens(text_by_sid[sid]) for sid in pool}
    base_rr = {sid: 1.0 / (20 + i) for i, sid in enumerate(pool, 1)}
    anchor_terms = set().union(*(session_terms[sid] for sid in protected))
    anchor_entities = set().union(*(session_entities[sid] for sid in protected))

    scored: list[tuple[float, int, str]] = []
    for sid in candidates:
        terms = session_terms[sid]
        entities = session_entities[sid]
        q_hits = terms & query_terms
        entity_hits = entities & query_entities
        anchor_term_overlap = len(terms & anchor_terms) / max(24.0, (len(terms) * len(anchor_terms)) ** 0.5)
        anchor_entity_overlap = len(entities & anchor_entities) / max(4.0, (len(entities) * len(anchor_entities)) ** 0.5)
        score = (
            0.75 * base_rr[sid]
            + 0.090 * anchor_term_overlap
            + 0.140 * anchor_entity_overlap
            + 0.020 * len(q_hits)
            + 0.055 * len(entity_hits)
        )
        scored.append((score, pool.index(sid), sid))
    scored.sort(key=lambda row: (-row[0], row[1]))

    fill = [sid for _score, _rank, sid in scored[: max(0, top_k - protected_n)]]
    if len(fill) < top_k - protected_n:
        for sid in baseline_tail:
            if sid not in fill:
                fill.append(sid)
            if len(fill) >= top_k - protected_n:
                break
    return (protected + fill)[:top_k]


def atom_sessions_from_chunks(chunks, scores: list[float], query: str) -> list[str]:
    priors = atom_type_priors(query)
    by_session: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for rank, (chunk, score) in enumerate(zip(chunks, scores, strict=False), start=1):
        sid = chunk.metadata.get("session_id") or chunk.metadata.get("source")
        if sid is None:
            continue
        atom_types = str(chunk.metadata.get("atom_types", "")).split(",")
        prior = max([priors.get(t, 1.0) for t in atom_types if t] or [1.0])
        value = prior * (float(score) + 1.0 / (20 + rank))
        sid = str(sid)
        if sid not in by_session or value > by_session[sid]:
            by_session[sid] = value
            best_rank[sid] = rank
    return sorted(by_session, key=lambda sid: (-by_session[sid], best_rank[sid]))


def confidence_fuse_sessions(session_chunks, session_scores: list[float], atom_chunks, atom_scores: list[float], query: str, top_k: int) -> list[str]:
    intents = query_memory_intents(query)
    session_rank = unique_sessions_from_chunks(session_chunks)
    atom_rank = atom_sessions_from_chunks(atom_chunks, atom_scores, query)
    session_pos = {str(sid): i for i, sid in enumerate(session_rank, start=1)}
    scores: dict[str, float] = {str(sid): 1.0 / (20 + i) for i, sid in enumerate(session_rank, start=1)}
    best: dict[str, int] = {str(sid): i for i, sid in enumerate(session_rank, start=1)}
    for arank, sid in enumerate(atom_rank, start=1):
        types = set()
        for chunk in atom_chunks:
            csid = chunk.metadata.get("session_id") or chunk.metadata.get("source")
            if str(csid) == str(sid):
                types.update(t for t in str(chunk.metadata.get("atom_types", "")).split(",") if t)
        if intents and not (types & intents):
            continue
        atom_bonus = 0.9 / (20 + arank)
        baseline_floor = 0.0 if str(sid) not in session_pos else 0.25 / (20 + session_pos[str(sid)])
        scores[str(sid)] = scores.get(str(sid), 0.0) + atom_bonus + baseline_floor
        best[str(sid)] = min(best.get(str(sid), 10**9), arank)
    return sorted(scores, key=lambda sid: (-scores[sid], best.get(sid, 10**9)))[:top_k]


def unique_sessions_from_chunks(chunks) -> list[str]:
    seen = set()
    out = []
    for chunk in chunks:
        sid = chunk.metadata.get("session_id") or chunk.metadata.get("source")
        if sid is None:
            continue
        if str(sid) not in seen:
            seen.add(str(sid))
            out.append(str(sid))
    return out


# ── temporal query detection ──────────────────────────────────────────────────
_TEMPORAL_Q_RE = re.compile(
    r"\b(current|currently|now|latest|most recent|recent|recently|last|updated?|changed?|switched|no longer|used to|today|this (week|month|year))\b",
    re.I,
)
_KNOWLEDGE_UPDATE_RE = re.compile(
    r"\b(still|anymore|nowadays|as of|at (this|the) (point|time|moment)|going forward|from now on)\b",
    re.I,
)


def _is_temporal_query(query: str) -> bool:
    return bool(_TEMPORAL_Q_RE.search(query) or _KNOWLEDGE_UPDATE_RE.search(query))


def _parse_question_date(query: str) -> datetime.date | None:
    """Extract question date from the runner's 'Question date:' header."""
    m = re.search(r"Question date:\s*(\d{4})[-/](\d{2})[-/](\d{2})", query)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _parse_session_date(date_str: str) -> datetime.date | None:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.datetime.strptime(str(date_str).strip(), fmt).date()
        except ValueError:
            continue
    return None


def scored_bm25_sessions(
    chunks,
    scores: list[float],
    query: str,
    session_dates: dict[str, str] | None = None,
    score_pool: str = "softmax_sum",   # "first" | "sum" | "softmax_sum"
    recency_weight: float = 0.15,
) -> list[str]:
    """Rank sessions by pooled BM25 chunk scores + optional date-aware recency boost.

    score_pool options:
      'first'       – original behaviour: rank by first-appearing chunk (pure dedup)
      'sum'         – sum all chunk scores per session
      'softmax_sum' – softmax-normalise scores before summing (prevents long sessions
                      from winning just by having more chunks)

    recency_weight: 0 disables recency; >0 blends in a recency signal for
    temporal queries.  Blend = (1-w)*bm25_rank_score + w*recency_score.
    """
    import math

    # ── collect per-session chunk scores ──────────────────────────────────────
    sess_raw: dict[str, list[float]] = {}
    sess_first_rank: dict[str, int] = {}
    for rank, (chunk, score) in enumerate(zip(chunks, scores), 1):
        sid = str(chunk.metadata.get("session_id") or chunk.metadata.get("source") or "")
        if not sid:
            continue
        sess_raw.setdefault(sid, []).append(float(score))
        if sid not in sess_first_rank:
            sess_first_rank[sid] = rank

    if not sess_raw:
        return []

    # ── pool chunk scores per session ─────────────────────────────────────────
    sess_pool: dict[str, float] = {}
    if score_pool == "first":
        for sid, rank in sess_first_rank.items():
            sess_pool[sid] = 1.0 / (20 + rank)
    elif score_pool == "sum":
        for sid, raw in sess_raw.items():
            sess_pool[sid] = sum(raw)
    else:  # softmax_sum
        for sid, raw in sess_raw.items():
            max_s = max(raw)
            exp_scores = [math.exp(s - max_s) for s in raw]
            total = sum(exp_scores)
            sess_pool[sid] = sum(e / total * s for e, s in zip(exp_scores, raw))

    # normalise pool scores to [0, 1]
    max_pool = max(sess_pool.values()) or 1.0
    sess_norm = {sid: v / max_pool for sid, v in sess_pool.items()}

    # ── optional date-aware recency bias ─────────────────────────────────────
    if recency_weight > 0 and session_dates and _is_temporal_query(query):
        q_date = _parse_question_date(query)
        if q_date is not None:
            # compute days-before-question for each session
            sess_days: dict[str, int] = {}
            for sid in sess_norm:
                d = _parse_session_date(session_dates.get(sid, ""))
                if d is not None and d <= q_date:
                    sess_days[sid] = (q_date - d).days
                else:
                    sess_days[sid] = 9999  # future or unknown → deprioritise
            # recency score: sessions closest to (but not after) question date score highest
            min_days = min(sess_days.values())
            max_days = max(v for v in sess_days.values() if v < 9999) or 1
            recency: dict[str, float] = {}
            for sid, days in sess_days.items():
                if days >= 9999:
                    recency[sid] = 0.0
                else:
                    # score decays with distance from question date, favouring recent
                    recency[sid] = 1.0 - (days - min_days) / (max_days - min_days + 1)
            # blend
            for sid in sess_norm:
                sess_norm[sid] = (
                    (1 - recency_weight) * sess_norm[sid]
                    + recency_weight * recency.get(sid, 0.0)
                )

    return sorted(sess_norm, key=lambda s: (-sess_norm[s], sess_first_rank.get(s, 9999)))


_NUM_WORDS = {
    "one": 1, "a": 1, "an": 1,
    "couple": 2, "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_WEEKDAYS = {name.lower(): i for i, name in enumerate(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])}


def _num_from_text(value: str) -> int | None:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    if value.isdigit():
        return int(value)
    if value == "a couple":
        return 2
    return _NUM_WORDS.get(value)


def _add_months(d: datetime.date, months: int) -> datetime.date:
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    last_day = [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    return datetime.date(year, month, min(d.day, last_day))


def _relative_date_window(query: str) -> tuple[datetime.date, datetime.date] | None:
    """Infer a generic relative-date window from natural temporal wording.

    Uses only the question date and phrases such as "10 days ago", "last
    Friday", "past two weeks", and "this month". It does not inspect answer
    labels or benchmark topics.
    """
    q_date = _parse_question_date(query)
    if q_date is None:
        return None
    body = query.split("Question:", 1)[-1].lower()

    num_pat = r"\d+|one|two|three|four|five|six|seven|eight|nine|ten|couple|a\s+couple"

    m = re.search(rf"\bpast\s+({num_pat})(?:\s+of)?\s+(day|days|week|weeks|month|months)\b", body)
    if m:
        n = _num_from_text(m.group(1)) or 1
        unit = m.group(2)
        days = n * (7 if unit.startswith("week") else 30 if unit.startswith("month") else 1)
        return (q_date - datetime.timedelta(days=days), q_date)

    if "this month" in body:
        return (q_date.replace(day=1), q_date)
    if "this week" in body:
        return (q_date - datetime.timedelta(days=q_date.weekday()), q_date)

    m = re.search(rf"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+({num_pat})\s+months?\s+ago\b", body)
    if m:
        target_wd = _WEEKDAYS[m.group(1)]
        n = _num_from_text(m.group(2)) or 1
        center = _add_months(q_date, -n)
        candidates = [center + datetime.timedelta(days=i) for i in range(-7, 8)]
        candidates = [d for d in candidates if d.weekday() == target_wd]
        if candidates:
            center = min(candidates, key=lambda d: abs((d - center).days))
        return (center - datetime.timedelta(days=2), center + datetime.timedelta(days=2))

    m = re.search(rf"\b({num_pat})(?:\s+of)?\s+(day|days|week|weeks|month|months)\s+ago\b", body)
    if m:
        n = _num_from_text(m.group(1)) or 1
        unit = m.group(2)
        if unit.startswith("day"):
            center = q_date - datetime.timedelta(days=n)
            pad = 1 if n <= 3 else 2
        elif unit.startswith("week"):
            center = q_date - datetime.timedelta(days=7 * n)
            pad = 3
        else:
            center = _add_months(q_date, -n)
            pad = 10
        return (center - datetime.timedelta(days=pad), center + datetime.timedelta(days=pad))

    m = re.search(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", body)
    if m:
        target_wd = _WEEKDAYS[m.group(1)]
        delta = (q_date.weekday() - target_wd) % 7 or 7
        center = q_date - datetime.timedelta(days=delta)
        return (center - datetime.timedelta(days=1), center + datetime.timedelta(days=1))

    return None


def augment_query_with_temporal_date_hint(query: str) -> str:
    """Add concrete date tokens for explicit relative-date questions."""
    window = _relative_date_window(query)
    if window is None:
        return query
    start, end = window
    days = (end - start).days
    if days < 0 or days > 21:
        return query
    labels = []
    d = start
    while d <= end:
        labels.append(f"{d:%Y/%m/%d} {d:%A}")
        d += datetime.timedelta(days=1)
    return query + "\nTemporal date hint: " + "; ".join(labels)


def structured_temporal_date_filters(
    query: str,
    question_type: str | None,
) -> list[dict[str, str]]:
    """Build metadata date predicates for explicit temporal questions."""
    if not str(question_type or "").startswith("temporal-reasoning"):
        return []
    window = _relative_date_window(query)
    if window is None:
        return []
    start, end = window
    return [
        {"field": "date", "op": "on_or_after", "value": start.isoformat()},
        {"field": "date", "op": "on_or_before", "value": end.isoformat()},
    ]


def temporal_date_rerank_sessions(
    query: str,
    session_dates: dict[str, str],
    base_order: list[str],
    top_k: int,
) -> list[str]:
    """Promote sessions whose dates match explicit relative-date queries."""
    window = _relative_date_window(query)
    if window is None:
        return base_order[:top_k]
    start, end = window
    center = start + (end - start) / 2
    half_width = max(1, (end - start).days / 2)
    rows: list[tuple[float, int, str]] = []
    for rank, sid in enumerate(base_order, 1):
        d = _parse_session_date(session_dates.get(str(sid), ""))
        base = 1.0 / rank
        date_score = 0.0
        if d is not None:
            if start <= d <= end:
                date_score = 1.0
            else:
                distance = min(abs((d - start).days), abs((d - end).days), abs((d - center).days))
                date_score = max(0.0, 1.0 - distance / max(7.0, half_width * 3.0))
        score = 0.58 * base + 0.42 * date_score
        rows.append((score, rank, str(sid)))
    rows.sort(key=lambda x: (-x[0], x[1]))
    return [sid for _score, _rank, sid in rows[:top_k]]


def scored_episode_sessions_lme(item: dict[str, Any], query: str, top_k: int) -> list[str]:
    scored = []
    for sid, sess in zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True):
        score = episode_relevance_score(query, sess)
        scored.append((score, sid))
    scored.sort(key=lambda x: -x[0])
    return [sid for score, sid in scored if score > 0][:top_k]


def eval_one(
    item: dict[str, Any],
    method: str,
    top_k_chunks: int,
    chunk_size: int,
    overlap: int,
    rank_by_session: bool,
    add_preference_facts: bool,
    add_memory_atoms: bool,
    retrieval_k: int,
    openai_vector_rerank: bool,
    openai_fusion: bool,
    token_native_rerank: bool,
    token_native_fusion: bool,
    token_chain_expand: bool,
    memory_atom_fusion: bool,
    episode_score: bool = False,
    episode_score_fusion: bool = False,
    structural_rerank: bool = False,
    query_auto: bool = False,
    evidence_atom_rerank: bool = False,
    score_pool: str = "first",
    recency_weight: float = 0.0,
    coverage_rerank: bool = False,
    targeted_expansion: bool = False,
    temporal_date_rerank: bool = False,
    relationship_boost: float = 1.0,
    conversation_chunks: bool = False,
    include_answer_marker: bool = False,
    conversation_parent: bool = False,
    structured_temporal_filters: bool = False,
    structured_filter_fusion: str = "none",
    filter_pushdown_threshold: float = 0.50,
    two_stage_sessions: bool = False,
) -> dict[str, Any]:
    tmp = Path(tempfile.mkdtemp(prefix="cf-lme-"))
    try:
        engine = RetrievalEngine.create(tmp)
        session_texts: list[tuple[str, str]] = []
        for sid, date, sess in zip(
            item["haystack_session_ids"],
            item["haystack_dates"],
            item["haystack_sessions"],
            strict=True,
        ):
            meta = {
                "session_id": sid,
                "date": date,
                "question_id": item["question_id"],
                "kind": "session",
            }
            full_text = session_to_text(
                sid,
                date,
                sess,
                include_answer_marker=include_answer_marker,
            )
            session_texts.append((sid, full_text))
            if conversation_chunks:
                engine.ingest_conversation(
                    sid,
                    date,
                    sess,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    metadata=meta,
                    update_indexes=True,
                    include_answer_marker=include_answer_marker,
                    include_parent=conversation_parent,
                )
            else:
                engine.ingest_text(
                    full_text,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    metadata=meta,
                    update_indexes=True,
                )
            if add_memory_atoms:
                atoms = extract_memory_atoms(sess, source_id=sid, source_date=date)
                if atoms:
                    atom_text = "\n\n".join(atom.to_index_text() for atom in atoms)
                    atom_meta = dict(meta)
                    atom_meta["kind"] = "memory_atoms"
                    atom_meta["atom_types"] = ",".join(sorted({atom.atom_type for atom in atoms}))
                    engine.ingest_text(
                        atom_text,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        metadata=atom_meta,
                        update_indexes=True,
                    )
            if add_preference_facts:
                pref = extract_preference_facts(sid, date, sess)
                if pref:
                    pref_meta = dict(meta)
                    pref_meta["kind"] = "preference_facts"
                    engine.ingest_text(
                        pref,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        metadata=pref_meta,
                        update_indexes=True,
                    )
        # Add question date to help temporal questions without using answer labels.
        query = f"Question date: {item.get('question_date','')}\nQuestion: {item['question']}"
        if temporal_date_rerank and str(item.get("question_type") or "").startswith("temporal-reasoning"):
            query = augment_query_with_temporal_date_hint(query)
        filters = (
            structured_temporal_date_filters(query, item.get("question_type"))
            if structured_temporal_filters
            else []
        )
        if two_stage_sessions:
            two_stage = engine.query_two_stage_sessions(
                query,
                top_k=top_k_chunks,
                broad_k=retrieval_k,
                precise_k=6,
                method=method,
                max_tokens=200_000,
            )
            retrieved_sessions = two_stage["session_ids"]
        elif query_auto:
            auto_result = engine.query_auto(
                query,
                top_k=top_k_chunks,
                retrieval_k=retrieval_k,
                method=method,
                max_tokens=200_000,
                evidence_atom_rerank=evidence_atom_rerank,
            )
            retrieved_sessions = auto_result["session_ids"]
        elif episode_score:
            retrieved_sessions = scored_episode_sessions_lme(item, query, top_k_chunks)
        elif episode_score_fusion:
            ep_sessions = scored_episode_sessions_lme(item, query, top_k_chunks)
            cf_result = engine.query(
                query,
                top_k=max(top_k_chunks, retrieval_k),
                method=method,
                max_tokens=200_000,
                filter_field=("kind", "session"),
                relationship_boost=relationship_boost,
                filters=filters,
                filter_pushdown_threshold=filter_pushdown_threshold,
            )
            cf_sessions = unique_sessions_from_chunks(cf_result.chunks)
            retrieved_sessions = reciprocal_rank_fusion([cf_sessions, ep_sessions])[:top_k_chunks]
        elif memory_atom_fusion:
            cf_result = engine.query(
                query,
                top_k=max(top_k_chunks, retrieval_k),
                method=method,
                max_tokens=200_000,
                filter_field=("kind", "session"),
                relationship_boost=relationship_boost,
                filters=filters,
                filter_pushdown_threshold=filter_pushdown_threshold,
            )
            atom_result = engine.query(
                augment_query_for_memory_atoms(query),
                top_k=max(top_k_chunks, retrieval_k),
                method=method,
                max_tokens=200_000,
                filter_field=("kind", "memory_atoms"),
                relationship_boost=relationship_boost,
            )
            retrieved_sessions = confidence_fuse_sessions(
                cf_result.chunks,
                cf_result.scores,
                atom_result.chunks,
                atom_result.scores,
                query,
                top_k_chunks,
            )
        elif token_chain_expand:
            cf_result = engine.query(
                query,
                top_k=max(top_k_chunks, retrieval_k),
                method=method,
                max_tokens=200_000,
                relationship_boost=relationship_boost,
                filters=filters,
                filter_pushdown_threshold=filter_pushdown_threshold,
            )
            retrieved_sessions = token_chain_expand_sessions(
                cf_result.chunks,
                cf_result.scores,
                top_k=top_k_chunks,
            )
        elif token_native_fusion:
            cf_result = engine.query(
                query,
                top_k=max(top_k_chunks, retrieval_k),
                method=method,
                max_tokens=200_000,
                relationship_boost=relationship_boost,
                filters=filters,
                filter_pushdown_threshold=filter_pushdown_threshold,
            )
            token_result = engine.query(
                query,
                top_k=max(top_k_chunks, retrieval_k),
                method=method,
                max_tokens=200_000,
                token_rerank=True,
                relationship_boost=relationship_boost,
                filters=filters,
                filter_pushdown_threshold=filter_pushdown_threshold,
            )
            cf_sessions = unique_sessions_from_chunks(cf_result.chunks)
            token_sessions = unique_sessions_from_chunks(token_result.chunks)
            retrieved_sessions = reciprocal_rank_fusion([cf_sessions, token_sessions])[:top_k_chunks]
        elif openai_vector_rerank or openai_fusion:
            vector_sessions = vector_rank_sessions(query, session_texts)
            if openai_fusion:
                cf_result = engine.query(
                    query,
                    top_k=max(top_k_chunks, retrieval_k),
                    method=method,
                    max_tokens=200_000,
                    token_rerank=token_native_rerank,
                    relationship_boost=relationship_boost,
                    filters=filters,
                    filter_pushdown_threshold=filter_pushdown_threshold,
                )
                cf_sessions = unique_sessions_from_chunks(cf_result.chunks)
                retrieved_sessions = reciprocal_rank_fusion([cf_sessions, vector_sessions])[:top_k_chunks]
            else:
                retrieved_sessions = vector_sessions[:top_k_chunks]
        elif rank_by_session:
            group_pool = top_k_chunks
            if coverage_rerank or temporal_date_rerank or evidence_atom_rerank or targeted_expansion:
                group_pool = max(top_k_chunks, 30)
            if targeted_expansion:
                group_pool = max(group_pool, min(retrieval_k, 50))
            if evidence_atom_rerank:
                group_pool = max(group_pool, min(retrieval_k, 50))
            if temporal_date_rerank:
                group_pool = max(group_pool, min(retrieval_k, 100))
            groups = engine.query_groups(
                query,
                group_by="session_id",
                top_k_groups=group_pool,
                retrieval_k=retrieval_k,
                method=method,
                max_tokens=200_000,
                relationship_boost=relationship_boost,
                filters=filters,
                filter_pushdown_threshold=filter_pushdown_threshold,
            )
            retrieved_sessions = [g["value"] for g in groups]
            if filters and structured_filter_fusion == "rrf":
                broad_groups = engine.query_groups(
                    query,
                    group_by="session_id",
                    top_k_groups=group_pool,
                    retrieval_k=retrieval_k,
                    method=method,
                    max_tokens=200_000,
                    relationship_boost=relationship_boost,
                )
                broad_sessions = [g["value"] for g in broad_groups]
                retrieved_sessions = reciprocal_rank_fusion(
                    [retrieved_sessions, broad_sessions]
                )
            if coverage_rerank and str(item.get("question_type") or "").startswith("multi-session"):
                retrieved_sessions = coverage_rerank_sessions(
                    query,
                    session_texts,
                    retrieved_sessions,
                    top_k=top_k_chunks,
                )
            if evidence_atom_rerank and str(item.get("question_type") or "").startswith("multi-session"):
                session_text_map = {sid: text for sid, text in session_texts}
                retrieved_sessions = rerank_sessions_by_evidence_atoms(
                    query,
                    retrieved_sessions,
                    session_text_map,
                    top_k=top_k_chunks,
                    candidate_k=max(top_k_chunks, retrieval_k),
                )
            if targeted_expansion and str(item.get("question_type") or "").startswith("multi-session"):
                retrieved_sessions = targeted_expansion_sessions(
                    query,
                    session_texts,
                    retrieved_sessions,
                    top_k=top_k_chunks,
                )
            if temporal_date_rerank and str(item.get("question_type") or "").startswith("temporal-reasoning"):
                session_dates = dict(zip(item["haystack_session_ids"], item["haystack_dates"], strict=True))
                retrieved_sessions = temporal_date_rerank_sessions(
                    query,
                    session_dates,
                    retrieved_sessions,
                    top_k=top_k_chunks,
                )
            else:
                retrieved_sessions = retrieved_sessions[:top_k_chunks]
        elif structural_rerank:
            # BM25 retrieval + token-native structural reranker (same as query_auto bm25 route)
            result = engine.query(
                query,
                top_k=max(top_k_chunks, retrieval_k),
                method=method,
                max_tokens=200_000,
                relationship_boost=relationship_boost,
                filters=filters,
                filter_pushdown_threshold=filter_pushdown_threshold,
            )
            bm25_order = unique_sessions_from_chunks(result.chunks)
            # Collect per-session chunk text for the reranker
            sess_text_map: dict[str, str] = {sid: text for sid, text in session_texts}
            retrieved_sessions = engine.rerank_sessions_by_structure(
                query, bm25_order, sess_text_map, top_k=top_k_chunks,
            )
        else:
            result = engine.query(
                query,
                top_k=max(top_k_chunks, retrieval_k),
                method=method,
                max_tokens=200_000,
                token_rerank=token_native_rerank,
                relationship_boost=relationship_boost,
                filters=filters,
                filter_pushdown_threshold=filter_pushdown_threshold,
            )
            if filters and structured_filter_fusion == "rrf":
                broad_result = engine.query(
                    query,
                    top_k=max(top_k_chunks, retrieval_k),
                    method=method,
                    max_tokens=200_000,
                    token_rerank=token_native_rerank,
                    relationship_boost=relationship_boost,
                )
                filtered_sessions = unique_sessions_from_chunks(result.chunks)
                broad_sessions = unique_sessions_from_chunks(broad_result.chunks)
                retrieved_sessions = reciprocal_rank_fusion(
                    [filtered_sessions, broad_sessions]
                )[:top_k_chunks]
            elif score_pool != "first" or recency_weight > 0:
                # Session-level score pooling + optional date-aware recency
                session_dates = dict(zip(item["haystack_session_ids"], item["haystack_dates"]))
                retrieved_sessions = scored_bm25_sessions(
                    result.chunks,
                    result.scores,
                    query,
                    session_dates=session_dates,
                    score_pool=score_pool,
                    recency_weight=recency_weight,
                )[:top_k_chunks]
            else:
                retrieved_sessions = unique_sessions_from_chunks(result.chunks)[:top_k_chunks]
        gold = set(item.get("answer_session_ids") or [])
        ranks = [retrieved_sessions.index(g) + 1 for g in gold if g in retrieved_sessions]
        return {
            "question_id": item["question_id"],
            "question_type": item.get("question_type"),
            "is_abstention": item["question_id"].endswith("_abs"),
            "n_sessions": len(item["haystack_sessions"]),
            "n_gold": len(gold),
            "retrieved_sessions": retrieved_sessions,
            "gold_sessions": sorted(gold),
            "best_rank": min(ranks) if ranks else None,
            "all_found_at": max(ranks) if len(ranks) == len(gold) and ranks else None,
            "structured_temporal_filters": filters,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def summarize(rows: list[dict[str, Any]], ks=(1, 3, 5, 10)) -> dict[str, Any]:
    eval_rows = [r for r in rows if not r["is_abstention"] and r["n_gold"] > 0]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in eval_rows:
        by_type[r.get("question_type") or "unknown"].append(r)

    def metrics(sub: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(sub)
        if not n:
            return {"n": 0}
        m: dict[str, Any] = {"n": n}
        for k in ks:
            m[f"any_recall@{k}"] = sum(r["best_rank"] is not None and r["best_rank"] <= k for r in sub) / n
            m[f"all_evidence@{k}"] = sum(r["all_found_at"] is not None and r["all_found_at"] <= k for r in sub) / n
        m["mrr"] = sum((1 / r["best_rank"]) if r["best_rank"] else 0 for r in sub) / n
        m["avg_sessions"] = sum(r["n_sessions"] for r in sub) / n
        return m

    return {
        "overall": metrics(eval_rows),
        "by_type": {qt: metrics(sub) for qt, sub in sorted(by_type.items())},
        "skipped_abstention": sum(r["is_abstention"] for r in rows),
        "total_rows": len(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--limit", type=int, default=50, help="number of examples to run; 0 = all")
    ap.add_argument("--method", choices=["exact", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--top-k-chunks", type=int, default=10)
    ap.add_argument("--retrieval-k", type=int, default=50, help="chunks to retrieve before session aggregation")
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--rank-by-session", action="store_true", help="aggregate chunk scores by session_id before ranking")
    ap.add_argument("--preference-facts", action="store_true", help="ingest lightweight extracted preference facts")
    ap.add_argument("--memory-atoms", action="store_true", help="ingest domain-neutral user memory atoms")
    ap.add_argument("--openai-vector-rerank", action="store_true", help="rank sessions by OpenAI embedding similarity; cached locally")
    ap.add_argument("--openai-fusion", action="store_true", help="reciprocal-rank fuse ContextFit retrieval with OpenAI vector session ranking")
    ap.add_argument("--token-native-rerank", action="store_true", help="rerank ContextFit candidates by token phrase/window/preference/temporal structure")
    ap.add_argument("--token-native-fusion", action="store_true", help="reciprocal-rank fuse baseline ContextFit ranking with token-native structural reranking")
    ap.add_argument("--token-chain-expand", action="store_true", help="promote sessions related to top anchors by rare token overlap")
    ap.add_argument("--memory-atom-fusion", action="store_true", help="RRF-fuse normal session retrieval with memory-atom retrieval")
    ap.add_argument("--episode-score", action="store_true", help="rank sessions by numeric episode relevance scorer (lexical+atom+entity) instead of ContextFit retrieval")
    ap.add_argument("--episode-score-fusion", action="store_true", help="RRF-fuse ContextFit retrieval with numeric episode relevance scorer")
    ap.add_argument("--structural-rerank", action="store_true", help="BM25 retrieval + token-native structural reranker (same as production query_auto bm25 route)")
    ap.add_argument("--query-auto", action="store_true", help="Use production deterministic query router and routed retrieval modes")
    ap.add_argument("--evidence-atom-rerank", action="store_true", help="use token-native evidence-atom/facet selection for multi-session reranking")
    ap.add_argument("--score-pool", choices=["first", "sum", "softmax_sum"], default="first", help="Session-level BM25 score pooling: 'first'=original dedup, 'sum'=sum all chunk scores, 'softmax_sum'=softmax-weighted sum")
    ap.add_argument("--recency-weight", type=float, default=0.0, help="Date-aware recency bias weight [0,1] for temporal queries. 0=disabled.")
    ap.add_argument("--coverage-rerank", action="store_true", help="greedily rerank session pools for complementary token/entity evidence coverage")
    ap.add_argument("--targeted-expansion", action="store_true", help="preserve top source anchors and fill tail slots with targeted companion sessions")
    ap.add_argument("--two-stage-sessions", action="store_true", help="broad session discovery followed by precise in-session retrieval")
    ap.add_argument("--temporal-date-rerank", action="store_true", help="rerank explicit relative-date temporal questions using question/session dates")
    ap.add_argument("--relationship-boost", type=float, default=1.0, help="optional derived entity/relationship backlink score boost; 1.0 disables")
    ap.add_argument(
        "--structured-temporal-filters",
        action="store_true",
        help="apply inferred metadata date filters to explicit temporal questions",
    )
    ap.add_argument(
        "--structured-filter-fusion",
        choices=["none", "rrf"],
        default="none",
        help="optionally RRF-fuse structured-filtered retrieval with broad retrieval",
    )
    ap.add_argument(
        "--filter-pushdown-threshold",
        type=float,
        default=0.50,
        help="maximum matched-chunk ratio that uses metadata filters as pushdown",
    )
    ap.add_argument(
        "--conversation-chunks",
        action="store_true",
        help="ingest haystack sessions as turn-aware conversation chunks instead of blind token windows",
    )
    ap.add_argument(
        "--conversation-parent",
        action="store_true",
        help="with --conversation-chunks, also index a full-session parent chunk",
    )
    ap.add_argument(
        "--include-answer-marker",
        action="store_true",
        help=(
            "include dataset answer labels in indexed text; for backwards-compatible diagnostics "
            "only, not valid benchmark reporting"
        ),
    )
    ap.add_argument("--out", type=Path, default=Path("benchmarks/longmemeval_contextfit_results.json"))
    args = ap.parse_args()

    data = json.loads(args.data.read_text())
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    rows = []
    t0 = time.time()
    for i, item in enumerate(data, 1):
        st = time.time()
        row = eval_one(
            item,
            args.method,
            args.top_k_chunks,
            args.chunk_size,
            args.overlap,
            args.rank_by_session,
            args.preference_facts,
            args.memory_atoms,
            args.retrieval_k,
            args.openai_vector_rerank,
            args.openai_fusion,
            args.token_native_rerank,
            args.token_native_fusion,
            args.token_chain_expand,
            args.memory_atom_fusion,
            episode_score=args.episode_score,
            episode_score_fusion=args.episode_score_fusion,
            structural_rerank=args.structural_rerank,
            query_auto=args.query_auto,
            evidence_atom_rerank=args.evidence_atom_rerank,
            score_pool=args.score_pool,
            recency_weight=args.recency_weight,
            coverage_rerank=args.coverage_rerank,
            targeted_expansion=args.targeted_expansion,
            temporal_date_rerank=args.temporal_date_rerank,
            relationship_boost=args.relationship_boost,
            conversation_chunks=args.conversation_chunks,
            include_answer_marker=args.include_answer_marker,
            conversation_parent=args.conversation_parent,
            structured_temporal_filters=args.structured_temporal_filters,
            structured_filter_fusion=args.structured_filter_fusion,
            filter_pushdown_threshold=args.filter_pushdown_threshold,
            two_stage_sessions=args.two_stage_sessions,
        )
        row["seconds"] = time.time() - st
        rows.append(row)
        print(
            f"[{i}/{len(data)}] {row['question_id']} {row['question_type']} "
            f"best_rank={row['best_rank']} sessions={row['n_sessions']} {row['seconds']:.2f}s",
            flush=True,
        )

    payload = {
        "benchmark": "LongMemEval_S_cleaned retrieval",
        "method": args.method,
        "limit": args.limit,
        "top_k_chunks": args.top_k_chunks,
        "retrieval_k": args.retrieval_k,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "rank_by_session": args.rank_by_session,
        "preference_facts": args.preference_facts,
        "memory_atoms": args.memory_atoms,
        "openai_vector_rerank": args.openai_vector_rerank,
        "openai_fusion": args.openai_fusion,
        "token_native_rerank": args.token_native_rerank,
        "token_native_fusion": args.token_native_fusion,
        "token_chain_expand": args.token_chain_expand,
        "memory_atom_fusion": args.memory_atom_fusion,
        "episode_score": args.episode_score,
        "episode_score_fusion": args.episode_score_fusion,
        "query_auto": args.query_auto,
        "evidence_atom_rerank": args.evidence_atom_rerank,
        "coverage_rerank": args.coverage_rerank,
        "targeted_expansion": args.targeted_expansion,
        "two_stage_sessions": args.two_stage_sessions,
        "temporal_date_rerank": args.temporal_date_rerank,
        "relationship_boost": args.relationship_boost,
        "structured_temporal_filters": args.structured_temporal_filters,
        "structured_filter_fusion": args.structured_filter_fusion,
        "filter_pushdown_threshold": args.filter_pushdown_threshold,
        "conversation_chunks": args.conversation_chunks,
        "conversation_parent": args.conversation_parent,
        "include_answer_marker": args.include_answer_marker,
        "elapsed_seconds": time.time() - t0,
        "summary": summarize(rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
