#!/usr/bin/env python3
"""Small domain-agnostic eval for AI-agent memory retrieval behaviors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from contextfit.retrieval.engine import RetrievalEngine
from contextfit.retrieval.memory_atoms import atom_type_priors, augment_query_for_memory_atoms, episode_context_text, episode_relevance_score, extract_memory_atoms, query_memory_intents
from contextfit.retrieval.query_router import describe_route, route_query


def session_to_text(session_id: str, date: str, turns: list[dict[str, Any]]) -> str:
    lines = [f"Session ID: {session_id}", f"Date: {date}", ""]
    for i, turn in enumerate(turns, 1):
        lines.append(f"Turn {i} ({turn.get('role', 'unknown')}): {turn.get('content', '')}")
    return "\n".join(lines)


def unique_sessions(chunks) -> list[str]:
    out = []
    seen = set()
    for chunk in chunks:
        sid = chunk.metadata.get("session_id") or chunk.metadata.get("source")
        if sid and sid not in seen:
            seen.add(sid)
            out.append(str(sid))
    return out


def atom_sessions(chunks, scores_: list[float], query: str) -> list[str]:
    priors = atom_type_priors(query)
    by_session: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for rank, (chunk, score) in enumerate(zip(chunks, scores_, strict=False), 1):
        sid = chunk.metadata.get("session_id") or chunk.metadata.get("source")
        if not sid:
            continue
        atom_types = str(chunk.metadata.get("atom_types", "")).split(",")
        prior = max([priors.get(t, 1.0) for t in atom_types if t] or [1.0])
        ranked_score = prior * (float(score) + 1.0 / (20 + rank))
        sid = str(sid)
        if sid not in by_session or ranked_score > by_session[sid]:
            by_session[sid] = ranked_score
            best_rank[sid] = rank
    return sorted(by_session, key=lambda sid: (-by_session[sid], best_rank[sid]))


def confidence_fuse(session_chunks, session_scores: list[float], atom_chunks, atom_scores: list[float], query: str, top_k: int) -> list[str]:
    """Promote atom hits only when query intent and atom type align strongly."""
    intents = query_memory_intents(query)
    session_rank = unique_sessions(session_chunks)
    atom_rank = atom_sessions(atom_chunks, atom_scores, query)

    session_pos = {sid: i for i, sid in enumerate(session_rank, 1)}
    scores: dict[str, float] = {sid: 1.0 / (20 + i) for i, sid in enumerate(session_rank, 1)}
    best: dict[str, int] = {sid: i for i, sid in enumerate(session_rank, 1)}

    for arank, sid in enumerate(atom_rank, 1):
        # Find atom types for this session. Any aligned type is enough to treat
        # the atom evidence as high confidence.
        types = set()
        for chunk in atom_chunks:
            csid = chunk.metadata.get("session_id") or chunk.metadata.get("source")
            if str(csid) == sid:
                types.update(t for t in str(chunk.metadata.get("atom_types", "")).split(",") if t)
        aligned = bool(types & intents) if intents else False
        if not aligned:
            continue
        # Conservative promotion: atom-only sessions can enter, but strongly;
        # aligned atoms mostly break ties / lift one or two places.
        atom_bonus = 0.9 / (20 + arank)
        baseline_floor = 0.0 if sid not in session_pos else 0.25 / (20 + session_pos[sid])
        scores[sid] = scores.get(sid, 0.0) + atom_bonus + baseline_floor
        best[sid] = min(best.get(sid, 10**9), arank)

    return sorted(scores, key=lambda sid: (-scores[sid], best.get(sid, 10**9)))[:top_k]


def _embed_cache_path(text: str, model: str, cache_dir: Path) -> Path:
    h = hashlib.sha256((model + "\0" + text).encode()).hexdigest()
    return cache_dir / f"{h}.json"


def embed_texts(texts: list[str], model: str, cache_dir: Path) -> list[list[float]]:
    """Embed texts with caching. Provider detected from model prefix."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[list[float] | None] = [None] * len(texts)
    missing: list[tuple[int, str, Path]] = []
    for i, text in enumerate(texts):
        p = _embed_cache_path(text, model, cache_dir)
        if p.exists():
            out[i] = json.loads(p.read_text())
        else:
            missing.append((i, text, p))

    if not missing:
        return out  # type: ignore[return-value]

    if model.startswith("voyage"):
        _embed_voyage(missing, model)
    elif model.startswith("embed-") or model.startswith("cohere"):
        _embed_cohere(missing, model)
    else:
        _embed_openai(missing, model)

    # Re-read from cache after writing
    for i, _, p in missing:
        out[i] = json.loads(p.read_text())
    return out  # type: ignore[return-value]


def _embed_openai(missing: list[tuple[int, str, Path]], model: str) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY required for openai_vector mode")
    for start in range(0, len(missing), 64):
        batch = missing[start: start + 64]
        payload = json.dumps({"model": model, "input": [t for _, t, _ in batch]}).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        for (i, _, p), item in zip(batch, data["data"], strict=True):
            p.write_text(json.dumps(item["embedding"], separators=(",", ":")))


def _embed_voyage(missing: list[tuple[int, str, Path]], model: str) -> None:
    import time as _time
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY required for voyage vector mode")
    for start in range(0, len(missing), 8):  # smaller batches to avoid rate limits
        batch = missing[start: start + 8]
        payload = json.dumps({
            "model": model,
            "input": [t for _, t, _ in batch],
            "input_type": "document",
        }).encode()
        for attempt in range(6):
            try:
                req = urllib.request.Request(
                    "https://api.voyageai.com/v1/embeddings",
                    data=payload,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = min(60, 4 ** attempt)  # 1, 4, 16, 60, 60, 60s
                    _time.sleep(wait)
                elif attempt == 5:
                    raise
                else:
                    _time.sleep(2 ** attempt)
            except Exception:
                if attempt == 5:
                    raise
                _time.sleep(2 ** attempt)
        for (i, _, p), item in zip(batch, data["data"], strict=True):
            p.write_text(json.dumps(item["embedding"], separators=(",", ":")))
        _time.sleep(1.0)  # 1s between batches to stay under rate limit


def _embed_cohere(missing: list[tuple[int, str, Path]], model: str) -> None:
    import time as _time
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY required for cohere vector mode")
    for start in range(0, len(missing), 16):  # smaller batches
        batch = missing[start: start + 16]
        payload = json.dumps({
            "model": model,
            "texts": [t for _, t, _ in batch],
            "input_type": "search_document",
            "embedding_types": ["float"],
        }).encode()
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    "https://api.cohere.com/v2/embed",
                    data=payload,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 2 ** attempt
                _time.sleep(wait)
        embeddings = data["embeddings"]["float"]
        for (i, _, p), emb in zip(batch, embeddings, strict=True):
            p.write_text(json.dumps(emb, separators=(",", ":")))
        _time.sleep(0.1)  # gentle pacing


def vector_rank_sessions(query: str, sessions: list[dict[str, Any]], model: str, cache_dir: Path) -> list[str]:
    """Rank sessions by cosine similarity to query embedding."""
    session_texts = [session_to_text(s["session_id"], s["date"], s["turns"]) for s in sessions]
    all_texts = [query] + [t[:24000] for t in session_texts]
    embeddings = embed_texts(all_texts, model, cache_dir)
    q = np.array(embeddings[0], dtype=np.float32)
    docs = np.array(embeddings[1:], dtype=np.float32)
    q /= max(float(np.linalg.norm(q)), 1e-9)
    docs /= np.maximum(np.linalg.norm(docs, axis=1, keepdims=True), 1e-9)
    sims = docs @ q
    order = np.argsort(-sims)
    return [sessions[int(i)]["session_id"] for i in order]


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking in rankings:
        for rank, sid in enumerate(ranking, 1):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank)
            best[sid] = min(best.get(sid, rank), rank)
    return sorted(scores, key=lambda sid: (-scores[sid], best[sid]))


def eval_mem0(item: dict[str, Any], query: str, top_k: int) -> tuple[list[str], float, float]:
    """Run Mem0 (LLM extraction + OpenAI embeddings) on a single eval case.

    Returns (retrieved_session_ids, ingest_ms, query_ms).
    """
    import tempfile
    try:
        from mem0 import Memory
    except ImportError:
        raise RuntimeError("mem0ai not installed: run: python -m pip install mem0ai qdrant-client")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY required for mem0 mode")

    with tempfile.TemporaryDirectory(prefix="cf-mem0-") as tmpdir:
        config = {
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-4o-mini", "api_key": api_key},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": "text-embedding-3-small", "api_key": api_key},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": f"eval_{item['id']}",
                    "path": tmpdir,
                    "on_disk": False,
                },
            },
            "history_db_path": f"{tmpdir}/mem0_history.db",
            "version": "v1.1",
        }
        m = Memory.from_config(config)

        import time as _time
        # Ingest phase — LLM extraction + embedding per session
        t_ingest_start = _time.perf_counter()
        for session in item["sessions"]:
            sid = session["session_id"]
            text = session_to_text(sid, session["date"], session["turns"])
            try:
                m.add(text, user_id="eval_user", metadata={"session_id": sid})
            except Exception:
                pass
        ingest_ms = (_time.perf_counter() - t_ingest_start) * 1000

        # Query phase
        t_query_start = _time.perf_counter()
        try:
            results = m.search(query, filters={"user_id": "eval_user"}, limit=top_k * 3)
        except Exception:
            return [], ingest_ms, 0.0
        query_ms = (_time.perf_counter() - t_query_start) * 1000

        seen: set[str] = set()
        retrieved: list[str] = []
        entries = results.get("results", []) if isinstance(results, dict) else results
        for r in entries:
            meta = r.get("metadata") or {}
            sid = meta.get("session_id")
            if sid and sid not in seen:
                seen.add(sid)
                retrieved.append(sid)
            if len(retrieved) >= top_k:
                break
        return retrieved, ingest_ms, query_ms


WORD_RE_RERANK = re.compile(r"[a-z0-9][a-z0-9'_-]*")
RERANK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "him", "his",
    "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "our", "she", "so",
    "that", "the", "their", "them", "they", "this", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "you", "your", "should", "help",
    "please", "about", "something", "anything", "someone", "tonight", "today", "week", "weekend",
}

DECISION_Q_RE = re.compile(r"\b(decide|decided|choose|chose|choice|pick|picked|select|selected|which .+ did|what did .+ decide)\b", re.I)
DECISION_E_RE = re.compile(r"\b(i|we)\s+(decided|chose|picked|selected|went with|will use|accepted)\b|\bdecided to\b", re.I)
PREF_Q_RE = re.compile(r"\b(recommend|suggest|would i like|do you think i would like|type of|kind of|favorite|prefer)\b", re.I)
PREF_E_RE = re.compile(r"\b(i|we)\s+(love|like|prefer|enjoy|hate|dislike|avoid)|favorite|go-?to|fan of|into\b", re.I)
GOAL_Q_RE = re.compile(r"\b(focus|progress|next step|how should|structure|plan|automate|work on|adjust|improve)\b", re.I)
GOAL_E_RE = re.compile(r"\b(i|we)\s+(want|need|hope|plan|aim|trying|looking|working|writing)|goal|trying to\b", re.I)
CONSTRAINT_Q_RE = re.compile(r"\b(should i|what should|serve|bring|order|make|budget|constraint|allergy|deadline|quick|cheap)\b", re.I)
CONSTRAINT_E_RE = re.compile(r"\b(can'?t|cannot|must|need to|have to|budget|deadline|allergy|limit|only have|trying to keep)\b", re.I)
TEMPORAL_Q_RE = re.compile(r"\b(current|currently|now|latest|these days|switched|changed|new|no longer)\b", re.I)
TEMPORAL_E_RE = re.compile(r"\b(now|currently|recently|switched|changed|updated|no longer|used to|started|stopped|quit|latest|these days)\b", re.I)
OPEN_Q_RE = re.compile(r"\b(outstanding|pending|follow up|supposed to|need to do|left undone|schedule|book|todo|to-do)\b", re.I)
OPEN_E_RE = re.compile(r"\b(todo|to-do|remind me|follow up|need to|haven'?t|still need|supposed to|deadline|schedule|book|renew|send|finish)\b", re.I)
ENTITY_Q_RE = re.compile(r"\b(friend|wife|husband|son|daughter|client|dog|cat|team|partner|parent|child|colleague|boss|coworker|sibling|sister|brother|spouse|relative|neighbor|mentor|manager)\b", re.I)
_EPISODIC_INTENT_RE = re.compile(
    r"\b(enjoy|enjoyed|like|likes|love|loves|gift|surprise|recommend|suggest|interest|hobby|passion|appreciate|fancy|into|outing|activity|workshop|class)\b",
    re.I,
)
PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
_PROPER_STOPS = {
    "What","How","Why","When","Where","Which","Who","Can","Could","Should","Would","Will",
    "Do","Does","Did","Is","Are","Was","Were","The","My","Your","Our","Their","His","Her",
    "This","That","These","Those","They","Have","Has","Had","Been","Just","Some","Any",
}


def _extract_proper_nouns(text: str) -> set[str]:
    """Extract lowercased proper-noun candidates from text."""
    return {w.lower() for w in PROPER_NOUN_RE.findall(text) if w not in _PROPER_STOPS}


def _rerank_words(text: str) -> list[str]:
    return [w for w in WORD_RE_RERANK.findall(str(text).lower()) if len(w) > 2 and w not in RERANK_STOPWORDS]


def _rerank_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / ((len(a) * len(b)) ** 0.5)


def _idf_lexical(q_words: list[str], sess_words: list[str], idf: dict[str, float]) -> float:
    """IDF-cosine style lexical overlap weighted by IDF scores."""
    if not q_words or not sess_words:
        return 0.0
    q_set = set(q_words)
    s_set = set(sess_words)
    shared = q_set & s_set
    if not shared:
        return 0.0
    numerator = sum(idf.get(w, 1.0) for w in shared)
    denom = (sum(idf.get(w, 1.0) for w in q_set) * sum(idf.get(w, 1.0) for w in s_set)) ** 0.5
    if denom == 0.0:
        return 0.0
    return numerator / denom


# --- Question-type slot matching patterns ---
_Q_WHO_RE = re.compile(r"\b(who|whose|someone)\b", re.I)
_E_WHO_RE = re.compile(
    r"\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})*\s+(?:is|was|told|said|recommended|suggested|asked))"
    r"|\b(he|she|they)\s+(told|said|asked|recommended|suggested|explained|mentioned)\b",
    re.I,
)
_Q_WHEN_RE = re.compile(r"\b(when|how long|what date|what time)\b", re.I)
_E_WHEN_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"|\b(last\s+(week|month|year)|next\s+(week|month|year)|yesterday|tomorrow)"
    r"|\b(\d{4})|\b(\d{1,2}/\d{1,2})",
    re.I,
)
_Q_WHERE_RE = re.compile(r"\b(where|what place|location)\b", re.I)
_E_WHERE_RE = re.compile(
    r"\b(at|in|near)\s+[A-Z][a-z]"
    r"|\b(address|location|city|town|street|avenue|boulevard|place|venue|restaurant|hotel|office|store)\b",
    re.I,
)
_Q_HOW_RE = re.compile(r"\b(how\s+(do|did|should|can|to)|steps|process|procedure)\b", re.I)
_E_HOW_RE = re.compile(
    r"\b(step\s*\d+|first[,.]|then[,.]|next[,.]|finally[,.]|\d+\.\s+[A-Z])"
    r"|\b(follow\s+these|instructions|procedure|process)\b",
    re.I,
)
_Q_RECOMMEND_RE = re.compile(r"\b(recommend|suggest|should i get|would i like|would you recommend)\b", re.I)
_E_RECOMMEND_RE = re.compile(
    r"\b(would\s+(love|enjoy|like)|perfect\s+for|you'?d\s+like|you'?ll\s+love|might\s+enjoy)"
    r"|\b(i\s+recommend|we\s+recommend|highly\s+recommend)\b",
    re.I,
)


def _slot_match(query: str, text: str) -> float:
    """Detect WH-question type and score matching answer-evidence in text.
    Returns 0.0-2.0 (capped).
    """
    score = 0.0
    if _Q_WHO_RE.search(query) and _E_WHO_RE.search(text):
        score += 1.0
    if _Q_WHEN_RE.search(query) and _E_WHEN_RE.search(text):
        score += 1.0
    if _Q_WHERE_RE.search(query) and _E_WHERE_RE.search(text):
        score += 1.0
    if _Q_HOW_RE.search(query) and _E_HOW_RE.search(text):
        score += 1.0
    if _Q_RECOMMEND_RE.search(query) and _E_RECOMMEND_RE.search(text):
        score += 1.2  # recommendation queries have a harder gap to close
    return min(score, 2.0)


def _window_density(query_words: set[str], text: str, window_size: int = 150) -> float:
    """Sliding window density: fraction of window_size words that are in query_words.
    Returns max density across all windows in [0.0, 1.0].
    """
    words = _rerank_words(text)
    if not words or not query_words:
        return 0.0
    if len(words) <= window_size:
        hits = sum(1 for w in words if w in query_words)
        return hits / window_size
    max_hits = 0
    # Count hits in first window
    hits = sum(1 for w in words[:window_size] if w in query_words)
    max_hits = hits
    for i in range(1, len(words) - window_size + 1):
        if words[i - 1] in query_words:
            hits -= 1
        if words[i + window_size - 1] in query_words:
            hits += 1
        if hits > max_hits:
            max_hits = hits
    return max_hits / window_size


def _user_session_text(session: dict[str, Any]) -> str:
    return "\n".join(str(t.get("content", "")) for t in session.get("turns", []) if str(t.get("role", "")).lower() == "user")


def scored_episode_sessions(item: dict[str, Any], query: str, top_k: int) -> list[str]:
    scored = []
    for rank, session in enumerate(item["sessions"], 1):
        score = episode_relevance_score(query, session["turns"])
        scored.append((score, -rank, session["session_id"]))
    scored.sort(reverse=True)
    return [sid for score, _rank, sid in scored if score > 0][:top_k]


def token_native_rerank_sessions(
    item: dict[str, Any],
    query: str,
    bm25_sessions: list[str],
    top_k: int,
    use_idf: bool = False,
    use_slot: bool = False,
    use_window: bool = False,
) -> list[str]:
    """Rerank all candidate sessions using free token-native structural features.

    This is a benchmark harness for the production reranker idea.  It does not
    use embeddings or benchmark labels; features are domain-neutral memory
    primitives: lexical overlap, BM25 rank, episode score, decision/preference/
    goal/constraint/temporal/open-loop markers, named entity overlap, and
    generic-distractor penalty.
    """
    q_words = set(_rerank_words(query))
    q_words_list = _rerank_words(query)
    q_entities = _extract_proper_nouns(query)  # e.g. {"maya", "stripe"}
    bm25_rank = {sid: i for i, sid in enumerate(bm25_sessions, 1)}
    ep_scored = {s["session_id"]: episode_relevance_score(query, s["turns"]) for s in item["sessions"]}
    max_ep = max(ep_scored.values() or [1.0]) or 1.0

    # Build IDF table across all sessions if IDF feature is requested
    idf: dict[str, float] = {}
    if use_idf:
        from math import log as _log
        N = len(item["sessions"])
        df: dict[str, int] = {}
        for session in item["sessions"]:
            text = _user_session_text(session)
            for w in set(_rerank_words(text)):
                df[w] = df.get(w, 0) + 1
        idf = {w: _log((N + 1) / (d + 1)) + 1.0 for w, d in df.items()}

    rows = []
    for pos, session in enumerate(item["sessions"], 1):
        sid = session["session_id"]
        text = _user_session_text(session)
        words = set(_rerank_words(text))
        words_list = _rerank_words(text)
        # Also extract proper nouns from the full session text (all turns).
        full_text = " ".join(str(t.get("content", "")) for t in session.get("turns", []))
        session_entities = _extract_proper_nouns(full_text)
        lexical = _rerank_overlap(q_words, words)
        # Reciprocal rank features.
        bm25 = 1.0 / bm25_rank[sid] if sid in bm25_rank else 0.0
        ep = ep_scored.get(sid, 0.0) / max_ep
        # Named entity overlap: query mentions a person/place/brand, session contains it.
        # This is the key signal for episodic inference (e.g. "Maya" in query → session).
        entity_hit = len(q_entities & session_entities) / max(len(q_entities), 1) if q_entities else 0.0
        entity_any = 1.0 if entity_hit > 0 else 0.0
        # Intent/evidence marker alignment.
        decision = 1.0 if DECISION_Q_RE.search(query) and DECISION_E_RE.search(text) else 0.0
        pref = 1.0 if PREF_Q_RE.search(query) and PREF_E_RE.search(text) else 0.0
        goal = 1.0 if GOAL_Q_RE.search(query) and GOAL_E_RE.search(text) else 0.0
        constraint = 1.0 if CONSTRAINT_Q_RE.search(query) and CONSTRAINT_E_RE.search(text) else 0.0
        temporal = 1.0 if TEMPORAL_Q_RE.search(query) and TEMPORAL_E_RE.search(text) else 0.0
        open_loop = 1.0 if OPEN_Q_RE.search(query) and OPEN_E_RE.search(text) else 0.0
        entity_role = 1.0 if ENTITY_Q_RE.search(query) and lexical > 0 else 0.0
        # Penalize generic advice sessions that have weak personal-memory markers.
        marker_sum = decision + pref + goal + constraint + temporal + open_loop + entity_any
        generic_penalty = 0.15 if marker_sum == 0 and lexical < 0.16 else 0.0

        # --- Feature 1: IDF-boosted lexical ---
        idf_lex = _idf_lexical(q_words_list, words_list, idf) if use_idf else 0.0

        # --- Feature 2: Question-type slot matching ---
        slot = _slot_match(query, full_text) if use_slot else 0.0

        # --- Feature 3: Evidence window density ---
        density = _window_density(q_words, full_text) if use_window else 0.0

        score = (
            1.05 * bm25
            + 0.85 * ep
            + 1.10 * lexical
            + 0.95 * decision
            + 0.75 * pref
            + 0.70 * goal
            + 0.70 * constraint
            + 0.80 * temporal
            + 0.85 * open_loop
            + 0.25 * entity_role
            + 1.20 * entity_any   # strong boost: person/place named in query appears in session
            + 0.60 * entity_hit   # partial credit for multi-entity partial matches
            - generic_penalty
            + 0.40 * idf_lex      # IDF-weighted lexical (Feature 1)
            + 0.35 * slot         # question-type slot match (Feature 2)
            + 0.45 * density      # evidence window density (Feature 3)
        )
        rows.append((score, -pos, sid))
    rows.sort(reverse=True)
    return [sid for score, _pos, sid in rows[:top_k] if score > 0]


def eval_one(
    item: dict[str, Any],
    mode: str,
    method: str,
    top_k: int,
    openai_model: str = "text-embedding-3-small",
    voyage_model: str = "voyage-3",
    cohere_model: str = "embed-english-v3.0",
    embed_cache: Path = Path("benchmarks/cache/openai_embeddings"),
    evidence_certificate_rerank: bool = False,
    typed_rescue: bool = False,
) -> dict[str, Any]:
    import time as _time
    tmp = Path(tempfile.mkdtemp(prefix="cf-agent-memory-"))
    try:
        engine = RetrievalEngine.create(tmp)
        t_ingest_start = _time.perf_counter()
        for session in item["sessions"]:
            sid = session["session_id"]
            date = session["date"]
            meta = {"session_id": sid, "date": date, "kind": "session", "eval_id": item["id"]}
            engine.ingest_text(
                session_to_text(sid, date, session["turns"]),
                chunk_size=4096,
                overlap=0,
                metadata=meta,
                update_indexes=True,
            )
            if mode in {"atoms", "fusion", "episode", "episode_fusion"}:
                atoms = extract_memory_atoms(session["turns"], source_id=sid, source_date=date)
                if atoms:
                    atom_meta = dict(meta)
                    atom_meta["kind"] = "memory_atoms"
                    atom_meta["atom_types"] = ",".join(sorted({a.atom_type for a in atoms}))
                    engine.ingest_text(
                        "\n\n".join(a.to_index_text() for a in atoms),
                        chunk_size=4096,
                        overlap=0,
                        metadata=atom_meta,
                        update_indexes=True,
                    )
            if mode in {"episode", "episode_fusion"}:
                episode_text = episode_context_text(session["turns"], source_id=sid, source_date=date)
                if episode_text:
                    episode_meta = dict(meta)
                    episode_meta["kind"] = "episode_context"
                    engine.ingest_text(
                        episode_text,
                        chunk_size=4096,
                        overlap=0,
                        metadata=episode_meta,
                        update_indexes=True,
                    )

        ingest_ms = (_time.perf_counter() - t_ingest_start) * 1000
        query = item["question"]
        t_query_start = _time.perf_counter()
        if mode == "baseline":
            result = engine.query(query, top_k=top_k, method=method, max_tokens=100_000)
            retrieved = unique_sessions(result.chunks)
        elif mode == "atoms":
            result = engine.query(
                augment_query_for_memory_atoms(query),
                top_k=top_k,
                method=method,
                max_tokens=100_000,
                filter_field=("kind", "memory_atoms"),
            )
            retrieved = atom_sessions(result.chunks, result.scores, query)
        elif mode == "episode_score":
            retrieved = scored_episode_sessions(item, query, top_k)
        elif mode == "episode_score_fusion":
            session_result = engine.query(
                query,
                top_k=top_k,
                method=method,
                max_tokens=100_000,
                filter_field=("kind", "session"),
            )
            retrieved = rrf([unique_sessions(session_result.chunks), scored_episode_sessions(item, query, top_k)])[:top_k]
        elif mode == "engine_episode_score":
            ranked = engine.rank_sessions_by_episode_score(query, top_k=top_k)
            retrieved = [r["session_id"] for r in ranked if r["score"] > 0]
        elif mode == "openai_vector":
            retrieved = vector_rank_sessions(query, item["sessions"], openai_model, embed_cache)[:top_k]
        elif mode == "openai_fusion":
            vec_ranked = vector_rank_sessions(query, item["sessions"], openai_model, embed_cache)
            session_result = engine.query(query, top_k=top_k, method=method, max_tokens=100_000, filter_field=("kind", "session"))
            cf_ranked = unique_sessions(session_result.chunks)
            retrieved = rrf([cf_ranked, vec_ranked])[:top_k]
        elif mode == "voyage_vector":
            retrieved = vector_rank_sessions(query, item["sessions"], voyage_model, embed_cache)[:top_k]
        elif mode == "cohere_vector":
            retrieved = vector_rank_sessions(query, item["sessions"], cohere_model, embed_cache)[:top_k]
        elif mode == "free_rerank":
            result = engine.query(query, top_k=50, method=method, max_tokens=100_000)
            bm25_sessions = unique_sessions(result.chunks)
            retrieved = token_native_rerank_sessions(item, query, bm25_sessions, top_k)
        elif mode == "free_rerank_idf":
            result = engine.query(query, top_k=50, method=method, max_tokens=100_000)
            bm25_sessions = unique_sessions(result.chunks)
            retrieved = token_native_rerank_sessions(item, query, bm25_sessions, top_k, use_idf=True)
        elif mode == "free_rerank_slot":
            result = engine.query(query, top_k=50, method=method, max_tokens=100_000)
            bm25_sessions = unique_sessions(result.chunks)
            retrieved = token_native_rerank_sessions(item, query, bm25_sessions, top_k, use_slot=True)
        elif mode == "free_rerank_window":
            result = engine.query(query, top_k=50, method=method, max_tokens=100_000)
            bm25_sessions = unique_sessions(result.chunks)
            retrieved = token_native_rerank_sessions(item, query, bm25_sessions, top_k, use_window=True)
        elif mode == "free_rerank_all":
            result = engine.query(query, top_k=50, method=method, max_tokens=100_000)
            bm25_sessions = unique_sessions(result.chunks)
            retrieved = token_native_rerank_sessions(item, query, bm25_sessions, top_k, use_idf=True, use_slot=True, use_window=True)
        elif mode == "free_rerank_fusion":
            # Adaptive fusion: heavy episode_score weight for episodic inference queries,
            # light blend otherwise.  Episodic inference = query about someone else's
            # interests/preferences (entity relationship term + recommendation language).
            result = engine.query(query, top_k=50, method=method, max_tokens=100_000)
            bm25_sessions = unique_sessions(result.chunks)
            reranked = token_native_rerank_sessions(item, query, bm25_sessions, top_k * 2)
            episode = scored_episode_sessions(item, query, top_k * 2)
            _has_entity = bool(ENTITY_Q_RE.search(query))
            _has_proper = bool(_extract_proper_nouns(query))  # named persons e.g. Maya
            _has_episodic_intent = bool(_EPISODIC_INTENT_RE.search(query))
            _episodic_q = (_has_entity or _has_proper) and _has_episodic_intent
            if _episodic_q:
                # Episode-dominant for third-party interest queries (2:1 episode weight)
                retrieved = rrf([episode, reranked, episode])[:top_k]
            else:
                retrieved = rrf([reranked, episode])[:top_k]
        elif mode == "mem0":
            retrieved, ingest_ms, query_ms = eval_mem0(item, query, top_k)
        elif mode == "auto":
            result = engine.query_auto(
                query,
                top_k=top_k,
                retrieval_k=50,
                method=method,
                max_tokens=100_000,
                evidence_certificate_rerank=evidence_certificate_rerank,
                typed_rescue=typed_rescue,
            )
            retrieved = result["session_ids"]
        elif mode == "two_stage":
            result = engine.query_two_stage_sessions(
                query,
                top_k=top_k,
                broad_k=50,
                precise_k=6,
                method=method,
                max_tokens=100_000,
            )
            retrieved = result["session_ids"]
        elif mode == "episode":

            result = engine.query(
                augment_query_for_memory_atoms(query),
                top_k=top_k,
                method=method,
                max_tokens=100_000,
                filter_field=("kind", "episode_context"),
            )
            retrieved = unique_sessions(result.chunks)
        elif mode == "episode_fusion":
            session_result = engine.query(
                query,
                top_k=top_k,
                method=method,
                max_tokens=100_000,
                filter_field=("kind", "session"),
            )
            episode_result = engine.query(
                augment_query_for_memory_atoms(query),
                top_k=top_k,
                method=method,
                max_tokens=100_000,
                filter_field=("kind", "episode_context"),
            )
            retrieved = rrf([unique_sessions(session_result.chunks), unique_sessions(episode_result.chunks)])[:top_k]
        elif mode == "fusion":
            session_result = engine.query(
                query,
                top_k=top_k,
                method=method,
                max_tokens=100_000,
                filter_field=("kind", "session"),
            )
            atom_result = engine.query(
                augment_query_for_memory_atoms(query),
                top_k=top_k,
                method=method,
                max_tokens=100_000,
                filter_field=("kind", "memory_atoms"),
            )
            retrieved = confidence_fuse(
                session_result.chunks,
                session_result.scores,
                atom_result.chunks,
                atom_result.scores,
                query,
                top_k,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

        if mode != "mem0":
            query_ms = (_time.perf_counter() - t_query_start) * 1000

        gold = set(item["answer_session_ids"])
        ranks = [retrieved.index(g) + 1 for g in gold if g in retrieved]
        n_sessions = len(item["sessions"])
        row = {
            "id": item["id"],
            "behavior": item["behavior"],
            "question": item["question"],
            "gold_sessions": sorted(gold),
            "retrieved_sessions": retrieved,
            "best_rank": min(ranks) if ranks else None,
            "n_sessions": n_sessions,
            "ingest_ms": round(ingest_ms, 1),
            "ingest_ms_per_session": round(ingest_ms / n_sessions, 1) if n_sessions else 0.0,
            "query_ms": round(query_ms, 1),
        }
        if mode == "auto":
            route = route_query(query)
            row["route"] = describe_route(route)
            certificates = result.get("details", {}).get("evidence_certificates") if "result" in locals() else None
            if certificates:
                row["evidence_certificates"] = certificates
        return row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def summarize(rows: list[dict[str, Any]], ks=(1, 3, 5)) -> dict[str, Any]:
    def metrics(sub):
        n = len(sub)
        out = {"n": n}
        for k in ks:
            out[f"recall@{k}"] = sum(r["best_rank"] is not None and r["best_rank"] <= k for r in sub) / n if n else 0
        out["mrr"] = sum((1 / r["best_rank"]) if r["best_rank"] else 0 for r in sub) / n if n else 0
        # Timing averages (skip rows that predate timing instrumentation)
        timed = [r for r in sub if "ingest_ms" in r]
        if timed:
            out["avg_ingest_ms"] = round(sum(r["ingest_ms"] for r in timed) / len(timed), 1)
            out["avg_ingest_ms_per_session"] = round(sum(r["ingest_ms_per_session"] for r in timed) / len(timed), 1)
            out["avg_query_ms"] = round(sum(r["query_ms"] for r in timed) / len(timed), 1)
        return out

    by = defaultdict(list)
    for r in rows:
        by[r["behavior"]].append(r)
    return {"overall": metrics(rows), "by_behavior": {k: metrics(v) for k, v in sorted(by.items())}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path, default=Path("benchmarks/data/agent_memory_eval.json"))
    ap.add_argument("--mode", choices=["baseline", "atoms", "fusion", "episode", "episode_fusion", "episode_score", "episode_score_fusion", "engine_episode_score", "openai_vector", "openai_fusion", "voyage_vector", "cohere_vector", "free_rerank", "free_rerank_idf", "free_rerank_slot", "free_rerank_window", "free_rerank_all", "free_rerank_fusion", "mem0", "auto", "two_stage"], default="baseline")
    ap.add_argument("--openai-model", default="text-embedding-3-small")
    ap.add_argument("--voyage-model", default="voyage-3")
    ap.add_argument("--cohere-model", default="embed-english-v3.0")
    ap.add_argument("--embed-cache", type=Path, default=Path("benchmarks/cache/openai_embeddings"))
    ap.add_argument("--method", choices=["exact", "bm25", "hybrid", "hybrid_rrf"], default="hybrid")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--evidence-certificate-rerank", action="store_true", help="enable production evidence-certificate rerank for auto mode")
    ap.add_argument("--typed-rescue", action="store_true", help="enable v5 typed rescue after evidence-certificate rerank for auto mode")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    items = json.loads(args.data.read_text())
    rows = []
    for item in items:
        row = eval_one(
            item,
            args.mode,
            args.method,
            args.top_k,
            openai_model=args.openai_model,
            voyage_model=args.voyage_model,
            cohere_model=args.cohere_model,
            embed_cache=args.embed_cache,
            evidence_certificate_rerank=args.evidence_certificate_rerank,
            typed_rescue=args.typed_rescue,
        )
        rows.append(row)
        print(f"{row['id']} {row['behavior']} best_rank={row['best_rank']} retrieved={row['retrieved_sessions']}")

    result = {
        "mode": args.mode,
        "method": args.method,
        "top_k": args.top_k,
        "evidence_certificate_rerank": args.evidence_certificate_rerank,
        "typed_rescue": args.typed_rescue,
        "summary": summarize(rows),
        "rows": rows,
    }
    print(json.dumps(result["summary"], indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
