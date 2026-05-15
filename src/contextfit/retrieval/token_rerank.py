"""Token-native structural reranking.

This module deliberately avoids embedding models.  It reranks retrieved chunks by
looking at token/phrase structure, local evidence windows, and lightweight
language markers such as preference and temporal/update phrases.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from contextfit.core.chunk import Chunk
from contextfit.core.tokenizer import Tokenizer


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers",
    "him", "his", "how", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "ours", "she", "so", "that", "the", "their", "theirs", "them", "then",
    "there", "these", "they", "this", "those", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "question", "date", "session", "turn", "user", "assistant",
}

PREFERENCE_QUERY_RE = re.compile(
    r"\b(favou?rite|prefer(?:red|ence)?|like|love|enjoy|go-?to|usual|normally|rather|pick|choose|choice|taste)\b",
    re.I,
)
PREFERENCE_EVIDENCE_RE = re.compile(
    r"\b("
    r"favou?rite|prefer(?:red|s|ence)?|like|likes|liked|love|loves|loved|enjoy|enjoys|go-?to|"
    r"usually|normally|always|tend to|rather|pick|choose|choice|works best|into|fan of"
    r")\b",
    re.I,
)
AVERSION_RE = re.compile(r"\b(hate|hates|dislike|dislikes|avoid|avoids|can't stand|cannot stand|not a fan)\b", re.I)
TEMPORAL_QUERY_RE = re.compile(
    r"\b(current|currently|now|latest|recent|recently|last|new|updated?|changed?|before|after|previous(?:ly)?|used to|no longer)\b",
    re.I,
)
PERSONAL_QUERY_RE = re.compile(r"\b(i|me|my|mine|for me|should i|can you recommend|any tips|any advice|what should i)\b", re.I)

TEMPORAL_EVIDENCE_RE = re.compile(
    r"\b(now|currently|these days|recently|latest|last|new|updated?|changed?|switched|instead|actually|used to|no longer|previously)\b",
    re.I,
)

WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*")


@dataclass
class RerankTrace:
    chunk_id: int
    base_score: float
    score: float
    features: dict[str, float] = field(default_factory=dict)


def _question_only(text: str) -> str:
    """Strip benchmark helper prefixes while remaining harmless for normal queries."""
    if "Question:" in text:
        return text.split("Question:", 1)[1].strip()
    return text


def _words(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


def _ngrams(items: list[int] | list[str], n: int) -> set[tuple]:
    if len(items) < n:
        return set()
    return {tuple(items[i : i + n]) for i in range(len(items) - n + 1)}


def _overlap_ratio(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def _user_turn_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if re.search(r"\bTurn\s+\d+\s+\(user\)", line):
            lines.append(line)
    return "\n".join(lines) or text


def _marker_windows(text: str, markers: Iterable[re.Pattern], radius_words: int = 28) -> list[str]:
    words = WORD_RE.findall(text.lower())
    if not words:
        return []
    # Approximate marker locations on the word stream. This is intentionally
    # simple and robust: if a marker phrase appears in a 5-word local string, it
    # contributes a nearby evidence window.
    windows: list[str] = []
    for i in range(len(words)):
        local = " ".join(words[i : i + 5])
        if any(rx.search(local) for rx in markers):
            lo = max(0, i - radius_words)
            hi = min(len(words), i + radius_words + 6)
            windows.append(" ".join(words[lo:hi]))
    return windows[:12]


class TokenNativeReranker:
    """Rerank candidate chunks using token/language structure, no embeddings."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def score_chunk(self, query: str, chunk: Chunk, base_score: float = 0.0) -> RerankTrace:
        q_text = _question_only(query)
        q_words = _words(q_text)
        q_word_set = set(q_words)
        try:
            raw_c_text = self.tokenizer.decode(chunk.tokens)
        except Exception:
            raw_c_text = ""
        # Personal-memory questions should match what the user said before,
        # not generic assistant advice that happens to share topic words.
        c_text = _user_turn_text(raw_c_text) if PERSONAL_QUERY_RE.search(q_text) else raw_c_text
        c_words = _words(c_text)
        c_word_set = set(c_words)

        q_tokens = self.tokenizer.encode(q_text).tolist()
        c_tokens = self.tokenizer.encode(c_text).tolist()

        # Token-native signatures.  These preserve BPE phrase structure rather
        # than collapsing text into vector space.
        q_tok_set = set(q_tokens)
        c_tok_set = set(c_tokens)
        token_overlap = _overlap_ratio(q_tok_set, c_tok_set)
        token_bigram = _overlap_ratio(_ngrams(q_tokens, 2), _ngrams(c_tokens, 2))
        token_trigram = _overlap_ratio(_ngrams(q_tokens, 3), _ngrams(c_tokens, 3))

        word_overlap = _overlap_ratio(q_word_set, c_word_set)
        word_bigram = _overlap_ratio(_ngrams(q_words, 2), _ngrams(c_words, 2))

        # Local answer-likelihood windows: maximum query-term density in any
        # small natural-language window, plus special windows around preference
        # and temporal markers.
        best_window = 0.0
        if q_word_set and c_words:
            width = 56
            step = 16
            for i in range(0, len(c_words), step):
                win = set(c_words[i : i + width])
                if not win:
                    continue
                best_window = max(best_window, len(q_word_set & win) / math.sqrt(len(q_word_set) * len(win)))

        preference_intent = 1.0 if PREFERENCE_QUERY_RE.search(q_text) else 0.0
        temporal_intent = 1.0 if TEMPORAL_QUERY_RE.search(q_text) else 0.0
        preference_marker = 1.0 if PREFERENCE_EVIDENCE_RE.search(c_text) else 0.0
        aversion_marker = 1.0 if AVERSION_RE.search(c_text) else 0.0
        temporal_marker = 1.0 if TEMPORAL_EVIDENCE_RE.search(c_text) else 0.0

        preference_window = 0.0
        if preference_intent:
            for win_text in _marker_windows(c_text, [PREFERENCE_EVIDENCE_RE, AVERSION_RE]):
                win_words = set(_words(win_text))
                preference_window = max(preference_window, _overlap_ratio(q_word_set, win_words))

        temporal_window = 0.0
        if temporal_intent:
            for win_text in _marker_windows(c_text, [TEMPORAL_EVIDENCE_RE]):
                win_words = set(_words(win_text))
                temporal_window = max(temporal_window, _overlap_ratio(q_word_set, win_words))

        # Query/date metadata should not dominate, but if a user asks latest or
        # previous, update/change markers are meaningful language structure.
        features = {
            "base": math.log1p(max(base_score, 0.0)) / 20.0,
            "token_overlap": token_overlap,
            "token_bigram": token_bigram,
            "token_trigram": token_trigram,
            "word_overlap": word_overlap,
            "word_bigram": word_bigram,
            "best_window": best_window,
            "preference": preference_intent * (0.20 * preference_marker + 0.10 * aversion_marker + 1.50 * preference_window),
            "temporal": temporal_intent * (0.20 * temporal_marker + 1.10 * temporal_window),
        }
        score = (
            features["base"]
            + 1.10 * features["token_overlap"]
            + 1.80 * features["token_bigram"]
            + 2.30 * features["token_trigram"]
            + 1.00 * features["word_overlap"]
            + 1.40 * features["word_bigram"]
            + 1.80 * features["best_window"]
            + features["preference"]
            + features["temporal"]
        )
        return RerankTrace(chunk_id=chunk.chunk_id, base_score=base_score, score=score, features=features)

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        base_scores: list[float] | None = None,
    ) -> list[tuple[Chunk, float, RerankTrace]]:
        """Rerank by fusing base retrieval rank with structural token rank.

        The structural score is intentionally a *reranker*, not a replacement
        retrieval model.  Baseline token/BM25 rank carries strong evidence; the
        language-structure pass should promote clear phrase/window matches
        without throwing away that prior.
        """
        base_scores = base_scores or [0.0] * len(chunks)
        scored = [
            (chunk, trace.score, trace)
            for chunk, base in zip(chunks, base_scores, strict=False)
            for trace in [self.score_chunk(query, chunk, base)]
        ]
        structural_order = sorted(scored, key=lambda item: (-item[1], -item[2].base_score, item[0].chunk_id))
        structural_rank = {item[0].chunk_id: rank for rank, item in enumerate(structural_order, start=1)}

        fused: list[tuple[Chunk, float, RerankTrace]] = []
        for base_rank, (chunk, _struct_score, trace) in enumerate(scored, start=1):
            srank = structural_rank[chunk.chunk_id]
            # Weighted reciprocal-rank fusion: preserve the original retrieval
            # prior while letting token-native structure break ties/promote
            # obvious local evidence windows.
            fused_score = 0.70 / (20 + base_rank) + 0.30 / (20 + srank)
            trace.features["base_rank"] = float(base_rank)
            trace.features["structural_rank"] = float(srank)
            trace.features["fused_rank_score"] = fused_score
            fused.append((chunk, fused_score, trace))

        fused.sort(key=lambda item: (-item[1], -item[2].base_score, item[0].chunk_id))
        return fused
