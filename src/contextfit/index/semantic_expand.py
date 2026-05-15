"""
SemanticExpander: Token-native query expansion using two strategies.

Strategy 1 — Corpus co-occurrence (zero dependencies):
    Tokens that appear together in similar chunks are semantically related
    in this domain. PMI-weighted expansion from corpus statistics alone.

Strategy 2 — LLM embedding matrix (index-time API, zero query-time cost):
    Embed each corpus token string once using the target LLM's embedding
    model. Cache vectors on disk. At query time: cosine lookup only (no API).
    This aligns retrieval with the LLM's own semantic space.

Both strategies are token-native and add zero latency at query time after
the index is built.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Token filter: only embed lexically meaningful BPE tokens
# ---------------------------------------------------------------------------

_LEADING_SPACE_ALPHA = re.compile(r"^\s?[A-Za-z][A-Za-z'\-]{1,}$")


def _is_lexical_token(s: str) -> bool:
    """
    Return True for BPE tokens worth embedding.

    Keeps:
      - space-prefixed alphabetic tokens (' rental', ' retirement')
      - plain alphabetic tokens ('rental', 'IRS') min-length 3

    Drops:
      - pure punctuation / digits / mixed garbage
      - very short tokens (<= 2 chars after stripping)
    """
    stripped = s.strip()
    if len(stripped) <= 2:
        return False
    return bool(_LEADING_SPACE_ALPHA.match(s))


# ---------------------------------------------------------------------------
# Strategy 1: Corpus co-occurrence (PMI)
# ---------------------------------------------------------------------------

class CorpusExpander:
    """
    Builds a token→related_tokens map from corpus co-occurrence statistics.

    Uses Positive PMI (PPMI):
        PPMI(t1, t2) = max(0, log( P(t1,t2) / (P(t1) * P(t2)) ))

    Tokens that co-occur more than chance across chunks are semantically
    related *in this corpus* — domain-specific, no external model needed.
    """

    def __init__(self, top_k_neighbors: int = 8):
        self.top_k = top_k_neighbors
        # token_id -> {neighbor_token_id: ppmi_score}
        self._neighbors: dict[int, list[tuple[int, float]]] = {}
        self._built = False

    def build(self, chunk_iter) -> "CorpusExpander":
        """
        Build co-occurrence map from an iterable of Chunk objects.
        Call once after ingest; then call save() to persist.
        """
        cooccur: dict[tuple[int, int], int] = defaultdict(int)
        token_freq: dict[int, int] = defaultdict(int)
        n_chunks = 0

        for chunk in chunk_iter:
            tokens = set(chunk.tokens.tolist() if hasattr(chunk.tokens, 'tolist')
                         else chunk.tokens)
            n_chunks += 1
            for t in tokens:
                token_freq[t] += 1
            # Count co-occurrences (unordered pairs only)
            token_list = sorted(tokens)
            for i, t1 in enumerate(token_list):
                for t2 in token_list[i + 1:]:
                    cooccur[(t1, t2)] += 1

        if n_chunks == 0:
            self._built = True
            return self

        # Compute PPMI for each pair
        ppmi: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for (t1, t2), cocount in cooccur.items():
            if cocount < 2:          # skip rare pairs
                continue
            p_t1 = token_freq[t1] / n_chunks
            p_t2 = token_freq[t2] / n_chunks
            p_joint = cocount / n_chunks
            pmi = math.log(p_joint / (p_t1 * p_t2))
            if pmi <= 0:
                continue
            ppmi[t1].append((t2, pmi))
            ppmi[t2].append((t1, pmi))

        # Keep top-K neighbors per token, sorted by PPMI descending
        for tok, neighbors in ppmi.items():
            neighbors.sort(key=lambda x: -x[1])
            self._neighbors[tok] = neighbors[:self.top_k]

        self._built = True
        return self

    def expand(
        self,
        query_tokens: list[int],
        min_ppmi: float = 0.5,
        max_expansions: int = 12,
    ) -> list[tuple[int, float]]:
        """
        Return (token_id, weight) expansion candidates for query tokens.
        Weights are relative (0–1); original query tokens should be weighted 1.0.
        """
        seen = set(query_tokens)
        candidates: dict[int, float] = {}

        for qt in query_tokens:
            for neighbor, ppmi in self._neighbors.get(qt, []):
                if neighbor in seen:
                    continue
                if ppmi < min_ppmi:
                    continue
                # Normalise PPMI to a 0–1 weight (soft cap at 3.0)
                weight = min(ppmi / 3.0, 1.0) * 0.6   # max 0.6 — originals stay dominant
                candidates[neighbor] = max(candidates.get(neighbor, 0.0), weight)

        ranked = sorted(candidates.items(), key=lambda x: -x[1])
        return ranked[:max_expansions]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        data = {str(k): [[t, s] for t, s in v] for k, v in self._neighbors.items()}
        tmp = path / "corpus_expander.json.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path / "corpus_expander.json")

    @classmethod
    def load(cls, path: Path | str, top_k: int = 8) -> "CorpusExpander":
        path = Path(path)
        exp = cls(top_k_neighbors=top_k)
        fpath = path / "corpus_expander.json"
        if not fpath.exists():
            return exp
        with open(fpath) as f:
            data = json.load(f)
        exp._neighbors = {int(k): [(t, s) for t, s in v] for k, v in data.items()}
        exp._built = True
        return exp

    def stats(self) -> dict:
        return {
            "tokens_with_neighbors": len(self._neighbors),
            "avg_neighbors": (
                sum(len(v) for v in self._neighbors.values()) / len(self._neighbors)
                if self._neighbors else 0
            ),
        }


# ---------------------------------------------------------------------------
# Strategy 2: LLM embedding matrix (index-time API, zero query-time cost)
# ---------------------------------------------------------------------------

class EmbeddingExpander:
    """
    Expands query tokens using pre-computed token string embeddings.

    Index time (one API call batch per ~200 tokens):
        - Decode each corpus token ID to its string via tiktoken
        - Embed all strings using the configured embedding model
        - Save vectors to disk as a numpy .npz

    Query time (zero API calls):
        - Look up query token vectors (local)
        - Cosine similarity against all corpus token vectors
        - Return top-K similar tokens as expansion candidates
    """

    EMBED_MODEL = "text-embedding-3-small"
    BATCH_SIZE = 200

    def __init__(self):
        self._token_ids: np.ndarray | None = None   # shape (V,)
        self._vectors: np.ndarray | None = None     # shape (V, D)  float32, L2-normalised
        self._id_to_row: dict[int, int] = {}
        # Set of token IDs that are word-initial (string starts with space, or
        # standalone alpha word ≥4 chars).  Only these are used as expansion pivots
        # to avoid BPE fragments like 'rent', 'al', 'preci' triggering noisy expansion.
        self._pivot_ids: set[int] = set()

    def build(
        self,
        corpus_token_ids: Iterable[int],
        tokenizer,
        api_key: str | None = None,
        cache_path: Path | None = None,
        filter_lexical: bool = True,
    ) -> "EmbeddingExpander":
        """
        Embed corpus token strings once. Saves to cache_path if given.

        With filter_lexical=True (default), only alphabetically meaningful
        BPE tokens are embedded — drops punctuation, digits, and very short
        subword fragments that carry no semantic signal.
        """
        from openai import OpenAI

        client = OpenAI(api_key=api_key)  # uses OPENAI_API_KEY env if None

        all_ids = sorted(set(corpus_token_ids))
        if filter_lexical:
            pairs = [(t, tokenizer.decode([t])) for t in all_ids]
            pairs = [(t, s) for t, s in pairs if _is_lexical_token(s)]
            token_ids = [t for t, _ in pairs]
            token_strings = [s for _, s in pairs]
            print(f"  Token filter: {len(all_ids)} total → {len(token_ids)} lexical tokens", flush=True)
        else:
            token_ids = all_ids
            token_strings = [tokenizer.decode([t]) for t in token_ids]

        print(f"  Embedding {len(token_ids)} corpus tokens via {self.EMBED_MODEL}...", flush=True)
        all_vectors = []
        for i in range(0, len(token_strings), self.BATCH_SIZE):
            batch = token_strings[i: i + self.BATCH_SIZE]
            resp = client.embeddings.create(model=self.EMBED_MODEL, input=batch)
            vecs = [e.embedding for e in sorted(resp.data, key=lambda x: x.index)]
            all_vectors.extend(vecs)
            print(f"    {min(i + self.BATCH_SIZE, len(token_strings))}/{len(token_strings)}", flush=True)

        mat = np.array(all_vectors, dtype=np.float32)
        # L2-normalise for fast cosine via dot product
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        mat /= norms

        self._token_ids = np.array(token_ids, dtype=np.int64)
        self._vectors = mat
        self._id_to_row = {tid: i for i, tid in enumerate(token_ids)}
        # Mark word-initial tokens as valid expansion pivots
        self._pivot_ids = {
            tid for tid, s in zip(token_ids, token_strings)
            if s.startswith(" ") or (s.isalpha() and len(s) >= 4)
        }

        if cache_path:
            self.save(cache_path)

        return self

    def expand(
        self,
        query_tokens: list[int],
        top_k: int = 8,
        min_similarity: float = 0.70,
        steep: float = 3.0,
    ) -> list[tuple[int, float]]:
        """Return (token_id, similarity_weight) for semantically similar corpus tokens.

        Uses a power curve so only tight neighbors (cosine ≥ ~0.85) get real
        weight.  Tokens just above min_similarity get near-zero weight and have
        negligible BM25 impact.

        Args:
            min_similarity: Hard floor — tokens below this are dropped entirely.
            steep: Exponent for the power curve.  Higher = sharper drop-off near
                   the floor.  Default 3.0 means sim=0.75 → weight≈0.03,
                   sim=0.85 → weight≈0.22, sim=0.95 → weight≈0.59.
        """
        if self._vectors is None or len(self._id_to_row) == 0:
            return []

        seen = set(query_tokens)
        candidates: dict[int, float] = {}

        for qt in query_tokens:
            row = self._id_to_row.get(qt)
            if row is None:
                continue
            # Only use word-initial tokens as expansion pivots.
            # BPE fragments like 'rent', 'al', 'preci' are in the matrix but
            # their neighbors are other fragments, not semantically useful words.
            if self._pivot_ids and qt not in self._pivot_ids:
                continue
            qvec = self._vectors[row]                      # (D,)
            sims = self._vectors @ qvec                    # (V,) cosine similarities
            top_rows = np.argpartition(sims, -top_k * 2)[-top_k * 2:]
            for r in top_rows:
                tid = int(self._token_ids[r])
                sim = float(sims[r])
                if tid in seen or sim < min_similarity:
                    continue
                # Power curve: ((sim - floor) / (1 - floor)) ^ steep, capped at 0.7
                norm = (sim - min_similarity) / (1.0 - min_similarity)
                weight = (norm ** steep) * 0.7
                candidates[tid] = max(candidates.get(tid, 0.0), weight)

        return sorted(candidates.items(), key=lambda x: -x[1])[:top_k]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Note: np.savez_compressed appends .npz if the filename doesn't end with it,
        # so we use a _tmp suffix (keeping the .npz extension) to get an atomic write.
        tmp = path / "embedding_expander_tmp.npz"
        pivot_arr = np.array(sorted(self._pivot_ids), dtype=np.int64)
        np.savez_compressed(
            tmp,
            token_ids=self._token_ids,
            vectors=self._vectors,
            pivot_ids=pivot_arr,
        )
        os.replace(tmp, path / "embedding_expander.npz")

    @classmethod
    def load(cls, path: Path | str) -> "EmbeddingExpander":
        path = Path(path)
        exp = cls()
        fpath = path / "embedding_expander.npz"
        if not fpath.exists():
            return exp
        data = np.load(fpath)
        exp._token_ids = data["token_ids"]
        exp._vectors = data["vectors"].astype(np.float32)
        exp._id_to_row = {int(tid): i for i, tid in enumerate(exp._token_ids)}
        if "pivot_ids" in data:
            exp._pivot_ids = set(int(x) for x in data["pivot_ids"])
        else:
            # Legacy cache without pivot_ids: recompute from token_ids only
            # (conservative fallback: treat all embedded tokens as pivots)
            exp._pivot_ids = set(int(tid) for tid in exp._token_ids)
        return exp

    def stats(self) -> dict:
        if self._vectors is None:
            return {"ready": False}
        return {
            "ready": True,
            "tokens_embedded": len(self._id_to_row),
            "vector_dim": int(self._vectors.shape[1]),
            "size_mb": round(self._vectors.nbytes / 1e6, 2),
        }


# ---------------------------------------------------------------------------
# Combined expander: tries embedding first, falls back to co-occurrence
# ---------------------------------------------------------------------------

class SemanticExpander:
    """
    Unified expander: embedding-based when available, co-occurrence fallback.
    """

    def __init__(
        self,
        corpus_expander: CorpusExpander | None = None,
        embedding_expander: EmbeddingExpander | None = None,
    ):
        self.corpus = corpus_expander or CorpusExpander()
        self.embeddings = embedding_expander or EmbeddingExpander()

    def expand(self, query_tokens: list[int]) -> list[tuple[int, float]]:
        """Return merged (token_id, weight) expansion list."""
        merged: dict[int, float] = {}

        # Embedding expansion (higher quality when available)
        for tid, w in self.embeddings.expand(query_tokens):
            merged[tid] = max(merged.get(tid, 0.0), w)

        # Co-occurrence expansion (always available)
        for tid, w in self.corpus.expand(query_tokens):
            merged[tid] = max(merged.get(tid, 0.0), w * 0.85)  # slightly lower weight

        seen = set(query_tokens)
        return [(t, w) for t, w in sorted(merged.items(), key=lambda x: -x[1]) if t not in seen]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        self.corpus.save(path)
        self.embeddings.save(path)

    @classmethod
    def load(cls, path: Path | str) -> "SemanticExpander":
        path = Path(path)
        return cls(
            corpus_expander=CorpusExpander.load(path),
            embedding_expander=EmbeddingExpander.load(path),
        )

    def stats(self) -> dict:
        return {
            "corpus": self.corpus.stats(),
            "embeddings": self.embeddings.stats(),
        }
