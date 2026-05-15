"""
SID generator: predict Semantic ID prefixes from query tokens.

This is the first retrieval-native generator for ContextFit. It does not decode
text and does not use float embeddings. It predicts SID prefixes by letting
strong lexical/minhash candidates vote for the SID neighborhoods they belong to.

The intended evolution path is:

1. Prototype generator (this file): token-native voting over chunks/SIDs.
2. Learned generator: train a small model to emit SID tokens directly.
3. LLM-native generator: prompt/fine-tune an LLM to generate SID token prefixes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from contextfit.graph.similarity import LSHIndex, MinHasher, MinHashSignature
from contextfit.index.bm25 import BM25Scorer
from contextfit.sid.semantic import SemanticIDIndex


@dataclass(frozen=True)
class SIDPrediction:
    """A generated SID prefix and the chunks that supported it."""

    prefix: tuple[int, ...]
    score: float
    depth: int
    support: int
    candidate_chunks: tuple[int, ...]


class SIDGenerator:
    """
    Generate likely Semantic ID prefixes for a query.

    The generator is intentionally discrete:
    - BM25 finds lexical candidate chunks using token IDs.
    - MinHash/LSH adds token-shingle neighbors.
    - Candidate chunks vote for their own SID prefixes.
    - Prefixes are resolved through the SID trie.

    No embeddings, no detokenization, and no external model are required.
    """

    def __init__(
        self,
        sid_index: SemanticIDIndex,
        bm25: BM25Scorer,
        hasher: MinHasher,
        lsh: LSHIndex,
        lexical_weight: float = 1.0,
        minhash_weight: float = 0.75,
        lsh_weight: float = 0.5,
    ):
        self.sid_index = sid_index
        self.bm25 = bm25
        self.hasher = hasher
        self.lsh = lsh
        self.lexical_weight = lexical_weight
        self.minhash_weight = minhash_weight
        self.lsh_weight = lsh_weight

    def _query_signature(self, query_tokens: list[int] | np.ndarray) -> MinHashSignature:
        return self.hasher.hash(-1, query_tokens)

    def _candidate_scores(
        self,
        query_tokens: list[int] | np.ndarray,
        candidate_k: int,
    ) -> dict[int, float]:
        """Collect chunk candidates and scores using token-only signals."""
        if isinstance(query_tokens, np.ndarray):
            query_tokens = query_tokens.tolist()

        scores: dict[int, float] = defaultdict(float)

        # Lexical candidates.
        bm25_results = self.bm25.top_k(query_tokens, k=candidate_k)
        max_bm25 = max((score for _, score in bm25_results), default=0.0)
        for rank, (chunk_id, score) in enumerate(bm25_results):
            rank_boost = 1.0 / (rank + 1)
            normalized = score / max_bm25 if max_bm25 > 0 else 0.0
            scores[chunk_id] += self.lexical_weight * (normalized + rank_boost)

        # Direct query SID prefix if it happens to map into known trie space.
        query_sig = self._query_signature(query_tokens)
        query_sid = self.sid_index.assign_from_signature(-1, query_sig, add=False)
        for chunk_id, prefix_score in self.sid_index.nearest_by_prefix(query_sid, k=candidate_k):
            scores[chunk_id] += prefix_score

        # LSH candidates from query signature.
        for chunk_id in self.lsh.query(query_sig, include_self=False):
            scores[chunk_id] += self.lsh_weight

        # MinHash similarity over currently known candidates.
        for chunk_id in list(scores.keys()):
            sig = self.lsh.get_signature(chunk_id)
            if sig is None:
                continue
            sim = self.hasher.jaccard(query_sig, sig)
            if sim > 0:
                scores[chunk_id] += self.minhash_weight * sim

        return dict(scores)

    def predict(
        self,
        query_tokens: list[int] | np.ndarray,
        top_k: int = 5,
        candidate_k: int = 32,
        max_prefix_depth: int | None = None,
    ) -> list[SIDPrediction]:
        """Predict top SID prefixes for a query."""
        max_prefix_depth = max_prefix_depth or self.sid_index.depth
        max_prefix_depth = min(max_prefix_depth, self.sid_index.depth)
        if max_prefix_depth <= 0:
            return []

        candidate_scores = self._candidate_scores(query_tokens, candidate_k=candidate_k)
        if not candidate_scores:
            return []

        prefix_scores: dict[tuple[int, ...], float] = defaultdict(float)
        prefix_chunks: dict[tuple[int, ...], set[int]] = defaultdict(set)

        for chunk_id, score in candidate_scores.items():
            sid = self.sid_index.get(chunk_id)
            if sid is None:
                continue

            for depth in range(1, min(max_prefix_depth, sid.depth) + 1):
                prefix = sid.prefix(depth)
                depth_weight = depth / self.sid_index.depth
                prefix_scores[prefix] += score * depth_weight
                prefix_chunks[prefix].add(chunk_id)

        predictions = [
            SIDPrediction(
                prefix=prefix,
                score=score,
                depth=len(prefix),
                support=len(prefix_chunks[prefix]),
                candidate_chunks=tuple(sorted(prefix_chunks[prefix])),
            )
            for prefix, score in prefix_scores.items()
        ]
        predictions.sort(key=lambda p: (-p.score, -p.depth, -p.support, p.prefix))
        return predictions[:top_k]

    def retrieve(
        self,
        query_tokens: list[int] | np.ndarray,
        top_k: int = 5,
        prediction_k: int | None = None,
        candidate_k: int = 32,
    ) -> list[tuple[int, float]]:
        """
        Generate SID prefixes, resolve them to chunks, and return scored chunks.
        """
        prediction_k = prediction_k or max(top_k, 5)
        predictions = self.predict(
            query_tokens,
            top_k=prediction_k,
            candidate_k=max(candidate_k, top_k * 4),
        )

        chunk_scores: dict[int, float] = defaultdict(float)
        for prediction in predictions:
            resolved = self.sid_index.query_sid(
                prediction.prefix,
                min_prefix_depth=max(1, prediction.depth),
            )
            for chunk_id, prefix_score in resolved:
                chunk_scores[chunk_id] += prediction.score * prefix_score

            # The generator's supporting chunks are strong local evidence even if
            # a prefix is too specific and resolves narrowly.
            for chunk_id in prediction.candidate_chunks:
                chunk_scores[chunk_id] += prediction.score * 0.25

        results = sorted(chunk_scores.items(), key=lambda x: (-x[1], x[0]))
        return results[:top_k]

    def stats(self) -> dict:
        """Return generator configuration."""
        return {
            "lexical_weight": self.lexical_weight,
            "minhash_weight": self.minhash_weight,
            "lsh_weight": self.lsh_weight,
        }
