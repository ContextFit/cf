"""
BM25 scoring directly on token IDs.

No text needed - token IDs are the terms.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from contextfit.index.inverted import InvertedIndex


class BM25Scorer:
    """
    BM25 relevance scoring using token IDs.
    
    Classic BM25 formula:
        score = Σ IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
    
    Where:
        - tf = term frequency in document
        - df = document frequency
        - dl = document length
        - avgdl = average document length
        - k1, b = tuning parameters
    
    Example:
        scorer = BM25Scorer(inverted_index)
        scores = scorer.score([token1, token2], [chunk_id1, chunk_id2])
    """
    
    def __init__(
        self,
        index: InvertedIndex,
        k1: float = 1.5,
        b: float = 0.90,
    ):
        self._index = index
        self.k1 = k1
        self.b = b
        
        # Precompute IDF values
        self._idf_cache: dict[int, float] = {}
    
    def _idf(self, token: int) -> float:
        """Compute IDF for a token (cached)."""
        if token in self._idf_cache:
            return self._idf_cache[token]
        
        N = self._index.total_chunks
        df = self._index.get_df(token)
        
        # BM25 IDF formula with smoothing
        if df == 0:
            idf = 0.0
        else:
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        
        self._idf_cache[token] = idf
        return idf
    
    def score_chunk(self, query_tokens: list[int], chunk_id: int) -> float:
        """
        Score a single chunk against query tokens.
        
        Args:
            query_tokens: List of token IDs in query
            chunk_id: Chunk to score
        
        Returns:
            BM25 score
        """
        dl = self._index.chunk_length(chunk_id)
        avgdl = self._index.avg_chunk_length
        
        if dl == 0 or avgdl == 0:
            return 0.0
        
        score = 0.0
        for token in query_tokens:
            tf = self._index.get_tf(token, chunk_id)
            if tf == 0:
                continue
            
            idf = self._idf(token)
            
            # BM25 score component
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
            score += idf * numerator / denominator
        
        return score
    
    def score(
        self,
        query_tokens: list[int] | np.ndarray,
        chunk_ids: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        """
        Score chunks against query tokens.

        When ``chunk_ids`` is None, uses a term-at-a-time accumulator that
        avoids iterating every candidate for every query token.  This is
        O(Σ posting_size) instead of O(|candidates| × |query_tokens|).

        Args:
            query_tokens: Query token IDs
            chunk_ids: Explicit candidate list (skips accumulation when given)

        Returns:
            List of (chunk_id, score) sorted by score descending
        """
        if isinstance(query_tokens, np.ndarray):
            query_tokens = query_tokens.tolist()

        if chunk_ids is not None:
            # Explicit candidate list: score directly (legacy path).
            results = []
            for chunk_id in chunk_ids:
                score = self.score_chunk(query_tokens, chunk_id)
                if score > 0:
                    results.append((chunk_id, score))
            results.sort(key=lambda x: -x[1])
            return results

        # ----------------------------------------------------------------
        # Term-at-a-time accumulator (fast path)
        # ----------------------------------------------------------------
        avgdl = self._index.avg_chunk_length
        k1 = self.k1
        b = self.b
        accum: dict[int, float] = {}

        for token in set(query_tokens):          # deduplicate query tokens
            posting = None
            # Trigger lazy load if needed.
            if hasattr(self._index, '_load_posting'):
                posting = self._index._load_posting(token)
            else:
                posting = self._index._index.get(token)
            if posting is None or len(posting.chunk_ids) == 0:
                continue

            idf = self._idf(token)
            if idf <= 0:
                continue

            # Iterate posting list once, accumulate BM25 contribution.
            for chunk_id in posting.chunk_ids:
                dl = self._index.chunk_length(chunk_id)
                if dl == 0 or avgdl == 0:
                    continue
                # TF: use positions length when available, else 1.
                positions = posting.positions.get(chunk_id)
                tf = len(positions) if positions else 1
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avgdl)
                accum[chunk_id] = accum.get(chunk_id, 0.0) + idf * numerator / denominator

        results = [(cid, score) for cid, score in accum.items() if score > 0]
        results.sort(key=lambda x: -x[1])
        return results

    def top_k(
        self,
        query_tokens: list[int] | np.ndarray,
        k: int = 10,
        chunk_ids: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Get top-k scoring chunks."""
        all_scores = self.score(query_tokens, chunk_ids)
        return all_scores[:k]
    
    def clear_cache(self) -> None:
        """Clear IDF cache (call after index updates)."""
        self._idf_cache.clear()
