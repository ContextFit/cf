"""
Semantic IDs (SIDs): hierarchical token prefixes for chunk retrieval.

A SID is a short sequence of reserved token IDs. Similar chunks should share
prefixes, which makes retrieval trie-like and LLM-native: a model or small
retriever can generate a SID prefix, then ContextFit resolves it to chunks.

This first implementation is intentionally discrete and embedding-free. It uses
MinHash signature bands as residual quantization buckets:

    chunk tokens → MinHash signature → band fingerprints → SID tokens

Because MinHash signatures preserve token/ngram-set similarity, chunks with
similar token patterns are more likely to share SID prefixes.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contextfit.graph.similarity import MinHasher, MinHashSignature

# Reserved high token range for storage/retrieval SIDs. These are not assumed to
# be valid vocabulary IDs for every LLM; they are retrieval-control tokens unless
# a downstream model explicitly reserves them as special tokens.
SID_TOKEN_BASE = 2_000_000_000


@dataclass(frozen=True)
class SemanticID:
    """A hierarchical sequence of reserved SID token IDs."""

    chunk_id: int
    tokens: tuple[int, ...]

    @property
    def depth(self) -> int:
        return len(self.tokens)

    def prefix(self, depth: int) -> tuple[int, ...]:
        """Return the SID prefix up to depth."""
        return self.tokens[:depth]

    def common_prefix_len(self, other: SemanticID | tuple[int, ...]) -> int:
        """Return common prefix length with another SID/prefix."""
        other_tokens = other.tokens if isinstance(other, SemanticID) else other
        count = 0
        for a, b in zip(self.tokens, other_tokens, strict=False):
            if a != b:
                break
            count += 1
        return count


class SemanticIDIndex:
    """
    Assign, store, and query hierarchical Semantic IDs.

    Parameters:
        depth: Number of SID tokens per chunk.
        branching_factor: Number of buckets per depth level.
        token_base: First reserved SID token ID.
        band_size: Number of MinHash values used per depth bucket. If omitted,
            it is derived from the MinHash signature size.

    SID token layout:
        token_base + level * branching_factor + bucket

    This means every level owns a contiguous token range, and equal prefixes are
    cheap integer tuple comparisons.
    """

    def __init__(
        self,
        depth: int = 4,
        branching_factor: int = 256,
        token_base: int = SID_TOKEN_BASE,
        band_size: int | None = None,
    ):
        if depth <= 0:
            raise ValueError("depth must be positive")
        if branching_factor <= 1:
            raise ValueError("branching_factor must be > 1")

        self.depth = depth
        self.branching_factor = branching_factor
        self.token_base = token_base
        self.band_size = band_size

        self._chunk_to_sid: dict[int, SemanticID] = {}
        self._prefix_to_chunks: dict[tuple[int, ...], set[int]] = defaultdict(set)

    def _stable_bucket(self, values: np.ndarray, level: int, parent: tuple[int, ...]) -> int:
        """Map a signature band + parent prefix to a deterministic bucket."""
        h = hashlib.blake2b(digest_size=8)
        h.update(level.to_bytes(2, "little", signed=False))
        for token in parent:
            h.update(int(token).to_bytes(8, "little", signed=False))
        h.update(np.asarray(values, dtype=np.uint64).tobytes())
        return int.from_bytes(h.digest(), "little") % self.branching_factor

    def assign(self, chunk_id: int, signature: MinHashSignature) -> SemanticID:
        """Assign a Semantic ID to a chunk from its MinHash signature."""
        sig = np.asarray(signature.signature, dtype=np.uint64)
        if sig.size == 0:
            raise ValueError("signature cannot be empty")

        band_size = self.band_size or max(1, sig.size // self.depth)
        sid_tokens: list[int] = []

        for level in range(self.depth):
            start = (level * band_size) % sig.size
            end = min(start + band_size, sig.size)
            band = sig[start:end]
            if band.size < band_size:
                band = np.concatenate([band, sig[: band_size - band.size]])

            bucket = self._stable_bucket(band, level, tuple(sid_tokens))
            sid_tokens.append(self.token_base + level * self.branching_factor + bucket)

        sid = SemanticID(chunk_id=chunk_id, tokens=tuple(sid_tokens))
        self.add(sid)
        return sid

    def assign_tokens(
        self,
        chunk_id: int,
        tokens: list[int] | np.ndarray,
        hasher: MinHasher,
    ) -> SemanticID:
        """Hash tokens with MinHasher and assign a SID."""
        return self.assign(chunk_id, hasher.hash(chunk_id, tokens))

    def add(self, sid: SemanticID) -> None:
        """Add an existing SID to the index."""
        old = self._chunk_to_sid.get(sid.chunk_id)
        if old:
            self.remove(sid.chunk_id)

        self._chunk_to_sid[sid.chunk_id] = sid
        for depth in range(1, sid.depth + 1):
            self._prefix_to_chunks[sid.prefix(depth)].add(sid.chunk_id)

    def remove(self, chunk_id: int) -> None:
        """Remove a chunk SID from the index."""
        sid = self._chunk_to_sid.pop(chunk_id, None)
        if not sid:
            return

        for depth in range(1, sid.depth + 1):
            prefix = sid.prefix(depth)
            chunks = self._prefix_to_chunks.get(prefix)
            if not chunks:
                continue
            chunks.discard(chunk_id)
            if not chunks:
                del self._prefix_to_chunks[prefix]

    def get(self, chunk_id: int) -> SemanticID | None:
        """Return the SID for a chunk."""
        return self._chunk_to_sid.get(chunk_id)

    def chunks_for_prefix(self, prefix: Iterable[int]) -> list[int]:
        """Return chunks matching a SID prefix exactly."""
        return sorted(self._prefix_to_chunks.get(tuple(prefix), set()))

    def query_sid(
        self,
        sid_or_prefix: SemanticID | Iterable[int],
        min_prefix_depth: int = 1,
        max_results: int | None = None,
    ) -> list[tuple[int, float]]:
        """
        Resolve a SID/prefix to chunk IDs with prefix-backoff.

        Starts with the longest prefix and backs off until it finds candidates.
        Score is normalized prefix length (1.0 for full SID match).
        """
        tokens = (
            sid_or_prefix.tokens
            if isinstance(sid_or_prefix, SemanticID)
            else tuple(sid_or_prefix)
        )
        if not tokens:
            return []

        full_depth = min(len(tokens), self.depth)
        min_prefix_depth = max(1, min(min_prefix_depth, full_depth))

        for depth in range(full_depth, min_prefix_depth - 1, -1):
            prefix = tuple(tokens[:depth])
            chunk_ids = self.chunks_for_prefix(prefix)
            if chunk_ids:
                score = depth / self.depth
                results = [(cid, score) for cid in chunk_ids]
                return results[:max_results] if max_results else results

        return []

    def query_tokens(
        self,
        tokens: list[int] | np.ndarray,
        hasher: MinHasher,
        min_prefix_depth: int = 1,
        max_results: int | None = None,
    ) -> list[tuple[int, float]]:
        """Generate a query SID from token IDs and resolve it."""
        query_sid = self.assign_from_signature(-1, hasher.hash(-1, tokens), add=False)
        return self.query_sid(query_sid, min_prefix_depth=min_prefix_depth, max_results=max_results)

    def assign_from_signature(
        self,
        chunk_id: int,
        signature: MinHashSignature,
        add: bool = True,
    ) -> SemanticID:
        """Assign a SID from a signature, optionally without indexing it."""
        sig = np.asarray(signature.signature, dtype=np.uint64)
        if sig.size == 0:
            raise ValueError("signature cannot be empty")

        band_size = self.band_size or max(1, sig.size // self.depth)
        sid_tokens: list[int] = []
        for level in range(self.depth):
            start = (level * band_size) % sig.size
            end = min(start + band_size, sig.size)
            band = sig[start:end]
            if band.size < band_size:
                band = np.concatenate([band, sig[: band_size - band.size]])
            bucket = self._stable_bucket(band, level, tuple(sid_tokens))
            sid_tokens.append(self.token_base + level * self.branching_factor + bucket)

        sid = SemanticID(chunk_id=chunk_id, tokens=tuple(sid_tokens))
        if add:
            self.add(sid)
        return sid

    def nearest_by_prefix(
        self,
        sid_or_prefix: SemanticID | Iterable[int],
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Return nearest indexed chunks by common SID prefix length.

        This scans indexed SIDs, so it is intended as a fallback/debug helper.
        Prefix trie lookup is preferred for production retrieval.
        """
        tokens = (
            sid_or_prefix.tokens
            if isinstance(sid_or_prefix, SemanticID)
            else tuple(sid_or_prefix)
        )
        scored = []
        for chunk_id, sid in self._chunk_to_sid.items():
            common = sid.common_prefix_len(tokens)
            if common > 0:
                scored.append((chunk_id, common / self.depth))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]

    def save(self, path: Path | str) -> None:
        """Save SID index to disk as compact JSON."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "depth": self.depth,
            "branching_factor": self.branching_factor,
            "token_base": self.token_base,
            "band_size": self.band_size,
            "sids": {
                str(chunk_id): list(sid.tokens)
                for chunk_id, sid in self._chunk_to_sid.items()
            },
        }
        tmp_path = path / "semantic_ids.json.tmp"
        tmp_path.write_text(json.dumps(data, separators=(",", ":")))
        os.replace(tmp_path, path / "semantic_ids.json")

    @classmethod
    def load(cls, path: Path | str) -> SemanticIDIndex:
        """Load SID index from disk."""
        path = Path(path)
        data = json.loads((path / "semantic_ids.json").read_text())
        index = cls(
            depth=data["depth"],
            branching_factor=data["branching_factor"],
            token_base=data["token_base"],
            band_size=data.get("band_size"),
        )
        for chunk_id, tokens in data.get("sids", {}).items():
            index.add(SemanticID(chunk_id=int(chunk_id), tokens=tuple(tokens)))
        return index

    def __len__(self) -> int:
        return len(self._chunk_to_sid)

    def __contains__(self, chunk_id: int) -> bool:
        return chunk_id in self._chunk_to_sid

    def stats(self) -> dict:
        """Return SID index statistics."""
        return {
            "sid_count": len(self),
            "depth": self.depth,
            "branching_factor": self.branching_factor,
            "token_base": self.token_base,
            "prefix_count": len(self._prefix_to_chunks),
        }
