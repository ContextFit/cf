"""
Similarity computation using MinHash and LSH.

No floating-point embeddings - pure integer operations on token sets/n-grams.
"""

from __future__ import annotations

import json
import pickle
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from datasketch import MinHash, MinHashLSH


@dataclass
class MinHashSignature:
    """MinHash signature for a chunk."""
    chunk_id: int
    signature: np.ndarray  # Array of hash values


class MinHasher:
    """
    Generate MinHash signatures for token sequences.
    
    Converts token sequences into compact integer signatures that preserve
    Jaccard similarity: similar sequences have similar signatures.
    
    Example:
        hasher = MinHasher(num_perm=128, ngram_size=3)
        sig1 = hasher.hash(chunk_id=0, tokens=[1, 2, 3, 4, 5])
        sig2 = hasher.hash(chunk_id=1, tokens=[2, 3, 4, 5, 6])
        similarity = hasher.jaccard(sig1, sig2)  # ~0.6
    """
    
    def __init__(
        self,
        num_perm: int = 128,
        ngram_size: int = 3,
        seed: int = 42,
    ):
        """
        Initialize MinHasher.
        
        Args:
            num_perm: Number of permutations (signature size)
            ngram_size: Size of n-grams to extract from tokens
            seed: Random seed for hash functions
        """
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self.seed = seed
    
    def _extract_ngrams(self, tokens: list[int] | np.ndarray) -> set[tuple[int, ...]]:
        """Extract n-grams from token sequence."""
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        
        if len(tokens) < self.ngram_size:
            return {tuple(tokens)} if tokens else set()
        
        ngrams = set()
        for i in range(len(tokens) - self.ngram_size + 1):
            ngram = tuple(tokens[i:i + self.ngram_size])
            ngrams.add(ngram)
        
        return ngrams
    
    def hash(self, chunk_id: int, tokens: list[int] | np.ndarray) -> MinHashSignature:
        """
        Generate MinHash signature for a token sequence.
        
        Args:
            chunk_id: Chunk identifier
            tokens: Token IDs
        
        Returns:
            MinHashSignature with integer signature array
        """
        ngrams = self._extract_ngrams(tokens)
        
        mh = MinHash(num_perm=self.num_perm, seed=self.seed)
        for ngram in ngrams:
            # Convert n-gram tuple to bytes for hashing
            mh.update(str(ngram).encode("utf-8"))
        
        return MinHashSignature(
            chunk_id=chunk_id,
            signature=np.array(mh.hashvalues, dtype=np.uint64),
        )
    
    def jaccard(self, sig1: MinHashSignature, sig2: MinHashSignature) -> float:
        """
        Estimate Jaccard similarity from signatures.
        
        This is an approximation based on MinHash property:
        P(min_hash(A) == min_hash(B)) ≈ J(A, B)
        """
        matches = np.sum(sig1.signature == sig2.signature)
        return matches / len(sig1.signature)
    
    def hash_batch(
        self,
        chunks: Iterable[tuple[int, list[int] | np.ndarray]],
    ) -> list[MinHashSignature]:
        """Hash multiple chunks."""
        return [self.hash(chunk_id, tokens) for chunk_id, tokens in chunks]


class LSHIndex:
    """
    Locality-Sensitive Hashing index for fast approximate nearest neighbor search.
    
    Groups similar chunks into buckets based on MinHash signatures.
    
    Example:
        lsh = LSHIndex(threshold=0.5, num_perm=128)
        lsh.add(chunk_id=0, signature=sig0)
        lsh.add(chunk_id=1, signature=sig1)
        
        neighbors = lsh.query(sig_query)  # Approximate neighbors
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        num_perm: int = 128,
    ):
        """
        Initialize LSH index.
        
        Args:
            threshold: Similarity threshold for bucketing
            num_perm: Number of permutations (must match MinHasher)
        """
        self.threshold = threshold
        self.num_perm = num_perm
        
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._signatures: dict[int, MinHashSignature] = {}
    
    def add(self, signature: MinHashSignature) -> None:
        """Add a signature to the index."""
        # Create MinHash object from signature array
        mh = MinHash(num_perm=self.num_perm)
        mh.hashvalues = signature.signature.copy()
        
        self._lsh.insert(str(signature.chunk_id), mh)
        self._signatures[signature.chunk_id] = signature
    
    def add_batch(self, signatures: Iterable[MinHashSignature]) -> None:
        """Add multiple signatures."""
        for sig in signatures:
            self.add(sig)
    
    def query(self, signature: MinHashSignature, include_self: bool = False) -> list[int]:
        """
        Find approximate neighbors.
        
        Args:
            signature: Query signature
            include_self: Include the query chunk if in index
        
        Returns:
            List of similar chunk IDs
        """
        mh = MinHash(num_perm=self.num_perm)
        mh.hashvalues = signature.signature.copy()
        
        results = self._lsh.query(mh)
        chunk_ids = [int(r) for r in results]
        
        if not include_self and signature.chunk_id in chunk_ids:
            chunk_ids.remove(signature.chunk_id)
        
        return chunk_ids
    
    def get_signature(self, chunk_id: int) -> MinHashSignature | None:
        """Get stored signature for a chunk."""
        return self._signatures.get(chunk_id)
    
    def find_pairs(self, min_similarity: float | None = None) -> Iterator[tuple[int, int, float]]:
        """
        Find all similar pairs above threshold.
        
        Yields:
            (chunk_id_1, chunk_id_2, estimated_similarity)
        """
        threshold = min_similarity or self.threshold
        seen = set()
        
        for chunk_id, sig in self._signatures.items():
            neighbors = self.query(sig, include_self=False)
            for neighbor_id in neighbors:
                if neighbor_id < chunk_id:  # Avoid duplicates
                    continue
                pair = (chunk_id, neighbor_id)
                if pair in seen:
                    continue
                seen.add(pair)
                
                neighbor_sig = self._signatures.get(neighbor_id)
                if neighbor_sig:
                    # Compute actual similarity
                    matches = np.sum(sig.signature == neighbor_sig.signature)
                    similarity = matches / len(sig.signature)
                    if similarity >= threshold:
                        yield (chunk_id, neighbor_id, similarity)
    
    def __len__(self) -> int:
        return len(self._signatures)
    
    def __contains__(self, chunk_id: int) -> bool:
        return chunk_id in self._signatures

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Persist signatures and LSH bucket state to *path* directory.

        Format:
          lsh_signatures.bin  – compact binary: [N: uint64][chunk_id: uint64, sig: num_perm×uint64] …
          lsh_meta.json       – threshold, num_perm, signature_count
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "threshold": self.threshold,
            "num_perm": self.num_perm,
            "signature_count": len(self._signatures),
        }
        (path / "lsh_meta.json").write_text(json.dumps(meta))

        tmp = path / "lsh_signatures.bin.tmp"
        with open(tmp, "wb") as f:
            # header: number of entries
            f.write(struct.pack("<Q", len(self._signatures)))
            for chunk_id, sig in self._signatures.items():
                # chunk_id (uint64) + signature (num_perm × uint64)
                f.write(struct.pack("<Q", chunk_id))
                f.write(sig.signature.astype(np.uint64).tobytes())
        tmp.replace(path / "lsh_signatures.bin")

        # Persist the datasketch LSH object itself so loads can skip re-inserting
        # all signatures one-by-one.
        with open(path / "lsh_datasketch.pkl", "wb") as f:
            pickle.dump(self._lsh, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path | str) -> "LSHIndex":
        """Load from a directory previously written by :meth:`save`.

        Reconstructs both the signature table and the datasketch LSH buckets
        in a single pass.
        """
        path = Path(path)
        meta = json.loads((path / "lsh_meta.json").read_text())
        threshold: float = meta["threshold"]
        num_perm: int = meta["num_perm"]

        index = cls(threshold=threshold, num_perm=num_perm)

        # Fast path: restore the datasketch LSH object directly.
        pkl = path / "lsh_datasketch.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                index._lsh = pickle.load(f)

        sig_bytes = (path / "lsh_signatures.bin").read_bytes()
        offset = 0
        (count,) = struct.unpack_from("<Q", sig_bytes, offset)
        offset += 8

        sig_size = num_perm * 8  # num_perm × uint64
        need_reinsert = not pkl.exists()
        for _ in range(count):
            (chunk_id,) = struct.unpack_from("<Q", sig_bytes, offset)
            offset += 8
            raw = sig_bytes[offset: offset + sig_size]
            offset += sig_size

            arr = np.frombuffer(raw, dtype=np.uint64).copy()
            sig = MinHashSignature(chunk_id=chunk_id, signature=arr)

            if need_reinsert:
                mh = MinHash(num_perm=num_perm)
                mh.hashvalues = arr.copy()
                index._lsh.insert(str(chunk_id), mh)
            index._signatures[chunk_id] = sig

        return index
