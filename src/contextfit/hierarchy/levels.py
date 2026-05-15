"""
Hierarchy builder: Geo-map-style multi-level navigation.

Level 0 = finest (raw chunks)
Level N = coarsest (global summaries)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from contextfit.core.chunk import Chunk, ChunkStore
from contextfit.graph.community import Community, CommunityDetector
from contextfit.graph.similarity import LSHIndex, MinHasher


@dataclass
class HierarchyLevel:
    """A single level in the hierarchy."""
    level: int
    chunks: list[Chunk]
    # Mapping from this level's chunks to children (lower level)
    children: dict[int, list[int]] = field(default_factory=dict)


class HierarchyBuilder:
    """
    Build a hierarchical structure over chunks.
    
    Creates multiple levels like a map:
    - Level 0: Raw chunks (finest detail)
    - Level 1: Clustered summaries
    - Level N: Global overview (coarsest)
    
    Navigation works like zooming a map: start at coarse level,
    find relevant areas, then zoom into details.
    
    Example:
        builder = HierarchyBuilder(
            chunk_store=store,
            hasher=MinHasher(),
            lsh=LSHIndex(threshold=0.3),
        )
        
        # Build hierarchy
        levels = builder.build(max_levels=3)
        
        # Navigate
        results = builder.navigate(query_tokens, start_level=2, end_level=0)
    """
    
    def __init__(
        self,
        chunk_store: ChunkStore,
        hasher: MinHasher,
        lsh: LSHIndex,
        summarizer: Callable[[list[int]], list[int]] | None = None,
    ):
        """
        Initialize builder.
        
        Args:
            chunk_store: Store for reading/writing chunks
            hasher: MinHash generator
            lsh: LSH index for similarity
            summarizer: Function to summarize multiple chunks into one.
                        Takes list of chunk IDs, returns summary tokens.
                        If None, uses extractive summarization (common tokens).
        """
        self.store = chunk_store
        self.hasher = hasher
        self.lsh = lsh
        self.summarizer = summarizer or self._extractive_summary
        
        self._levels: dict[int, HierarchyLevel] = {}
    
    def _extractive_summary(self, chunk_ids: list[int]) -> list[int]:
        """
        Simple extractive summarization: collect most common tokens.
        
        In production, you might use an LLM here.
        """
        from collections import Counter
        
        all_tokens: list[int] = []
        for chunk_id in chunk_ids:
            chunk = self.store.get(chunk_id)
            if chunk:
                all_tokens.extend(chunk.tokens.tolist())
        
        if not all_tokens:
            return []
        
        # Get most common tokens (deduplicated by position)
        counter = Counter(all_tokens)
        # Keep unique tokens in order of frequency
        common = [token for token, _ in counter.most_common()]
        
        # Limit summary size
        max_summary = 256
        return common[:max_summary]
    
    def build(self, max_levels: int = 3) -> list[HierarchyLevel]:
        """
        Build the full hierarchy.
        
        Returns:
            List of HierarchyLevel from finest (0) to coarsest (N)
        """
        # Level 0: Get all raw chunks
        level_0_chunks = list(self.store.iter_chunks(level=0))
        
        if not level_0_chunks:
            return []
        
        self._levels[0] = HierarchyLevel(level=0, chunks=level_0_chunks)
        
        # Build LSH index from level 0
        for chunk in level_0_chunks:
            sig = self.hasher.hash(chunk.chunk_id, chunk.tokens)
            self.lsh.add(sig)
        
        # Build higher levels via community detection
        current_level = 0
        current_chunk_ids = [c.chunk_id for c in level_0_chunks]
        
        while current_level < max_levels - 1 and len(current_chunk_ids) > 1:
            # Detect communities
            detector = CommunityDetector(resolution=1.0 + current_level * 0.5)
            
            for id1, id2, sim in self.lsh.find_pairs():
                if id1 in current_chunk_ids and id2 in current_chunk_ids:
                    detector.add_edge(id1, id2, sim)
            
            # Add isolated nodes
            for chunk_id in current_chunk_ids:
                detector.add_node(chunk_id)
            
            communities = detector.detect()
            
            if len(communities) <= 1:
                # Can't split further
                break
            
            # Create summary chunks for each community
            next_level = current_level + 1
            level_chunks = []
            children_map: dict[int, list[int]] = {}
            
            for comm in communities:
                # Summarize community
                summary_tokens = self.summarizer(comm.chunk_ids)
                
                if not summary_tokens:
                    continue
                
                # Store as new chunk
                summary_chunk = self.store.add(
                    tokens=np.array(summary_tokens, dtype=np.uint32),
                    level=next_level,
                    metadata={"community_id": comm.community_id},
                )
                
                level_chunks.append(summary_chunk)
                children_map[summary_chunk.chunk_id] = comm.chunk_ids
                
                # Add to LSH for next level
                sig = self.hasher.hash(summary_chunk.chunk_id, summary_chunk.tokens)
                self.lsh.add(sig)
            
            if not level_chunks:
                break
            
            self._levels[next_level] = HierarchyLevel(
                level=next_level,
                chunks=level_chunks,
                children=children_map,
            )
            
            current_level = next_level
            current_chunk_ids = [c.chunk_id for c in level_chunks]
        
        return [self._levels[i] for i in sorted(self._levels.keys())]
    
    def get_level(self, level: int) -> HierarchyLevel | None:
        """Get a specific hierarchy level."""
        return self._levels.get(level)
    
    def get_children(self, chunk_id: int) -> list[int]:
        """Get child chunk IDs for a higher-level chunk."""
        for level_data in self._levels.values():
            if chunk_id in level_data.children:
                return level_data.children[chunk_id]
        return []
    
    def navigate(
        self,
        query_tokens: list[int] | np.ndarray,
        start_level: int | None = None,
        end_level: int = 0,
        top_k: int = 5,
        scorer: Callable[[list[int], Chunk], float] | None = None,
    ) -> list[Chunk]:
        """
        Navigate from coarse to fine level.
        
        Args:
            query_tokens: Query token IDs
            start_level: Starting level (None = highest available)
            end_level: Target level (0 = finest)
            top_k: Number of chunks to keep at each level
            scorer: Optional scoring function (query_tokens, chunk) -> score
        
        Returns:
            List of relevant chunks at end_level
        """
        if isinstance(query_tokens, np.ndarray):
            query_tokens = query_tokens.tolist()
        
        if not self._levels:
            return []
        
        # Default scorer: token overlap
        if scorer is None:
            def scorer(q: list[int], c: Chunk) -> float:
                q_set = set(q)
                c_set = set(c.tokens.tolist())
                return len(q_set & c_set) / max(1, len(q_set | c_set))
        
        # Start at highest level
        if start_level is None:
            start_level = max(self._levels.keys())
        
        current_level = start_level
        candidates: list[int] = []
        
        # Get initial candidates at start level
        if current_level in self._levels:
            level_data = self._levels[current_level]
            scored = [(c, scorer(query_tokens, c)) for c in level_data.chunks]
            scored.sort(key=lambda x: -x[1])
            candidates = [c.chunk_id for c, _ in scored[:top_k]]
        
        # Navigate down
        while current_level > end_level and candidates:
            # Expand to children
            child_ids: set[int] = set()
            for cid in candidates:
                children = self.get_children(cid)
                child_ids.update(children)
            
            if not child_ids:
                # No children, we're at the bottom
                break
            
            current_level -= 1
            
            # Score children
            child_chunks = []
            for cid in child_ids:
                chunk = self.store.get(cid)
                if chunk:
                    child_chunks.append(chunk)
            
            if not child_chunks:
                break
            
            scored = [(c, scorer(query_tokens, c)) for c in child_chunks]
            scored.sort(key=lambda x: -x[1])
            candidates = [c.chunk_id for c, _ in scored[:top_k]]
        
        # Return final chunks
        result = []
        for cid in candidates:
            chunk = self.store.get(cid)
            if chunk:
                result.append(chunk)
        
        return result
    
    @property
    def num_levels(self) -> int:
        return len(self._levels)
    
    def stats(self) -> dict:
        """Return hierarchy statistics."""
        return {
            "num_levels": self.num_levels,
            "chunks_per_level": {
                level: len(data.chunks)
                for level, data in sorted(self._levels.items())
            },
        }
