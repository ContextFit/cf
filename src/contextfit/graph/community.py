"""
Community detection on the chunk similarity graph.

Identifies clusters of related chunks for hierarchical organization
and commonality mining.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator

import networkx as nx
import numpy as np


@dataclass
class Community:
    """A detected community of related chunks."""
    community_id: int
    chunk_ids: list[int]
    # Representative tokens (most common across community)
    common_tokens: list[int]
    # Parent community in hierarchy
    parent_id: int | None = None
    # Child communities
    children: list[int] | None = None


class CommunityDetector:
    """
    Detect communities in the chunk similarity graph.
    
    Uses Louvain algorithm for community detection, then extracts
    common patterns across chunks in each community.
    
    Example:
        detector = CommunityDetector()
        
        # Build graph from similarity pairs
        for id1, id2, sim in lsh_index.find_pairs():
            detector.add_edge(id1, id2, sim)
        
        # Detect communities
        communities = detector.detect()
        
        # Find common patterns
        for comm in communities:
            print(f"Community {comm.community_id}: {len(comm.chunk_ids)} chunks")
            print(f"  Common tokens: {comm.common_tokens[:10]}")
    """
    
    def __init__(self, resolution: float = 1.0):
        """
        Initialize detector.
        
        Args:
            resolution: Louvain resolution parameter (higher = more communities)
        """
        self.resolution = resolution
        self._graph = nx.Graph()
    
    def add_edge(self, chunk_id_1: int, chunk_id_2: int, weight: float) -> None:
        """Add a similarity edge between chunks."""
        self._graph.add_edge(chunk_id_1, chunk_id_2, weight=weight)
    
    def add_node(self, chunk_id: int) -> None:
        """Add an isolated node (no edges)."""
        self._graph.add_node(chunk_id)
    
    def build_from_pairs(
        self,
        pairs: Iterable[tuple[int, int, float]],
        min_weight: float = 0.0,
    ) -> None:
        """
        Build graph from similarity pairs.
        
        Args:
            pairs: Iterable of (chunk_id_1, chunk_id_2, similarity)
            min_weight: Minimum similarity to include edge
        """
        for id1, id2, sim in pairs:
            if sim >= min_weight:
                self.add_edge(id1, id2, sim)
    
    def detect(self) -> list[Community]:
        """
        Detect communities using Louvain algorithm.
        
        Returns:
            List of Community objects
        """
        if len(self._graph) == 0:
            return []
        
        # Run Louvain
        communities_dict = nx.community.louvain_communities(
            self._graph,
            resolution=self.resolution,
            seed=42,
        )
        
        communities = []
        for i, chunk_ids in enumerate(communities_dict):
            community = Community(
                community_id=i,
                chunk_ids=sorted(chunk_ids),
                common_tokens=[],  # Filled in by analyze_tokens
            )
            communities.append(community)
        
        return communities
    
    def detect_hierarchical(self, max_levels: int = 3) -> list[Community]:
        """
        Detect hierarchical communities (multiple levels).
        
        Returns communities with parent/children relationships.
        """
        all_communities = []
        current_graph = self._graph.copy()
        community_offset = 0
        
        for level in range(max_levels):
            if len(current_graph) < 2:
                break
            
            # Detect communities at this level
            communities_dict = nx.community.louvain_communities(
                current_graph,
                resolution=self.resolution * (level + 1),
                seed=42,
            )
            
            if len(communities_dict) <= 1:
                break
            
            # Create Community objects
            level_communities = []
            for i, chunk_ids in enumerate(communities_dict):
                community = Community(
                    community_id=community_offset + i,
                    chunk_ids=sorted(chunk_ids),
                    common_tokens=[],
                    children=[] if level > 0 else None,
                )
                level_communities.append(community)
            
            # Link to parent communities from previous level
            if level > 0 and all_communities:
                for comm in level_communities:
                    # Find parent based on chunk overlap
                    for parent in all_communities:
                        if parent.parent_id is None and any(
                            c in parent.chunk_ids for c in comm.chunk_ids
                        ):
                            comm.parent_id = parent.community_id
                            if parent.children is None:
                                parent.children = []
                            parent.children.append(comm.community_id)
                            break
            
            all_communities.extend(level_communities)
            community_offset += len(level_communities)
            
            # Build coarser graph for next level
            # Each community becomes a node
            new_graph = nx.Graph()
            for comm in level_communities:
                new_graph.add_node(comm.community_id)
            
            # Add edges between communities based on cross-community edges
            for comm1 in level_communities:
                for comm2 in level_communities:
                    if comm1.community_id >= comm2.community_id:
                        continue
                    
                    # Sum weights of edges between communities
                    total_weight = 0.0
                    count = 0
                    for c1 in comm1.chunk_ids:
                        for c2 in comm2.chunk_ids:
                            if current_graph.has_edge(c1, c2):
                                total_weight += current_graph[c1][c2]["weight"]
                                count += 1
                    
                    if count > 0:
                        avg_weight = total_weight / count
                        new_graph.add_edge(
                            comm1.community_id,
                            comm2.community_id,
                            weight=avg_weight,
                        )
            
            current_graph = new_graph
        
        return all_communities
    
    def analyze_tokens(
        self,
        communities: list[Community],
        chunk_tokens: dict[int, list[int]],
        top_k: int = 20,
    ) -> None:
        """
        Find common tokens for each community.
        
        Args:
            communities: Communities to analyze
            chunk_tokens: Mapping from chunk_id to token list
            top_k: Number of top tokens to keep
        """
        for community in communities:
            # Count token frequencies across community
            token_counts: dict[int, int] = defaultdict(int)
            total_chunks = len(community.chunk_ids)
            
            for chunk_id in community.chunk_ids:
                tokens = chunk_tokens.get(chunk_id, [])
                unique_tokens = set(tokens)
                for token in unique_tokens:
                    token_counts[token] += 1
            
            # Keep tokens that appear in many chunks (>30% of community)
            threshold = max(1, total_chunks * 0.3)
            common = [
                (token, count)
                for token, count in token_counts.items()
                if count >= threshold
            ]
            common.sort(key=lambda x: -x[1])
            
            community.common_tokens = [token for token, _ in common[:top_k]]
    
    def find_commonalities(
        self,
        chunk_tokens: dict[int, list[int]],
        min_support: float = 0.1,
        ngram_size: int = 3,
    ) -> list[tuple[tuple[int, ...], list[int]]]:
        """
        Find frequent n-gram patterns across chunks.
        
        Returns:
            List of (ngram, [chunk_ids_containing_it])
        """
        ngram_chunks: dict[tuple[int, ...], set[int]] = defaultdict(set)
        
        for chunk_id, tokens in chunk_tokens.items():
            if len(tokens) < ngram_size:
                continue
            
            for i in range(len(tokens) - ngram_size + 1):
                ngram = tuple(tokens[i:i + ngram_size])
                ngram_chunks[ngram].add(chunk_id)
        
        # Filter by support
        total_chunks = len(chunk_tokens)
        min_count = max(2, int(total_chunks * min_support))
        
        frequent = [
            (ngram, sorted(chunks))
            for ngram, chunks in ngram_chunks.items()
            if len(chunks) >= min_count
        ]
        
        # Sort by frequency
        frequent.sort(key=lambda x: -len(x[1]))
        
        return frequent
    
    @property
    def num_nodes(self) -> int:
        return len(self._graph)
    
    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()
    
    def stats(self) -> dict:
        """Return graph statistics."""
        return {
            "nodes": self.num_nodes,
            "edges": self.num_edges,
            "density": nx.density(self._graph) if self.num_nodes > 1 else 0,
            "components": nx.number_connected_components(self._graph),
        }
