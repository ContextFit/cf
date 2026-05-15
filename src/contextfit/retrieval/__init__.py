"""Retrieval layer: Query processing and result assembly."""

from contextfit.retrieval.engine import RetrievalEngine, RetrievalResult
# rerank_sessions_by_structure is a method on RetrievalEngine; no separate export needed.
from contextfit.retrieval.memory_atoms import (
    MemoryAtom,
    augment_query_for_memory_atoms,
    atom_type_priors,
    episode_context_text,
    episode_relevance_score,
    episode_relevance_features,
    extract_memory_atoms,
    query_memory_intents,
)
from contextfit.retrieval.query_router import (
    QueryRoute,
    describe_route,
    route_query,
)
from contextfit.retrieval.token_rerank import TokenNativeReranker

__all__ = [
    # Engine
    "RetrievalEngine",
    "RetrievalResult",
    # Memory atoms
    "MemoryAtom",
    "augment_query_for_memory_atoms",
    "atom_type_priors",
    "episode_context_text",
    "episode_relevance_score",
    "episode_relevance_features",
    "extract_memory_atoms",
    "query_memory_intents",
    # Query router
    "QueryRoute",
    "describe_route",
    "route_query",
    # Reranker
    "TokenNativeReranker",
]
