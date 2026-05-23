"""Retrieval layer: Query processing and result assembly."""

from contextfit.retrieval.engine import RetrievalEngine, RetrievalResult
# rerank_sessions_by_structure is a method on RetrievalEngine; no separate export needed.
from contextfit.retrieval.evidence_atoms import (
    EvidenceAtom,
    extract_evidence_atoms,
    query_evidence_facets,
    rerank_sessions_by_evidence_atoms,
)
from contextfit.retrieval.evidence_compiler import (
    EvidenceContext,
    EvidenceSource,
    MISSING_ANSWER,
    build_count_list_ledger,
    build_deterministic_aggregation_assembly,
    build_evidence_packet,
    build_fusion_evidence_map,
    build_multi_session_evidence_ledger,
    build_multi_session_evidence_set,
    build_targeted_source_highlights,
    build_token_evidence_table,
    build_typed_evidence_answer_prompt,
    prepare_evidence_sources,
    promote_evidence_sources_for_count_list,
    should_use_evidence_packet,
    should_use_fusion_evidence_map,
    typed_answer_policy,
)
from contextfit.retrieval.memory_atoms import (
    MemoryAtom,
    PreferenceSupport,
    augment_query_for_memory_atoms,
    atom_type_priors,
    build_preference_support_view,
    episode_context_text,
    episode_relevance_score,
    episode_relevance_features,
    extract_preference_support,
    extract_memory_atoms,
    normalize_memory_query,
    query_memory_intents,
)
from contextfit.retrieval.query_router import (
    QueryRoute,
    describe_route,
    route_query,
)
from contextfit.retrieval.query_spec import MetadataPredicate, QuerySpec
from contextfit.retrieval.token_rerank import TokenNativeReranker

__all__ = [
    # Engine
    "RetrievalEngine",
    "RetrievalResult",
    "EvidenceAtom",
    "extract_evidence_atoms",
    "query_evidence_facets",
    "rerank_sessions_by_evidence_atoms",
    "EvidenceContext",
    "EvidenceSource",
    "MISSING_ANSWER",
    "build_count_list_ledger",
    "build_deterministic_aggregation_assembly",
    "build_evidence_packet",
    "build_fusion_evidence_map",
    "build_multi_session_evidence_ledger",
    "build_multi_session_evidence_set",
    "build_targeted_source_highlights",
    "build_token_evidence_table",
    "build_typed_evidence_answer_prompt",
    "prepare_evidence_sources",
    "promote_evidence_sources_for_count_list",
    "should_use_evidence_packet",
    "should_use_fusion_evidence_map",
    "typed_answer_policy",
    # Memory atoms
    "MemoryAtom",
    "PreferenceSupport",
    "augment_query_for_memory_atoms",
    "atom_type_priors",
    "build_preference_support_view",
    "episode_context_text",
    "episode_relevance_score",
    "episode_relevance_features",
    "extract_preference_support",
    "extract_memory_atoms",
    "normalize_memory_query",
    "query_memory_intents",
    # Query router
    "QueryRoute",
    "describe_route",
    "route_query",
    "MetadataPredicate",
    "QuerySpec",
    # Reranker
    "TokenNativeReranker",
]
