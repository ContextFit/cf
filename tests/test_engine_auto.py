"""Integration tests for RetrievalEngine.rank_sessions_by_episode_score() and query_auto()."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from contextfit.retrieval import RetrievalEngine, route_query, QueryRoute


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_engine(tmp_path):
    """A fresh engine with two ingested sessions."""
    engine = RetrievalEngine.create(tmp_path)

    # Session A: user expressed a clear food preference
    engine.ingest_text(
        "Session ID: s_spicy\nDate: 2026/01/10\n\n"
        "Turn 1 (user): I love spicy food. My favourite cuisines are Thai and Sichuan.\n"
        "Turn 2 (assistant): Noted — you enjoy bold heat and complex spice profiles.",
        metadata={"session_id": "s_spicy", "kind": "session"},
        update_indexes=True,
    )

    # Session B: user mentioned a decision
    engine.ingest_text(
        "Session ID: s_decision\nDate: 2026/01/15\n\n"
        "Turn 1 (user): I decided to use PostgreSQL for the new project. We evaluated"
        " MySQL and SQLite but Postgres won on reliability.\n"
        "Turn 2 (assistant): PostgreSQL is a solid choice for production workloads.",
        metadata={"session_id": "s_decision", "kind": "session"},
        update_indexes=True,
    )

    # Session C: generic, should score low for both queries
    engine.ingest_text(
        "Session ID: s_generic\nDate: 2026/01/20\n\n"
        "Turn 1 (user): What time does the library close?\n"
        "Turn 2 (assistant): Most public libraries close at 6 PM on weekdays.",
        metadata={"session_id": "s_generic", "kind": "session"},
        update_indexes=True,
    )

    yield engine


# ---------------------------------------------------------------------------
# rank_sessions_by_episode_score
# ---------------------------------------------------------------------------

def test_rank_sessions_returns_list(tmp_engine):
    results = tmp_engine.rank_sessions_by_episode_score("What should I eat tonight?", top_k=5)
    assert isinstance(results, list)


def test_rank_sessions_result_shape(tmp_engine):
    results = tmp_engine.rank_sessions_by_episode_score("What should I eat tonight?", top_k=5)
    for r in results:
        assert "session_id" in r
        assert "score" in r
        assert "rank" in r
        assert isinstance(r["score"], float)


def test_rank_sessions_food_preference_surfaces_spicy(tmp_engine):
    results = tmp_engine.rank_sessions_by_episode_score("What should I eat tonight?", top_k=5)
    ids = [r["session_id"] for r in results]
    assert "s_spicy" in ids
    # s_spicy should rank above s_generic
    spicy_rank = next(r["rank"] for r in results if r["session_id"] == "s_spicy")
    generic_rank = next((r["rank"] for r in results if r["session_id"] == "s_generic"), 999)
    assert spicy_rank < generic_rank


def test_rank_sessions_top_k_respected(tmp_engine):
    results = tmp_engine.rank_sessions_by_episode_score("anything", top_k=2)
    assert len(results) <= 2


def test_rank_sessions_ranks_are_ascending(tmp_engine):
    results = tmp_engine.rank_sessions_by_episode_score("food recommendation", top_k=5)
    ranks = [r["rank"] for r in results]
    assert ranks == sorted(ranks)


# ---------------------------------------------------------------------------
# query_auto
# ---------------------------------------------------------------------------

def test_query_auto_returns_expected_keys(tmp_engine):
    result = tmp_engine.query_auto("What should I eat tonight?", top_k=3)
    assert "route" in result
    assert "session_ids" in result
    assert "details" in result
    assert isinstance(result["session_ids"], list)


def test_query_auto_route_is_query_route(tmp_engine):
    result = tmp_engine.query_auto("What should I eat tonight?", top_k=3)
    assert isinstance(result["route"], QueryRoute)


def test_query_auto_vague_advice_routes_episode_score(tmp_engine):
    result = tmp_engine.query_auto("What should I eat tonight?", top_k=3)
    assert result["route"].mode == "episode_score"


def test_query_auto_decision_routes_bm25(tmp_engine):
    result = tmp_engine.query_auto("Which database did I decide to use?", top_k=3)
    assert result["route"].mode == "bm25"


def test_query_auto_episode_score_surfaces_spicy_session(tmp_engine):
    result = tmp_engine.query_auto("What should I eat tonight?", top_k=3)
    # episode_score mode — s_spicy should be in session_ids
    assert "s_spicy" in result["session_ids"]


def test_query_auto_open_loop_routes_episode_score(tmp_engine):
    result = tmp_engine.query_auto("Is there anything I was supposed to follow up on?", top_k=3)
    assert result["route"].mode == "episode_score"


def test_query_auto_temporal_fact_routes_fusion(tmp_engine):
    result = tmp_engine.query_auto("Which database am I currently using?", top_k=3)
    assert result["route"].mode == "episode_bm25_fusion"


def test_query_auto_details_has_route_description(tmp_engine):
    result = tmp_engine.query_auto("Recommend something to watch", top_k=3)
    assert "route" in result["details"]
    assert "mode=" in result["details"]["route"]


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------

def test_retrieval_init_exports():
    from contextfit.retrieval import (
        RetrievalEngine, RetrievalResult,
        MemoryAtom, extract_memory_atoms, episode_relevance_score,
        route_query, QueryRoute, describe_route,
        TokenNativeReranker,
    )
    assert RetrievalEngine is not None
    assert route_query is not None
    assert QueryRoute is not None


def test_query_auto_preference_rerank_prefers_taste_over_topic(tmp_path):
    engine = RetrievalEngine.create(tmp_path)
    engine.ingest_text(
        "Session ID: s_podcast_pref\nDate: 2026/01/20\n\n"
        "Turn 1 (user): I really enjoy long-form interview podcasts, especially ones about science, technology, and human psychology.\n",
        metadata={"session_id": "s_podcast_pref", "kind": "session"},
        update_indexes=True,
    )
    engine.ingest_text(
        "Session ID: s_commute_time\nDate: 2026/01/21\n\n"
        "Turn 1 (user): My commute is about 40 minutes each way by train.\n",
        metadata={"session_id": "s_commute_time", "kind": "session"},
        update_indexes=True,
    )
    result = engine.query_auto("Can you recommend a podcast for my commute?", top_k=2, retrieval_k=10)
    assert result["route"].mode == "preference_rerank"
    assert result["session_ids"][0] == "s_podcast_pref"


def test_query_auto_preference_rerank_preserves_token_prefix_for_large_k(tmp_path):
    engine = RetrievalEngine.create(tmp_path)
    for i in range(1, 7):
        engine.ingest_text(
            f"Session ID: s_token_{i}\nDate: 2026/01/{i:02d}\n\n"
            f"Turn 1 (user): I am comparing podcast options for my commute, candidate {i}.\n",
            metadata={"session_id": f"s_token_{i}", "kind": "session"},
            update_indexes=True,
        )
    engine.ingest_text(
        "Session ID: s_pref\nDate: 2026/01/20\n\n"
        "Turn 1 (user): I really enjoy long-form interview podcasts about science and psychology.\n",
        metadata={"session_id": "s_pref", "kind": "session"},
        update_indexes=True,
    )

    result = engine.query_auto("Can you recommend a podcast for my commute?", top_k=10, retrieval_k=10)

    assert result["route"].mode == "preference_rerank"
    assert result["details"]["preference_protected_token_top_k"] == 5
    assert all(sid.startswith("s_token_") for sid in result["session_ids"][:5])
    assert "s_pref" in result["session_ids"]


def test_query_auto_preference_rerank_uses_ingested_memory_atoms_as_tail(tmp_path):
    engine = RetrievalEngine.create(tmp_path)
    for i in range(1, 7):
        engine.ingest_text(
            f"Session ID: s_token_{i}\nDate: 2026/01/{i:02d}\n\n"
            f"Turn 1 (user): I am comparing movie options for tonight, candidate {i}.\n",
            metadata={"session_id": f"s_token_{i}", "kind": "session"},
            update_indexes=True,
        )
    engine.ingest_text(
        "Session ID: s_pref\nDate: 2026/01/20\n\n"
        "Turn 1 (user): I am practicing a stage routine this month.\n",
        metadata={"session_id": "s_pref", "kind": "session"},
        update_indexes=True,
    )
    engine.ingest_text(
        "Memory atom type: user_interest\n"
        "Source session: s_pref\n"
        "User memory: I enjoy storytelling comedy specials on streaming services.\n"
        "Retrieval hints: preference recommendations entertainment watch movie show",
        metadata={"session_id": "s_pref", "source_id": "s_pref", "kind": "memory_atoms"},
        update_indexes=True,
    )

    result = engine.query_auto("Can you recommend a movie or show for tonight?", top_k=10, retrieval_k=10)

    assert result["route"].mode == "preference_rerank"
    assert result["details"]["preference_atom_candidate_sessions"] >= 1
    assert result["details"]["preference_protected_token_top_k"] == 5
    assert "s_pref" in result["session_ids"]


def test_query_auto_multi_session_rerank_covers_complementary_facets(tmp_path):
    engine = RetrievalEngine.create(tmp_path)
    engine.ingest_text(
        "Session ID: s_goal\nDate: 2026/02/01\n\n"
        "Turn 1 (user): I want to improve my endurance for a marathon this spring.\n",
        metadata={"session_id": "s_goal", "kind": "session"},
        update_indexes=True,
    )
    engine.ingest_text(
        "Session ID: s_constraint\nDate: 2026/02/02\n\n"
        "Turn 1 (user): I need to avoid high-impact workouts because my knee has been sore.\n",
        metadata={"session_id": "s_constraint", "kind": "session"},
        update_indexes=True,
    )
    engine.ingest_text(
        "Session ID: s_generic\nDate: 2026/02/03\n\n"
        "Turn 1 (user): What are some popular fitness trends and general gym tips?\n",
        metadata={"session_id": "s_generic", "kind": "session"},
        update_indexes=True,
    )
    result = engine.query_auto("What should I focus on to improve my endurance?", top_k=3, retrieval_k=10)
    assert result["route"].mode == "multi_session_rerank"
    assert result["session_ids"][:2] == ["s_goal", "s_constraint"]


def test_query_auto_can_use_evidence_atom_rerank_for_multi_session(tmp_path):
    engine = RetrievalEngine.create(tmp_path)
    engine.ingest_text(
        "Session ID: s_goal\nDate: 2026/02/01\n\n"
        "Turn 1 (user): I want to improve my endurance for a marathon this spring.\n",
        metadata={"session_id": "s_goal", "kind": "session"},
        update_indexes=True,
    )
    engine.ingest_text(
        "Session ID: s_constraint\nDate: 2026/02/02\n\n"
        "Turn 1 (user): I need to avoid high-impact workouts because my knee has been sore.\n",
        metadata={"session_id": "s_constraint", "kind": "session"},
        update_indexes=True,
    )
    engine.ingest_text(
        "Session ID: s_generic\nDate: 2026/02/03\n\n"
        "Turn 1 (user): What are some popular endurance tips and general gym ideas?\n",
        metadata={"session_id": "s_generic", "kind": "session"},
        update_indexes=True,
    )

    result = engine.query_auto(
        "What should I focus on to improve my endurance?",
        top_k=3,
        retrieval_k=10,
        evidence_atom_rerank=True,
    )

    assert result["details"]["evidence_atom_rerank"] is True
    assert result["session_ids"][:2] == ["s_goal", "s_constraint"]


def test_query_two_stage_sessions_searches_inside_discovered_parent(tmp_path):
    engine = RetrievalEngine.create(tmp_path)
    engine.ingest_text(
        "Session ID: s_launch\nDate: 2026/02/01\n\n"
        "Turn 1 (user): Project Orion launch planning notes.\n"
        "Turn 2 (assistant): The checklist mentions buying batteries and badge clips.\n",
        metadata={"session_id": "s_launch", "kind": "session"},
        update_indexes=True,
    )
    engine.ingest_text(
        "Session ID: s_generic\nDate: 2026/02/02\n\n"
        "Turn 1 (user): Project Orion had a long generic planning discussion.\n"
        "Turn 2 (assistant): We covered milestones, morale, and broad timelines.\n",
        metadata={"session_id": "s_generic", "kind": "session"},
        update_indexes=True,
    )
    engine.ingest_text(
        "Session ID: s_supplies\nDate: 2026/02/03\n\n"
        "Turn 1 (user): Office supply notes.\n"
        "Turn 2 (assistant): Buying folders and pens is unrelated to the launch.\n",
        metadata={"session_id": "s_supplies", "kind": "session"},
        update_indexes=True,
    )

    result = engine.query_two_stage_sessions(
        "For Project Orion launch, what buying checklist items came up?",
        top_k=2,
        broad_k=10,
        precise_k=3,
    )

    assert result["route"] == "two_stage_sessions"
    assert result["session_ids"][0] == "s_launch"
    assert result["details"]["candidates"][0]["precise_hits"] > 0
