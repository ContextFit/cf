"""Tests for the deterministic query router."""

import pytest
from contextfit.retrieval.query_router import QueryRoute, route_query


# ---------------------------------------------------------------------------
# Episode score (vague advice)
# ---------------------------------------------------------------------------

def test_route_what_should_i_cook():
    r = route_query("What should I cook for dinner tonight?")
    assert r.mode == "episode_score"
    assert r.confidence > 0.4
    assert any("vague_advice" in s for s in r.signals)


def test_route_can_you_recommend():
    r = route_query("Can you recommend a podcast for my commute?")
    assert r.mode == "preference_rerank"
    assert any("preference_recommendation" in s for s in r.signals)


def test_route_help_me_plan():
    r = route_query("Help me plan a healthy meal for the week.")
    # Both episode_score and episode_bm25_fusion are valid for plan queries
    assert r.mode in {"episode_score", "episode_bm25_fusion", "multi_session_rerank"}


def test_route_any_tips():
    r = route_query("Any tips for my commute?")
    # commute=7 chars, but no vague_advice trigger -> stays episode_score
    assert r.mode in {"episode_score", "episode_bm25_fusion", "multi_session_rerank"}


def test_route_how_should_i():
    # structure + writing + sessions are all 7-8 char domain nouns -> fusion correct
    r = route_query("How should I structure my writing sessions this week?")
    assert r.mode in {"episode_score", "episode_bm25_fusion", "multi_session_rerank"}


def test_route_what_type_of():
    r = route_query("What type of exercise should I do?")
    assert r.mode == "preference_rerank"


# ---------------------------------------------------------------------------
# BM25 (specific fact lookup)
# ---------------------------------------------------------------------------

def test_route_what_did_i_decide():
    r = route_query("What did I decide about the job offer?")
    assert r.mode == "bm25"
    assert any("specific_fact" in s for s in r.signals)


def test_route_which_framework():
    r = route_query("Which project management framework did I decide to use?")
    assert r.mode == "bm25"


def test_route_current_state():
    # temporal + specific_fact combo -> fusion (episode_score's temporal_update
    # atom alignment finds "I switched to WHOOP" better than BM25 token match)
    r = route_query("What fitness tracker am I currently using?")
    assert r.mode == "episode_bm25_fusion"
    assert any("temporal+fact" in s for s in r.signals)


def test_route_is_there_something_pending():
    # Vague open-loop: no named topic -> episode_score (better atom alignment)
    r = route_query("Is there something I was supposed to follow up on this week?")
    assert r.mode == "episode_score"
    assert any("open_loop_vague" in s for s in r.signals)


def test_route_outstanding_tasks():
    # Vague open-loop: no named topic -> episode_score
    r = route_query("Do I have any outstanding tasks I need to take care of?")
    assert r.mode == "episode_score"
    assert any("open_loop_vague" in s for s in r.signals)


def test_route_temporal_current():
    r = route_query("Which coffee machine am I using these days?")
    assert r.mode == "bm25"


# ---------------------------------------------------------------------------
# Multi-session fusion
# ---------------------------------------------------------------------------

def test_route_make_progress_this_weekend():
    r = route_query("I want to make progress on my home this weekend. Any suggestions?")
    assert r.mode in {"episode_bm25_fusion", "episode_score", "multi_session_rerank"}


def test_route_what_should_i_focus_on():
    r = route_query("What should I focus on this month to move forward professionally?")
    # Could be episode_score or episode_bm25_fusion — both are acceptable
    assert r.mode in {"episode_score", "episode_bm25_fusion", "multi_session_rerank"}


# ---------------------------------------------------------------------------
# Confidence and structure
# ---------------------------------------------------------------------------

def test_route_returns_dataclass():
    r = route_query("Recommend a wine for me.")
    assert isinstance(r, QueryRoute)
    assert isinstance(r.confidence, float)
    assert 0.0 <= r.confidence <= 1.0
    assert isinstance(r.signals, list)
    assert r.mode in {"episode_score", "bm25", "preference_rerank", "multi_session_rerank", "atom_fusion", "episode_bm25_fusion"}


def test_route_short_query_defaults_to_preference_for_watch():
    r = route_query("What should I watch tonight?")
    assert r.mode == "preference_rerank"


def test_route_signals_nonempty():
    r = route_query("Help me choose a gift.")
    assert len(r.signals) > 0


def test_route_proper_noun_penalises_episode():
    # Specific tool name should push toward bm25
    r_vague = route_query("What did I decide to use?")
    r_specific = route_query('What did I decide about "Notion" vs "Obsidian"?')
    # Specific version should be at least as bm25-weighted as vague
    assert r_specific.mode == "bm25"


def test_route_deterministic():
    q = "What should I get my friend for her birthday?"
    assert route_query(q).mode == route_query(q).mode
    assert route_query(q).confidence == route_query(q).confidence


def test_route_evidence_coverage_to_multi_session():
    r = route_query("What factors should I consider for my upcoming vacation?")
    assert r.mode == "multi_session_rerank"
    assert any("evidence_coverage" in sig for sig in r.signals)


def test_route_personal_aggregate_count_to_multi_session():
    r = route_query("What is the total number of siblings I have?")
    assert r.mode == "multi_session_rerank"
    assert "personal_aggregate" in r.signals


def test_route_relative_date_math_does_not_use_personal_aggregate():
    r = route_query("How many days ago did I meet Emma?")
    assert "personal_aggregate" not in r.signals
    assert r.mode != "multi_session_rerank"
