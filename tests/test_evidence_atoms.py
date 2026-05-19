from contextfit.retrieval.evidence_atoms import (
    extract_evidence_atoms,
    query_evidence_facets,
    rerank_sessions_by_evidence_atoms,
)


def test_extract_evidence_atoms_keeps_source_and_facets():
    atoms = extract_evidence_atoms(
        "Session ID: s1\nTurn 1 (user): I want to improve endurance, but I need to avoid high-impact workouts.",
        source_id="s1",
    )

    assert {a.facet for a in atoms} >= {"goal", "constraint"}
    assert all(a.source_id == "s1" for a in atoms)
    assert any("endurance" in a.tokens for a in atoms)


def test_query_evidence_facets_for_advice_requests_are_general():
    facets = query_evidence_facets("What should I focus on to improve endurance?")

    assert {"goal", "constraint", "entity_context"} <= facets
    assert "marathon" not in facets


def test_evidence_atom_rerank_selects_complementary_session_before_generic_duplicate():
    sessions = {
        "s_goal": "Turn 1 (user): I want to improve endurance for a spring marathon.",
        "s_generic": "Turn 1 (user): What are popular endurance tips and gym ideas?",
        "s_constraint": "Turn 1 (user): I need to avoid high-impact workouts because my knee is sore.",
    }
    ranked = rerank_sessions_by_evidence_atoms(
        "What should I focus on to improve endurance?",
        ["s_goal", "s_generic", "s_constraint"],
        sessions,
        top_k=3,
    )

    assert ranked[:2] == ["s_goal", "s_constraint"]
