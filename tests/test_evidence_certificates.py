from contextfit.retrieval.evidence_certificates import (
    apply_typed_rescue,
    rerank_with_evidence_certificates,
    rerank_with_optional_typed_rescue,
)


def test_multi_count_target_fact_promotes_specific_count_evidence():
    texts = {
        "s1": "Turn 1 (user): I talked about appointments.",
        "s2": "Turn 1 (user): I mentioned a health topic.",
        "s3": "Turn 1 (user): I had a visit last month.",
        "s4": "Turn 1 (user): We discussed the calendar.",
        "s5": "Turn 1 (user): I visited several places this year.",
        "s6": "Turn 1 (user): I visited three different doctors, including a dermatologist and physician.",
    }
    result = rerank_with_evidence_certificates(
        "Question: How many different doctors did I visit?",
        texts,
        ["s1", "s2", "s3", "s4", "s5", "s6"],
        5,
        route_mode="multi_session_rerank",
    )

    assert result.session_ids[:5] == ["s1", "s2", "s3", "s4", "s6"]
    assert result.traces()[0]["certificate"] == "multi_count_target_fact"
    assert result.traces()[0]["displaced_source_id"] == "s5"


def test_answer_shaped_tail_is_protected_from_non_answer_companion():
    texts = {
        "s1": "Turn 1 (user): I talked about appointments.",
        "s2": "Turn 1 (user): I mentioned a health topic.",
        "s3": "Turn 1 (user): I had a visit last month.",
        "s4": "Turn 1 (user): We discussed the calendar.",
        "answer_tail": "Turn 1 (assistant) [HAS_ANSWER]: The answer was two doctors.",
        "s6": "Turn 1 (user): I visited three doctors and one specialist.",
    }
    result = rerank_with_evidence_certificates(
        "Question: How many different doctors did I visit?",
        texts,
        ["s1", "s2", "s3", "s4", "answer_tail", "s6"],
        5,
        route_mode="multi_session_rerank",
    )

    assert result.session_ids[:5] == ["s1", "s2", "s3", "s4", "answer_tail"]
    assert result.traces()[0]["certificate"] == "answer_evidence_tail_protection"
    assert result.traces()[0]["action"] == "protect"


def test_no_promotion_for_generic_overlap_only():
    texts = {
        "s1": "Turn 1 (user): I talked about appointments.",
        "s2": "Turn 1 (user): I mentioned a health topic.",
        "s3": "Turn 1 (user): I had a visit last month.",
        "s4": "Turn 1 (user): We discussed the calendar.",
        "s5": "Turn 1 (user): I visited places and had visits.",
        "s6": "Turn 1 (user): Visit count number total different many.",
    }
    result = rerank_with_evidence_certificates(
        "Question: How many different doctors did I visit?",
        texts,
        ["s1", "s2", "s3", "s4", "s5", "s6"],
        5,
        route_mode="multi_session_rerank",
    )

    assert result.session_ids[:5] == ["s1", "s2", "s3", "s4", "s5"]
    assert all(trace["action"] != "promote" for trace in result.traces())


def test_typed_temporal_rescue_is_opt_in():
    texts = {
        "s1": "Turn 1 (user): I had coffee.",
        "s2": "Turn 1 (user): I checked my email.",
        "s3": "Turn 1 (user): I read a note.",
        "s4": "Turn 1 (user): I walked downtown.",
        "s5": "Turn 1 (user): I made lunch.",
        "s7": "Turn 1 (user): I paid a bill.",
        "s8": "Turn 1 (user): I read the news.",
        "s9": "Turn 1 (user): I sent a note.",
        "s10": "Turn 1 (user): I checked the calendar.",
        "s11": "Turn 1 (user): I packed a bag.",
        "s6": "Turn 1 (user): I met Emma at lunch yesterday.",
    }
    current = ["s1", "s2", "s3", "s4", "s5"]
    base_order = ["s1", "s2", "s3", "s4", "s5", "s7", "s8", "s9", "s10", "s11", "s6"]

    off = rerank_with_optional_typed_rescue(
        "Date: 2026-05-24\nQuestion: When did I meet Emma at lunch?",
        texts,
        base_order,
        5,
        candidate_k=20,
        route_mode="temporal-reasoning",
        typed_rescue=False,
    )
    on = apply_typed_rescue(
        "Date: 2026-05-24\nQuestion: When did I meet Emma at lunch?",
        texts,
        base_order,
        current,
        5,
        candidate_k=20,
        route_mode="temporal-reasoning",
    )

    assert off.session_ids == current
    assert on.session_ids[:5] == ["s1", "s2", "s3", "s4", "s6"]
    assert on.traces()[0]["certificate"] == "temporal_entity_action_rescue"


def test_typed_rescue_rejects_generic_preference_words():
    texts = {
        "s1": "Turn 1 (user): I read about restaurants.",
        "s2": "Turn 1 (user): I checked dinner hours.",
        "s3": "Turn 1 (user): I looked at a map.",
        "s4": "Turn 1 (user): I made a reservation.",
        "s5": "Turn 1 (user): I asked about dinner plans.",
        "s6": "Turn 1 (user): Restaurant dinner recommendation preference words only, with no personal context.",
    }

    result = apply_typed_rescue(
        "Question: Can you recommend a restaurant for dinner?",
        texts,
        ["s1", "s2", "s3", "s4", "s5", "s6"],
        ["s1", "s2", "s3", "s4", "s5"],
        5,
        candidate_k=10,
        route_mode="preference_rerank",
    )

    assert result.session_ids == ["s1", "s2", "s3", "s4", "s5"]
    assert result.traces() == []


def test_typed_rescue_rejects_temporal_entity_word_salad():
    texts = {
        "s1": "Turn 1 (user): I had coffee near the office.",
        "s2": "Turn 1 (user): I checked email after lunch.",
        "s3": "Turn 1 (user): I read a note about Jordan Street.",
        "s4": "Turn 1 (user): I walked downtown.",
        "s5": "Turn 1 (user): I made lunch at home.",
        "s6": "Turn 1 (user): Jordan lunch when calendar question note.",
    }

    result = apply_typed_rescue(
        "Date: 2026-05-24\nQuestion: When did I meet Jordan for lunch?",
        texts,
        ["s1", "s2", "s3", "s4", "s5", "s6"],
        ["s1", "s2", "s3", "s4", "s5"],
        5,
        candidate_k=10,
        route_mode="temporal-reasoning",
    )

    assert result.session_ids == ["s1", "s2", "s3", "s4", "s5"]
    assert result.traces() == []
