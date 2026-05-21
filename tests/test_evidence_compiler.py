from __future__ import annotations

from types import SimpleNamespace

from contextfit.retrieval.evidence_compiler import (
    EvidenceSource,
    build_targeted_source_highlights,
    build_typed_evidence_answer_prompt,
    build_deterministic_aggregation_assembly,
    build_fusion_evidence_map,
    build_multi_session_evidence_ledger,
    build_token_evidence_table,
    effective_fusion_evidence_map_mode,
    prepare_evidence_sources,
    should_use_evidence_packet,
    should_use_fusion_evidence_map,
)


def test_reusable_evidence_compiler_builds_source_linked_views() -> None:
    item = {
        "question": "Which museum did I visit after the concert?",
        "question_type": "temporal-reasoning",
    }
    selected = ["s1", "s2"]
    date_by_sid = {"s1": "2026-05-01", "s2": "2026-05-03"}
    turns_by_sid = {
        "s1": [{"role": "user", "content": "On May 1 I attended a concert downtown."}],
        "s2": [{"role": "user", "content": "On May 3 I visited the science museum after lunch."}],
    }

    table = build_token_evidence_table(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        max_lines=3,
        include_signals=True,
    )
    ledger = build_multi_session_evidence_ledger(item, selected, date_by_sid, turns_by_sid)
    evidence_map = build_fusion_evidence_map(item, selected, date_by_sid, turns_by_sid)

    assert "Token Evidence Table" in table
    assert "Multi-Session Evidence Ledger" in ledger
    assert "Fusion Evidence Map" in evidence_map
    assert "science museum" in evidence_map


def test_reusable_evidence_compiler_routing_predicates() -> None:
    temporal_item = {
        "question": "What changed after May?",
        "question_type": "temporal-reasoning",
    }
    preference_item = {
        "question": "What hotel should I choose?",
        "question_type": "single-session-preference",
    }
    args = SimpleNamespace(expert_ensemble="moe", fusion_evidence_map="off")

    assert should_use_evidence_packet(temporal_item, "general") is True
    assert should_use_evidence_packet(preference_item, "general") is False
    assert should_use_fusion_evidence_map(temporal_item, "moe") is True
    assert should_use_fusion_evidence_map(preference_item, "moe") is False
    assert effective_fusion_evidence_map_mode(args) == "moe"


def test_prepare_evidence_sources_slices_oversized_real_world_memory() -> None:
    item = {
        "question": "How long between transaction management and final deployment?",
        "question_type": "temporal-reasoning",
    }
    long_text = " ".join(
        [
            "Unrelated frontend note about button colors.",
            "Another unrelated note about dashboard cards.",
            "January 15, 2024: Transaction management features were completed.",
            "March 15, 2024: Final deployment was completed.",
            "A separate April 15 deadline appeared in an older planning note.",
        ]
        * 40
    )

    context = prepare_evidence_sources(
        item,
        [EvidenceSource(source_id="beam-like-memory", text=long_text, date="2024-03-15")],
        max_source_chars=700,
        max_total_chars=700,
        max_sentences_per_source=4,
    )

    prepared = context.turns_by_sid["beam-like-memory"][0]["content"]
    assert len(prepared) <= 700
    assert "Transaction management" in prepared
    assert "Final deployment" in prepared
    assert context.selected == ["beam-like-memory"]
    assert context.date_by_sid["beam-like-memory"] == "2024-03-15"


def test_prepare_evidence_sources_enforces_total_budget() -> None:
    item = {
        "question": "What changed after launch?",
        "question_type": "knowledge-update",
    }
    sources = [
        EvidenceSource(source_id=f"s{i}", text="After launch, the user changed the deployment plan. " * 20)
        for i in range(5)
    ]

    context = prepare_evidence_sources(
        item,
        sources,
        max_source_chars=500,
        max_total_chars=900,
        max_sentences_per_source=5,
    )

    total = sum(len(context.turns_by_sid[sid][0]["content"]) for sid in context.selected)
    assert total <= 900
    assert 1 <= len(context.selected) < len(sources)


def test_aggregation_assembly_keeps_pickup_return_obligations_separate() -> None:
    item = {
        "question": "How many items of clothing do I need to pick up or return from a store?",
        "question_type": "multi-session",
    }
    selected = ["dry-cleaning", "zara"]
    date_by_sid = {"dry-cleaning": "2023/02/15", "zara": "2023/02/15"}
    turns_by_sid = {
        "dry-cleaning": [
            {
                "role": "user",
                "content": (
                    "I still need to pick up my dry cleaning for the navy blue blazer "
                    "I wore to a meeting."
                ),
            }
        ],
        "zara": [
            {
                "role": "user",
                "content": (
                    "I need to return some boots to Zara. I exchanged them for a larger size "
                    "and just haven't had a chance to pick up the new pair yet."
                ),
            }
        ],
    }

    assembly = build_deterministic_aggregation_assembly(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        max_candidates=20,
        max_chars=8000,
    )

    text = assembly["text"]
    assert "pickup obligation: pick up dry cleaning for navy blue blazer" in text
    assert "return obligation: return some boots to Zara" in text
    assert "pickup obligation: pick up new pair" in text


def test_aggregation_assembly_supports_store_obligation_synonyms() -> None:
    item = {
        "question": "How many clothing errands do I need to collect, retrieve, or drop off?",
        "question_type": "multi-session",
    }
    selected = ["errands"]
    date_by_sid = {"errands": "2023/02/16"}
    turns_by_sid = {
        "errands": [
            {
                "role": "user",
                "content": (
                    "I need to collect the altered jacket from the tailor. "
                    "I also need to drop off my jeans at Zara."
                ),
            }
        ],
    }

    assembly = build_deterministic_aggregation_assembly(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        max_candidates=20,
        max_chars=8000,
    )

    text = assembly["text"]
    assert "pickup obligation: pick up altered jacket" in text
    assert "return obligation: return my jeans to Zara" in text


def test_prepare_evidence_sources_budgets_by_rank_before_chronological_output() -> None:
    item = {
        "question": "What is the current dashboard response time?",
        "question_type": "knowledge-update",
    }
    sources = [
        EvidenceSource(
            source_id="new-high-rank",
            text="April 5, 2024: The dashboard API response time improved to 250ms.",
            date="2024-04-05",
        ),
        EvidenceSource(
            source_id="old-low-rank",
            text="March 15, 2024: Older setup notes mention unrelated login code. " * 20,
            date="2024-03-15",
        ),
    ]

    context = prepare_evidence_sources(
        item,
        sources,
        max_source_chars=500,
        max_total_chars=500,
        sort_chronologically=True,
    )

    assert "new-high-rank" in context.selected
    assert "old-low-rank" not in context.selected


def test_typed_answer_prompt_is_less_abstention_biased_for_summarization() -> None:
    prompt = build_typed_evidence_answer_prompt(
        question="Summarize the project timeline.",
        question_type="summarization",
        evidence_packet="Deterministic Evidence Packet:\n- March 15: deployment completed.",
    )

    assert "Do not abstain merely because the summary is incomplete" in prompt
    assert "coverage-oriented summary" in prompt


def test_typed_answer_prompt_keeps_strict_abstention_policy() -> None:
    prompt = build_typed_evidence_answer_prompt(
        question="What private detail did the user never mention?",
        question_type="abstention",
        evidence_packet="Deterministic Evidence Packet:\n- no event candidates detected",
    )

    assert "This is an abstention/no-evidence task" in prompt
    assert "return the missing-answer string exactly" in prompt


def test_typed_answer_prompt_extracts_partial_direct_evidence() -> None:
    prompt = build_typed_evidence_answer_prompt(
        question="Which tool and date were mentioned?",
        question_type="information_extraction",
        evidence_packet="Deterministic Evidence Packet:\n- [S1] March 15: Flask deployment.",
    )

    assert "Extract exact names, dates, numbers" in prompt
    assert "Use partial direct evidence when available" in prompt


def test_targeted_source_highlights_recovers_lower_rank_schema_change() -> None:
    item = {
        "question": "How many new columns did I want to add to the transactions table?",
        "question_type": "multi_session_reasoning",
    }
    sources = [
        EvidenceSource(
            source_id="top",
            text="Generic Flask setup notes for the transactions table.",
            date="2024-04-05",
            metadata={"rank": 1},
        ),
        EvidenceSource(
            source_id="late",
            text="I need to add a 'notes' TEXT column to the transactions table without downtime.",
            date="2024-04-25",
            metadata={"rank": 88},
        ),
    ]

    highlights = build_targeted_source_highlights(item, sources)

    assert "Targeted Source Highlights" in highlights
    assert "notes" in highlights
    assert "transactions table" in highlights


def test_targeted_source_highlights_recovers_generic_schema_fields() -> None:
    item = {
        "question": "How many new fields did I ask to add to the invoices table?",
        "question_type": "multi_session_reasoning",
    }
    sources = [
        EvidenceSource(
            source_id="top",
            text="General invoice workflow notes.",
            date="2024-04-05",
            metadata={"rank": 1},
        ),
        EvidenceSource(
            source_id="schema",
            text=(
                "I need to add a 'due_date' DATE column to the invoices table. "
                "I also want a status field so we can track payment state."
            ),
            date="2024-04-25",
            metadata={"rank": 42},
        ),
    ]

    highlights = build_targeted_source_highlights(item, sources)

    assert "User-Requested Schema Fields" in highlights
    assert "due_date" in highlights
    assert "status" in highlights
    assert "invoices table" in highlights


def test_targeted_source_highlights_keeps_matching_temporal_endpoints() -> None:
    item = {
        "question": "How many weeks between finishing transaction management features and final deployment?",
        "question_type": "temporal_reasoning",
    }
    sources = [
        EvidenceSource(
            source_id="timeline",
            text=(
                "Milestones: Dec 16, 2023 - Jan 15, 2024: develop transaction management features. "
                "Feb 16 - Mar 15, 2024: final adjustments, testing, and deployment."
            ),
            date="2024-03-15",
            metadata={"rank": 21},
        )
    ]

    highlights = build_targeted_source_highlights(item, sources)

    assert "Jan 15" in highlights
    assert "Mar 15" in highlights
    assert "transaction management" in highlights


def test_targeted_source_highlights_matches_question_derived_temporal_endpoints() -> None:
    item = {
        "question": "How many days between contract signing and first invoice payment?",
        "question_type": "temporal_reasoning",
    }
    sources = [
        EvidenceSource(
            source_id="timeline",
            text=(
                "Project timeline: May 2, 2024: contract signing completed. "
                "May 17, 2024: first invoice payment arrived."
            ),
            date="2024-05-17",
            metadata={"rank": 3},
        )
    ]

    highlights = build_targeted_source_highlights(item, sources)

    assert "Same-Source Timeline Candidates" in highlights
    assert "contract signing" in highlights
    assert "first invoice payment" in highlights
    assert "15 days" in highlights


def test_targeted_source_highlights_prefers_direct_extraction_over_update() -> None:
    item = {
        "question": "When does my first sprint end?",
        "question_type": "information_extraction",
    }
    sources = [
        EvidenceSource(
            source_id="direct",
            text="The first sprint ends March 29, focusing on user registration and login.",
            date="2024-03-15",
            metadata={"rank": 1},
        ),
        EvidenceSource(
            source_id="update",
            text="The first sprint now targets completion by March 31 for extra testing.",
            date="2024-03-15",
            metadata={"rank": 2},
        ),
    ]

    highlights = build_targeted_source_highlights(item, sources)

    assert "Exact Extraction Candidates" in highlights
    assert highlights.index("March 29") < highlights.index("March 31")


def test_instruction_policy_requires_unindented_language_fences() -> None:
    prompt = build_typed_evidence_answer_prompt(
        question="Show me how to implement login.",
        question_type="instruction_following",
        evidence_packet="Deterministic Evidence Packet:\n- Login implementation requested.",
    )

    assert "start with an explicit language tag" in prompt
    assert "no indentation" in prompt


def test_targeted_source_highlights_builds_contradiction_pair() -> None:
    item = {
        "question": "Have I worked with Flask routes and handled HTTP requests in this project?",
        "question_type": "contradiction_resolution",
    }
    sources = [
        EvidenceSource(
            source_id="negative",
            text="I have never written any Flask routes or handled HTTP requests in this project.",
            date="2024-03-15",
            metadata={"rank": 1},
        ),
        EvidenceSource(
            source_id="positive",
            text="I also mentioned implementing a basic homepage route with Flask using @app.route('/').",
            date="2024-03-15",
            metadata={"rank": 2},
        ),
    ]

    highlights = build_targeted_source_highlights(item, sources)

    assert "Contradiction Pair Candidates" in highlights
    assert "never written any Flask routes" in highlights
    assert "basic homepage route with Flask" in highlights
    assert "Which statement is correct?" in highlights
