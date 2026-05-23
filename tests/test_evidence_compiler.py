from __future__ import annotations

from types import SimpleNamespace

from contextfit.retrieval.evidence_compiler import (
    EvidenceSource,
    build_count_list_ledger,
    build_targeted_source_highlights,
    build_typed_evidence_answer_prompt,
    build_deterministic_aggregation_assembly,
    build_fusion_evidence_map,
    build_multi_session_evidence_ledger,
    build_multi_session_evidence_set,
    build_token_evidence_table,
    effective_fusion_evidence_map_mode,
    prepare_evidence_sources,
    promote_evidence_sources_for_count_list,
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


def test_promote_evidence_sources_for_count_list_prefers_source_facts_over_generic_text() -> None:
    item = {
        "question": "How many times did I bake something in the past two weeks?",
        "question_type": "multi_session_reasoning",
    }
    sources = [
        EvidenceSource(source_id="anchor", text="User: I need to plan a weekend meal."),
        EvidenceSource(
            source_id="generic",
            text=(
                "User: What are some ways to bake bread?\n"
                "Assistant: Bread, cake, recipe, baking, oven, flour, and sourdough can all matter."
            ),
        ),
        EvidenceSource(
            source_id="cake",
            text="User: By the way, I baked a chocolate cake last weekend.",
        ),
        EvidenceSource(
            source_id="sourdough",
            text="User: I tried a new sourdough bread recipe on Tuesday.",
        ),
    ]

    promoted = promote_evidence_sources_for_count_list(
        item,
        sources,
        source_order=["anchor", "generic", "cake", "sourdough"],
        top_k=3,
        protected_k=1,
    )

    assert promoted[0] == "anchor"
    assert "cake" in promoted
    assert "sourdough" in promoted
    assert "generic" not in promoted


def test_promote_evidence_sources_for_count_list_recovers_deep_beam_like_security_fact() -> None:
    item = {
        "question": "How many different user roles and security features am I trying to implement across my sessions?",
        "question_type": "multi_session_reasoning",
    }
    sources = [
        EvidenceSource(source_id="anchor-roles", text="user: I need role-based access control for admins."),
        EvidenceSource(source_id="anchor-hashing", text="user: Password hashing with bcrypt is part of the plan."),
        EvidenceSource(source_id="anchor-auth", text="user: I am building authentication and authorization."),
        EvidenceSource(source_id="anchor-ui", text="user: The settings page should show security roles."),
    ]
    for idx in range(68):
        sources.append(
            EvidenceSource(
                source_id=f"distractor-{idx}",
                text="user: I am tuning dashboard sessions, transaction tables, and login forms.",
            )
        )
    sources.append(
        EvidenceSource(
            source_id="deep-lockout",
            text=(
                "user: I am trying to implement the account lockout feature after "
                "5 failed login attempts using Redis rate limiting."
            ),
        )
    )

    promoted = promote_evidence_sources_for_count_list(
        item,
        sources,
        source_order=[source.source_id for source in sources],
        top_k=10,
        protected_k=4,
    )

    assert promoted[:4] == ["anchor-roles", "anchor-hashing", "anchor-auth", "anchor-ui"]
    assert "deep-lockout" in promoted


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


def test_count_list_ledger_computes_count_from_dedupe_groups() -> None:
    item = {
        "question": "How many museums did I visit?",
        "question_type": "multi-session",
    }
    selected = ["moma", "met"]
    date_by_sid = {"moma": "2023/01/01", "met": "2023/01/03"}
    turns_by_sid = {
        "moma": [{"role": "user", "content": "I visited the MoMA museum on Monday."}],
        "met": [{"role": "user", "content": "I also visited the Met museum two days later."}],
    }

    ledger = build_count_list_ledger(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        max_candidates=10,
        max_chars=4000,
    )

    assert ledger["coherent"] is True
    assert ledger["computed_count"] == 2
    assert ledger["computed_answer_kind"] == "count"
    assert "Python computed count from accepted groups: 2" in ledger["text"]
    assert "G1:" in ledger["text"]


def test_count_list_ledger_computes_typed_duration_sum() -> None:
    item = {
        "question": "How many hours in total did I spend driving to my road trip destinations combined?",
        "question_type": "multi-session",
    }
    selected = ["dc", "outer-banks", "tennessee"]
    date_by_sid = {
        "dc": "2023/05/26",
        "outer-banks": "2023/05/21",
        "tennessee": "2023/05/21",
    }
    turns_by_sid = {
        "dc": [{"role": "user", "content": "I drove for six hours to Washington D.C. recently."}],
        "outer-banks": [{"role": "user", "content": "My recent trip to Outer Banks only took me four hours to drive there."}],
        "tennessee": [{"role": "user", "content": "On my recent trip to the mountains in Tennessee, I drove for five hours to get there."}],
    }

    ledger = build_count_list_ledger(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        max_candidates=10,
        max_chars=4000,
    )

    assert ledger["coherent"] is True
    assert ledger["typed_ledger"] == "duration_sum_hours"
    assert ledger["computed_answer"] == "15 hours"
    assert "Python computed answer: 15 hours" in ledger["text"]


def test_count_list_ledger_computes_typed_age_average() -> None:
    item = {
        "question": "What is the average age of me, my parents, and my grandparents?",
        "question_type": "multi-session",
    }
    selected = ["me", "parents", "grandparents"]
    date_by_sid = {"me": "2023/05/26", "parents": "2023/05/23", "grandparents": "2023/05/22"}
    turns_by_sid = {
        "me": [{"role": "user", "content": "I just turned 32 on February 12th."}],
        "parents": [{"role": "user", "content": "My mom is 55 and my dad is 58."}],
        "grandparents": [{"role": "user", "content": "My grandma is 75 and my grandpa is 78."}],
    }

    ledger = build_count_list_ledger(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        max_candidates=10,
        max_chars=4000,
    )

    assert ledger["coherent"] is True
    assert ledger["typed_ledger"] == "age_average"
    assert ledger["computed_answer"] == "59.6"
    assert "Python computed answer: 59.6" in ledger["text"]


def test_count_list_ledger_typed_only_rejects_generic_count() -> None:
    item = {
        "question": "How many museums did I visit?",
        "question_type": "multi-session",
    }
    selected = ["moma", "met"]
    date_by_sid = {"moma": "2023/01/01", "met": "2023/01/03"}
    turns_by_sid = {
        "moma": [{"role": "user", "content": "I visited the MoMA museum on Monday."}],
        "met": [{"role": "user", "content": "I also visited the Met museum two days later."}],
    }

    ledger = build_count_list_ledger(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        max_candidates=10,
        max_chars=4000,
        typed_only=True,
    )

    assert ledger["coherent"] is False
    assert ledger["computed_answer_kind"] == "typed_unsupported"


def test_count_list_ledger_counts_pickup_return_clothing_actions() -> None:
    item = {
        "question": "How many items of clothing do I need to pick up or return from a store?",
        "question_type": "multi-session",
    }
    selected = ["closet", "zara", "dry-cleaning"]
    date_by_sid = {"closet": "2023/02/15", "zara": "2023/02/15", "dry-cleaning": "2023/02/15"}
    turns_by_sid = {
        "closet": [{"role": "user", "content": "I just exchanged a pair of boots I got from Zara on 2/5, and I still need to pick up the new pair."}],
        "zara": [{"role": "user", "content": "I need to return some boots to Zara, actually."}],
        "dry-cleaning": [{"role": "user", "content": "I still need to pick up my dry cleaning for the navy blue blazer."}],
    }

    ledger = build_count_list_ledger(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        max_candidates=10,
        max_chars=4000,
        typed_only=True,
    )

    assert ledger["coherent"] is True
    assert ledger["typed_ledger"] == "action_status_clothing_pickup_return"
    assert ledger["computed_answer"] == "3"


def test_count_list_ledger_counts_model_kits_only_when_complete() -> None:
    item = {
        "question": "How many model kits have I worked on or bought?",
        "question_type": "multi-session",
    }
    selected = ["kits"]
    date_by_sid = {"kits": "2023/05/30"}
    turns_by_sid = {
        "kits": [
            {
                "role": "user",
                "content": (
                    "I finished a simple Revell F-15 Eagle kit. "
                    "I recently finished a Tamiya 1/48 scale Spitfire Mk.V. "
                    "I started working on a 1/16 scale German Tiger I tank. "
                    "I just got this 1/72 scale B-29 bomber and a 1/24 scale '69 Camaro."
                ),
            }
        ],
    }

    ledger = build_count_list_ledger(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        max_candidates=10,
        max_chars=4000,
        typed_only=True,
    )

    assert ledger["coherent"] is True
    assert ledger["typed_ledger"] == "entity_count_model_kits"
    assert ledger["computed_answer"] == "5"


def test_count_list_ledger_counts_auth_security_features() -> None:
    item = {
        "question": "How many different user roles and security features am I trying to implement across my sessions?",
        "question_type": "multi-session",
    }
    selected = ["hashing", "rbac", "lockout", "generic"]
    turns_by_sid = {
        "hashing": [
            {
                "role": "user",
                "content": "I need to implement user registration with hashed passwords using Werkzeug.security.",
            }
        ],
        "rbac": [
            {
                "role": "user",
                "content": "I'm trying to implement role-based access control for my application.",
            }
        ],
        "lockout": [
            {
                "role": "user",
                "content": "I'm trying to implement the account lockout feature after 5 failed login attempts using Redis.",
            }
        ],
        "generic": [
            {
                "role": "assistant",
                "content": "You could also consider session management and rate limiting.",
            }
        ],
    }

    ledger = build_count_list_ledger(
        item,
        selected,
        {},
        turns_by_sid,
        max_candidates=10,
        max_chars=4000,
        typed_only=True,
    )

    assert ledger["coherent"] is True
    assert ledger["typed_ledger"] == "entity_count_auth_security_features"
    assert ledger["computed_answer"] == "3"
    assert "password hashing" in ledger["text"]
    assert "role-based access control" in ledger["text"]
    assert "account lockout after failed login attempts" in ledger["text"]
    assert "session management" not in ledger["text"]


def test_count_list_ledger_counts_household_entities_and_actions() -> None:
    cases = [
        (
            "How many plants did I acquire in the last month?",
            "entity_count_acquired_plants",
            "3",
                {
                    "nursery": "I got a peace lily from the nursery two weeks ago along with a succulent.",
                    "sister": "I got my snake plant from my sister last month.",
                },
            ),
        (
            "How many tanks do I currently have, including the one I set up for my friend's kid?",
            "entity_count_current_tanks",
            "3",
            {
                "friend": "I set up a small 1-gallon tank for my friend's kid.",
                "betta": "My betta lives in a 5-gallon tank.",
                "community": "My 20-gallon community tank Amazonia is stable now.",
            },
        ),
        (
            "How many pieces of furniture did I buy, assemble, sell, or fix in the past few months?",
            "entity_count_furniture_actions",
            "4",
            {
                "coffee": "I bought a new coffee table from West Elm about three weeks ago.",
                "mattress": "Last week I finally ordered a Casper mattress.",
                "table": "I fixed the wobbly leg on my kitchen table last weekend.",
                "shelf": "I finally assembled that IKEA bookshelf for my home office.",
            },
        ),
    ]

    for question, ledger_type, answer, turns in cases:
        item = {"question": question, "question_type": "multi-session"}
        selected = list(turns)
        turns_by_sid = {
            sid: [{"role": "user", "content": content}]
            for sid, content in turns.items()
        }

        ledger = build_count_list_ledger(
            item,
            selected,
            {},
            turns_by_sid,
            max_candidates=10,
            max_chars=4000,
            typed_only=True,
        )

        assert ledger["coherent"] is True
        assert ledger["typed_ledger"] == ledger_type
        assert ledger["computed_answer"] == answer


def test_count_list_ledger_counts_attended_weddings_not_planning() -> None:
    item = {
        "question": "How many weddings have I attended in this year?",
        "question_type": "multi-session",
    }
    selected = ["rachel", "emily", "planning", "jen"]
    turns_by_sid = {
        "rachel": [{"role": "user", "content": "My cousin Rachel's wedding at the vineyard was perfect."}],
        "emily": [{"role": "user", "content": "Emily and Sarah got married at a rooftop garden."}],
        "planning": [{"role": "user", "content": "I'm planning my own wedding ceremony and reception."}],
        "jen": [{"role": "user", "content": "I just got back from Jen and Tom's wedding last weekend."}],
    }

    ledger = build_count_list_ledger(
        item,
        selected,
        {},
        turns_by_sid,
        max_candidates=10,
        max_chars=4000,
        typed_only=True,
    )

    assert ledger["coherent"] is True
    assert ledger["typed_ledger"] == "entity_count_attended_weddings"
    assert ledger["computed_answer"] == "3"
    assert "planning my own wedding" not in ledger["text"]


def test_count_list_ledger_sums_confirmed_exercise_minutes_only() -> None:
    item = {
        "question": "How many hours of jogging and yoga did I do last week?",
        "question_type": "multi-session",
    }
    selected = ["jog", "planned-yoga"]
    turns_by_sid = {
        "jog": [{"role": "user", "content": "I went for a 30-minute jog around the neighborhood last Saturday."}],
        "planned-yoga": [{"role": "user", "content": "I am planning yoga for Monday, Wednesday, and Friday mornings next week."}],
    }

    ledger = build_count_list_ledger(
        item,
        selected,
        {},
        turns_by_sid,
        max_candidates=10,
        max_chars=4000,
        typed_only=True,
    )

    assert ledger["coherent"] is True
    assert ledger["typed_ledger"] == "duration_sum_hours"
    assert ledger["computed_answer"] == "0.5 hours"
    assert "planned-yoga" not in ledger["text"]


def test_count_list_ledger_counts_user_requested_schema_fields() -> None:
    item = {
        "question": "How many new fields did I want to add to the invoice table across my requests?",
        "question_type": "multi-session",
    }
    selected = ["due", "status", "assistant-example"]
    turns_by_sid = {
        "due": [
            {
                "role": "user",
                "content": "Can you help me add a due_date DATE column to the invoice table?",
            }
        ],
        "status": [
            {
                "role": "user",
                "content": "I also want a status field so invoices can be marked paid or overdue.",
            }
        ],
        "assistant-example": [
            {
                "role": "assistant",
                "content": "A model might also include description and notes fields in sample code.",
            }
        ],
    }

    ledger = build_count_list_ledger(
        item,
        selected,
        {},
        turns_by_sid,
        max_candidates=10,
        max_chars=4000,
        typed_only=True,
    )

    assert ledger["coherent"] is True
    assert ledger["typed_ledger"] == "schema_field_count"
    assert ledger["computed_answer"] == "2"
    assert "due_date" in ledger["text"]
    assert "status" in ledger["text"]
    assert "description" not in ledger["text"]


def test_count_list_ledger_sums_semantic_money_without_object_specific_rules() -> None:
    item = {
        "question": "What is the total amount of money I raised from charity events?",
        "question_type": "multi-session",
    }
    selected = ["auction", "run", "future"]
    turns_by_sid = {
        "auction": [{"role": "user", "content": "The charity auction raised $2,400 for the shelter."}],
        "run": [{"role": "user", "content": "The charity run raised $1,350 last weekend."}],
        "future": [{"role": "user", "content": "I might plan another charity event that could raise $500."}],
    }

    ledger = build_count_list_ledger(
        item,
        selected,
        {},
        turns_by_sid,
        max_candidates=10,
        max_chars=4000,
        typed_only=True,
        semantic_counting=True,
    )

    assert ledger["coherent"] is True
    assert ledger["typed_ledger"] == "semantic_numeric_sum_dollars"
    assert ledger["computed_answer"] == "$3750"
    assert "might plan" not in ledger["text"]


def test_count_list_ledger_sums_semantic_page_counts_without_book_specific_rules() -> None:
    item = {
        "question": "What was the page count of the two novels I finished?",
        "question_type": "multi-session",
    }
    selected = ["novel-a", "novel-b"]
    turns_by_sid = {
        "novel-a": [{"role": "user", "content": "I finished the first novel, which was 416 pages."}],
        "novel-b": [{"role": "user", "content": "The second novel I finished was 440 pages."}],
    }

    ledger = build_count_list_ledger(
        item,
        selected,
        {},
        turns_by_sid,
        max_candidates=10,
        max_chars=4000,
        typed_only=True,
        semantic_counting=True,
    )

    assert ledger["coherent"] is True
    assert ledger["typed_ledger"] == "semantic_numeric_sum_pages"
    assert ledger["computed_answer"] == "856 pages"


def test_count_list_ledger_counts_quantified_entities_without_object_specific_rules() -> None:
    item = {
        "question": "How many parts are in both repair boxes?",
        "question_type": "multi-session",
    }
    selected = ["box-a", "box-b"]
    turns_by_sid = {
        "box-a": [{"role": "user", "content": "The first repair box has 2 hinges and 3 brackets."}],
        "box-b": [{"role": "user", "content": "The second repair box has one latch."}],
    }

    ledger = build_count_list_ledger(
        item,
        selected,
        {},
        turns_by_sid,
        max_candidates=10,
        max_chars=4000,
        typed_only=True,
        semantic_counting=True,
    )

    assert ledger["coherent"] is True
    assert ledger["typed_ledger"] == "semantic_quantified_entity_count"
    assert ledger["computed_answer"] == "6"


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


def test_multi_session_evidence_set_scans_broader_pool_than_prompt_context() -> None:
    item = {
        "question": "How many model kits did I assemble?",
        "question_type": "multi-session",
    }
    selected = [f"s{i}" for i in range(1, 8)]
    date_by_sid = {sid: f"2023/01/0{i}" for i, sid in enumerate(selected, start=1)}
    turns_by_sid = {
        "s1": [{"role": "user", "content": "I assembled the Saturn V model kit."}],
        "s2": [{"role": "user", "content": "I bought groceries and cleaned the garage."}],
        "s3": [{"role": "user", "content": "I assembled the Spitfire model kit."}],
        "s4": [{"role": "user", "content": "We discussed dinner plans."}],
        "s5": [{"role": "user", "content": "I assembled the Gundam model kit."}],
        "s6": [{"role": "user", "content": "I assembled the locomotive model kit."}],
        "s7": [{"role": "user", "content": "I assembled the lunar rover model kit."}],
    }

    evidence_set = build_multi_session_evidence_set(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        base_top_k=2,
        max_candidates=8,
        max_chars=4000,
    )

    assert evidence_set["coherent"] is True
    assert evidence_set["confidence"] >= 0.72
    assert "three_plus_dedupe_groups" in evidence_set["confidence_reasons"]
    assert evidence_set["selected_source_ranks"]["s7"] == 7
    assert evidence_set["novel_decisive_group_count"] >= 1
    assert evidence_set["marginal_utility"] > 0
    assert "Marginal utility:" in evidence_set["text"]
    assert "Saturn V model kit" in evidence_set["text"]
    assert "Selector confidence:" in evidence_set["text"]
    assert "lunar rover model kit" in evidence_set["text"]
    assert "s7" in evidence_set["selected_source_ids"]


def test_multi_session_evidence_set_reports_no_marginal_utility_when_base_covers_selected_groups() -> None:
    item = {
        "question": "How many model kits did I assemble?",
        "question_type": "multi-session",
    }
    selected = ["s1", "s2", "s3"]
    date_by_sid = {sid: f"2023/01/0{i}" for i, sid in enumerate(selected, start=1)}
    turns_by_sid = {
        "s1": [{"role": "user", "content": "I assembled the Saturn V model kit."}],
        "s2": [{"role": "user", "content": "I assembled the Spitfire model kit."}],
        "s3": [{"role": "user", "content": "I cleaned the garage."}],
    }

    evidence_set = build_multi_session_evidence_set(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        base_top_k=3,
        max_candidates=8,
        max_chars=4000,
    )

    assert evidence_set["coherent"] is True
    assert evidence_set["pool_added_group_count"] == 0
    assert evidence_set["novel_decisive_group_count"] == 0
    assert evidence_set["marginal_utility"] == 0


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


def test_fusion_evidence_map_preserves_relationship_constraints() -> None:
    item = {
        "question": "How many months have passed since I last visited a museum with a friend?",
        "question_type": "temporal-reasoning",
    }
    selected = ["friend-visit", "dad-visit"]
    date_by_sid = {"friend-visit": "2022/10/22", "dad-visit": "2023/02/18"}
    turns_by_sid = {
        "friend-visit": [{"role": "user", "content": "I visited the Science Museum with a friend."}],
        "dad-visit": [{"role": "user", "content": "I visited the Natural History Museum with my dad."}],
    }

    evidence_map = build_fusion_evidence_map(item, selected, date_by_sid, turns_by_sid)

    assert "Temporal Constraints" in evidence_map
    assert "Question companion/person constraints: friend" in evidence_map
    assert "do not substitute conflicting relationship terms" in evidence_map
    assert "relationship_constraint_mismatch" in evidence_map
    assert "Natural History Museum with my dad" in evidence_map


def test_fusion_evidence_map_anchors_recent_actions_to_session_date() -> None:
    item = {
        "question": "How many days ago did I buy a smoker?",
        "question_type": "temporal-reasoning",
    }
    selected = ["smoker", "cookout"]
    date_by_sid = {"smoker": "2023/03/15", "cookout": "2023/03/20"}
    turns_by_sid = {
        "smoker": [{"role": "user", "content": "I just got a smoker for the backyard."}],
        "cookout": [{"role": "user", "content": "I used the smoker for a cookout."}],
    }

    evidence_map = build_fusion_evidence_map(item, selected, date_by_sid, turns_by_sid)

    assert "session_date_anchor" in evidence_map
    assert "date=2023/03/15" in evidence_map
    assert "I just got a smoker" in evidence_map


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
