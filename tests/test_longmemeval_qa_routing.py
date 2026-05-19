from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_qa_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "longmemeval_contextfit_qa.py"
    spec = importlib.util.spec_from_file_location("longmemeval_contextfit_qa", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evidence_packet_general_skips_multisession_count_list() -> None:
    qa = _load_qa_module()
    item = {
        "question": "How many items are currently on my reading list?",
        "question_type": "multi-session",
    }

    assert qa.should_use_evidence_packet(item, "general") is False
    assert qa.should_use_evidence_packet(item, "all") is True


def test_evidence_packet_general_skips_multisession_non_count_list() -> None:
    qa = _load_qa_module()
    item = {
        "question": "What did I decide after comparing both conversations?",
        "question_type": "multi-session",
    }

    assert qa.should_use_evidence_packet(item, "general") is False
    assert qa.should_use_evidence_packet(item, "all") is True


def test_evidence_packet_general_keeps_temporal_update_route() -> None:
    qa = _load_qa_module()
    item = {
        "question": "What podcast do I listen to now for news?",
        "question_type": "temporal-reasoning",
    }

    assert qa.should_use_evidence_packet(item, "general") is True


def test_evidence_packet_general_keeps_non_multisession_count_list() -> None:
    qa = _load_qa_module()
    item = {
        "question": "How many vacation days did I mention?",
        "question_type": "single-session-user",
    }

    assert qa.should_use_evidence_packet(item, "general") is True


def test_evidence_packet_general_skips_preference_rows() -> None:
    qa = _load_qa_module()
    item = {
        "question": "What hotel should I pick based on what I like?",
        "question_type": "single-session-preference",
    }

    assert qa.should_use_evidence_packet(item, "general") is False
    assert qa.should_use_evidence_packet(item, "all") is True


def test_verifier_flags_final_answer_after_missing_source_note() -> None:
    qa = _load_qa_module()
    item = {
        "question": "Where did I buy the tennis racket?",
        "question_type": "single-session-user",
    }
    hypothesis = (
        "Source Notes:\n"
        "Retrieved Session 2: The store is not mentioned.\n\n"
        "Final Answer:\n"
        "You bought it downtown."
    )

    report = qa.deterministic_verification_report(item, hypothesis)

    assert report["needs_correction"] is True
    assert "final_answer_despite_missing_source_evidence" in report["flags"]


def test_verifier_accepts_supported_source_answer() -> None:
    qa = _load_qa_module()
    item = {
        "question": "Where did I buy the tennis racket?",
        "question_type": "single-session-user",
    }
    hypothesis = (
        "Source Notes:\n"
        "Retrieved Session 6: The user said the new tennis racket was bought from a sports store downtown.\n\n"
        "Final Answer:\n"
        "You bought it from a sports store downtown."
    )

    report = qa.deterministic_verification_report(item, hypothesis)

    assert report["needs_correction"] is False
    assert report["flags"] == []


def test_verifier_accepts_markdown_heading_sections() -> None:
    qa = _load_qa_module()
    item = {
        "question": "What is the Spotify playlist name?",
        "question_type": "single-session-user",
    }
    hypothesis = (
        "## Required Evidence\n"
        "Find the exact playlist name.\n\n"
        "## Source Notes\n"
        "Retrieved Session 4: The user said the Spotify playlist was called Summer Vibes.\n\n"
        "## Sufficiency: sufficient\n\n"
        "## Final Answer\n"
        "The playlist was called Summer Vibes."
    )

    report = qa.deterministic_verification_report(item, hypothesis)

    assert report["needs_correction"] is False
    assert report["flags"] == []


def test_verifier_accepts_corrected_candidate_evidence_sections() -> None:
    qa = _load_qa_module()
    item = {
        "question": "How many playlists do I have?",
        "question_type": "single-session-user",
    }
    hypothesis = (
        "## Candidate Evidence\n"
        "Retrieved Session 3: The user explicitly states they have 20 playlists.\n\n"
        "## Deduped Set\n"
        "- One direct count candidate: 20 playlists\n\n"
        "## Final Answer\n"
        "You have 20 playlists."
    )

    report = qa.deterministic_verification_report(item, hypothesis)

    assert report["needs_correction"] is False
    assert report["flags"] == []


def test_verifier_flags_count_list_without_aggregation_trace() -> None:
    qa = _load_qa_module()
    item = {
        "question": "How many vacation days did I mention?",
        "question_type": "single-session-user",
    }
    hypothesis = (
        "Source Notes:\n"
        "Retrieved Session 1: The user talked about vacation days.\n\n"
        "Final Answer:\n"
        "Five."
    )

    report = qa.deterministic_verification_report(item, hypothesis)

    assert report["needs_correction"] is True
    assert "count_list_lacks_aggregation_trace" in report["flags"]


def test_multi_session_evidence_ledger_keeps_source_rows_and_dedupe_keys() -> None:
    qa = _load_qa_module()
    item = {
        "question": "How many museums did I visit?",
        "question_type": "multi-session",
    }
    selected = ["s1", "s2"]
    date_by_sid = {"s1": "2023/01/01", "s2": "2023/01/03"}
    turns_by_sid = {
        "s1": [{"role": "user", "content": "I visited the MoMA museum on January 1."}],
        "s2": [{"role": "user", "content": "I also visited the Met museum two days later."}],
    }

    ledger = qa.build_multi_session_evidence_ledger(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        max_rows=10,
        max_chars=4000,
    )

    assert "Multi-Session Evidence Ledger" in ledger
    assert "E1: source=S1" in ledger
    assert "E2: source=S2" in ledger
    assert "dedupe_key=" in ledger


def test_multi_session_compiler_prompt_is_strict_and_not_generic_packet() -> None:
    qa = _load_qa_module()
    item = {
        "question_id": "q1",
        "question": "How many museums did I visit?",
        "question_type": "multi-session",
        "question_date": "2023/01/04",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2023/01/01"],
        "haystack_sessions": [[{"role": "user", "content": "I visited the MoMA museum."}]],
    }

    prompt = qa.build_multi_session_compiler_prompt(
        item,
        ["s1"],
        top_k=10,
        max_session_chars=2000,
        max_ledger_rows=10,
        max_ledger_chars=4000,
    )

    assert "strict evidence compilation contract" in prompt
    assert "Multi-Session Evidence Ledger" in prompt
    assert "Candidate Set, Deduped Set" in prompt
    assert "Deterministic Evidence Packet" not in prompt


def test_multi_session_guided_prompt_keeps_full_history_authoritative() -> None:
    qa = _load_qa_module()
    item = {
        "question_id": "q1",
        "question": "How many museums did I visit?",
        "question_type": "multi-session",
        "question_date": "2023/01/04",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2023/01/01"],
        "haystack_sessions": [[{"role": "user", "content": "I visited the MoMA museum."}]],
    }

    prompt = qa.build_multi_session_guided_prompt(
        item,
        ["s1"],
        top_k=10,
        max_session_chars=2000,
        max_ledger_rows=10,
        max_ledger_chars=4000,
    )

    assert "full History Chats are authoritative" in prompt
    assert "Source Notes" in prompt
    assert "Deterministic Evidence Packet" not in prompt


def test_source_set_aware_prompt_separates_primary_and_companion_sources() -> None:
    qa = _load_qa_module()
    item = {
        "question_id": "q1",
        "question": "How many museums did I visit?",
        "question_type": "multi-session",
        "question_date": "2023/01/04",
        "haystack_session_ids": ["s1", "s2", "s3"],
        "haystack_dates": ["2023/01/01", "2023/01/02", "2023/01/03"],
        "haystack_sessions": [
            [{"role": "user", "content": "I visited the MoMA museum."}],
            [{"role": "user", "content": "I had lunch near a museum."}],
            [{"role": "user", "content": "I also visited the Met museum."}],
        ],
    }

    prompt = qa.build_source_set_aware_prompt(
        item,
        primary_sessions=["s1", "s2"],
        expanded_sessions=["s1", "s3", "s2"],
        primary_top_k=2,
        companion_top_k=5,
        max_session_chars=2000,
    )

    assert "Primary Sources" in prompt
    assert "Added Companion Sources" in prompt
    assert "Companion Audit" in prompt
    assert "changes_candidate_set: yes" in prompt
    assert "Candidate Set" in prompt
    assert "Session ID: s3" in prompt
    assert prompt.count("Session ID: s1") == 1


def test_aggregation_assembly_builds_candidate_rows_and_groups() -> None:
    qa = _load_qa_module()
    item = {
        "question_id": "q1",
        "question": "How many museums did I visit?",
        "question_type": "multi-session",
        "question_date": "2023/01/04",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2023/01/01", "2023/01/03"],
        "haystack_sessions": [
            [{"role": "user", "content": "I visited the MoMA museum on Monday."}],
            [{"role": "user", "content": "I also visited the Met museum two days later."}],
        ],
    }
    date_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_dates"], strict=True))
    turns_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True))

    assembly = qa.build_deterministic_aggregation_assembly(
        item,
        ["s1", "s2"],
        date_by_sid,
        turns_by_sid,
        max_candidates=10,
        max_chars=4000,
    )

    assert assembly["coherent"] is True
    assert assembly["candidate_count"] == 2
    assert "C1:" in assembly["text"]
    assert "Dedupe Groups" in assembly["text"]
    assert "MoMA museum" in assembly["text"]
    assert "Met museum" in assembly["text"]


def test_aggregation_assembly_prompt_requires_deduped_set() -> None:
    qa = _load_qa_module()
    item = {
        "question_id": "q1",
        "question": "How many museums did I visit?",
        "question_type": "multi-session",
        "question_date": "2023/01/04",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2023/01/01"],
        "haystack_sessions": [[{"role": "user", "content": "I visited the MoMA museum."}]],
    }

    prompt, report = qa.build_aggregation_assembly_prompt(
        item,
        ["s1"],
        top_k=10,
        max_session_chars=2000,
        aggregation_max_candidates=10,
        aggregation_max_chars=4000,
    )

    assert report["coherent"] is True
    assert "Deterministic Aggregation Assembly" in prompt
    assert "Candidate Review" in prompt
    assert "Deduped Set" in prompt
    assert "Final Answer" in prompt


def test_aggregation_assembly_router_is_narrow() -> None:
    qa = _load_qa_module()
    count_item = {
        "question": "How many museums did I visit?",
        "question_type": "multi-session",
    }
    non_count_item = {
        "question": "Which museum did I visit first?",
        "question_type": "multi-session",
    }

    assert qa.should_use_aggregation_assembly(count_item, "multi_session_count_list") is True
    assert qa.should_use_aggregation_assembly(non_count_item, "multi_session_count_list") is False
    assert qa.should_use_aggregation_assembly(count_item, "off") is False
