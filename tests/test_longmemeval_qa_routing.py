from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace
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


def test_temporal_evidence_packet_can_be_disabled_per_item() -> None:
    qa = _load_qa_module()
    temporal_item = {
        "question": "How long was it between signing and launch?",
        "question_type": "temporal-reasoning",
    }
    update_item = {
        "question": "What podcast do I listen to now for news?",
        "question_type": "knowledge-update",
    }
    args = SimpleNamespace(evidence_packet="general", temporal_evidence_packet="off")

    assert qa.should_use_evidence_packet_for_item(temporal_item, args) is False
    assert qa.should_use_evidence_packet_for_item(update_item, args) is True


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


def test_effective_top_k_context_uses_task_specific_overrides() -> None:
    qa = _load_qa_module()
    args = SimpleNamespace(top_k_context=5, multi_session_top_k_context=10, temporal_top_k_context=8)

    assert qa.effective_top_k_context({"question_type": "multi-session"}, args) == 10
    assert qa.effective_top_k_context({"question_type": "temporal-reasoning"}, args) == 8
    assert qa.effective_top_k_context({"question_type": "single-session-preference"}, args) == 5


def test_openai_compatible_sanitizer_trims_chat_role_echo_after_answer() -> None:
    qa = _load_qa_module()
    text = "42\nuser\nassistant\nI'm sorry, but I can't help with that."

    assert qa.sanitize_openai_compatible_text(text) == "42"


def test_openai_compatible_sanitizer_preserves_normal_multiline_answer() -> None:
    qa = _load_qa_module()
    text = "Source Notes:\nRetrieved Session 1: Evidence.\n\nFinal Answer:\n42"

    assert qa.sanitize_openai_compatible_text(text) == text


def test_openai_compatible_sanitizer_skips_leading_role_echo() -> None:
    qa = _load_qa_module()
    text = "assistant\nqwen-ready\n<|im_end|>\nuser\nrepeat"

    assert qa.sanitize_openai_compatible_text(text) == "qwen-ready"


def test_parse_yes_no_label_uses_first_label() -> None:
    qa = _load_qa_module()

    assert qa.parse_yes_no_label("No. The answer mentions yes later.") is False
    assert qa.parse_yes_no_label("YES\nReason: supported.") is True


def test_completion_cache_key_includes_seed_and_prompt() -> None:
    qa = _load_qa_module()
    messages = [{"role": "user", "content": "Question"}]

    key_a = qa.completion_cache_key(
        purpose="judge",
        qid="q1",
        model="gpt-4o-2024-08-06",
        messages=messages,
        max_tokens=10,
        temperature=0.0,
        seed=123,
    )
    key_b = qa.completion_cache_key(
        purpose="judge",
        qid="q1",
        model="gpt-4o-2024-08-06",
        messages=messages,
        max_tokens=10,
        temperature=0.0,
        seed=456,
    )

    assert key_a != key_b


def test_sanitized_args_excludes_api_key() -> None:
    qa = _load_qa_module()
    args = SimpleNamespace(api_key="secret", openai_key="secret", data=Path("data.json"), top_k_context=5)

    sanitized = qa.sanitized_args_dict(args)

    assert "api_key" not in sanitized
    assert "openai_key" not in sanitized
    assert sanitized["data"] == "data.json"
    assert sanitized["top_k_context"] == 5


def test_summary_includes_run_provenance(tmp_path: Path) -> None:
    qa = _load_qa_module()
    data_path = tmp_path / "data.json"
    retrieval_path = tmp_path / "retrieval.json"
    judged_path = tmp_path / "judged.jsonl"
    hypotheses_path = tmp_path / "hypotheses.jsonl"
    summary_path = tmp_path / "summary.json"
    answerability_path = tmp_path / "answerability.jsonl"
    extract_path = tmp_path / "extract.jsonl"
    qid_path = tmp_path / "qids.txt"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    data_path.write_text(json.dumps([{"question_id": "q1", "question_type": "temporal-reasoning"}]))
    retrieval_path.write_text(json.dumps({"rows": []}))
    qid_path.write_text("q1\n")
    judged_path.write_text(json.dumps({"question_id": "q1", "route": "source_aware", "autoeval_label": {"label": True}}) + "\n")
    (cache_dir / "generation.json").write_text(
        json.dumps(
            {
                "purpose": "generation",
                "question_id": "q1",
                "model": "gpt-4o-2024-08-06",
                "seed": 123,
                "metadata": {"system_fingerprint": "fp_test"},
                "text": "answer",
            }
        )
    )
    args = SimpleNamespace(
        data=data_path,
        retrieval_artifact=retrieval_path,
        question_id_file=qid_path,
        supporting_retrieval_artifact=None,
        primary_retrieval_artifact=None,
        hypotheses_out=hypotheses_path,
        judged_out=judged_path,
        summary_out=summary_path,
        answerability_out=answerability_path,
        extract_out=extract_path,
        completion_cache_dir=cache_dir,
        top_k_context=5,
        multi_session_top_k_context=10,
        temporal_top_k_context=8,
        preference_support_packet="off",
        preference_support_max_items=10,
        preference_support_per_source=2,
        multi_session_evidence_set="off",
        multi_session_evidence_set_min_confidence=0.72,
        evidence_packet="general",
        temporal_evidence_packet="off",
        fusion_evidence_map="temporal",
        aggregation_assembly="off",
        source_aware=True,
        source_sufficiency=False,
        generation_model="gpt-4o-2024-08-06",
        judge_model="gpt-4o-2024-08-06",
        answerability_model="gpt-4o-2024-08-06",
        extraction_model="gpt-4o-2024-08-06",
        generation_seed=123,
        judge_seed=456,
        answerability_seed=789,
        api_key="secret",
    )

    summary = qa.summarize(judged_path, data_path, args)

    provenance = summary["provenance"]
    assert provenance["run_hash"]
    assert "api_key" not in provenance["args"]
    assert provenance["inputs"]["data"]
    assert provenance["route_settings"]["temporal_top_k_context"] == 8
    assert provenance["route_settings"]["multi_session_evidence_set_min_confidence"] == 0.72
    assert provenance["cache"]["by_purpose"]["generation"]["system_fingerprints"] == ["fp_test"]


def test_preference_support_packet_uses_only_retrieved_sources() -> None:
    qa = _load_qa_module()
    item = {
        "question_id": "q1",
        "question": "Any tips on what to bake for colleagues?",
        "question_type": "single-session-preference",
        "question_date": "2023/01/04",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2023/01/01", "2023/01/03"],
        "haystack_sessions": [
            [{"role": "user", "content": "I've made a lemon poppyseed cake before, and it was a hit."}],
            [{"role": "user", "content": "I have a pasta maker."}],
        ],
    }

    prompt = qa.build_answer_prompt(
        item,
        ["s1"],
        top_k=1,
        max_session_chars=2000,
        cot=False,
        source_aware=True,
        source_sufficiency=False,
        token_evidence=False,
        evidence_packet=False,
        fusion_evidence_map=False,
        fusion_evidence_map_max_items=0,
        fusion_evidence_map_max_chars=0,
        evidence_packet_max_items=0,
        evidence_packet_max_chars=0,
        token_evidence_max_lines=0,
        token_evidence_signals=False,
        count_list_mode=False,
        strict_missing_final_answer=False,
        preference_support_packet="retrieved",
        preference_support_max_items=4,
        preference_support_per_source=2,
    )

    assert "Preference support candidates" in prompt
    assert "lemon poppyseed cake" in prompt
    assert "pasta maker" not in prompt


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


def test_count_list_ledger_prompt_exposes_python_computed_count() -> None:
    qa = _load_qa_module()
    item = {
        "question_id": "q1",
        "question": "How many museums did I visit?",
        "question_type": "multi-session",
        "question_date": "2023/01/04",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2023/01/01", "2023/01/03"],
        "haystack_sessions": [
            [{"role": "user", "content": "I visited the MoMA museum."}],
            [{"role": "user", "content": "I visited the Met museum."}],
        ],
    }

    prompt, report = qa.build_count_list_ledger_prompt(
        item,
        ["s1", "s2"],
        top_k=10,
        max_session_chars=2000,
        max_candidates=10,
        max_chars=4000,
    )

    assert report["coherent"] is True
    assert report["computed_count"] == 2
    assert "Python Count/List Ledger" in prompt
    assert "Python computed count is 2" in prompt
    assert "Accepted Groups" in prompt
    assert "Final Answer" in prompt


def test_count_list_ledger_prompt_exposes_typed_computed_answer() -> None:
    qa = _load_qa_module()
    item = {
        "question_id": "q1",
        "question": "What is the average age of me, my parents, and my grandparents?",
        "question_type": "multi-session",
        "question_date": "2023/05/30",
        "haystack_session_ids": ["me", "parents", "grandparents"],
        "haystack_dates": ["2023/05/26", "2023/05/23", "2023/05/22"],
        "haystack_sessions": [
            [{"role": "user", "content": "I just turned 32 on February 12th."}],
            [{"role": "user", "content": "My mom is 55 and my dad is 58."}],
            [{"role": "user", "content": "My grandma is 75 and my grandpa is 78."}],
        ],
    }

    prompt, report = qa.build_count_list_ledger_prompt(
        item,
        ["me", "parents", "grandparents"],
        top_k=10,
        max_session_chars=2000,
        max_candidates=10,
        max_chars=4000,
    )

    assert report["coherent"] is True
    assert report["computed_answer"] == "59.6"
    assert report["typed_ledger"] == "age_average"
    assert "Python computed answer is 59.6" in prompt
    assert "Typed ledger: age_average" in prompt


def test_count_list_agent_extraction_prompt_uses_source_ids() -> None:
    qa = _load_qa_module()
    item = {
        "question_id": "q1",
        "question": "How many museums did I visit?",
        "question_type": "multi-session",
        "question_date": "2023/01/04",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2023/01/01", "2023/01/03"],
        "haystack_sessions": [
            [{"role": "user", "content": "I visited the MoMA museum."}],
            [{"role": "user", "content": "I visited the Met museum."}],
        ],
    }

    prompt, source_ids = qa.build_count_list_agent_extraction_prompt(
        item,
        ["s1", "s2"],
        top_k=10,
        max_session_chars=2000,
    )

    assert source_ids == {"S1": "s1", "S2": "s2"}
    assert "Return JSON only" in prompt
    assert '"operation": "count|sum|average|difference|duration|list|unknown"' in prompt
    assert "remaining-to-go questions" in prompt
    assert "combined totals" in prompt
    assert "calculation may only use" not in prompt
    assert "### S1" in prompt
    assert "### S2" in prompt


def test_count_list_agent_extraction_prompt_can_expose_calculation_plan() -> None:
    qa = _load_qa_module()
    item = {
        "question_id": "q1",
        "question": "How many total pages did I read?",
        "question_type": "multi-session",
        "question_date": "2023/01/04",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2023/01/01"],
        "haystack_sessions": [[{"role": "user", "content": "I read a 416-page novel."}]],
    }

    prompt, _ = qa.build_count_list_agent_extraction_prompt(
        item,
        ["s1"],
        top_k=10,
        max_session_chars=2000,
        include_calculation_plan=True,
    )

    assert "calculation may only use" in prompt
    assert '"calculation": {"function": "count_rows|sum_values|difference|average_values|list_entities"' in prompt


def test_count_list_agent_extraction_validation_computes_sum() -> None:
    qa = _load_qa_module()
    raw = json.dumps(
        {
            "operation": "sum",
            "unit": "rides",
            "answerability": "sufficient",
            "accepted": [
                {"entity": "first park visit", "value": 3, "source": "S1", "evidence": "three rides"},
                {"entity": "second park visit", "value": "2", "source": "S2", "evidence": "two rides"},
            ],
            "rejected": [],
            "dedupe_notes": [],
            "missing_or_uncertain": [],
        }
    )

    report = qa.validate_count_list_agent_extraction(raw, {"S1": "s1", "S2": "s2"})

    assert report["coherent"] is True
    assert report["operation"] == "sum"
    assert report["computed_answer"] == "5"
    assert report["accepted_count"] == 2
    assert report["accepted"][0]["session_id"] == "s1"


def test_count_list_agent_extraction_validation_sums_count_multiplicities() -> None:
    qa = _load_qa_module()
    raw = json.dumps(
        {
            "operation": "count",
            "unit": "rides",
            "answerability": "sufficient",
            "accepted": [
                {"entity": "first coaster", "value": 3, "source": "S1", "evidence": "rode it three times"},
                {"entity": "second coaster", "value": 2, "source": "S2", "evidence": "rode twice"},
            ],
            "rejected": [],
            "dedupe_notes": [],
            "missing_or_uncertain": [],
        }
    )

    report = qa.validate_count_list_agent_extraction(raw, {"S1": "s1", "S2": "s2"})

    assert report["coherent"] is True
    assert report["operation"] == "count"
    assert report["computed_answer"] == "5"


def test_count_list_agent_extraction_keeps_distinct_same_source_rows() -> None:
    qa = _load_qa_module()
    raw = json.dumps(
        {
            "operation": "count",
            "unit": "repairs",
            "answerability": "sufficient",
            "accepted": [
                {
                    "entity": "appliance upgrade",
                    "value": 1,
                    "source": "S1",
                    "evidence": "donated the old appliance and enjoy the upgrade",
                },
                {
                    "entity": "appliance upgrade",
                    "value": 1,
                    "source": "S1",
                    "evidence": "replaced the old appliance with a larger countertop oven",
                },
            ],
            "rejected": [],
            "dedupe_notes": [],
            "missing_or_uncertain": [],
        }
    )

    report = qa.validate_count_list_agent_extraction(raw, {"S1": "s1"})

    assert report["coherent"] is True
    assert report["accepted_count"] == 2
    assert report["computed_answer"] == "2"


def test_count_list_agent_extraction_validation_computes_difference() -> None:
    qa = _load_qa_module()
    raw = json.dumps(
        {
            "operation": "difference",
            "unit": "points",
            "answerability": "sufficient",
            "accepted": [
                {"entity": "current rewards balance", "value": 200, "source": "S1", "evidence": "total to 200 points"},
                {"entity": "redemption target", "value": 300, "source": "S2", "evidence": "need 300 points"},
            ],
            "rejected": [],
            "dedupe_notes": [],
            "missing_or_uncertain": [],
        }
    )

    report = qa.validate_count_list_agent_extraction(raw, {"S1": "s1", "S2": "s2"})

    assert report["coherent"] is True
    assert report["operation"] == "difference"
    assert report["computed_answer"] == "100"


def test_count_list_agent_extraction_validation_accepts_precomputed_difference() -> None:
    qa = _load_qa_module()
    raw = json.dumps(
        {
            "operation": "difference",
            "unit": "points",
            "answerability": "sufficient",
            "accepted": [
                {
                    "entity": "remaining points needed",
                    "value": 100,
                    "source": "S1",
                    "evidence": "need 100 more points to reach the reward target",
                }
            ],
            "rejected": [],
            "dedupe_notes": [],
            "missing_or_uncertain": [],
        }
    )

    report = qa.validate_count_list_agent_extraction(raw, {"S1": "s1"})

    assert report["coherent"] is True
    assert report["operation"] == "difference"
    assert report["computed_answer"] == "100"


def test_count_list_agent_extraction_validation_executes_sum_plan() -> None:
    qa = _load_qa_module()
    raw = json.dumps(
        {
            "operation": "list",
            "unit": "pages",
            "answerability": "sufficient",
            "accepted": [
                {"entity": "January novel", "value": 416, "source": "S1", "evidence": "416-page novel"},
                {"entity": "March novel", "value": 440, "source": "S2", "evidence": "440 pages"},
            ],
            "rejected": [],
            "dedupe_notes": [],
            "missing_or_uncertain": [],
            "calculation": {"function": "sum_values", "rows": ["A1", "A2"]},
        }
    )

    report = qa.validate_count_list_agent_extraction(
        raw,
        {"S1": "s1", "S2": "s2"},
        allow_calculation_plan=True,
    )

    assert report["coherent"] is True
    assert report["computed_answer"] == "856"
    assert report["calculation"]["function"] == "sum_values"


def test_count_list_agent_extraction_validation_derives_schema_sum() -> None:
    qa = _load_qa_module()
    raw = json.dumps(
        {
            "operation": "list",
            "unit": "pages",
            "answerability": "sufficient",
            "accepted": [
                {"entity": "January novel", "value": 416, "source": "S1", "evidence": "416-page novel"},
                {"entity": "March novel", "value": 440, "source": "S2", "evidence": "440 pages"},
            ],
            "rejected": [],
            "dedupe_notes": [],
            "missing_or_uncertain": [],
        }
    )

    report = qa.validate_count_list_agent_extraction(
        raw,
        {"S1": "s1", "S2": "s2"},
        derive_schema_operation=True,
        question="How many total pages did I read across those books?",
    )

    assert report["coherent"] is True
    assert report["operation"] == "sum"
    assert report["computed_answer"] == "856"
    assert report["schema_calculation"]["function"] == "sum_values"


def test_count_list_agent_extraction_validation_rejects_unknown_plan_function() -> None:
    qa = _load_qa_module()
    raw = json.dumps(
        {
            "operation": "sum",
            "unit": "pages",
            "answerability": "sufficient",
            "accepted": [
                {"entity": "January novel", "value": 416, "source": "S1", "evidence": "416 pages"}
            ],
            "rejected": [],
            "dedupe_notes": [],
            "missing_or_uncertain": [],
            "calculation": {"function": "eval_python", "rows": ["A1"]},
        }
    )

    report = qa.validate_count_list_agent_extraction(
        raw,
        {"S1": "s1"},
        allow_calculation_plan=True,
    )

    assert report["coherent"] is True
    assert report["computed_answer"] == "416"
    assert report["calculation"]["coherent"] is False


def test_count_list_agent_hypothesis_formats_usd_without_generation() -> None:
    qa = _load_qa_module()
    report = {
        "accepted": [
            {"entity": "charity walk", "source": "S1", "session_id": "s1", "value": 250.0, "evidence": "raised $250"},
        ],
        "rejected": [],
        "unit": "USD",
        "computed_answer": "250",
    }

    hypothesis = qa.build_count_list_agent_hypothesis(report)

    assert "Accepted Rows" in hypothesis
    assert "Final Answer: $250" in hypothesis


def test_count_list_agent_extraction_validation_rejects_unknown_source() -> None:
    qa = _load_qa_module()
    raw = json.dumps(
        {
            "operation": "count",
            "unit": "museums",
            "answerability": "sufficient",
            "accepted": [{"entity": "MoMA", "value": 1, "source": "S9", "evidence": "visited MoMA"}],
            "rejected": [],
            "dedupe_notes": [],
            "missing_or_uncertain": [],
        }
    )

    report = qa.validate_count_list_agent_extraction(raw, {"S1": "s1"})

    assert report["coherent"] is False
    assert report["accepted_count"] == 0
    assert report["computed_answer"] is None


def test_count_list_agent_extraction_validation_accepts_duration_sources() -> None:
    qa = _load_qa_module()
    raw = json.dumps(
        {
            "operation": "duration",
            "unit": "days",
            "answerability": "sufficient",
            "accepted": [
                {
                    "entity": "order to delivery interval",
                    "value": 5,
                    "source": "S1, S2",
                    "evidence": "ordered Monday and delivered Saturday",
                }
            ],
            "rejected": [],
            "dedupe_notes": [],
            "missing_or_uncertain": [],
        }
    )

    report = qa.validate_count_list_agent_extraction(raw, {"S1": "ordered", "S2": "delivered"})

    assert report["coherent"] is True
    assert report["operation"] == "duration"
    assert report["computed_answer"] == "5"
    assert report["accepted"][0]["session_id"] == "ordered,delivered"


def test_count_list_agent_route_rejects_uncertain_reports() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "operation": "count",
        "unit": "courses",
        "accepted_count": 2,
        "missing_or_uncertain": ["One candidate is ambiguous."],
    }

    assert qa.is_safe_count_list_agent_route(report) is False


def test_count_list_agent_route_rejects_broad_item_counts() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "operation": "count",
        "unit": "kitchen items",
        "accepted_count": 4,
        "missing_or_uncertain": [],
    }

    assert qa.is_safe_count_list_agent_route(report) is False


def test_count_list_agent_route_accepts_specific_multi_candidate_counts() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "operation": "count",
        "unit": "rides",
        "accepted_count": 4,
        "missing_or_uncertain": [],
    }

    assert qa.is_safe_count_list_agent_route(report) is True


def test_count_list_agent_route_rejects_broad_money_sums() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "operation": "sum",
        "unit": "USD",
        "accepted_count": 5,
        "missing_or_uncertain": [],
    }

    assert qa.is_safe_count_list_agent_route(report) is False


def test_count_list_agent_route_rejects_broad_item_sums() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "operation": "sum",
        "unit": "items",
        "accepted_count": 3,
        "missing_or_uncertain": [],
    }

    assert qa.is_safe_count_list_agent_route(report) is False


def test_count_list_agent_route_rejects_unmatched_dimensional_modifier() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "operation": "count",
        "unit": "fish",
        "accepted_count": 3,
        "missing_or_uncertain": [],
        "question": "How many fish are there in my 30-gallon tank?",
        "accepted": [
            {
                "entity": "neon tetras",
                "evidence": "10 neon tetras in the aquarium",
            }
        ],
    }

    assert qa.is_safe_count_list_agent_route(report) is False


def test_count_list_agent_route_accepts_matched_dimensional_modifier() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "operation": "count",
        "unit": "fish",
        "accepted_count": 3,
        "missing_or_uncertain": [],
        "question": "How many fish are there in my 30-gallon tank?",
        "accepted": [
            {
                "entity": "30-gallon tank fish",
                "evidence": "In my 30 gallon tank, I have 10 neon tetras.",
            }
        ],
    }

    assert qa.is_safe_count_list_agent_route(report) is True


def test_count_list_agent_plan_route_requires_safe_calculation() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "operation": "sum",
        "unit": "pages",
        "accepted_count": 2,
        "missing_or_uncertain": [],
        "calculation": {
            "coherent": True,
            "function": "sum_values",
            "computed_answer": "856",
        },
    }

    assert qa.is_safe_count_list_agent_plan_route(report) is True


def test_count_list_agent_plan_route_rejects_count_rows() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "operation": "count",
        "unit": "classes",
        "accepted_count": 4,
        "missing_or_uncertain": [],
        "calculation": {
            "coherent": True,
            "function": "count_rows",
            "computed_answer": "4",
        },
    }

    assert qa.is_safe_count_list_agent_plan_route(report) is False


def test_count_list_agent_plan_route_rejects_zero_difference() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "operation": "difference",
        "unit": "days",
        "accepted_count": 2,
        "missing_or_uncertain": [],
        "calculation": {
            "coherent": True,
            "function": "difference",
            "computed_answer": "0",
        },
    }

    assert qa.is_safe_count_list_agent_plan_route(report) is False


def test_count_list_agent_schema_route_requires_schema_calculation() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "operation": "sum",
        "unit": "pages",
        "accepted_count": 2,
        "missing_or_uncertain": [],
        "schema_calculation": {
            "coherent": True,
            "function": "sum_values",
            "computed_answer": "856",
        },
    }

    assert qa.is_safe_count_list_agent_schema_route(report) is True


def test_count_list_agent_schema_route_falls_back_without_schema_calculation() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "operation": "sum",
        "unit": "pages",
        "accepted_count": 2,
        "missing_or_uncertain": [],
        "schema_calculation": None,
    }

    assert qa.is_safe_count_list_agent_schema_route(report) is True


def test_count_list_agent_blend_route_rejects_times_count() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "operation": "count",
        "unit": "times",
        "accepted_count": 3,
        "missing_or_uncertain": [],
        "accepted": [
            {"source": "S1", "entity": "cake"},
            {"source": "S2", "entity": "bread"},
            {"source": "S3", "entity": "cookies"},
        ],
        "schema_calculation": {
            "coherent": True,
            "function": "count_rows",
            "computed_answer": "3",
        },
    }

    assert qa.is_safe_count_list_agent_blend_route(report) is False


def test_count_list_agent_blend_route_rejects_many_rows_from_too_few_sources() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "operation": "count",
        "unit": "types",
        "accepted_count": 5,
        "missing_or_uncertain": [],
        "accepted": [
            {"source": "S1", "entity": "lime"},
            {"source": "S1", "entity": "orange"},
            {"source": "S1", "entity": "lemon"},
            {"source": "S2", "entity": "grapefruit"},
            {"source": "S2", "entity": "citrus"},
        ],
        "schema_calculation": {
            "coherent": True,
            "function": "count_rows",
            "computed_answer": "5",
        },
    }

    assert qa.is_safe_count_list_agent_blend_route(report) is False


def test_count_list_agent_blend_route_keeps_distributed_count() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "operation": "count",
        "unit": "festivals",
        "accepted_count": 4,
        "missing_or_uncertain": [],
        "accepted": [
            {"source": "S1", "entity": "Austin Film Festival"},
            {"source": "S2", "entity": "Seattle International Film Festival"},
            {"source": "S3", "entity": "Portland Film Festival"},
            {"source": "S4", "entity": "AFI Fest"},
        ],
        "schema_calculation": {
            "coherent": True,
            "function": "count_rows",
            "computed_answer": "4",
        },
    }

    assert qa.is_safe_count_list_agent_blend_route(report) is True


def test_count_list_ledger_rejects_percentage_comparison_questions() -> None:
    qa = _load_qa_module()

    assert (
        qa.is_unsupported_count_list_ledger_question(
            "Did I receive a higher percentage discount from Acme Meal Kits compared to Northwind Delivery?"
        )
        is True
    )


def test_count_list_ledger_rejects_percentage_of_ratio_questions() -> None:
    qa = _load_qa_module()

    assert (
        qa.is_unsupported_count_list_ledger_question(
            "What percentage of the property price is the renovation cost?"
        )
        is True
    )


def test_count_list_ledger_keeps_non_percentage_count_questions() -> None:
    qa = _load_qa_module()

    assert qa.is_unsupported_count_list_ledger_question("How many projects are still open?") is False


def test_multi_session_evidence_set_prompt_uses_broad_pool_sources() -> None:
    qa = _load_qa_module()
    item = {
        "question_id": "q1",
        "question": "How many model kits did I assemble?",
        "question_type": "multi-session",
        "question_date": "2023/01/08",
        "haystack_session_ids": ["s1", "s2", "s3", "s4"],
        "haystack_dates": ["2023/01/01", "2023/01/02", "2023/01/03", "2023/01/04"],
        "haystack_sessions": [
            [{"role": "user", "content": "I assembled the Saturn V model kit."}],
            [{"role": "user", "content": "I cleaned the garage."}],
            [{"role": "user", "content": "I assembled the Spitfire model kit."}],
            [{"role": "user", "content": "I assembled the lunar rover model kit."}],
        ],
    }

    prompt, report = qa.build_multi_session_evidence_set_prompt(
        item,
        ["s1", "s2", "s3", "s4"],
        top_k=2,
        pool_k=4,
        max_session_chars=2000,
        max_candidates=8,
        max_chars=4000,
    )

    assert report["coherent"] is True
    assert "Multi-Session Evidence Set" in prompt
    assert "lunar rover model kit" in prompt
    assert "s4" in report["selected_source_ids"]
    assert "Evidence Used" in prompt


def test_multi_session_source_select_prompt_adds_broad_pool_context() -> None:
    qa = _load_qa_module()
    item = {
        "question_id": "q1",
        "question": "How many model kits did I assemble?",
        "question_type": "multi-session",
        "question_date": "2023/01/08",
        "haystack_session_ids": ["s1", "s2", "s3"],
        "haystack_dates": ["2023/01/01", "2023/01/02", "2023/01/03"],
        "haystack_sessions": [
            [{"role": "user", "content": "I assembled the Saturn V model kit."}],
            [{"role": "user", "content": "I cleaned the garage."}],
            [{"role": "user", "content": "I assembled the lunar rover model kit."}],
        ],
    }

    prompt, report = qa.build_multi_session_source_select_prompt(
        item,
        ["s1", "s2", "s3"],
        top_k=1,
        pool_k=3,
        max_session_chars=2000,
        max_candidates=6,
        max_chars=3000,
    )

    assert report["coherent"] is True
    assert "s3" in report["selected_source_ids"]
    assert "lunar rover model kit" in prompt
    assert "Source Notes" in prompt


def test_confidence_gated_multi_session_selector_requires_threshold() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "confidence": 0.68,
        "novel_decisive_group_count": 1,
        "marginal_utility": 1,
    }

    assert qa.should_use_multi_session_evidence_set_selector(
        report,
        "source_select",
        min_confidence=0.72,
    ) is True
    assert qa.should_use_multi_session_evidence_set_selector(
        report,
        "confidence_source_select",
        min_confidence=0.72,
    ) is False
    report["confidence"] = 0.73
    assert qa.should_use_multi_session_evidence_set_selector(
        report,
        "confidence_source_select",
        min_confidence=0.72,
    ) is True


def test_confidence_gated_multi_session_selector_requires_marginal_utility() -> None:
    qa = _load_qa_module()
    report = {
        "coherent": True,
        "confidence": 0.96,
        "novel_decisive_group_count": 0,
        "marginal_utility": 0,
    }

    assert qa.should_use_multi_session_evidence_set_selector(
        report,
        "confidence_source_select",
        min_confidence=0.72,
    ) is False

    report["novel_decisive_group_count"] = 1
    report["marginal_utility"] = -1
    assert qa.should_use_multi_session_evidence_set_selector(
        report,
        "confidence_source_select",
        min_confidence=0.72,
    ) is False

    report["marginal_utility"] = 1
    assert qa.should_use_multi_session_evidence_set_selector(
        report,
        "confidence_source_select",
        min_confidence=0.72,
    ) is True


def test_confidence_gated_multi_session_selector_rejects_incoherent_report() -> None:
    qa = _load_qa_module()

    assert qa.should_use_multi_session_evidence_set_selector(
        {"coherent": False, "confidence": 1.0, "novel_decisive_group_count": 1, "marginal_utility": 1},
        "confidence_source_select",
        min_confidence=0.72,
    ) is False


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


def test_fusion_evidence_map_moe_is_conservative() -> None:
    qa = _load_qa_module()
    update_item = {
        "question": "What podcast do I listen to now for news?",
        "question_type": "knowledge-update",
    }
    temporal_item = {
        "question": "Which museum did I visit first?",
        "question_type": "temporal-reasoning",
    }
    preference_item = {
        "question": "What hotel should I pick based on what I like?",
        "question_type": "single-session-preference",
    }
    multi_item = {
        "question": "What did I decide after comparing both conversations?",
        "question_type": "multi-session",
    }

    assert qa.should_use_fusion_evidence_map(update_item, "moe") is True
    assert qa.should_use_fusion_evidence_map(temporal_item, "moe") is True
    assert qa.should_use_fusion_evidence_map(preference_item, "moe") is False
    assert qa.should_use_fusion_evidence_map(multi_item, "moe") is False


def test_fusion_evidence_map_temporal_is_surgical() -> None:
    qa = _load_qa_module()
    update_item = {
        "question": "What podcast do I listen to now for news?",
        "question_type": "knowledge-update",
    }
    temporal_item = {
        "question": "Which museum did I visit first?",
        "question_type": "temporal-reasoning",
    }
    preference_item = {
        "question": "What hotel should I pick based on what I like?",
        "question_type": "single-session-preference",
    }
    multi_item = {
        "question": "What did I decide after comparing both conversations?",
        "question_type": "multi-session",
    }

    assert qa.should_use_fusion_evidence_map(update_item, "temporal") is False
    assert qa.should_use_fusion_evidence_map(temporal_item, "temporal") is True
    assert qa.should_use_fusion_evidence_map(preference_item, "temporal") is False
    assert qa.should_use_fusion_evidence_map(multi_item, "temporal") is False


def test_expert_ensemble_moe_resolves_conservative_defaults() -> None:
    qa = _load_qa_module()
    args = SimpleNamespace(
        expert_ensemble="moe",
        fusion_evidence_map="off",
        aggregation_assembly="off",
    )

    assert qa.effective_fusion_evidence_map_mode(args) == "moe"
    assert qa.effective_aggregation_assembly_mode(args) == "off"


def test_expert_ensemble_does_not_override_explicit_modes() -> None:
    qa = _load_qa_module()
    args = SimpleNamespace(
        expert_ensemble="moe",
        fusion_evidence_map="targeted",
        aggregation_assembly="multi_session_count_list",
    )

    assert qa.effective_fusion_evidence_map_mode(args) == "targeted"
    assert qa.effective_aggregation_assembly_mode(args) == "multi_session_count_list"
