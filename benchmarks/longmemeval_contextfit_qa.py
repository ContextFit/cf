#!/usr/bin/env python3
"""Generate and judge LongMemEval QA answers from ContextFit retrieval artifacts.

This is the end-to-end QA companion to ``longmemeval_contextfit.py``. It keeps
retrieval fixed by consuming a saved ContextFit retrieval artifact, then asks an
LLM to answer from the retrieved sessions and uses the LongMemEval GPT judge
prompts to score the resulting hypotheses.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_GENERATION_MODEL = "gpt-4o-2024-08-06"
DEFAULT_JUDGE_MODEL = "gpt-4o-2024-08-06"
UNANSWERABLE_RESPONSE = "The information is not available in the provided history."
MISSING_EVIDENCE_MARKERS = (
    "not mentioned",
    "no mention",
    "no information",
    "not available",
    "not provided",
    "does not mention",
    "doesn't mention",
    "cannot determine",
    "can't determine",
    "insufficient",
)
STRUCTURED_FALLBACK_MARKERS = (
    "not available",
    "not provided",
    "cannot determine",
    "can't determine",
    "not enough information",
    "insufficient information",
    "unknown",
)
TEMPORAL_QUERY_RE = re.compile(
    r"\b("
    r"current|currently|latest|last|previous|previously|before|after|when|"
    r"since|ago|now|no longer|changed|switch(?:ed)?|update(?:d)?|"
    r"how long|how many (?:days|weeks|months|years)"
    r")\b",
    re.I,
)
QUESTION_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "been",
    "before",
    "being",
    "between",
    "could",
    "current",
    "does",
    "during",
    "from",
    "have",
    "having",
    "into",
    "like",
    "many",
    "much",
    "need",
    "only",
    "should",
    "that",
    "their",
    "there",
    "these",
    "thing",
    "this",
    "those",
    "through",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}
NUMBER_RE = re.compile(r"\b(?:\d+(?:[.,]\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", re.I)
DATE_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}|"
    r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?)\b",
    re.I,
)
TEMPORAL_RE = re.compile(r"\b(?:current|currently|latest|now|previous|before|after|then|today|yesterday|tomorrow|updated|changed|switched|no longer|used to|last|next)\b", re.I)
PREFERENCE_RE = re.compile(r"\b(?:like|love|prefer|favorite|favourite|enjoy|hate|dislike|avoid|interested|want|need|goal|constraint|allergic|can't|cannot)\b", re.I)


def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 0.0,
    api_key: str | None = None,
    retries: int = 6,
) -> str:
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return str(data["choices"][0]["message"]["content"]).strip()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                body = exc.read().decode("utf-8", errors="replace")[:1000]
                raise RuntimeError(f"OpenAI HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"OpenAI request failed after {retries} retries: {last_error}") from last_error


def scrub_turn(turn: dict[str, Any]) -> dict[str, str]:
    return {
        "role": str(turn.get("role", "unknown")),
        "content": " ".join(str(turn.get("content", "")).split()),
    }


def session_to_text(session_id: str, date: str, turns: list[dict[str, Any]], max_chars: int) -> str:
    lines = [f"Session ID: {session_id}", f"Session Date: {date}", "Session Content:"]
    for turn in turns:
        clean = scrub_turn(turn)
        if clean["content"]:
            lines.append(f"{clean['role']}: {clean['content']}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[: max_chars - 80].rstrip() + "\n[session truncated for context budget]"
    return text


def question_keywords(question: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", question.lower())
    return {word.strip("'") for word in words if len(word) >= 4 and word not in QUESTION_STOPWORDS}


def split_fact_candidates(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [" ".join(part.split()) for part in parts if part.strip()]


def score_fact(sentence: str, keywords: set[str], question_type: str) -> int:
    lower = sentence.lower()
    score = sum(2 for word in keywords if word in lower)
    if NUMBER_RE.search(sentence):
        score += 2
    if DATE_RE.search(sentence):
        score += 2
    if TEMPORAL_RE.search(sentence):
        score += 2 if question_type in {"temporal-reasoning", "knowledge-update"} else 1
    if PREFERENCE_RE.search(sentence):
        score += 2 if question_type == "single-session-preference" else 1
    if re.search(r"\b(?:but|however|instead|actually|rather than|not|no longer)\b", lower):
        score += 1
    return score


def compact_fact(sentence: str, max_chars: int = 260) -> str:
    sentence = sentence.strip()
    if len(sentence) <= max_chars:
        return sentence
    return sentence[: max_chars - 1].rstrip() + "..."


def build_token_evidence_table(
    item: dict[str, Any],
    selected: list[str],
    date_by_sid: dict[str, str],
    turns_by_sid: dict[str, list[dict[str, Any]]],
    *,
    max_lines: int,
    include_signals: bool,
) -> str:
    """Build deterministic source-linked evidence hints from retrieved sessions.

    This intentionally uses only token/regex signals. It is not an answerer; it
    gives the LLM a compact correlation layer over the same retrieved context.
    """
    keywords = question_keywords(item["question"])
    question_type = item["question_type"]
    lines = [
        "Token Evidence Table:",
        f"- Question type: {question_type}",
        f"- Question keywords: {', '.join(sorted(keywords)) if keywords else '(none)'}",
    ]
    global_numbers: list[str] = []
    global_dates: list[str] = []
    global_temporal: list[str] = []
    seen_facts: set[str] = set()

    for source_idx, sid in enumerate(selected, start=1):
        scored: list[tuple[int, int, str]] = []
        numbers: list[str] = []
        dates: list[str] = []
        temporal_markers: list[str] = []
        for turn_idx, turn in enumerate(turns_by_sid[sid], start=1):
            clean = scrub_turn(turn)
            content = clean["content"]
            if not content:
                continue
            numbers.extend(match.group(0) for match in NUMBER_RE.finditer(content))
            dates.extend(match.group(0) for match in DATE_RE.finditer(content))
            temporal_markers.extend(match.group(0) for match in TEMPORAL_RE.finditer(content))
            for sentence in split_fact_candidates(content):
                score = score_fact(sentence, keywords, question_type)
                if score:
                    scored.append((score, turn_idx, f"{clean['role']}: {compact_fact(sentence)}"))
        top_facts: list[str] = []
        for _score, turn_idx, sentence in sorted(scored, key=lambda row: (-row[0], row[1], row[2])):
            dedupe_key = sentence.lower()
            if dedupe_key in seen_facts:
                continue
            seen_facts.add(dedupe_key)
            top_facts.append(f"  - turn {turn_idx}: {sentence}")
            if len(top_facts) >= max_lines:
                break
        if top_facts or (include_signals and (numbers or dates or temporal_markers)):
            lines.append(f"Source {source_idx} ({sid}, {date_by_sid.get(sid, '')}):")
            if include_signals and dates:
                unique_dates = list(dict.fromkeys(dates))[:8]
                lines.append(f"  - dates/times: {', '.join(unique_dates)}")
                global_dates.extend(unique_dates)
            if include_signals and numbers:
                unique_numbers = list(dict.fromkeys(numbers))[:10]
                lines.append(f"  - numeric/count tokens: {', '.join(unique_numbers)}")
                global_numbers.extend(unique_numbers)
            if include_signals and temporal_markers:
                unique_temporal = list(dict.fromkeys(marker.lower() for marker in temporal_markers))[:8]
                lines.append(f"  - temporal/update markers: {', '.join(unique_temporal)}")
                global_temporal.extend(unique_temporal)
            lines.extend(top_facts or ["  - no high-overlap fact sentence found"])

    if include_signals and (global_numbers or global_dates or global_temporal):
        lines.append("Cross-source token signals:")
        if global_dates:
            lines.append(f"- dates/times seen: {', '.join(list(dict.fromkeys(global_dates))[:20])}")
        if global_numbers:
            lines.append(f"- numeric/count tokens seen: {', '.join(list(dict.fromkeys(global_numbers))[:20])}")
        if global_temporal:
            lines.append(f"- temporal/update markers seen: {', '.join(list(dict.fromkeys(global_temporal))[:20])}")
    lines.append(
        "Reducer instruction: use this table to notice correlations, counts, dates, updates, and repeated facts; verify every final answer against the full retrieved sessions below."
    )
    return "\n".join(lines)


def is_count_list_question(question: str) -> bool:
    return bool(
        re.search(
            r"\b(?:how many|number of|total|count|list|which .*\b(?:ones|things|items)|what .*\b(?:items|things|events)|including|include)\b",
            question.lower(),
        )
    )


def build_answer_prompt(
    item: dict[str, Any],
    retrieved_sessions: list[str],
    *,
    top_k: int,
    max_session_chars: int,
    cot: bool,
    source_aware: bool,
    source_sufficiency: bool,
    token_evidence: bool,
    token_evidence_max_lines: int,
    token_evidence_signals: bool,
    count_list_mode: bool,
    strict_missing_final_answer: bool,
) -> str:
    date_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_dates"], strict=True))
    turns_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True))
    selected = [sid for sid in retrieved_sessions[:top_k] if sid in turns_by_sid]

    # LongMemEval's generation script sorts selected context chronologically.
    selected.sort(key=lambda sid: date_by_sid.get(sid, ""))

    context_parts = []
    for i, sid in enumerate(selected, start=1):
        context_parts.append(
            f"### Retrieved Session {i}\n"
            + session_to_text(sid, date_by_sid[sid], turns_by_sid[sid], max_session_chars)
        )
    context = "\n\n".join(context_parts)
    evidence_table = ""
    if token_evidence:
        evidence_table = (
            build_token_evidence_table(
                item,
                selected,
                date_by_sid,
                turns_by_sid,
                max_lines=token_evidence_max_lines,
                include_signals=token_evidence_signals,
            )
            + "\n\n"
        )
    use_count_list_mode = count_list_mode and is_count_list_question(item["question"])
    if use_count_list_mode:
        answer_instruction = (
            "Answer the question based on the provided chat history. "
            "This is a count/list aggregation question. Treat each retrieved session as a separate evidence source. "
            "First write Candidate Evidence: inspect every retrieved session and list each concrete candidate item, event, amount, date, or occurrence that could affect the answer, with its source session. "
            "Then write Deduped Set: merge repeated mentions of the same real-world item/event, keep distinct items/events separate, and explain any exclusion. "
            "For temporal wording such as current, last month, before, after, latest, or currently, only include candidates that satisfy that time constraint. "
            "If the question asks for a count, the Final Answer must be the count implied by the Deduped Set. "
            "If the question asks for a list, the Final Answer must list exactly the deduped items. "
            "Do not stop after the first matching session. Do not estimate from general knowledge. "
            "If the provided history truly does not contain enough relevant information, say that the information is not available in the provided history."
        )
        answer_label = "Candidate Evidence, Deduped Set, and Final Answer:"
    elif source_sufficiency:
        answer_instruction = (
            "Answer the question based on the provided chat history. "
            "Treat each retrieved session as a separate evidence source. "
            "First write Required Evidence: identify the exact kind of facts needed to answer the question, such as dates, counts, quantities, places, roles, entities, or comparison sides. "
            "Then write Source Notes: inspect every retrieved session and list all concrete facts that could help answer the question. Do not omit relevant numeric amounts, dates, named entities, or candidate items. "
            "For multi-session, counting, total, list, and comparison questions, scan all retrieved sessions and combine distinct facts across sources; do not stop after the first matching session. "
            "For when/date questions, preserve the most specific date expression available, including holidays such as Valentine's Day when that identifies a date. "
            "Use dates and temporal wording to decide whether facts are current, previous, updated, or superseded. "
            "If several sessions mention the same item, count it once unless the question asks for mentions or events. "
            "After Source Notes, write exactly one line: Sufficiency: sufficient or Sufficiency: insufficient. "
            "Mark Sufficiency: insufficient if the source notes do not contain every required entity, value, date, place, role, event, or side of a comparison needed to answer the question. "
            "For comparison questions, if any requested side is missing, mark insufficient. "
            "For questions about a specific role, place, course, item, or event, mark insufficient when the notes only support a similar but different role, place, course, item, or event. "
            "If Sufficiency is insufficient, the Final Answer must be exactly: The information is not available in the provided history. "
            "If Sufficiency is sufficient, write Final Answer with the concise answer. "
            "For recommendation or preference questions, Sufficiency is sufficient when the source notes contain personal context that can reasonably support the recommendation."
        )
        answer_label = "Required Evidence, Source Notes, Sufficiency, and Final Answer:"
    elif source_aware:
        answer_instruction = (
            "Answer the question based on the provided chat history. "
            "Treat each retrieved session as a separate evidence source. "
            "If a Token Evidence Table is provided, use it as a deterministic hint layer for correlations, counts, dates, updates, and repeated facts, but verify the final answer against the full retrieved sessions. "
            "First write Source Notes: for every retrieved session that contains relevant evidence, list the concrete facts from that session. "
            "For multi-session, counting, total, list, and comparison questions, scan all retrieved sessions and combine distinct facts across sources; do not stop after the first matching session. "
            "Use dates and temporal wording to decide whether facts are current, previous, updated, or superseded. "
            "If several sessions mention the same item, count it once unless the question asks for mentions or events. "
            "Then write Final Answer with the concise answer. "
            "For recommendation or preference questions, use indirect personal context such as the user's interests, tools, constraints, goals, and prior activities. "
            "If the provided history truly does not contain enough relevant information, say that the information is not available in the provided history."
        )
        if strict_missing_final_answer:
            answer_instruction += (
                " If your Source Notes explicitly say that a required value, entity, date, place, role, event, or side of a comparison is missing or not mentioned, the Final Answer must be exactly: The information is not available in the provided history."
            )
        answer_label = "Source Notes and Final Answer:"
    elif cot:
        answer_instruction = (
            "Answer the question based on the relevant chat history. "
            "First extract the relevant information from the provided sessions, then reason over that information to get the answer. "
            "For recommendation or preference questions, use indirect personal context such as the user's interests, tools, constraints, goals, and prior activities. "
            "If the provided history truly does not contain enough relevant information, say that the information is not available in the provided history."
        )
        answer_label = "Answer (relevant information, reasoning, final answer):"
    else:
        answer_instruction = (
            "Answer the question based only on the provided relevant chat history. "
            "If the provided history does not contain enough information to answer, say that the information is not available in the provided history. "
            "Be concise and give the final answer directly."
        )
        answer_label = "Answer:"
    return (
        "I will give you several history chats between you and a user. "
        f"{answer_instruction}\n\n"
        f"{evidence_table}"
        f"History Chats:\n\n{context}\n\n"
        f"Current Date: {item.get('question_date', '')}\n"
        f"Question: {item['question']}\n"
        f"{answer_label}"
    )


def build_structured_extraction_prompt(
    item: dict[str, Any],
    retrieved_sessions: list[str],
    *,
    top_k: int,
    max_session_chars: int,
    include_question_type: bool,
) -> str:
    date_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_dates"], strict=True))
    turns_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True))
    selected = [sid for sid in retrieved_sessions[:top_k] if sid in turns_by_sid]
    selected.sort(key=lambda sid: date_by_sid.get(sid, ""))

    context_parts = []
    for i, sid in enumerate(selected, start=1):
        context_parts.append(
            f"### Retrieved Session {i}\n"
            + session_to_text(sid, date_by_sid[sid], turns_by_sid[sid], max_session_chars)
        )
    context = "\n\n".join(context_parts)
    question_type_hint = f"Question Type: {item['question_type']}\n" if include_question_type else ""
    return (
        "Extract source-grounded facts needed to answer the question from retrieved chat history.\n"
        "Do not answer the question yet.\n\n"
        "Return compact JSON only with this shape:\n"
        "{\n"
        '  "question_kind": "infer from the question text",\n'
        '  "required_facts": ["..."],\n'
        '  "facts": [\n'
        '    {"source": "Retrieved Session N", "date": "...", "fact": "...", "value": "...", "entity": "...", "event": "...", "currentness": "current|previous|unknown"}\n'
        "  ],\n"
        '  "dedupe_notes": ["..."],\n'
        '  "temporal_order": ["..."],\n'
        '  "missing_or_uncertain": ["..."]\n'
        "}\n\n"
        "Rules:\n"
        "- Extract only facts supported by the retrieved sessions.\n"
        "- For counts/lists/totals, include every candidate item/event/value that may affect the result.\n"
        "- For temporal/current/latest questions, preserve dates and note whether facts are current, previous, or superseded.\n"
        "- For comparisons, extract both sides and the attributes being compared.\n"
        "- For preference/recommendation questions, extract user-specific preferences, interests, constraints, goals, and prior activities.\n"
        "- If the evidence is incomplete, record the gap in missing_or_uncertain.\n\n"
        f"History Chats:\n\n{context}\n\n"
        f"Current Date: {item.get('question_date', '')}\n"
        f"{question_type_hint}"
        f"Question: {item['question']}\n"
        "JSON:"
    )


def build_structured_answer_prompt(
    item: dict[str, Any],
    structured_facts: str,
    *,
    include_question_type: bool,
) -> str:
    question_type_hint = f"Question Type: {item['question_type']}\n" if include_question_type else ""
    return (
        "Answer the question using the structured facts extracted from retrieved chat history.\n"
        "The structured facts may contain candidate facts, dedupe notes, temporal ordering, and missing/uncertain evidence.\n\n"
        "Rules:\n"
        "- Base the answer only on the structured facts.\n"
        "- For counts/lists/totals, dedupe repeated mentions of the same real-world item/event and count distinct included items/events.\n"
        "- For temporal/current/latest questions, use temporal_order and currentness to choose the correct fact.\n"
        "- For comparison questions, make sure every requested side is represented.\n"
        "- If missing_or_uncertain shows a required fact is absent, say: The information is not available in the provided history.\n"
        "- Give a concise final answer.\n\n"
        f"{question_type_hint}"
        f"Question: {item['question']}\n"
        f"Structured Facts:\n{structured_facts}\n\n"
        "Final Answer:"
    )


def should_use_temporal_hybrid(item: dict[str, Any], router: str) -> bool:
    if router == "question_type":
        return item["question_type"] == "temporal-reasoning"
    if router == "query":
        return bool(TEMPORAL_QUERY_RE.search(item["question"]))
    raise ValueError(f"unknown temporal router: {router}")


def should_fallback_from_structured_answer(item: dict[str, Any], hypothesis: str) -> bool:
    text = hypothesis.lower()
    if any(marker in text for marker in STRUCTURED_FALLBACK_MARKERS):
        return True
    question = item["question"].lower()
    # These query-shape fallbacks are benchmark-ablation guardrails, not product
    # routing logic. They protect known weak structured-extraction shapes while
    # we measure whether the path is worth turning into a general query router.
    if re.search(r"\b(?:ago|since)\b", question):
        return True
    if re.search(r"\bhow many\b.+\bbefore\b", question):
        return True
    return False


def get_anscheck_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    if not abstention:
        if task in {"single-session-user", "single-session-assistant", "multi-session"}:
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the answer, answer no. \n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the answer, answer no. "
                "In addition, do not penalize off-by-one errors for the number of days. "
                "If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response from a model. "
                "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                "The model does not need to reflect all the points in the rubric. "
                "The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
                "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        raise NotImplementedError(task)
    template = (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. "
        "The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\n"
        "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    )
    return template.format(question, answer, response)


def get_answerability_prompt(item: dict[str, Any], hypothesis: str) -> str:
    return (
        "You are checking whether a model's answer is sufficiently supported by its own cited source notes.\n"
        "The model was asked to answer from retrieved chat history only. It may include Source Notes and a Final Answer.\n\n"
        "Answer YES only if the response contains concrete source-backed evidence for the requested answer. "
        "Answer NO if the response says the information is unavailable, if the source notes do not contain the requested fact, "
        "or if the final answer relies on inference not grounded in the source notes.\n\n"
        "For counting/list/total questions, answer YES only if the source notes enumerate enough distinct items/events/facts to support the count or list. "
        "For temporal questions, answer YES only if the source notes include the dates or update ordering needed for the answer. "
        "For comparison questions involving two or more named entities, events, locations, roles, or quantities, answer NO if any requested side is missing. "
        "If the notes say one requested entity/event/value is not mentioned, answer NO even if the model still gives a final answer. "
        "If the question asks about a specific role, place, course, item, or event, answer NO when the notes only support a similar but different role, place, course, item, or event. "
        "For recommendation/preference questions, answer YES if the source notes contain personal context that could reasonably support the recommendation.\n\n"
        f"Question: {item['question']}\n\n"
        f"Model Response:\n{hypothesis}\n\n"
        "Do the source notes sufficiently support an answer to the question? Answer YES or NO only."
    )


def answerability_label(args: argparse.Namespace, item: dict[str, Any], hypothesis: str) -> dict[str, Any]:
    response = chat_completion(
        args.answerability_model,
        [{"role": "user", "content": get_answerability_prompt(item, hypothesis)}],
        max_tokens=args.answerability_max_tokens,
        api_key=args.api_key,
    )
    return {
        "model": args.answerability_model,
        "answerable": "yes" in response.lower(),
        "raw": response,
    }


def should_run_answerability_gate(args: argparse.Namespace, hypothesis: str) -> bool:
    if not args.answerability_trigger_missing:
        return True
    text = hypothesis.lower()
    return any(marker in text for marker in MISSING_EVIDENCE_MARKERS)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        print(json.dumps(row, ensure_ascii=False), file=f, flush=True)


def generate_hypotheses(args: argparse.Namespace) -> None:
    data = json.loads(args.data.read_text())
    data = filter_items(data, args)
    retrieval = json.loads(args.retrieval_artifact.read_text())
    rows_by_qid = {row["question_id"]: row for row in retrieval["rows"]}

    done = {row["question_id"] for row in read_jsonl(args.hypotheses_out)}
    extracted_done = {row["question_id"]: row for row in read_jsonl(args.extract_out)}

    for i, item in enumerate(data, start=1):
        qid = item["question_id"]
        if qid in done:
            continue
        retrieved = rows_by_qid[qid]["retrieved_sessions"]
        route = "source_aware"
        if args.temporal_hybrid and should_use_temporal_hybrid(item, args.temporal_hybrid_router):
            route = "structured_temporal"
            if qid in extracted_done:
                structured_facts = extracted_done[qid]["structured_facts"]
            else:
                extract_prompt = build_structured_extraction_prompt(
                    item,
                    retrieved,
                    top_k=args.top_k_context,
                    max_session_chars=args.max_session_chars,
                    include_question_type=args.include_question_type_in_prompts,
                )
                structured_facts = chat_completion(
                    args.extraction_model,
                    [{"role": "user", "content": extract_prompt}],
                    max_tokens=args.extraction_max_tokens,
                    api_key=args.api_key,
                )
                extract_row = {"question_id": qid, "structured_facts": structured_facts}
                append_jsonl(args.extract_out, extract_row)
                extracted_done[qid] = extract_row
            prompt = build_structured_answer_prompt(
                item,
                structured_facts,
                include_question_type=args.include_question_type_in_prompts,
            )
            hypothesis = chat_completion(
                args.generation_model,
                [{"role": "user", "content": prompt}],
                max_tokens=args.generation_max_tokens,
                api_key=args.api_key,
            )
            if should_fallback_from_structured_answer(item, hypothesis):
                structured_hypothesis = hypothesis
                fallback_prompt = build_answer_prompt(
                    item,
                    retrieved,
                    top_k=args.top_k_context,
                    max_session_chars=args.max_session_chars,
                    cot=False,
                    source_aware=True,
                    source_sufficiency=False,
                    token_evidence=False,
                    token_evidence_max_lines=args.token_evidence_max_lines,
                    token_evidence_signals=False,
                    count_list_mode=False,
                    strict_missing_final_answer=False,
                )
                hypothesis = chat_completion(
                    args.generation_model,
                    [{"role": "user", "content": fallback_prompt}],
                    max_tokens=args.generation_max_tokens,
                    api_key=args.api_key,
                )
                route = "source_aware_fallback_after_structured_unavailable"
                append_jsonl(
                    args.hypotheses_out,
                    {
                        "question_id": qid,
                        "hypothesis": hypothesis,
                        "route": route,
                        "structured_hypothesis": structured_hypothesis,
                    },
                )
            else:
                append_jsonl(args.hypotheses_out, {"question_id": qid, "hypothesis": hypothesis, "route": route})
            print(f"[generate {i}/{len(data)}] {qid} {item['question_type']} route={route}", flush=True)
            continue
        if args.structured_extract:
            route = "structured"
            if qid in extracted_done:
                structured_facts = extracted_done[qid]["structured_facts"]
            else:
                extract_prompt = build_structured_extraction_prompt(
                    item,
                    retrieved,
                    top_k=args.top_k_context,
                    max_session_chars=args.max_session_chars,
                    include_question_type=args.include_question_type_in_prompts,
                )
                structured_facts = chat_completion(
                    args.extraction_model,
                    [{"role": "user", "content": extract_prompt}],
                    max_tokens=args.extraction_max_tokens,
                    api_key=args.api_key,
                )
                extract_row = {"question_id": qid, "structured_facts": structured_facts}
                append_jsonl(args.extract_out, extract_row)
                extracted_done[qid] = extract_row
            prompt = build_structured_answer_prompt(
                item,
                structured_facts,
                include_question_type=args.include_question_type_in_prompts,
            )
        else:
            prompt = build_answer_prompt(
                item,
                retrieved,
                top_k=args.top_k_context,
                max_session_chars=args.max_session_chars,
                cot=args.cot,
                source_aware=args.source_aware,
                source_sufficiency=args.source_sufficiency,
                token_evidence=args.token_evidence,
                token_evidence_max_lines=args.token_evidence_max_lines,
                token_evidence_signals=args.token_evidence_signals,
                count_list_mode=args.count_list_mode,
                strict_missing_final_answer=args.strict_missing_final_answer,
            )
            if not args.source_aware:
                route = "standard"
        hypothesis = chat_completion(
            args.generation_model,
            [{"role": "user", "content": prompt}],
            max_tokens=args.generation_max_tokens,
            api_key=args.api_key,
        )
        append_jsonl(args.hypotheses_out, {"question_id": qid, "hypothesis": hypothesis, "route": route})
        print(f"[generate {i}/{len(data)}] {qid} {item['question_type']} route={route}", flush=True)


def judge_hypotheses(args: argparse.Namespace) -> None:
    data = json.loads(args.data.read_text())
    data = filter_items(data, args)
    by_qid = {item["question_id"]: item for item in data}
    hypotheses = read_jsonl(args.hypotheses_out)
    allowed_qids = set(by_qid)
    hypotheses = [row for row in hypotheses if row["question_id"] in allowed_qids]
    done = {row["question_id"] for row in read_jsonl(args.judged_out)}
    answerability_done = {row["question_id"]: row for row in read_jsonl(args.answerability_out)}

    for i, row in enumerate(hypotheses, start=1):
        qid = row["question_id"]
        if qid in done:
            continue
        item = by_qid[qid]
        row_for_judge = dict(row)
        if args.answerability_check and should_run_answerability_gate(args, row["hypothesis"]):
            if qid in answerability_done:
                gate = answerability_done[qid]["answerability_check"]
            else:
                gate = answerability_label(args, item, row["hypothesis"])
                answerability_row = {"question_id": qid, "answerability_check": gate}
                append_jsonl(args.answerability_out, answerability_row)
                answerability_done[qid] = answerability_row
            row_for_judge["answerability_check"] = gate
            if not gate["answerable"]:
                row_for_judge["original_hypothesis"] = row_for_judge["hypothesis"]
                row_for_judge["hypothesis"] = UNANSWERABLE_RESPONSE
        prompt = get_anscheck_prompt(
            item["question_type"],
            item["question"],
            item["answer"],
            row_for_judge["hypothesis"],
            abstention=qid.endswith("_abs"),
        )
        response = chat_completion(
            args.judge_model,
            [{"role": "user", "content": prompt}],
            max_tokens=10,
            api_key=args.api_key,
        )
        judged = dict(row_for_judge)
        judged["autoeval_label"] = {
            "model": args.judge_model,
            "label": "yes" in response.lower(),
            "raw": response,
        }
        append_jsonl(args.judged_out, judged)
        print(f"[judge {i}/{len(hypotheses)}] {qid} label={judged['autoeval_label']['label']}", flush=True)


def summarize(judged_path: Path, data_path: Path) -> dict[str, Any]:
    judged = read_jsonl(judged_path)
    data = {item["question_id"]: item for item in json.loads(data_path.read_text())}
    type2acc: dict[str, list[int]] = defaultdict(list)
    route_counts: dict[str, int] = defaultdict(int)
    abstention: list[int] = []
    all_acc: list[int] = []
    for row in judged:
        item = data[row["question_id"]]
        val = 1 if row["autoeval_label"]["label"] else 0
        type2acc[item["question_type"]].append(val)
        route_counts[row.get("route", "unknown")] += 1
        all_acc.append(val)
        if row["question_id"].endswith("_abs"):
            abstention.append(val)
    by_type = {
        qt: {"n": len(vals), "accuracy": (sum(vals) / len(vals) if vals else 0.0)}
        for qt, vals in sorted(type2acc.items())
    }
    non_empty_types = [v["accuracy"] for v in by_type.values() if v["n"]]
    return {
        "n": len(all_acc),
        "overall_accuracy": sum(all_acc) / len(all_acc) if all_acc else 0.0,
        "task_averaged_accuracy": sum(non_empty_types) / len(non_empty_types) if non_empty_types else 0.0,
        "abstention_accuracy": sum(abstention) / len(abstention) if abstention else None,
        "abstention_n": len(abstention),
        "by_type": by_type,
        "route_counts": dict(sorted(route_counts.items())),
    }


def filter_items(data: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.question_id_file:
        allowed_qids = {
            line.strip()
            for line in args.question_id_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        data = [item for item in data if item["question_id"] in allowed_qids]
    if args.abstention_only:
        data = [item for item in data if item["question_id"].endswith("_abs")]
    if args.non_abstention_only:
        data = [item for item in data if not item["question_id"].endswith("_abs")]
    if args.question_type:
        allowed = set(args.question_type)
        data = [item for item in data if item["question_type"] in allowed]
    if args.count_list_only:
        data = [item for item in data if is_count_list_question(item["question"])]
    if args.limit:
        data = data[: args.limit]
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("benchmarks/data/longmemeval_s_cleaned.json"))
    ap.add_argument("--retrieval-artifact", type=Path, default=Path("benchmarks/longmemeval_token_only_leaderboard_run_20260516.json"))
    ap.add_argument("--hypotheses-out", type=Path, default=Path("benchmarks/longmemeval_contextfit_token_only_qa_hypotheses_20260516.jsonl"))
    ap.add_argument("--judged-out", type=Path, default=Path("benchmarks/longmemeval_contextfit_token_only_qa_judged_20260516.jsonl"))
    ap.add_argument("--generation-model", default=DEFAULT_GENERATION_MODEL)
    ap.add_argument("--extraction-model", default=DEFAULT_GENERATION_MODEL)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--answerability-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--generation-max-tokens", type=int, default=500)
    ap.add_argument("--extraction-max-tokens", type=int, default=1200)
    ap.add_argument("--answerability-max-tokens", type=int, default=10)
    ap.add_argument("--structured-extract", action="store_true", help="run a two-pass structured fact extraction before answering")
    ap.add_argument(
        "--temporal-hybrid",
        action="store_true",
        help="experimental: use structured extraction for temporal rows, falling back to source-aware answers when structured output is unavailable",
    )
    ap.add_argument(
        "--temporal-hybrid-router",
        choices=("question_type", "query"),
        default="question_type",
        help="router for --temporal-hybrid; question_type reproduces LongMemEval-label experiments, query avoids dataset-label routing",
    )
    ap.add_argument(
        "--include-question-type-in-prompts",
        action="store_true",
        help="include LongMemEval question_type in structured prompts; off by default to avoid label leakage in cleaner runs",
    )
    ap.add_argument(
        "--extract-out",
        type=Path,
        default=Path("benchmarks/longmemeval_contextfit_token_only_qa_extract_20260516.jsonl"),
    )
    ap.add_argument("--cot", action="store_true", help="use extract-then-reason answer prompt")
    ap.add_argument("--source-aware", action="store_true", help="use source-notes plus final-answer prompt")
    ap.add_argument("--source-sufficiency", action="store_true", help="use source notes plus an in-answer sufficiency decision")
    ap.add_argument("--count-list-mode", action="store_true", help="use a specialized candidate/deduped-set prompt for count/list questions")
    ap.add_argument("--token-evidence", action="store_true", help="prepend a deterministic token evidence table to the answer prompt")
    ap.add_argument("--token-evidence-max-lines", type=int, default=5, help="max fact lines per retrieved source in the token evidence table")
    ap.add_argument("--token-evidence-signals", action="store_true", help="include raw date/number/temporal token dumps in the token evidence table")
    ap.add_argument(
        "--strict-missing-final-answer",
        action="store_true",
        help="force unavailable final answers when source notes explicitly say required evidence is missing",
    )
    ap.add_argument("--answerability-check", action="store_true", help="gate unsupported answers before judging")
    ap.add_argument(
        "--answerability-trigger-missing",
        action="store_true",
        help="only run the answerability gate when the response itself mentions missing evidence",
    )
    ap.add_argument(
        "--answerability-out",
        type=Path,
        default=Path("benchmarks/longmemeval_contextfit_token_only_qa_answerability_20260516.jsonl"),
    )
    ap.add_argument(
        "--question-type",
        action="append",
        help="only run rows with this LongMemEval question_type; may be repeated",
    )
    ap.add_argument("--question-id-file", type=Path, help="only run question ids listed in this newline-delimited file")
    ap.add_argument("--abstention-only", action="store_true")
    ap.add_argument("--non-abstention-only", action="store_true")
    ap.add_argument("--count-list-only", action="store_true", help="only run questions matching count/list wording")
    ap.add_argument("--top-k-context", type=int, default=5)
    ap.add_argument("--max-session-chars", type=int, default=16_000)
    ap.add_argument("--limit", type=int, default=0, help="0 means all rows")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--summary-out", type=Path, default=Path("benchmarks/longmemeval_contextfit_token_only_qa_summary_20260516.json"))
    args = ap.parse_args()
    args.limit = args.limit if args.limit > 0 else 0
    args.api_key = os.environ.get("OPENAI_API_KEY")
    if not args.api_key and (not args.skip_generate or not args.skip_judge):
        raise RuntimeError("OPENAI_API_KEY is required")

    if not args.skip_generate:
        generate_hypotheses(args)
    if not args.skip_judge:
        judge_hypotheses(args)
    summary = summarize(args.judged_out, args.data)
    args.summary_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
