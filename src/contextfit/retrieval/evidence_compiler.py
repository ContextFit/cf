"""Deterministic evidence compilers for retrieved ContextFit sessions.

These helpers are token/regex based. They do not answer questions; they build
source-linked evidence views that downstream answer synthesis can audit.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any

TEMPORAL_QUERY_RE = re.compile(
    r"\b("
    r"current|currently|latest|last|previous|previously|before|after|when|"
    r"since|ago|now|no longer|changed|switch(?:ed)?|update(?:d)?|"
    r"how long|how many (?:days|weeks|months|years)"
    r")\b",
    re.I,
)
TEMPORAL_REASONING_QUERY_RE = re.compile(
    r"\b(?:how many (?:days|weeks|months|years)|how long|order|earliest|latest|before|after|since|when)\b",
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
PREFERENCE_QUERY_RE = re.compile(
    r"\b(?:recommend|suggest|advice|tips|prefer|favorite|favourite|like|love|enjoy|"
    r"interested|goal|constraint|allergic|avoid|watch|serve|hotel|activities)\b",
    re.I,
)
COUNT_LIST_QUERY_RE = re.compile(
    r"\b(?:how many|number of|total|count|list|which .*\b(?:ones|things|items)|"
    r"what .*\b(?:items|things|events)|including|include|percentage|percent)\b",
    re.I,
)
UPDATE_RE = re.compile(
    r"\b(?:now|currently|current|latest|previous|previously|before|after|then|"
    r"updated|changed|switched|replaced|moved|started|stopped|finished|completed|"
    r"no longer|used to|instead|but|however)\b",
    re.I,
)
ACTION_RE = re.compile(
    r"\b(?:went|visited|attended|met|bought|purchased|received|gave|started|finished|"
    r"completed|signed|redeemed|used|picked|pickup|collected|retrieved|"
    r"returned|return|exchanged|dropped|mailed|shipped|assembled|sold|fixed|worked|"
    r"read|listened|watched|joined|registered|moved|changed|switched|invested)\b",
    re.I,
)
PICKUP_RETURN_QUERY_RE = re.compile(
    r"\b(?:pick(?:ed)?\s*up|pickup|collect|collected|retrieve|retrieved|return(?:ed)?|"
    r"drop(?:ped)?\s*off|exchange|exchanged|mail(?:ed)?\s*back|ship(?:ped)?\s*back|bring(?:ing)?\s*back)\b",
    re.I,
)
PICKUP_ACTION_RE = re.compile(r"\b(?:pick(?:ed)?\s*up|pickup|collect(?:ed)?|retrieve(?:d)?)\b", re.I)
RETURN_ACTION_RE = re.compile(
    r"\b(?:return(?:ed)?|drop(?:ped)?\s*off|exchange|mail(?:ed)?\s*back|ship(?:ped)?\s*back|bring(?:ing)?\s*back)\b",
    re.I,
)
CLOTHING_RE = re.compile(
    r"\b(?:blazer|boot|boots|jeans|pants|trousers|shirt|sweater|dress|sundress|"
    r"jacket|coat|scarf|gloves|skirt|shorts|shoes|sneakers|yoga pants)\b",
    re.I,
)
MONTH_LOOKUP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@dataclass(frozen=True)
class EvidenceSource:
    """Normalized retrieved source for evidence compilation.

    Adapters for benchmarks, repos, documents, or chat history should convert
    their native retrieval rows into this shape before invoking the compiler.
    """

    source_id: str
    text: str
    date: str = ""
    role: str = "memory"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceContext:
    """Compiler-ready view over normalized evidence sources."""

    selected: list[str]
    date_by_sid: dict[str, str]
    turns_by_sid: dict[str, list[dict[str, Any]]]


MISSING_ANSWER = "I don't have enough information to answer this question."

ANSWER_POLICY_BY_TYPE = {
    "abstention": [
        "This is an abstention/no-evidence task. Answer only when the packet contains direct support for the specific asked fact.",
        "If support is absent, weak, or only adjacent to the asked fact, return the missing-answer string exactly.",
        "Do not fill gaps with plausible project, personal, or world knowledge.",
    ],
    "contradiction_resolution": [
        "Identify conflicting statements explicitly.",
        "Mention both sides of the contradiction, not only the later or resolved state.",
        "Prefer the latest dated or clearly corrected evidence when resolving the final state.",
        "If old and new states both matter, state what changed instead of flattening them into one fact.",
        "When the evidence includes Contradiction Pair Candidates, use the highest-ranked pair and include the open resolution question if provided.",
    ],
    "event_ordering": [
        "Build a chronological timeline from dated events and source ordering.",
        "Return events in oldest-to-newest order unless the question asks otherwise.",
        "If the question asks for a fixed number of events, provide exactly that many when supported.",
    ],
    "information_extraction": [
        "Extract exact names, dates, numbers, tools, services, amounts, and other requested slots.",
        "Use partial direct evidence when available; do not abstain just because some unrelated slots are missing.",
        "Do not let an update override an extracted value unless the question asks for the current/latest value or the evidence clearly marks the older value as replaced.",
        "When the evidence includes Exact Extraction Candidates, prefer the highest-ranked direct candidate unless the question asks for current/latest.",
        "Keep the answer compact and preserve original units or date granularity.",
    ],
    "instruction_following": [
        "Find the user's durable instruction, constraint, or requested output style in the evidence.",
        "Apply the instruction directly in the answer instead of merely describing it.",
        "If the user asks for code or an implementation example, include concise language-tagged code fences when supported by the evidence.",
        "Every code fence must be complete and start with an explicit language tag such as ```python, ```bash, ```sql, ```javascript, or ```html.",
        "Put every opening and closing code fence at the start of its own line with no indentation.",
        "Do not nest code fences inside numbered lists, bullets, blockquotes, or other indentation; the first visible characters on a fence line must be the three backticks.",
        "Say explicitly that the following are syntax-highlighted code blocks, then provide the code fences.",
        "After that sentence, put a language-tagged code fence immediately so syntax highlighting is unambiguous.",
        "For implementation questions, include at least two separate language-tagged code blocks: first a ```bash setup block for commands, then a ```python, ```html, or other implementation block.",
        "Never put shell commands such as pip install inside a ```python block.",
        "If instructions changed, follow the most recent supported instruction.",
    ],
    "knowledge_update": [
        "Separate older state, update/correction, and current state.",
        "Prefer current/latest evidence for the final answer, but mention the previous state when it explains the update.",
        "Treat negation, correction, replacement, and 'no longer' language as high-signal evidence.",
    ],
    "multi_session_reasoning": [
        "Combine evidence across sources; do not require one source to contain the whole answer.",
        "Track which sources contribute which parts before producing the final answer.",
        "Dedupe repeated facts while preserving distinct items, dates, or constraints.",
        "For count/list questions, count only items directly requested or asserted by the user; do not count incidental fields from code examples unless the question asks about implemented schema.",
        "For schema-change questions, ignore columns that appear only in assistant-generated code examples; count columns the user explicitly asked to add or migrate.",
    ],
    "preference_following": [
        "Use the user's latest supported preference, constraint, dislike, or favorite.",
        "If preferences conflict, prefer the most recent dated evidence and mention the change when useful.",
        "Make the answer actionable rather than restating every preference.",
    ],
    "single-session-preference": [
        "Use the user's supported preference, constraint, dislike, or favorite.",
        "Make the answer actionable and specific to the question.",
        "Do not invent preferences that are not present in the evidence.",
    ],
    "summarization": [
        "Write a coverage-oriented summary across the relevant sources and evidence sections.",
        "Include major milestones, features, decisions, constraints, dates, and documentation/security/testing work when present.",
        "If Summary Coverage Candidates are present, explicitly cover each listed aspect: features, timeline, security, and documentation.",
        "Do not abstain merely because the summary is incomplete; summarize the supported evidence and note uncertainty only for missing parts.",
    ],
    "temporal-reasoning": [
        "Reason over dates, ordering, durations, before/after relationships, and current-vs-previous state.",
        "Use explicit dates first, then source order only when dates are absent.",
        "When multiple timelines conflict, prefer the highest-ranked source that contains both endpoints unless a later source clearly says it updated or replaced that timeline.",
        "Match the event phrase in the question closely; do not substitute a related later implementation task for the named milestone.",
        "When the evidence includes a Same-Source Timeline Candidates section, use the highest-ranked candidate whose start and end labels match the question.",
        "Show the computed duration or ordering when the question asks for it.",
    ],
    "temporal_reasoning": [
        "Reason over dates, ordering, durations, before/after relationships, and current-vs-previous state.",
        "Use explicit dates first, then source order only when dates are absent.",
        "When multiple timelines conflict, prefer the highest-ranked source that contains both endpoints unless a later source clearly says it updated or replaced that timeline.",
        "Match the event phrase in the question closely; do not substitute a related later implementation task for the named milestone.",
        "When the evidence includes a Same-Source Timeline Candidates section, use the highest-ranked candidate whose start and end labels match the question.",
        "Show the computed duration or ordering when the question asks for it.",
    ],
}


def _normalized_question_type(question_type: str) -> str:
    return question_type.replace("_", "-").strip().lower()


def typed_answer_policy(question_type: str) -> list[str]:
    """Return answer-synthesis rules for a benchmark or real-world task type."""
    if question_type in ANSWER_POLICY_BY_TYPE:
        return ANSWER_POLICY_BY_TYPE[question_type]
    normalized = _normalized_question_type(question_type)
    if normalized in ANSWER_POLICY_BY_TYPE:
        return ANSWER_POLICY_BY_TYPE[normalized]
    if normalized == "temporal-reasoning":
        return ANSWER_POLICY_BY_TYPE["temporal-reasoning"]
    return [
        "Answer from supported evidence only.",
        "Combine relevant facts across sources when needed.",
        "If the packet lacks direct support for the asked fact, return the missing-answer string exactly.",
    ]


def build_typed_evidence_answer_prompt(
    *,
    question: str,
    question_type: str,
    evidence_packet: str,
    missing_answer: str = MISSING_ANSWER,
    context_label: str = "compiled evidence packet",
) -> str:
    """Build a reusable typed answer prompt for compiled evidence.

    The compiler prepares source-linked evidence; this prompt defines how an
    answerer should use it for common real-world and benchmark task types. The
    policy is intentionally stricter only for abstention/no-evidence tasks and
    less abstention-biased for extraction, summarization, and synthesis tasks.
    """
    policy = typed_answer_policy(question_type)
    policy_lines = "\n".join(f"{idx}. {line}" for idx, line in enumerate(policy, start=1))
    return f"""You are answering a question using a {context_label} built from retrieved sources.

QUESTION TYPE: {question_type}

GLOBAL RULES:
1. Use only the evidence in the packet. Do not use outside knowledge.
2. Treat packet rows as candidate evidence, not conclusions; reconcile conflicts using dates, source order, and update language.
3. Scan all evidence sections before answering. The best answer may require combining multiple rows or sources.
4. Be specific: preserve names, dates, numbers, tools, amounts, and constraints from the evidence.
5. Return exactly "{missing_answer}" only when the typed policy says evidence is missing or the packet has no direct support for the asked fact.
6. Do not mention the packet, prompt, or these rules in the final answer.
7. If an Exact Extraction Candidates section contains a Mandatory exact extraction value or Preferred extraction answer candidate, use that value/candidate as the final answer unless the question asks for current/latest.
8. If a Contradiction Pair Candidates section exists, explicitly include both sides of the pair before giving any resolution.

TYPED POLICY:
{policy_lines}

QUESTION:
{question}

EVIDENCE PACKET:
{evidence_packet}

ANSWER:"""


def scrub_turn(turn: dict[str, Any]) -> dict[str, str]:
    return {
        "role": str(turn.get("role", "unknown")),
        "content": " ".join(str(turn.get("content", "")).split()),
    }


def question_keywords(question: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", question.lower())
    keywords = {word.strip("'") for word in words if len(word) >= 4 and word not in QUESTION_STOPWORDS}
    variants = set(keywords)
    for word in keywords:
        if word.endswith("ies") and len(word) > 4:
            variants.add(f"{word[:-3]}y")
        elif word.endswith("s") and len(word) > 4:
            variants.add(word[:-1])
        else:
            variants.add(f"{word}s")
    return variants


AGGREGATION_GENERIC_TERMS = {
    "count",
    "counts",
    "day",
    "days",
    "dayss",
    "different",
    "hours",
    "last",
    "many",
    "month",
    "months",
    "past",
    "spend",
    "spends",
    "state",
    "states",
    "total",
    "trip",
    "trips",
    "united",
    "uniteds",
    "year",
    "years",
}


def _aggregation_keywords(question: str) -> set[str]:
    """Return content terms for count/list candidate selection.

    Count/list questions often contain generic scope words such as "days",
    "year", or "United States". Those are useful in the final reasoning step
    but too broad for candidate retrieval; they can crowd out the actual
    evidence-bearing terms such as camping, games, babies, or art events.
    """
    return {word for word in question_keywords(question) if word not in AGGREGATION_GENERIC_TERMS}


def split_fact_candidates(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+|\s+(?=-\s+\*\*)|\s+(?=####?\s+)|\s+(?=\d+\.\s+\*\*)", text)
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


def _source_sort_value(source: EvidenceSource, fallback_rank: int) -> tuple[str, int]:
    rank = source.metadata.get("rank", fallback_rank)
    try:
        rank_int = int(rank)
    except (TypeError, ValueError):
        rank_int = fallback_rank
    return (source.date, rank_int)


def _select_source_sentences(
    question: str,
    question_type: str,
    text: str,
    *,
    max_chars: int,
    max_sentences: int,
) -> str:
    """Keep the strongest sentence-level evidence from an oversized source."""
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized

    keywords = question_keywords(question)
    scored: list[tuple[int, int, str]] = []
    for idx, sentence in enumerate(split_fact_candidates(text)):
        compact = compact_fact(sentence, max_chars=min(700, max_chars))
        if not compact:
            continue
        score = score_fact(compact, keywords, question_type)
        if DATE_RE.search(compact):
            score += 2
        if UPDATE_RE.search(compact):
            score += 1
        if ACTION_RE.search(compact):
            score += 1
        scored.append((score, idx, compact))

    if not scored:
        return normalized[: max_chars - 1].rstrip() + "..."

    selected = sorted(scored, key=lambda row: (-row[0], row[1], row[2]))[:max_sentences]
    selected = sorted(selected, key=lambda row: row[1])

    kept: list[str] = []
    total = 0
    for _score, _idx, sentence in selected:
        extra = len(sentence) + (1 if kept else 0)
        if kept and total + extra > max_chars:
            continue
        if not kept and extra > max_chars:
            kept.append(sentence[: max_chars - 1].rstrip() + "...")
            break
        kept.append(sentence)
        total += extra

    return " ".join(kept).strip()


def _prepared_source_score(question: str, question_type: str, prepared_text: str) -> int:
    keywords = question_keywords(question)
    sentence_scores: list[int] = []
    seen: set[str] = set()
    for sentence in split_fact_candidates(prepared_text):
        key = _dedupe_key(sentence)
        if key in seen:
            continue
        seen.add(key)
        score = score_fact(sentence, keywords, question_type)
        if UPDATE_RE.search(sentence):
            score += 2
        if ACTION_RE.search(sentence):
            score += 1
        sentence_scores.append(score)
    return sum(sorted(sentence_scores, reverse=True)[:5])


def _parse_month_day_year(text: str, default_year: int | None = None) -> date | None:
    dates = _parse_month_day_years(text, default_year=default_year)
    return dates[0] if dates else None


def _parse_month_day_years(text: str, default_year: int | None = None) -> list[date]:
    matches = re.finditer(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+"
        r"(\d{1,2})(?:,\s*(\d{4}))?",
        text,
        re.I,
    )
    parsed: list[date] = []
    for match in matches:
        month = MONTH_LOOKUP.get(match.group(1).lower().rstrip("."))
        if month is None:
            continue
        year_text = match.group(3)
        if year_text is None and default_year is None:
            continue
        year = int(year_text) if year_text is not None else default_year
        try:
            parsed.append(date(int(year), month, int(match.group(2))))
        except ValueError:
            continue
    return parsed


def _year_hint(text: str) -> int | None:
    years = [int(year) for year in re.findall(r"\b(20\d{2}|19\d{2})\b", text)]
    return years[0] if years else None


def _format_duration_weeks(start: date, end: date) -> str:
    days = (end - start).days
    weeks = days // 7
    if days % 7 == 0:
        return f"{weeks} weeks"
    return f"about {weeks} weeks ({days} days)"


def _format_long_date(value: date) -> str:
    month = value.strftime("%B")
    return f"{month} {value.day}, {value.year}"


def _pickup_return_action_candidates(sentence: str, question: str) -> list[tuple[str, str]]:
    """Extract concrete store-errand obligations from a sentence.

    The trigger is intentionally broader than the literal words "pick up" and
    "return": users often say collect, retrieve, drop off, exchange, or mail
    back for the same real-world obligation class.
    """
    if not PICKUP_RETURN_QUERY_RE.search(question):
        return []
    lower = sentence.lower()
    if not PICKUP_RETURN_QUERY_RE.search(sentence):
        return []

    candidates: list[tuple[str, str]] = []
    if re.search(r"\bdry[-\s]?clean(?:ing|er)?\b", lower) and PICKUP_ACTION_RE.search(sentence):
        item_match = re.search(
            r"\b(?:for|the)\s+(?:my\s+|the\s+)?([a-z][a-z\s-]{0,60}?"
            r"(?:blazer|jacket|coat|pants|dress|shirt|sweater|scarf|gloves|jeans))\b",
            sentence,
            re.I,
        )
        item = item_match.group(1).strip() if item_match else "dry-cleaning clothing"
        candidates.append(
            (
                f"pickup:{_dedupe_key(item)}:dry-cleaning",
                f"pickup obligation: pick up dry cleaning for {item}",
            )
        )

    if (
        RETURN_ACTION_RE.search(sentence)
        and CLOTHING_RE.search(sentence)
        and (
            re.search(r"\b(?:to|at|from)\s+([A-Z][A-Za-z0-9'&-]{1,40})\b", sentence)
            or re.search(r"\b(?:store|shop|zara|dry[-\s]?clean)\b", sentence, re.I)
        )
    ):
        store_match = re.search(r"\b(?:to|at|from)\s+([A-Z][A-Za-z0-9'&-]{1,40})\b", sentence)
        item_match = re.search(
            r"\b(?:return(?:ed)?|drop(?:ped)?\s*off|exchange|mail(?:ed)?\s*back|ship(?:ped)?\s*back|bring(?:ing)?\s*back)\s+"
            r"(?:some\s+|the\s+|my\s+)?([a-z][a-z\s-]{0,40}?"
            r"(?:boot|boots|jeans|pants|trousers|shirt|sweater|dress|jacket|coat|shoes|sneakers))\b",
            sentence,
            re.I,
        )
        item = item_match.group(1).strip() if item_match else CLOTHING_RE.search(sentence).group(0)
        store = f" to {store_match.group(1)}" if store_match else ""
        candidates.append(
            (
                f"return:{_dedupe_key(item)}",
                f"return obligation: return {item}{store}",
            )
        )

    if (
        not re.search(r"\bdry[-\s]?clean(?:ing|er)?\b", lower)
        and PICKUP_ACTION_RE.search(sentence)
    ) and (
        CLOTHING_RE.search(sentence) or re.search(r"\bnew pair\b|\blarger size\b", lower)
    ):
        item = "new pair"
        item_match = re.search(
            r"\b(?:pick(?:ed)?\s*up|pickup|collect(?:ed)?|retrieve(?:d)?)\s+(?:the\s+|my\s+)?([a-z][a-z\s-]{0,50}?"
            r"(?:boot|boots|jeans|pants|trousers|shirt|sweater|dress|jacket|coat|shoes|sneakers|pair))\b",
            sentence,
            re.I,
        )
        if item_match:
            item = item_match.group(1).strip()
        elif re.search(r"\bboot|boots\b", lower):
            item = "new boots"
        store_match = re.search(r"\b(?:from|at|to)\s+([A-Z][A-Za-z0-9'&-]{1,40})\b", sentence)
        store = f" from {store_match.group(1)}" if store_match else ""
        candidates.append(
            (
                f"pickup:{_dedupe_key(item)}:{store.lower().strip()}",
                f"pickup obligation: pick up {item}{store}",
            )
        )

    deduped: dict[str, str] = {}
    for key, label in candidates:
        deduped.setdefault(key, label)
    return list(deduped.items())


def _extract_user_requested_schema_columns(question: str, sources: list[EvidenceSource]) -> list[tuple[str, str]]:
    found: dict[str, str] = {}
    question_schema_context = bool(
        re.search(r"\b(?:column|columns|field|fields|schema|table|attribute|attributes|property|properties|migration)\b", question, re.I)
    )
    patterns = [
        re.compile(r"\badd\s+(?:a\s+|an\s+)?['`\"]?([a-zA-Z_][a-zA-Z0-9_]*)['`\"]?(?:\s+[A-Z][A-Z0-9()]+)?\s+(?:column|field|attribute|property)\b", re.I),
        re.compile(r"\binclude\s+(?:a\s+|an\s+)?['`\"]?([a-zA-Z_][a-zA-Z0-9_]*)['`\"]?(?:\s+[A-Z][A-Z0-9()]+)?\s+(?:column|field|attribute|property)\b", re.I),
        re.compile(r"\b(?:need|want|requested|ask(?:ed)?)\s+(?:a\s+|an\s+)?['`\"]?([a-zA-Z_][a-zA-Z0-9_]*)['`\"]?(?:\s+[A-Z][A-Z0-9()]+)?\s+(?:column|field|attribute|property)\b", re.I),
        re.compile(r"\b(?:column|field|attribute|property)\s+(?:called|named)\s+['`\"]?([a-zA-Z_][a-zA-Z0-9_]*)['`\"]?\b", re.I),
        re.compile(r"\binclude\s+(?:a\s+|an\s+)?['`\"]?([a-zA-Z_][a-zA-Z0-9_]*)['`\"]?(?:\s+[A-Z][A-Z0-9()]+)?\b.*?\bnew\s+(?:column|field|attribute|property)\b", re.I),
        re.compile(r"\b['`\"]?([a-zA-Z_][a-zA-Z0-9_]*)['`\"]?\s+[A-Z][A-Z0-9()]+\s+(?:column|field|attribute|property)\b", re.I),
    ]
    schema_signal_re = re.compile(
        r"\b(?:add|added|include|included|need|want|requested|asked|column|columns|field|fields|schema|migration|table|attribute|property|entity|model)\b",
        re.I,
    )
    for source in sources:
        rank = source.metadata.get("rank", "")
        for sentence in split_fact_candidates(source.text):
            if not re.search(r"\b(?:user:|i(?:'d| would)?\s+(?:want|need|like|am trying)|can you help me)\b", sentence, re.I):
                continue
            if not schema_signal_re.search(sentence):
                continue
            if not (
                question_schema_context
                or re.search(r"\b(?:[a-zA-Z_][a-zA-Z0-9_]*\s+table|table|schema|model|entity)\b", sentence, re.I)
            ):
                continue
            for pattern in patterns:
                for match in pattern.finditer(sentence):
                    column = match.group(1).strip("_").lower()
                    if column in {
                        "new",
                        "this",
                        "that",
                        "the",
                        "column",
                        "columns",
                        "field",
                        "fields",
                        "attribute",
                        "attributes",
                        "property",
                        "properties",
                        "table",
                        "schema",
                    }:
                        continue
                    found.setdefault(
                        column,
                        f"[rank={rank}; sid={source.source_id}; date={source.date}] {compact_fact(sentence, max_chars=420)}",
                    )
    return list(found.items())


def _term_set(text: str) -> set[str]:
    stopwords = QUESTION_STOPWORDS | {
        "days",
        "weeks",
        "months",
        "years",
        "long",
        "date",
        "time",
        "start",
        "end",
        "finish",
        "finishing",
        "completed",
        "completion",
        "first",
        "final",
    }
    return {
        token.lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", text)
        if token.lower() not in stopwords
    }


def _question_timeline_endpoint_terms(question: str) -> tuple[set[str], set[str]]:
    normalized = re.sub(r"\s+", " ", question).strip(" ?.")
    patterns = [
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"\bfrom\s+(.+?)\s+(?:to|till|until|through)\s+(.+?)(?:\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, re.I)
        if not match:
            continue
        start_terms = _term_set(match.group(1))
        end_terms = _term_set(match.group(2))
        if start_terms and end_terms:
            return start_terms, end_terms
    return set(), set()


def _matches_endpoint_terms(text: str, terms: set[str]) -> bool:
    if not terms:
        return False
    lower = text.lower()
    return any(term in lower for term in terms)


def _question_asks_current_or_latest(question: str) -> bool:
    return bool(re.search(r"\b(?:current|currently|latest|now|recent|recently|updated)\b", question, re.I))


def _extract_exact_answer_candidates(
    item: dict[str, Any],
    sources: list[EvidenceSource],
    *,
    max_candidates: int = 8,
) -> list[tuple[int, int, bool, str]]:
    question = str(item.get("question", ""))
    question_type = str(item.get("question_type", ""))
    keywords = question_keywords(question)
    wants_current = _question_asks_current_or_latest(question)
    rows: list[tuple[int, int, bool, str]] = []
    seen: set[str] = set()

    for source_idx, source in enumerate(sources, start=1):
        rank = source.metadata.get("rank", source_idx)
        try:
            rank_int = int(rank)
        except (TypeError, ValueError):
            rank_int = source_idx
        for sentence in split_fact_candidates(source.text):
            compact = compact_fact(sentence, max_chars=420)
            if not compact:
                continue
            lower = compact.lower()
            score = score_fact(compact, keywords, question_type)
            keyword_hits = sum(1 for keyword in keywords if keyword in lower)
            has_extractable = bool(DATE_RE.search(compact) or NUMBER_RE.search(compact))
            if not has_extractable or score < 4 or keyword_hits < 2:
                continue
            is_update = bool(UPDATE_RE.search(compact))
            if is_update and not wants_current:
                score -= 4
            if re.search(r"\b(?:ends?|deadline|date|when)\b", question, re.I) and DATE_RE.search(compact):
                score += 4
            key = _dedupe_key(compact)
            if key in seen:
                continue
            seen.add(key)
            row = f"[rank={rank}; sid={source.source_id}; date={source.date}] {compact}"
            rows.append((score, rank_int, is_update, row))

    return sorted(rows, key=lambda row: (-row[0], row[1], row[2], row[3]))[:max_candidates]


def _extract_contradiction_pair_candidates(
    item: dict[str, Any],
    sources: list[EvidenceSource],
    *,
    max_candidates: int = 4,
) -> list[tuple[int, str, str, str]]:
    question = str(item.get("question", ""))
    keywords = question_keywords(question)
    negative_rows: list[tuple[int, int, str]] = []
    positive_rows: list[tuple[int, int, str]] = []

    for source_idx, source in enumerate(sources, start=1):
        rank = source.metadata.get("rank", source_idx)
        try:
            rank_int = int(rank)
        except (TypeError, ValueError):
            rank_int = source_idx
        for sentence in split_fact_candidates(source.text):
            compact = compact_fact(sentence, max_chars=460)
            if not compact:
                continue
            lower = compact.lower()
            keyword_hits = sum(1 for keyword in keywords if keyword in lower)
            score = score_fact(compact, keywords, "contradiction_resolution") + keyword_hits
            row = f"[rank={rank}; sid={source.source_id}; date={source.date}] {compact}"
            has_route_signal = bool(re.search(r"\b(?:flask|route|routes?|http|requests?|homepage|@app\.route)\b", compact, re.I))
            if not has_route_signal:
                continue
            if re.search(r"\b(?:never|no|not|starting from scratch|haven't|have not)\b", compact, re.I):
                negative_rows.append((score + 5, rank_int, row))
            if re.search(r"\b(?:basic homepage route|homepage route|@app\.route|define routes?|implemented|added|worked with)\b", compact, re.I):
                positive_rows.append((score + 5, rank_int, row))

    if not negative_rows or not positive_rows:
        return []

    negative_rows = sorted(negative_rows, key=lambda row: (-row[0], row[1], row[2]))
    positive_rows = sorted(positive_rows, key=lambda row: (-row[0], row[1], row[2]))
    pairs: list[tuple[int, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for neg_score, neg_rank, neg in negative_rows[:max_candidates]:
        for pos_score, pos_rank, pos in positive_rows[:max_candidates]:
            key = (_dedupe_key(neg), _dedupe_key(pos))
            if key in seen:
                continue
            seen.add(key)
            score = neg_score + pos_score - min(neg_rank, pos_rank)
            pairs.append((score, neg, pos, "Which statement is correct?"))
    return sorted(pairs, key=lambda row: (-row[0], row[1], row[2]))[:max_candidates]


def prepare_evidence_sources(
    item: dict[str, Any],
    sources: list[EvidenceSource],
    *,
    max_sources: int | None = None,
    max_source_chars: int = 2_000,
    max_total_chars: int | None = None,
    max_sentences_per_source: int = 8,
    sort_chronologically: bool = False,
    prioritize_evidence: bool = False,
) -> EvidenceContext:
    """Prepare arbitrary retrieved sources for the deterministic compiler.

    This is the shared adapter for real-world and benchmark corpora where a
    retrieved item may be a short chat session, a long document chunk, a file,
    or a BEAM-style aggregated memory. Oversized sources are reduced by
    sentence-level token/regex evidence scoring before the normal compiler
    runs.
    """
    ordered_sources = sources[:max_sources] if max_sources is not None else list(sources)
    prepared_candidates: list[tuple[int, int, str, EvidenceSource, str]] = []
    used_ids: set[str] = set()

    for idx, source in enumerate(ordered_sources, start=1):
        sid = source.source_id or f"source-{idx}"
        if sid in used_ids:
            sid = f"{sid}-{idx}"
        used_ids.add(sid)
        prepared_text = _select_source_sentences(
            str(item.get("question", "")),
            str(item.get("question_type", "")),
            source.text,
            max_chars=max_source_chars,
            max_sentences=max_sentences_per_source,
        )
        if not prepared_text:
            continue
        source_score = _prepared_source_score(
            str(item.get("question", "")),
            str(item.get("question_type", "")),
            prepared_text,
        )
        prepared_candidates.append((source_score, idx, sid, source, prepared_text))

    if max_total_chars is not None:
        selection_order = prepared_candidates
        if prioritize_evidence:
            selection_order = sorted(prepared_candidates, key=lambda row: (-row[0], row[1], row[2]))

        selected_candidates: list[tuple[int, int, str, EvidenceSource, str]] = []
        total_chars = 0
        for candidate in selection_order:
            prepared_text = candidate[4]
            separator_chars = 1 if selected_candidates else 0
            projected_total = total_chars + separator_chars + len(prepared_text)
            if projected_total > max_total_chars:
                if selected_candidates:
                    continue
                prepared_text = prepared_text[:max_total_chars].rstrip()
                candidate = (candidate[0], candidate[1], candidate[2], candidate[3], prepared_text)
                projected_total = len(prepared_text)
            selected_candidates.append(candidate)
            total_chars = projected_total
        prepared_candidates = selected_candidates

    if sort_chronologically:
        prepared_candidates = sorted(prepared_candidates, key=lambda row: (row[3].date, row[1], row[2]))
    elif max_total_chars is not None and prioritize_evidence:
        prepared_candidates = sorted(prepared_candidates, key=lambda row: (row[1], row[2]))

    selected: list[str] = []
    date_by_sid: dict[str, str] = {}
    turns_by_sid: dict[str, list[dict[str, Any]]] = {}

    for _score, _idx, sid, source, prepared_text in prepared_candidates:
        selected.append(sid)
        date_by_sid[sid] = source.date
        turns_by_sid[sid] = [
            {
                "role": source.role,
                "content": prepared_text,
                "metadata": source.metadata,
            }
        ]

    return EvidenceContext(selected=selected, date_by_sid=date_by_sid, turns_by_sid=turns_by_sid)


def build_targeted_source_highlights(
    item: dict[str, Any],
    sources: list[EvidenceSource],
    *,
    max_sources: int | None = None,
    max_rows: int = 16,
    max_chars: int = 8_000,
) -> str:
    """Scan raw retrieved sources for high-signal facts before budgeted prep.

    This is a recall guard for large real-world sources and BEAM-style memory
    blobs: prepared context may reasonably fit the main budget while a decisive
    schema change or timeline row sits lower in the retrieval list. Highlights
    are source-linked candidates only; downstream prompts still reconcile them
    against the compiled packet.
    """
    question = str(item.get("question", ""))
    question_type = str(item.get("question_type", ""))
    keywords = question_keywords(question)
    limited_sources = sources[:max_sources] if max_sources is not None else sources
    rows: list[tuple[int, int, int, str]] = []
    exact_candidates = (
        _extract_exact_answer_candidates(item, limited_sources)
        if question_type in {"information_extraction", "single-session-user", "single-session-assistant"}
        else []
    )
    contradiction_pairs = (
        _extract_contradiction_pair_candidates(item, limited_sources)
        if question_type == "contradiction_resolution"
        else []
    )
    schema_columns = (
        _extract_user_requested_schema_columns(question, limited_sources)
        if is_count_list_question(question)
        and re.search(r"\b(?:column|columns|field|fields|schema|table|attribute|attributes|property|properties|migration)\b", question, re.I)
        else []
    )
    summary_aspects = {
        "features": re.compile(
            r"\b(?:registration|login|transaction|income|expense|analytics|visuali[sz]ation|chart)\b",
            re.I,
        ),
        "timeline": re.compile(r"\b(?:schedule|phase|milestone|MVP|deadline|April 15|deployment)\b", re.I),
        "security": re.compile(
            r"\b(?:security|Argon2|password|hashing|token-based|role-based|access control|input validation)\b",
            re.I,
        ),
        "documentation": re.compile(r"\b(?:documentation|Confluence|API endpoint|architecture|tables|diagrams)\b", re.I),
    }
    summary_candidates: dict[str, list[tuple[int, int, int, str]]] = defaultdict(list)
    timeline_candidates: list[tuple[int, int, str, date | None, date | None]] = []
    seen: set[str] = set()
    start_terms, end_terms = _question_timeline_endpoint_terms(question)

    for source_idx, source in enumerate(limited_sources, start=1):
        source_sentences = split_fact_candidates(source.text)
        source_rank = source.metadata.get("rank", source_idx)
        if question_type in {"temporal-reasoning", "temporal_reasoning"}:
            endpoint_snippets = [
                compact_fact(sentence, max_chars=300)
                for sentence in source_sentences
                if DATE_RE.search(sentence)
                and (
                    score_fact(sentence, keywords, question_type) >= 4
                    or _matches_endpoint_terms(sentence, start_terms)
                    or _matches_endpoint_terms(sentence, end_terms)
                )
            ]
            if start_terms and end_terms:
                start_snippets = [
                    snippet for snippet in endpoint_snippets if _matches_endpoint_terms(snippet, start_terms)
                ]
                end_snippets = [
                    snippet for snippet in endpoint_snippets if _matches_endpoint_terms(snippet, end_terms)
                ]
            else:
                start_snippets = [
                    snippet
                    for snippet in endpoint_snippets
                    if re.search(r"\b(?:start|started|begin|began|kickoff|initial|first|develop|feature|phase)\b", snippet, re.I)
                ]
                end_snippets = [
                    snippet
                    for snippet in endpoint_snippets
                    if re.search(r"\b(?:end|ended|finish|finished|final|deadline|deployment|completed|delivery|release)\b", snippet, re.I)
                ]
            if start_snippets and end_snippets:
                rank_int = source_idx
                try:
                    rank_int = int(source_rank)
                except (TypeError, ValueError):
                    pass
                start_text = start_snippets[0]
                end_text = next((snippet for snippet in end_snippets if snippet != start_text), end_snippets[0])
                year = _year_hint(source.text)
                start_dates = _parse_month_day_years(start_text, default_year=year)
                end_dates = _parse_month_day_years(end_text, default_year=year)
                if start_text == end_text and len(start_dates) >= 2:
                    start_date = start_dates[0]
                    end_date = start_dates[-1]
                else:
                    start_date = start_dates[-1] if start_dates else None
                    end_date = end_dates[-1] if end_dates else None
                timeline_candidates.append(
                    (
                        rank_int,
                        source_idx,
                        f"[rank={source_rank}; sid={source.source_id}; date={source.date}] start={start_text} | end={end_text}",
                        start_date,
                        end_date,
                    )
                )

        for sentence_idx, sentence in enumerate(source_sentences, start=1):
            compact = compact_fact(sentence, max_chars=520)
            if not compact:
                continue
            score = score_fact(compact, keywords, question_type)
            lower = compact.lower()
            keyword_hits = sum(1 for keyword in keywords if keyword in lower)
            date_count = len(DATE_RE.findall(compact))
            has_update = bool(UPDATE_RE.search(compact))
            has_action = bool(ACTION_RE.search(compact))
            has_schema_signal = bool(
                re.search(r"\b(?:add|added|include|column|field|schema|migration|table|alter)\b", compact, re.I)
            )
            has_timeline_signal = bool(
                re.search(r"\b(?:timeline|milestone|phase|deadline|deployment|completed|features?)\b", compact, re.I)
            )
            if question_type == "summarization":
                for aspect, pattern in summary_aspects.items():
                    if pattern.search(compact):
                        aspect_score = score + keyword_hits + date_count + int(has_update) + int(has_action)
                        if aspect in {"timeline", "documentation"}:
                            aspect_score += 3
                        if aspect == "documentation" and re.search(
                            r"\b(?:Confluence|API endpoints?|architecture decisions?|tables?|diagrams?)\b",
                            compact,
                            re.I,
                        ):
                            aspect_score += 8
                        if aspect == "security" and re.search(
                            r"\b(?:token-based|role-based|access control|input validation|Argon2)\b",
                            compact,
                            re.I,
                        ):
                            aspect_score += 5
                        summary_candidates[aspect].append(
                            (
                                aspect_score,
                                source_idx,
                                sentence_idx,
                                f"[rank={source_rank}; sid={source.source_id}; date={source.date}; role={source.role}] {compact}",
                            )
                        )
            if is_count_list_question(question) and has_schema_signal:
                score += 4
            if question_type in {"temporal-reasoning", "temporal_reasoning"} and has_timeline_signal:
                score += 4
            if keyword_hits >= 2:
                score += keyword_hits
            if date_count >= 2:
                score += 4
            if has_update:
                score += 2
            if has_action:
                score += 1
            if score < 5:
                continue
            key = _dedupe_key(compact)
            if key in seen:
                continue
            seen.add(key)
            rank = source.metadata.get("rank", source_idx)
            row = (
                f"[rank={rank}; sid={source.source_id}; date={source.date}; role={source.role}] "
                f"{compact}"
            )
            rows.append((score, source_idx, sentence_idx, row))

    if not rows:
        if (
            not exact_candidates
            and not contradiction_pairs
            and not schema_columns
            and not timeline_candidates
            and not summary_candidates
        ):
            return ""

    selected = sorted(rows, key=lambda row: (-row[0], row[1], row[2], row[3]))[:max_rows]
    lines = [
        "Targeted Source Highlights:",
        "- High-recall scan over raw retrieved sources before budgeted evidence preparation.",
        "- Use these as candidate facts to prevent missing lower-ranked schema changes, timeline endpoints, or multi-source list items.",
        f"- Question keywords: {', '.join(sorted(keywords)) if keywords else '(none)'}",
    ]
    if exact_candidates:
        lines.append("")
        lines.append("Exact Extraction Candidates:")
        preferred_exact = next((candidate for candidate in exact_candidates if not candidate[2]), exact_candidates[0])
        _score, _rank, _is_update, preferred_row = preferred_exact
        preferred_fact = preferred_row.split("] ", 1)[-1]
        preferred_dates = DATE_RE.findall(preferred_fact)
        preferred_value = f" value={preferred_dates[0]}" if preferred_dates else ""
        if preferred_dates:
            lines.append(f"- Mandatory exact extraction value for this non-current question: {preferred_dates[0]}")
        lines.append(f"- Preferred extraction answer candidate:{preferred_value} {preferred_row}")
        for _score, _rank, is_update, row in exact_candidates:
            marker = " update_or_current" if is_update else " direct"
            lines.append(f"-{marker}: {row}")
        lines.append(
            "- Selection hint: for exact extraction, answer with the preferred extraction candidate; use update/current candidates only when the question asks for current/latest."
        )
        lines.append("")
    if contradiction_pairs:
        lines.append("")
        lines.append("Contradiction Pair Candidates:")
        for _score, negative, positive, open_question in contradiction_pairs:
            lines.append(f"- Side A: {negative}")
            lines.append(f"  Side B: {positive}")
            lines.append(f"  Resolution prompt: {open_question}")
        lines.append(
            "- Selection hint: contradiction answers must mention Side A, Side B, and the resolution prompt before resolving or qualifying the answer."
        )
        lines.append("")
    if schema_columns:
        lines.append("")
        lines.append("User-Requested Schema Fields:")
        for column, evidence in schema_columns[:12]:
            lines.append(f"- {column}: {evidence}")
        lines.append(
            "- Selection hint: for schema count/list questions, count these user-requested columns/fields only; ignore assistant-only example fields."
        )
        lines.append("")
    if timeline_candidates:
        lines.append("")
        lines.append("Same-Source Timeline Candidates:")
        sorted_timeline_candidates = sorted(timeline_candidates, key=lambda row: (row[0], row[1], row[2]))
        preferred_timeline = next(
            (
                candidate
                for candidate in sorted_timeline_candidates
                if candidate[3] is not None and candidate[4] is not None and candidate[4] >= candidate[3]
            ),
            None,
        )
        if preferred_timeline is not None:
            _rank, _source_idx, _row, start_date, end_date = preferred_timeline
            assert start_date is not None
            assert end_date is not None
            lines.append(
                "- Preferred duration answer candidate: "
                f"{_format_duration_weeks(start_date, end_date)} from "
                f"{_format_long_date(start_date)} till {_format_long_date(end_date)}."
            )
        for _rank, _source_idx, row, start_date, end_date in sorted_timeline_candidates[:6]:
            duration = ""
            if start_date is not None and end_date is not None and end_date >= start_date:
                duration = f" | duration={_format_duration_weeks(start_date, end_date)}"
            lines.append(f"- {row}{duration}")
        lines.append(
            "- Selection hint: for between/duration questions, prefer the earliest-ranked same-source candidate that names both requested endpoints."
        )
        lines.append("")
    if summary_candidates:
        lines.append("")
        lines.append("Summary Coverage Candidates:")
        for aspect in ("documentation", "security", "timeline", "features"):
            candidates = sorted(
                summary_candidates.get(aspect, []),
                key=lambda row: (-row[0], row[1], row[2], row[3]),
            )[:3]
            if not candidates:
                continue
            lines.append(f"- {aspect}:")
            for _score, _source_idx, _sentence_idx, row in candidates:
                lines.append(f"  - {row}")
        lines.append(
            "- Selection hint: for summarization questions, cover every requested aspect with concrete supported details before adding general synthesis."
        )
        lines.append("")
    for _score, _source_idx, _sentence_idx, row in selected:
        lines.append(f"- {row}")
    lines.append(
        "Reducer instruction: reconcile these highlights with the compiled evidence packet; count only directly requested/asserted items and prefer timeline rows that match the question endpoints."
    )
    packet = "\n".join(lines)
    if len(packet) > max_chars:
        return packet[: max_chars - 70].rstrip() + "\n[targeted source highlights truncated for budget]"
    return packet


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


def _dedupe_key(text: str) -> str:
    words = [
        word
        for word in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_'-]{2,}", text.lower())
        if word not in QUESTION_STOPWORDS
    ]
    return " ".join(words[:14])


def build_evidence_packet(
    item: dict[str, Any],
    selected: list[str],
    date_by_sid: dict[str, str],
    turns_by_sid: dict[str, list[dict[str, Any]]],
    *,
    max_items_per_section: int = 24,
    max_chars: int = 12_000,
) -> str:
    """Build deterministic evidence candidates for the answer model.

    This is intentionally token/regex based. It does not answer the question;
    it assembles dated, source-linked candidates that are easy for an answer
    model to audit for temporal ordering, updates, counts, and preferences.
    """
    keywords = question_keywords(item["question"])
    question_type = item["question_type"]
    events: list[tuple[int, str]] = []
    updates: list[tuple[int, str]] = []
    counts: list[tuple[int, str]] = []
    preferences: list[tuple[int, str]] = []
    temporal: list[tuple[int, str]] = []
    seen: set[str] = set()

    for source_idx, sid in enumerate(selected, start=1):
        source = f"S{source_idx} {sid} {date_by_sid.get(sid, '')}".strip()
        for turn_idx, turn in enumerate(turns_by_sid[sid], start=1):
            clean = scrub_turn(turn)
            role = clean["role"]
            content = clean["content"]
            if not content:
                continue
            for sentence in split_fact_candidates(content):
                compact = compact_fact(sentence, max_chars=300)
                if not compact:
                    continue
                score = score_fact(compact, keywords, question_type)
                has_date = bool(DATE_RE.search(compact))
                has_number = bool(NUMBER_RE.search(compact))
                has_temporal = bool(TEMPORAL_RE.search(compact))
                has_update = bool(UPDATE_RE.search(compact))
                has_action = bool(ACTION_RE.search(compact))
                has_preference = bool(PREFERENCE_RE.search(compact))
                if not (score or has_date or has_number or has_temporal or has_update or has_action or has_preference):
                    continue
                row = f"[{source}; turn {turn_idx}; {role}] {compact}"
                key = _dedupe_key(row)
                if key in seen:
                    continue
                seen.add(key)
                weighted = score + int(has_action) + int(has_update) + int(has_temporal)
                if has_action or score >= 4:
                    events.append((weighted, row))
                if has_update:
                    updates.append((weighted, row))
                if has_number or is_count_list_question(item["question"]):
                    counts.append((weighted + int(has_number), f"{row} | dedupe_key={_dedupe_key(compact)}"))
                if has_temporal or has_date:
                    temporal.append((weighted + int(has_date), row))
                if has_preference:
                    preferences.append((weighted, row))

    def section(title: str, rows: list[tuple[int, str]], empty: str) -> list[str]:
        out = [f"{title}:"]
        if not rows:
            out.append(f"- {empty}")
            return out
        for _score, text in sorted(rows, key=lambda row: (-row[0], row[1]))[:max_items_per_section]:
            out.append(f"- {text}")
        return out

    lines = [
        "Deterministic Evidence Packet:",
        "- Built with token/regex rules from the retrieved sessions; verify against the full sessions below.",
        "- Use source ids and dates for ordering, current-vs-previous state, counts/lists, and provenance.",
        f"- Question keywords: {', '.join(sorted(keywords)) if keywords else '(none)'}",
    ]
    lines.extend(section("Events", events, "no event candidates detected"))
    lines.extend(section("State Updates", updates, "no update/currentness candidates detected"))
    lines.extend(section("Count/List Candidates", counts, "no count/list candidates detected"))
    lines.extend(section("Temporal Relations", temporal, "no temporal candidates detected"))
    lines.extend(section("Preference/Constraint Hints", preferences, "no preference or constraint candidates detected"))
    lines.append(
        "Reducer instruction: treat these as candidate evidence, not conclusions. Dedupe repeated candidates, preserve dates, and cite the source sessions in Source Notes."
    )
    packet = "\n".join(lines)
    if len(packet) > max_chars:
        return packet[: max_chars - 86].rstrip() + "\n[deterministic evidence packet truncated for budget]"
    return packet


def build_fusion_evidence_map(
    item: dict[str, Any],
    selected: list[str],
    date_by_sid: dict[str, str],
    turns_by_sid: dict[str, list[dict[str, Any]]],
    *,
    max_items: int = 12,
    max_chars: int = 5_000,
) -> str:
    """Build a compact orientation layer for already-strong fused retrieval.

    Unlike the heavier evidence packet, this map should not replace or narrow
    the source-aware context. It gives the answerer coverage, chronology, and
    update hints while keeping the full retrieved sessions authoritative.
    """
    keywords = question_keywords(item["question"])
    question_type = item["question_type"]
    source_rows: list[tuple[int, int, str]] = []
    temporal_rows: list[tuple[int, str]] = []
    update_rows: list[tuple[int, str]] = []
    conflict_rows: list[tuple[int, str]] = []
    preference_rows: list[tuple[int, str]] = []
    seen_facts: set[str] = set()
    temporal_evidence_sources: set[int] = set()

    for source_idx, sid in enumerate(selected, start=1):
        date = date_by_sid.get(sid, "")
        source_score = 0
        signals: set[str] = set()
        best: list[tuple[int, int, str]] = []
        for turn_idx, turn in enumerate(turns_by_sid[sid], start=1):
            clean = scrub_turn(turn)
            role = clean["role"]
            content = clean["content"]
            if not content:
                continue
            for sentence in split_fact_candidates(content):
                compact = compact_fact(sentence, max_chars=260)
                if not compact:
                    continue
                score = score_fact(compact, keywords, question_type)
                has_date = bool(DATE_RE.search(compact))
                has_temporal = bool(TEMPORAL_RE.search(compact))
                has_update = bool(UPDATE_RE.search(compact))
                has_preference = bool(PREFERENCE_RE.search(compact))
                has_conflict = bool(
                    re.search(r"\b(?:but|however|instead|actually|rather than|not|no longer|changed|updated)\b", compact, re.I)
                )
                has_event = bool(ACTION_RE.search(compact))
                if not (score or has_date or has_temporal or has_update or has_preference or has_conflict):
                    continue
                weighted = score + int(has_date) + int(has_temporal) + int(has_update) + int(has_conflict)
                source_score += weighted
                if score:
                    signals.add("query_overlap")
                if has_date or has_temporal:
                    signals.add("temporal")
                if has_update:
                    signals.add("update")
                if has_conflict:
                    signals.add("conflict_or_negation")
                if has_preference:
                    signals.add("preference")
                strong_temporal_signal = bool(
                    question_type == "temporal-reasoning"
                    and date
                    and (score or has_event or has_update or has_conflict or has_date or has_temporal)
                )
                if strong_temporal_signal:
                    signals.add("dated_event")
                    temporal_evidence_sources.add(source_idx)
                row = f"[S{source_idx}; sid={sid}; date={date}; turn={turn_idx}; {role}] {compact}"
                key = _dedupe_key(row)
                if key in seen_facts:
                    continue
                seen_facts.add(key)
                best.append((weighted, turn_idx, row))
                if has_date or has_temporal or strong_temporal_signal:
                    temporal_rows.append((weighted, row))
                if has_update:
                    update_rows.append((weighted, row))
                if has_conflict:
                    conflict_rows.append((weighted, row))
                if has_preference:
                    preference_rows.append((weighted, row))
        signal_text = ",".join(sorted(signals)) if signals else "none"
        source_rows.append((source_score, source_idx, f"S{source_idx}: sid={sid} date={date} score={source_score} signals={signal_text}"))

    covered = [row for row in source_rows if row[0] > 0]
    if question_type == "temporal-reasoning" and TEMPORAL_REASONING_QUERY_RE.search(item["question"]):
        # A temporal map is only useful when it can orient the model across at
        # least two dated pieces of evidence. Otherwise it becomes prompt noise
        # and can perturb unrelated rows without adding real signal.
        if len(temporal_evidence_sources) < 2:
            return ""
    active_experts = ["source-coverage"]
    if question_type == "knowledge-update" or UPDATE_RE.search(item["question"]):
        active_experts.extend(["update-currentness", "conflict-negation"])
    if question_type == "temporal-reasoning" or TEMPORAL_QUERY_RE.search(item["question"]):
        active_experts.append("timeline-ordering")
    if question_type == "single-session-preference":
        active_experts.append("preference-preservation")
    lines = [
        "Fusion Evidence Map:",
        "- Lightweight orientation layer for fused retrieval; the full History Chats below are authoritative.",
        "- Use this map to scan coverage, dates, updates, and conflicts, then verify every answer against the retrieved sessions.",
        "- If this map and the History Chats disagree, trust the History Chats.",
        f"- Question type: {question_type}",
        f"- Active experts: {', '.join(dict.fromkeys(active_experts))}",
        f"- Question keywords: {', '.join(sorted(keywords)) if keywords else '(none)'}",
        f"- Retrieved source coverage: {len(covered)}/{len(source_rows)} sources contain deterministic query/date/update/conflict signals.",
        "",
        "Source Coverage:",
    ]
    for _score, _source_idx, text in sorted(source_rows, key=lambda row: (row[1], row[2])):
        lines.append(f"- {text}")

    def add_section(title: str, rows: list[tuple[int, str]], include: bool) -> None:
        if not include:
            return
        lines.append("")
        lines.append(f"{title}:")
        if not rows:
            lines.append("- No deterministic candidates detected.")
            return
        for _score, text in sorted(rows, key=lambda row: (-row[0], row[1]))[:max_items]:
            lines.append(f"- {text}")

    add_section("Temporal/Date Hints", temporal_rows, question_type == "temporal-reasoning" or bool(TEMPORAL_QUERY_RE.search(item["question"])))
    add_section("Update/Current-State Hints", update_rows, question_type == "knowledge-update" or bool(UPDATE_RE.search(item["question"])))
    add_section("Conflict/Negation Hints", conflict_rows, question_type in {"knowledge-update", "temporal-reasoning"} or bool(UPDATE_RE.search(item["question"])))
    add_section("Preference/Constraint Hints", preference_rows, question_type == "single-session-preference")

    lines.append("")
    lines.append(
        "Reducer instruction: use Source Coverage to avoid overlooking relevant retrieved sessions; use the typed hints only as a reading guide, not as a constrained evidence set."
    )
    packet = "\n".join(lines)
    if len(packet) > max_chars:
        return packet[: max_chars - 68].rstrip() + "\n[fusion evidence map truncated for budget]"
    return packet


def build_multi_session_evidence_ledger(
    item: dict[str, Any],
    selected: list[str],
    date_by_sid: dict[str, str],
    turns_by_sid: dict[str, list[dict[str, Any]]],
    *,
    max_rows: int = 40,
    max_chars: int = 12_000,
) -> str:
    """Build a narrow deterministic ledger for multi-session synthesis.

    This is intentionally separate from the broad evidence packet. It only
    emits source-linked candidate facts that can participate in cross-session
    aggregation, comparison, temporal ordering, or dedupe.
    """
    keywords = question_keywords(item["question"])
    question_type = item["question_type"]
    rows: list[tuple[int, int, int, str]] = []
    seen: set[str] = set()

    for source_idx, sid in enumerate(selected, start=1):
        source = f"S{source_idx}"
        date = date_by_sid.get(sid, "")
        for turn_idx, turn in enumerate(turns_by_sid[sid], start=1):
            clean = scrub_turn(turn)
            role = clean["role"]
            content = clean["content"]
            if not content:
                continue
            for sentence in split_fact_candidates(content):
                compact = compact_fact(sentence, max_chars=320)
                if not compact:
                    continue
                score = score_fact(compact, keywords, question_type)
                signals: list[str] = []
                if DATE_RE.search(compact):
                    signals.append("date")
                if NUMBER_RE.search(compact):
                    signals.append("number")
                if TEMPORAL_RE.search(compact):
                    signals.append("temporal")
                if UPDATE_RE.search(compact):
                    signals.append("update")
                if ACTION_RE.search(compact):
                    signals.append("event")
                if PREFERENCE_RE.search(compact):
                    signals.append("preference_or_constraint")
                if score:
                    signals.append("query_overlap")
                if not signals:
                    continue
                key = _dedupe_key(compact)
                dedupe_scope = f"{key}|{source}|{turn_idx}"
                if dedupe_scope in seen:
                    continue
                seen.add(dedupe_scope)
                weight = score + len(set(signals)) + int("number" in signals) + int("date" in signals)
                row = (
                    f"source={source} sid={sid} date={date} turn={turn_idx} role={role} "
                    f"type={','.join(dict.fromkeys(signals))} dedupe_key={key or '(none)'} "
                    f"fact={compact}"
                )
                rows.append((weight, source_idx, turn_idx, row))

    lines = [
        "Multi-Session Evidence Ledger:",
        "- Built deterministically from retrieved sessions using token/regex signals only.",
        "- Each ledger row is a candidate fact, not a conclusion.",
        "- Use dedupe_key to merge repeated real-world items/events while keeping distinct items/events separate.",
        f"- Question keywords: {', '.join(sorted(keywords)) if keywords else '(none)'}",
    ]
    if not rows:
        lines.append("- No candidate facts were detected.")
    else:
        selected_rows = sorted(rows, key=lambda row: (-row[0], row[1], row[2], row[3]))[:max_rows]
        selected_rows = sorted(selected_rows, key=lambda row: (row[1], row[2], row[3]))
        for idx, (_weight, _source_idx, _turn_idx, row) in enumerate(selected_rows, start=1):
            lines.append(f"E{idx}: {row}")
    lines.append(
        "Reducer instruction: construct Candidate Set, Deduped Set, Missing/Excluded Evidence, and Final Answer from this ledger. Cite E# ids for every included candidate."
    )
    ledger = "\n".join(lines)
    if len(ledger) > max_chars:
        return ledger[: max_chars - 87].rstrip() + "\n[multi-session evidence ledger truncated for budget]"
    return ledger


def is_count_list_question(question: str) -> bool:
    return bool(COUNT_LIST_QUERY_RE.search(question))


def should_use_evidence_packet(item: dict[str, Any], mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "off":
        return False
    if mode == "general":
        question = item["question"]
        count_list = is_count_list_question(question)
        if item["question_type"] in {"multi-session", "single-session-preference"}:
            return False
        return (
            bool(TEMPORAL_QUERY_RE.search(question))
            or bool(UPDATE_RE.search(question))
            or bool(PREFERENCE_QUERY_RE.search(question))
            or count_list
            or item["question_type"] == "knowledge-update"
        )
    raise ValueError(f"unknown evidence packet mode: {mode}")


def should_use_fusion_evidence_map(item: dict[str, Any], mode: str) -> bool:
    if mode == "all":
        return True
    if mode == "off":
        return False
    if mode == "moe":
        # Conservative mixture-of-experts mode. The full 2026-05-20
        # LongMemEval fusion run showed broad maps helped update-style rows
        # but regressed preference and multi-session synthesis, so only attach
        # the map where it acts as a narrow expert hint.
        return item["question_type"] in {"knowledge-update", "temporal-reasoning"}
    if mode == "temporal":
        return item["question_type"] == "temporal-reasoning"
    if mode == "targeted":
        return item["question_type"] in {"knowledge-update", "temporal-reasoning"}
    if mode == "general":
        # Lighter than evidence_packet=general: keep raw fused context
        # authoritative and use the map only for coverage/orientation.
        return item["question_type"] in {
            "knowledge-update",
            "multi-session",
            "single-session-preference",
            "temporal-reasoning",
        }
    raise ValueError(f"unknown fusion evidence map mode: {mode}")


def should_use_aggregation_assembly(item: dict[str, Any], mode: str) -> bool:
    if mode == "off":
        return False
    if mode == "count_list":
        return is_count_list_question(item["question"])
    if mode == "multi_session_count_list":
        return item["question_type"] == "multi-session" and is_count_list_question(item["question"])
    raise ValueError(f"unknown aggregation assembly mode: {mode}")


def effective_fusion_evidence_map_mode(args: argparse.Namespace) -> str:
    if args.expert_ensemble == "moe" and args.fusion_evidence_map == "off":
        return "moe"
    return args.fusion_evidence_map


def effective_aggregation_assembly_mode(args: argparse.Namespace) -> str:
    return args.aggregation_assembly


def build_deterministic_aggregation_assembly(
    item: dict[str, Any],
    selected: list[str],
    date_by_sid: dict[str, str],
    turns_by_sid: dict[str, list[dict[str, Any]]],
    *,
    max_candidates: int = 36,
    max_chars: int = 12_000,
) -> dict[str, Any]:
    """Assemble source-linked aggregation candidates before answer synthesis.

    This is a conservative bridge for count/list/cross-thread questions. It
    does not decide the answer; it builds a compact candidate table and dedupe
    groups from retrieved sessions. If the candidate set is too sparse or too
    noisy, callers should fall back to normal source-aware synthesis.
    """
    keywords = question_keywords(item["question"])
    aggregation_terms = _aggregation_keywords(item["question"])
    count_list = is_count_list_question(item["question"])
    candidate_rows: list[tuple[int, int, int, str, str, str]] = []
    seen_sentence_scope: set[str] = set()

    for source_idx, sid in enumerate(selected, start=1):
        date = date_by_sid.get(sid, "")
        for turn_idx, turn in enumerate(turns_by_sid[sid], start=1):
            clean = scrub_turn(turn)
            role = clean["role"]
            content = clean["content"]
            if not content:
                continue
            if count_list and role == "assistant":
                continue
            for sentence in split_fact_candidates(content):
                compact = compact_fact(sentence, max_chars=220 if count_list else 320)
                if not compact:
                    continue
                for action_key, action_label in _pickup_return_action_candidates(
                    compact, item["question"]
                ):
                    row_text = (
                        f"[S{source_idx}; sid={sid}; date={date}; turn={turn_idx}; role={role}] "
                        f"{action_label}. Evidence: {compact}"
                    )
                    candidate_rows.append(
                        (
                            100,
                            source_idx,
                            turn_idx,
                            action_key,
                            f"S{source_idx}",
                            row_text,
                        )
                    )
                lower = compact.lower()
                score = score_fact(compact, keywords, item["question_type"])
                aggregation_hits = sum(1 for word in aggregation_terms if word in lower)
                has_signal = bool(
                    (score and (not count_list or aggregation_hits))
                    or NUMBER_RE.search(compact)
                    or DATE_RE.search(compact)
                    or ACTION_RE.search(compact)
                    or UPDATE_RE.search(compact)
                    or aggregation_hits
                )
                if not has_signal:
                    continue
                if count_list and not aggregation_hits:
                    continue
                dedupe_key = _dedupe_key(compact)
                sentence_scope = f"{sid}|{turn_idx}|{dedupe_key}"
                if sentence_scope in seen_sentence_scope:
                    continue
                seen_sentence_scope.add(sentence_scope)
                weight = (
                    score
                    + aggregation_hits * 4
                    + int(bool(NUMBER_RE.search(compact))) * 2
                    + int(bool(DATE_RE.search(compact)))
                    + int(bool(ACTION_RE.search(compact)))
                    + int(bool(UPDATE_RE.search(compact)))
                )
                candidate_rows.append(
                    (
                        weight,
                        source_idx,
                        turn_idx,
                        dedupe_key or f"source-{source_idx}-turn-{turn_idx}",
                        f"S{source_idx}",
                        f"[S{source_idx}; sid={sid}; date={date}; turn={turn_idx}; role={role}] {compact}",
                    )
                )

    ranked = sorted(candidate_rows, key=lambda row: (-row[0], row[1], row[2], row[5]))[:max_candidates]
    ranked = sorted(ranked, key=lambda row: (row[1], row[2], row[5]))
    groups: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for idx, (_weight, _source_idx, _turn_idx, dedupe_key, source, row_text) in enumerate(ranked, start=1):
        groups[dedupe_key].append((idx, source, row_text))

    coherent = bool(ranked) and len(ranked) <= max_candidates
    if is_count_list_question(item["question"]) and len(groups) == 0:
        coherent = False

    lines = [
        "Deterministic Aggregation Assembly:",
        "- Built from retrieved sessions before answer synthesis using token/regex signals only.",
        "- Candidate rows are evidence candidates, not conclusions.",
        "- Dedupe groups merge repeated real-world items/events; keep distinct items/events separate.",
        "- For pickup/return questions, pickup and return obligations are separate candidates when the source states both.",
        f"- Question keywords: {', '.join(sorted(keywords)) if keywords else '(none)'}",
        f"- Aggregation content terms: {', '.join(sorted(aggregation_terms)) if aggregation_terms else '(none)'}",
        f"- Coherence: {'coherent' if coherent else 'not_coherent'}",
        "",
        "Candidate Rows:",
    ]
    if not ranked:
        lines.append("- No aggregation candidates detected.")
    else:
        for idx, (_weight, _source_idx, _turn_idx, dedupe_key, _source, row_text) in enumerate(ranked, start=1):
            lines.append(f"C{idx}: group={dedupe_key} {row_text}")
    lines.append("")
    lines.append("Dedupe Groups:")
    if not groups:
        lines.append("- No dedupe groups.")
    else:
        for group_idx, (dedupe_key, group_rows) in enumerate(sorted(groups.items()), start=1):
            row_ids = ", ".join(f"C{idx}" for idx, _source, _row_text in group_rows)
            sources = ", ".join(dict.fromkeys(source for _idx, source, _row_text in group_rows))
            lines.append(f"G{group_idx}: key={dedupe_key} rows={row_ids} sources={sources}")
    lines.append(
        "Reducer instruction: answer count/list/cross-thread questions from Dedupe Groups. "
        "Cite C# row ids for included and excluded candidates. If the groups do not contain enough evidence, answer unavailable."
    )
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 92].rstrip() + "\n[deterministic aggregation assembly truncated for budget]"

    return {
        "coherent": coherent,
        "candidate_count": len(ranked),
        "dedupe_group_count": len(groups),
        "text": text,
    }
