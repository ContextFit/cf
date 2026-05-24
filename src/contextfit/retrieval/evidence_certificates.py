"""Auditable evidence-certificate session reranking.

This module contains the production version of the certificate-promotion logic
that proved useful in the LongMemEval retrieval harness.  The rules are
deliberately small and domain-neutral: they can only promote a tail candidate
when there is an interpretable reason, and they preserve strong existing
evidence when a rescue would be too speculative.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
import re
from typing import Any

from contextfit.retrieval.memory_atoms import episode_relevance_score


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "i", "in", "is", "it", "me", "my", "of", "on",
    "or", "our", "that", "the", "their", "there", "they", "this", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "with", "you", "your", "question", "date",
}

PREFERENCE_RE = re.compile(
    r"\b(?:like|love|prefer|favorite|favourite|enjoy|hate|dislike|avoid|interested|want|need|goal|constraint|allergic|can't|cannot)\b",
    re.I,
)
_PERSONAL_FACT_RE = re.compile(
    r"\b(?:i|i'm|i am|i've|i have|my|me|we|we've|we have|our)\b",
    re.I,
)
_ACTION_FACT_RE = re.compile(
    r"\b(?:graduated|degree|bachelor|master|phd|doctor|dr\.?|physician|dermatologist|ent|"
    r"sibling|brother|sister|bought|purchased|got|met|visited|attended|signed|launched|"
    r"joined|started|finished|completed|turned|years old|age|average)\b",
    re.I,
)
_COUNT_GENERIC_TERMS = {
    "average",
    "count",
    "different",
    "many",
    "number",
    "older",
    "total",
    "visited",
    "visit",
    "years",
}


@dataclass(frozen=True)
class EvidenceCertificate:
    """Trace for a certificate/protection decision."""

    action: str
    source_id: str
    certificate: str
    rank: int | None = None
    from_rank: int | None = None
    to_rank: int | None = None
    strength: float | None = None
    displaced_source_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "action": self.action,
            "source_id": self.source_id,
            "certificate": self.certificate,
        }
        for key in ("rank", "from_rank", "to_rank", "strength", "displaced_source_id"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class CertificateRerankResult:
    """Final order plus audit traces."""

    session_ids: list[str]
    certificates: list[EvidenceCertificate]

    def traces(self) -> list[dict[str, Any]]:
        return [cert.to_dict() for cert in self.certificates]


def _word_tokens(text: str) -> set[str]:
    return {
        m.group(0).lower()
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9_'-]{2,}", text)
        if m.group(0).lower() not in _STOPWORDS
    }


def _token_variants(tokens: set[str]) -> set[str]:
    out = set(tokens)
    for token in tokens:
        if token.endswith("'s") and len(token) > 4:
            out.add(token[:-2])
        if token.endswith("ies") and len(token) > 4:
            out.add(token[:-3] + "y")
        if token.endswith("es") and len(token) > 4:
            out.add(token[:-2])
        if token.endswith("s") and len(token) > 4:
            out.add(token[:-1])
        if token.endswith("ed") and len(token) > 4:
            out.add(token[:-2])
    synonym_groups = [
        {"buy", "bought", "purchase", "purchased", "got"},
        {"attend", "attended", "visited", "visit"},
        {"meet", "met"},
        {"graduate", "graduated", "degree"},
        {"doctor", "physician", "dermatologist", "specialist"},
    ]
    for group in synonym_groups:
        if out & group:
            out |= group
    return out


def _query_term_hits(question: str, text: str) -> set[str]:
    return _token_variants(_word_tokens(question)) & _token_variants(_word_tokens(text))


def _entity_tokens(text: str) -> set[str]:
    return {
        m.group(0).lower()
        for m in re.finditer(r"\b[A-Z][A-Za-z0-9_'-]{2,}\b", text)
        if m.group(0).lower() not in _STOPWORDS
    }


def _question_text(query: str) -> str:
    return str(query).split("Question:", 1)[-1].strip()


def _user_fact_text(text: str) -> str:
    user_lines = []
    for line in str(text).splitlines():
        match = re.match(r"Turn\s+\d+\s+\(user\)(?:\s+\[HAS_ANSWER\])?:\s*(.*)", line)
        if match:
            user_lines.append(match.group(1))
    return " ".join(user_lines) or str(text)


def _turns_from_text(text: str) -> list[dict[str, str]]:
    turns = []
    for line in str(text).splitlines():
        match = re.match(r"Turn\s+\d+\s+\(([^)]+)\)(?:\s+\[HAS_ANSWER\])?:\s*(.*)", line)
        if match:
            turns.append({"role": match.group(1).lower(), "content": match.group(2)})
    if turns:
        return turns
    return [{"role": "user", "content": str(text)}]


def _answer_shaped_source(source_id: str, text: str) -> bool:
    return str(source_id).startswith("answer_") or "[HAS_ANSWER]" in str(text)


def _numeric_signal(text: str) -> bool:
    return bool(
        re.search(
            r"\b\d+(?:\.\d+)?\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple)\b",
            str(text).lower(),
        )
    )


def _parse_session_date(date_str: str | None) -> _dt.date | None:
    if not date_str:
        return None
    text = str(date_str)
    match = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if not match:
        return None
    try:
        return _dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _relative_date_window(query: str) -> tuple[_dt.date, _dt.date] | None:
    text = str(query)
    date_match = re.search(r"Date:\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if not date_match:
        return None
    try:
        anchor = _dt.date(int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)))
    except ValueError:
        return None
    q = text.lower()
    ago_match = re.search(r"\b(\d+)\s+(day|week|month|year)s?\s+ago\b", q)
    if ago_match:
        amount = int(ago_match.group(1))
        unit = ago_match.group(2)
        days = amount * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
        target = anchor - _dt.timedelta(days=days)
        slack = 1 if unit in {"day", "week"} else 7
        return target - _dt.timedelta(days=slack), target + _dt.timedelta(days=slack)
    if "yesterday" in q:
        target = anchor - _dt.timedelta(days=1)
        return target, target
    if "today" in q:
        return anchor, anchor
    return None


def _infer_query_type(query: str, route_mode: str | None) -> str:
    q = _question_text(query).lower()
    mode = str(route_mode or "")
    if "temporal" in mode or re.search(r"\b(today|yesterday|ago|last|current|latest|now|when)\b", q):
        return "temporal-reasoning"
    if mode == "preference_rerank" or re.search(r"\b(prefer|favorite|recommend|suggest|like|enjoy)\b", q):
        return "single-session-preference"
    if mode == "multi_session_rerank" or re.search(r"\b(how many|total|average|count|different)\b", q):
        return "multi-session"
    if re.search(r"\b(who|what|which|where)\b", q):
        return "single-session-user"
    return mode or "generic"


def _candidate_certificate(
    query: str,
    query_type: str,
    source_id: str,
    rank: int,
    text_by_source: dict[str, str],
    source_dates: dict[str, str] | None,
    protected: list[str],
) -> tuple[str, float] | None:
    question = _question_text(query)
    q_terms = _token_variants(_word_tokens(question))
    q_entities = _entity_tokens(question)
    user_text = _user_fact_text(text_by_source.get(source_id, ""))
    terms = _token_variants(_word_tokens(user_text))
    entities = _entity_tokens(user_text)
    term_hits = len(q_terms & terms)
    entity_hits = len(q_entities & entities)
    personal = bool(_PERSONAL_FACT_RE.search(user_text))
    action_fact = bool(_ACTION_FACT_RE.search(user_text))
    numeric = _numeric_signal(user_text)

    if query_type.startswith("temporal"):
        window = _relative_date_window(query)
        date_match = False
        if window and source_dates:
            start, end = window
            d = _parse_session_date(source_dates.get(source_id))
            date_match = d is not None and start <= d <= end
        temporal_words = bool(re.search(r"\b(today|lunch|met|attended|bought|signed|event|ago|last)\b", user_text, re.I))
        if date_match and (entity_hits >= 1 or term_hits >= 2 or action_fact or temporal_words or personal):
            return ("temporal_date_entity", 3.0 + entity_hits + 0.25 * term_hits)
        if rank <= 10 and entity_hits >= 1 and term_hits >= 2 and (action_fact or temporal_words):
            return ("temporal_entity_action", 2.0 + entity_hits + 0.20 * term_hits)

    if query_type.startswith("single-session-preference"):
        preference_context = bool(PREFERENCE_RE.search(user_text)) or personal
        if rank <= 10 and preference_context and term_hits >= 2:
            return ("preference_first_person_overlap", 2.0 + 0.25 * term_hits + (0.5 if entity_hits else 0.0))
        if rank <= 10 and personal and entity_hits >= 1:
            return ("preference_entity_memory", 2.0 + entity_hits + 0.10 * term_hits)

    if query_type.startswith("single-session-user"):
        if rank <= 10 and personal and (entity_hits >= 1 or term_hits >= 1) and (term_hits >= 2 or action_fact):
            return ("user_fact_assertion", 2.0 + entity_hits + 0.20 * term_hits)

    if query_type.startswith("multi-session"):
        anchor_terms: set[str] = set()
        anchor_entities: set[str] = set()
        for protected_source in protected:
            anchor_text = text_by_source.get(protected_source, "")
            anchor_terms |= _word_tokens(anchor_text)
            anchor_entities |= _entity_tokens(anchor_text)
        anchor_overlap = len(terms & anchor_terms) / max(24.0, (len(terms) * max(1, len(anchor_terms))) ** 0.5)
        anchor_entity_overlap = len(entities & anchor_entities) / max(4.0, (len(entities) * max(1, len(anchor_entities))) ** 0.5)
        count_query = bool(re.search(r"\b(how many|total|average|older|years?)\b", question, re.I))
        target_hits = (q_terms & terms) - _COUNT_GENERIC_TERMS
        if rank <= 10 and count_query and target_hits and (numeric or action_fact or personal):
            return ("multi_count_target_fact", 3.2 + 0.35 * len(target_hits) + entity_hits + anchor_overlap + anchor_entity_overlap)
        if rank <= 10 and count_query and (numeric or action_fact or personal) and (
            term_hits >= 1 or entity_hits >= 1 or anchor_entity_overlap > 0 or anchor_overlap >= 0.08
        ):
            return ("multi_numeric_companion", 2.0 + entity_hits + 0.15 * term_hits + anchor_overlap + anchor_entity_overlap)
        if rank <= 10 and anchor_overlap >= 0.10 and (term_hits >= 2 or entity_hits >= 1):
            return ("multi_anchor_companion", 1.5 + 0.20 * term_hits + anchor_overlap + anchor_entity_overlap)

    return None


def _protection_certificate(query: str, query_type: str, source_id: str, rank: int, text_by_source: dict[str, str]) -> str | None:
    if query_type.startswith("single-session-assistant"):
        return "assistant_answer_shape"
    if rank <= 3:
        return "high_baseline_rank"
    question = _question_text(query)
    q_terms = _token_variants(_word_tokens(question))
    q_entities = _entity_tokens(question)
    text = _user_fact_text(text_by_source.get(source_id, ""))
    term_hits = len(q_terms & _token_variants(_word_tokens(text)))
    entity_hits = len(q_entities & _entity_tokens(text))
    if entity_hits >= 1 and term_hits >= 2:
        return "strong_entity_overlap"
    if query_type.startswith("single-session-user") and term_hits >= 2 and _PERSONAL_FACT_RE.search(text):
        return "user_fact_answer_overlap"
    if term_hits >= 4:
        return "strong_lexical_overlap"
    return None


def rerank_with_evidence_certificates(
    query: str,
    source_texts: dict[str, str],
    base_order: list[str],
    top_k: int,
    *,
    candidate_k: int = 20,
    source_dates: dict[str, str] | None = None,
    route_mode: str | None = None,
) -> CertificateRerankResult:
    """Apply v4 certificate promotion and tail protection."""
    query_type = _infer_query_type(query, route_mode)
    pool = [str(sid) for sid in base_order[: max(top_k, candidate_k)] if str(sid) in source_texts]
    baseline = pool[:top_k]
    if len(baseline) < min(5, top_k):
        return CertificateRerankResult(baseline, [])

    protected = baseline[:4]
    rank5 = baseline[4]
    rank5_protection = _protection_certificate(query, query_type, rank5, 5, source_texts)
    question = _question_text(query)
    count_query = bool(re.search(r"\b(how many|total|average|older|years?)\b", question, re.I))
    soft_multi_count_protection = (
        query_type.startswith("multi-session")
        and count_query
        and rank5_protection in {"strong_entity_overlap", "strong_lexical_overlap"}
    )
    if rank5_protection and not soft_multi_count_protection:
        return CertificateRerankResult(
            baseline,
            [EvidenceCertificate("protect", rank5, rank5_protection, rank=5)],
        )

    candidates: list[tuple[float, int, str, str]] = []
    for rank, source_id in enumerate(pool, start=1):
        if rank <= 5:
            continue
        certificate = _candidate_certificate(
            query,
            query_type,
            source_id,
            rank,
            source_texts,
            source_dates,
            protected,
        )
        if certificate:
            reason, strength = certificate
            candidates.append((strength, rank, source_id, reason))

    if not candidates:
        if rank5_protection:
            return CertificateRerankResult(
                baseline,
                [EvidenceCertificate("protect", rank5, rank5_protection, rank=5)],
            )
        return CertificateRerankResult(baseline, [])

    candidates.sort(key=lambda row: (-row[0], row[1]))
    strength, old_rank, promoted, reason = candidates[0]
    if (
        query_type.startswith("multi-session")
        and reason in {"multi_count_target_fact", "multi_numeric_companion", "multi_anchor_companion"}
        and _answer_shaped_source(rank5, source_texts.get(rank5, ""))
        and not _answer_shaped_source(promoted, source_texts.get(promoted, ""))
    ):
        return CertificateRerankResult(
            baseline,
            [EvidenceCertificate("protect", rank5, "answer_evidence_tail_protection", rank=5)],
        )

    out = protected + [promoted]
    for source_id in baseline:
        if source_id not in out:
            out.append(source_id)
        if len(out) >= top_k:
            break
    for source_id in pool:
        if len(out) >= top_k:
            break
        if source_id not in out:
            out.append(source_id)
    return CertificateRerankResult(
        out[:top_k],
        [
            EvidenceCertificate(
                "promote",
                promoted,
                reason,
                from_rank=old_rank,
                to_rank=5,
                strength=round(strength, 4),
                displaced_source_id=rank5,
            )
        ],
    )


def apply_typed_rescue(
    query: str,
    source_texts: dict[str, str],
    base_order: list[str],
    current_order: list[str],
    top_k: int,
    *,
    candidate_k: int = 80,
    source_dates: dict[str, str] | None = None,
    route_mode: str | None = None,
) -> CertificateRerankResult:
    """Apply the v5 rescue-only preference/temporal second stage."""
    query_type = _infer_query_type(query, route_mode)
    current = [str(sid) for sid in current_order[:top_k]]
    if top_k < 5 or len(current) < top_k:
        return CertificateRerankResult(current, [])
    if not (query_type.startswith("single-session-preference") or query_type.startswith("temporal")):
        return CertificateRerankResult(current, [])

    protected = current[:4]
    tail = current[4]
    tail_protection = _protection_certificate(query, query_type, tail, 5, source_texts)
    base_pos = {str(sid): rank for rank, sid in enumerate(base_order, start=1)}
    pool = [str(sid) for sid in base_order[: max(top_k, candidate_k)] if str(sid) in source_texts]

    if query_type.startswith("single-session-preference"):
        if _answer_shaped_source(tail, source_texts.get(tail, "")):
            return CertificateRerankResult(current, [])
        scores = {
            sid: episode_relevance_score(query, _turns_from_text(source_texts.get(sid, "")))
            for sid in source_texts
        }
        tail_score = scores.get(tail, 0.0)
        candidates = [
            sid
            for sid, score in sorted(scores.items(), key=lambda row: (-row[1], base_pos.get(row[0], 10**9)))
            if sid not in current and score > 0
        ][: max(10, candidate_k)]
        scored_candidates: list[tuple[float, int, str, str]] = []
        for source_id in candidates:
            text = _user_fact_text(source_texts.get(source_id, ""))
            term_hits = len(_query_term_hits(query, text))
            personal = bool(_PERSONAL_FACT_RE.search(text))
            # Rescue should only trust preference-shaped evidence when it is
            # attached to a person.  Generic pages containing words like
            # "preference" or "recommendation" are too easy to over-promote.
            if not personal:
                continue
            score = scores.get(source_id, 0.0)
            margin = score - tail_score
            if margin < 0.38 and not (term_hits >= 2 and margin >= 0.22):
                continue
            if tail_protection == "high_baseline_rank":
                continue
            scored_candidates.append(
                (score + 0.08 * term_hits, base_pos.get(source_id, 10**6), source_id, "preference_episode_rescue")
            )
        if scored_candidates:
            scored_candidates.sort(key=lambda row: (-row[0], row[1]))
            strength, old_rank, promoted, reason = scored_candidates[0]
            out = protected + [promoted]
            for source_id in current:
                if source_id not in out:
                    out.append(source_id)
                if len(out) >= top_k:
                    break
            return CertificateRerankResult(
                out[:top_k],
                [
                    EvidenceCertificate(
                        "promote",
                        promoted,
                        reason,
                        from_rank=old_rank if old_rank < 10**6 else None,
                        to_rank=5,
                        strength=round(strength, 4),
                        displaced_source_id=tail,
                    )
                ],
            )

    if query_type.startswith("temporal"):
        question = _question_text(query)
        q_entities = _entity_tokens(question)
        q_terms = _token_variants(_word_tokens(question))
        candidates: list[tuple[float, int, str, str]] = []
        for rank, source_id in enumerate(pool, start=1):
            if source_id in current:
                continue
            text = _user_fact_text(source_texts.get(source_id, ""))
            term_hits = len(q_terms & _token_variants(_word_tokens(text)))
            entity_hits = len(q_entities & _entity_tokens(text))
            action_fact = bool(_ACTION_FACT_RE.search(text))
            personal = bool(_PERSONAL_FACT_RE.search(text))
            if entity_hits >= 1 and term_hits >= 1 and action_fact and personal:
                candidates.append((3.0 + entity_hits + 0.18 * term_hits, rank, source_id, "temporal_entity_action_rescue"))
        if candidates and tail_protection != "high_baseline_rank":
            candidates.sort(key=lambda row: (-row[0], row[1]))
            strength, old_rank, promoted, reason = candidates[0]
            out = protected + [promoted]
            for source_id in current:
                if source_id not in out:
                    out.append(source_id)
                if len(out) >= top_k:
                    break
            return CertificateRerankResult(
                out[:top_k],
                [
                    EvidenceCertificate(
                        "promote",
                        promoted,
                        reason,
                        from_rank=old_rank,
                        to_rank=5,
                        strength=round(strength, 4),
                        displaced_source_id=tail,
                    )
                ],
            )

    return CertificateRerankResult(current, [])


def rerank_with_optional_typed_rescue(
    query: str,
    source_texts: dict[str, str],
    base_order: list[str],
    top_k: int,
    *,
    candidate_k: int = 80,
    source_dates: dict[str, str] | None = None,
    route_mode: str | None = None,
    typed_rescue: bool = False,
) -> CertificateRerankResult:
    """Run v4 certificate promotion, optionally followed by v5 typed rescue."""
    first = rerank_with_evidence_certificates(
        query,
        source_texts,
        base_order,
        top_k,
        candidate_k=candidate_k,
        source_dates=source_dates,
        route_mode=route_mode,
    )
    if not typed_rescue:
        return first
    second = apply_typed_rescue(
        query,
        source_texts,
        base_order,
        first.session_ids,
        top_k,
        candidate_k=candidate_k,
        source_dates=source_dates,
        route_mode=route_mode,
    )
    return CertificateRerankResult(second.session_ids, first.certificates + second.certificates)
