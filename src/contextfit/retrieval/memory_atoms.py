"""Domain-agnostic agent memory atoms.

Memory atoms are compact, source-linked records extracted from user-authored
text.  They are meant to improve AI-agent retrieval generally: preferences,
goals, constraints, decisions, facts, temporal updates, and open loops.

This module is deliberately deterministic and label-free.  It does not encode
LongMemEval topics or benchmark answers; it extracts broad memory primitives
that should be useful for any agent memory corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class MemoryAtom:
    """A compact, source-linked memory primitive."""

    atom_type: str
    text: str
    source_id: str | None = None
    source_date: str | None = None
    turn_index: int | None = None
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_index_text(self) -> str:
        """Render atom as text suitable for token-native indexing."""
        parts = [f"Memory atom type: {self.atom_type}"]
        if self.source_date:
            parts.append(f"Date: {self.source_date}")
        if self.source_id:
            parts.append(f"Source session: {self.source_id}")
        if self.turn_index is not None:
            parts.append(f"Source turn: {self.turn_index}")
        parts.append(f"User memory: {self.text}")
        if self.atom_type == "user_preference":
            parts.append("Retrieval hints: preference likes dislikes favorites go-to recommendations taste choice")
        elif self.atom_type == "user_goal":
            parts.append("Retrieval hints: goal plan wants trying looking for future intent")
        elif self.atom_type == "user_constraint":
            parts.append("Retrieval hints: constraint requirement limitation budget allergy avoid cannot must need")
        elif self.atom_type == "decision":
            parts.append("Retrieval hints: decision decided chose will going to committed")
        elif self.atom_type == "temporal_update":
            parts.append("Retrieval hints: update changed now currently no longer switched latest")
        elif self.atom_type == "open_loop":
            parts.append("Retrieval hints: follow up reminder pending later next action")
        elif self.atom_type == "entity_fact":
            parts.append("Retrieval hints: user context entity fact possession state resource available relevant background")
        return "\n".join(parts)


# Broad language markers, intentionally domain-neutral.
ATOM_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    (
        "user_preference",
        re.compile(
            r"\b("
            r"i\s+(?:really\s+|usually\s+|generally\s+|always\s+|tend\s+to\s+)?"
            r"(?:like|love|prefer|enjoy|hate|dislike|avoid|can't stand|cannot stand)"
            r"|my\s+(?:favorite|favourite|go-?to|preference|preferred)"
            r"|works\s+(?:best|better)\s+for\s+me"
            r"|i'?m\s+(?:a\s+)?(?:fan|not\s+a\s+fan)\s+of"
            r")\b",
            re.I,
        ),
        0.80,
    ),
    (
        "user_interest",
        re.compile(
            r"\b("
            r"i\s+(?:am\s+|was\s+|have\s+been\s+)?(?:interested\s+in|into|getting\s+into|learning|exploring|working\s+on|reading\s+about|listening\s+to|watching)"
            r"|my\s+\w+\s+(?:is|has\s+been|keeps)\s+(?:into|interested\s+in|getting\s+into|talking\s+about|working\s+on)"
            r")\b",
            re.I,
        ),
        0.68,
    ),
    (
        "entity_fact",
        re.compile(
            r"\b("
            r"i\s+(?:have|own|bought|got|use|carry|harvested|grew|made|built)"
            r"|(?:my|our|the)\s+\w+\s+(?:is|are|has|have|keeps|seems)"
            r")\b",
            re.I,
        ),
        0.60,
    ),
    (
        "user_goal",
        re.compile(
            r"\b("
            r"i\s+(?:want|need|hope|plan|intend|aim|would\s+like|am\s+trying|am\s+looking|am\s+thinking|am\s+considering)"
            r"|i\s+am\s+(?:trying|looking|planning|hoping|thinking|considering)"
            r"|i'?m\s+(?:trying|looking|planning|hoping|thinking|considering)"
            r")\b",
            re.I,
        ),
        0.72,
    ),
    (
        "user_constraint",
        re.compile(
            r"\b("
            r"i\s+(?:can'?t|cannot|must|need\s+to|have\s+to|shouldn'?t)"
            r"|my\s+(?:budget|deadline|constraint|requirement|allergy|limit|limitation)"
            r"|(?:budget|deadline|constraint|requirement|allergy|limit|limitation)\s+(?:is|of)"
            r")\b",
            re.I,
        ),
        0.72,
    ),
    (
        "decision",
        re.compile(
            r"\b("
            r"i\s+(?:decided|chose|picked|selected|will|am\s+going\s+to|am\s+planning\s+to)"
            r"|let'?s\s+(?:go|do|use|choose|pick|try)"
            r")\b",
            re.I,
        ),
        0.68,
    ),
    (
        "temporal_update",
        re.compile(
            r"\b("
            r"now\s+i|i\s+(?:now|currently|recently|used\s+to|no\s+longer|changed|switched|updated)"
            r"|these\s+days|from\s+now\s+on|going\s+forward"
            r")\b",
            re.I,
        ),
        0.70,
    ),
    (
        "open_loop",
        re.compile(
            r"\b("
            r"remind\s+me|follow\s+up|check\s+back|later|next\s+time|tomorrow|next\s+week|pending|todo|to-do"
            r")\b",
            re.I,
        ),
        0.62,
    ),
]

QUESTION_INTEREST_RE = re.compile(
    r"\b(can you recommend|recommend|suggest|any tips|any advice|what should i|do you think|should i|help me|i'?m looking for)\b",
    re.I,
)

WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*")
EPISODE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "him", "his", "how", "i",
    "if", "in", "is", "it", "me", "my", "of", "on", "or", "our", "she", "so", "that", "the",
    "their", "them", "they", "this", "to", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your",
}


def _compact(text: str, max_chars: int = 700) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _split_sentences(text: str) -> list[str]:
    text = " ".join(str(text).split())
    if not text:
        return []
    # Keep this intentionally simple and dependency-free.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _is_user_turn(turn: dict[str, Any]) -> bool:
    role = str(turn.get("role", "")).lower()
    return role in {"user", "human", "client", "customer"}


def query_memory_intents(query: str) -> set[str]:
    """Infer broad memory intents from query wording."""
    q = str(query)
    intents: set[str] = set()
    if re.search(r"\b(watch|read|listen|eat|drink|buy|choose|pick|recommend|suggest|gift)\b", q, re.I):
        intents.update({"user_preference", "user_interest", "entity_fact"})
    if re.search(r"\b(prepare|plan|train|practice|work on|task|project|write|writing|how should i|what should i do|next steps)\b", q, re.I):
        intents.update({"user_goal", "user_constraint", "entity_fact"})
    if re.search(r"\b(can't|cannot|budget|deadline|allergy|avoid|constraint|requirement|snack|party|serve|dinner|cook|phone|travel)\b", q, re.I):
        intents.update({"user_constraint", "entity_fact"})
    if re.search(r"\b(now|current|currently|latest|these days|changed|switched)\b", q, re.I):
        intents.add("temporal_update")
    if re.search(r"\b(decide|decided|choice|choose|pick|picked|what database|what did we use)\b", q, re.I):
        intents.add("decision")
    if re.search(r"\b(follow up|remind|pending|next week|tomorrow|later)\b", q, re.I):
        intents.add("open_loop")
    return intents


def atom_type_priors(query: str) -> dict[str, float]:
    """Return general atom-type priors from query intent, not topic words."""
    intents = query_memory_intents(query)
    priors = {
        "user_preference": 1.0,
        "user_interest": 1.0,
        "user_goal": 1.0,
        "user_constraint": 1.0,
        "decision": 1.0,
        "temporal_update": 1.0,
        "open_loop": 1.0,
        "entity_fact": 1.0,
    }
    if {"user_preference", "user_interest"} & intents:
        priors["user_preference"] = 1.35
        priors["user_interest"] = 1.25
        priors["entity_fact"] = 1.15
    if {"user_goal", "user_constraint"} & intents:
        priors["user_goal"] = 1.35
        priors["user_constraint"] = 1.20
        priors["entity_fact"] = 1.10
    if "user_constraint" in intents:
        priors["user_constraint"] = 1.45
    if "temporal_update" in intents:
        priors["temporal_update"] = 1.50
    if "decision" in intents:
        priors["decision"] = 1.50
    if "open_loop" in intents:
        priors["open_loop"] = 1.50
    return priors


def augment_query_for_memory_atoms(query: str) -> str:
    """Add domain-neutral memory-intent hints for atom retrieval.

    This does not add topical synonyms. It only tells token retrieval which
    general memory primitive is useful for broad advice/recommendation queries.
    """
    q = str(query)
    hints: list[str] = []
    if QUESTION_INTEREST_RE.search(q):
        hints.extend([
            "user memory",
            "personal context",
            "user interest",
            "user preference",
            "user goal",
            "user constraint",
        ])
    if re.search(r"\b(now|current|currently|latest|these days|changed|switched)\b", q, re.I):
        hints.extend(["temporal update", "current user memory", "changed switched now"])
    if re.search(r"\b(decide|decided|choice|choose|pick|picked|use)\b", q, re.I):
        hints.extend(["decision", "chose picked selected"])
    if re.search(r"\b(follow up|remind|pending|next week|tomorrow|later)\b", q, re.I):
        hints.extend(["open loop", "follow up reminder pending"])
    if not hints:
        return q
    return q + "\nMemory retrieval intent: " + "; ".join(dict.fromkeys(hints))


def _salient_words(text: str, limit: int = 32) -> list[str]:
    words = [w for w in WORD_RE.findall(text.lower()) if len(w) > 2 and w not in EPISODE_STOPWORDS]
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for i, word in enumerate(words):
        counts[word] = counts.get(word, 0) + 1
        first.setdefault(word, i)
    ranked = sorted(counts, key=lambda w: (-counts[w], first[w], w))
    return ranked[:limit]


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / ((len(a) * len(b)) ** 0.5)


def episode_relevance_features(query: str, turns: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Numeric, domain-neutral relevance features for a prior episode.

    This is deliberately not a topic dictionary. It measures whether a prior
    user episode has the kind of memory signal requested by the query, plus
    token overlap over salient user-authored terms.
    """
    turns = list(turns)
    user_text = " ".join(str(t.get("content", "")) for t in turns if _is_user_turn(t))
    atoms = extract_memory_atoms(turns)
    intents = query_memory_intents(query)
    priors = atom_type_priors(query)
    q_terms = set(_salient_words(query, limit=24))
    e_terms = set(_salient_words(user_text, limit=48))
    atom_types = {a.atom_type for a in atoms}
    aligned = atom_types & intents if intents else set()
    atom_prior = max([priors.get(a.atom_type, 1.0) * a.confidence for a in atoms] or [0.0])
    aligned_conf = max([a.confidence for a in atoms if a.atom_type in aligned] or [0.0])
    lexical = _overlap_ratio(q_terms, e_terms)
    # Entity/context facts are often the bridge for vague advice (gift, cook,
    # pack, bring up, prepare) even when no explicit preference is stated.
    entity_context = 1.0 if "entity_fact" in atom_types and ("entity_fact" in intents or QUESTION_INTEREST_RE.search(query)) else 0.0
    specificity = min(len(e_terms) / 24.0, 1.0)
    return {
        "lexical": lexical,
        "atom_prior": atom_prior,
        "aligned_conf": aligned_conf,
        "aligned_count": float(len(aligned)),
        "entity_context": entity_context,
        "specificity": specificity,
    }


def episode_relevance_score(query: str, turns: Iterable[dict[str, Any]]) -> float:
    f = episode_relevance_features(query, turns)
    return (
        1.80 * f["lexical"]
        + 1.25 * f["aligned_conf"]
        + 0.45 * min(f["aligned_count"], 2.0)
        + 0.55 * f["entity_context"]
        + 0.35 * f["atom_prior"]
        + 0.10 * f["specificity"]
    )


def episode_context_text(
    turns: Iterable[dict[str, Any]],
    source_id: str | None = None,
    source_date: str | None = None,
) -> str:
    """Render a whole prior user episode as future-advice context.

    Unlike atoms, this keeps the episode as a unit. It helps later vague
    recommendation/advice queries find prior situations, interests, people,
    resources, and problems that may not be explicit preferences.
    """
    user_lines = []
    for i, turn in enumerate(turns, 1):
        if _is_user_turn(turn):
            text = _compact(str(turn.get("content", "")), max_chars=900)
            if text:
                user_lines.append(f"User turn {i}: {text}")
    if not user_lines:
        return ""
    joined = "\n".join(user_lines)
    atom_types = sorted({atom.atom_type for atom in extract_memory_atoms(turns, source_id, source_date)})
    salient = _salient_words(joined)
    parts = ["Episode memory: prior user context useful for future advice, recommendations, planning, and decisions"]
    if source_date:
        parts.append(f"Date: {source_date}")
    if source_id:
        parts.append(f"Source session: {source_id}")
    if atom_types:
        parts.append("Episode signals: " + ", ".join(atom_types))
    if salient:
        parts.append("Salient episode terms: " + ", ".join(salient))
    parts.append(joined)
    parts.append("Retrieval hints: prior episode background context situation interests people resources problems constraints goals useful for advice recommendation planning what should I do")
    return "\n".join(parts)


def extract_memory_atoms(
    turns: Iterable[dict[str, Any]],
    source_id: str | None = None,
    source_date: str | None = None,
    max_atoms_per_turn: int = 4,
) -> list[MemoryAtom]:
    """Extract domain-neutral memory atoms from user-authored turns."""
    atoms: list[MemoryAtom] = []
    seen: set[tuple[str, str]] = set()
    for turn_index, turn in enumerate(turns, 1):
        if not _is_user_turn(turn):
            continue
        content = str(turn.get("content", ""))
        sentences = _split_sentences(content)
        emitted = 0
        for sentence in sentences:
            if emitted >= max_atoms_per_turn:
                break
            matches: list[tuple[str, float]] = []
            for atom_type, pattern, confidence in ATOM_PATTERNS:
                if pattern.search(sentence):
                    matches.append((atom_type, confidence))
            # Advice/recommendation questions often reveal an interest even if
            # they do not contain explicit preference verbs.
            if QUESTION_INTEREST_RE.search(sentence):
                matches.append(("user_interest", 0.58))
            for atom_type, confidence in matches[:2]:
                text = _compact(sentence)
                key = (atom_type, text.lower())
                if key in seen:
                    continue
                seen.add(key)
                atoms.append(
                    MemoryAtom(
                        atom_type=atom_type,
                        text=text,
                        source_id=source_id,
                        source_date=source_date,
                        turn_index=turn_index,
                        confidence=confidence,
                    )
                )
                emitted += 1
                if emitted >= max_atoms_per_turn:
                    break
    return atoms
