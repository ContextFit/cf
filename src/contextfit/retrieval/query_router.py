"""Lightweight, deterministic query-type router for agent memory retrieval.

Routes incoming queries to the most appropriate retrieval mode without any
LLM calls or external embeddings.  Signals are structural and domain-neutral:
query syntax, intent verbs, specificity markers, and memory-type hints derived
from the same primitives used in memory_atoms.py.

Modes
-----
episode_score
    Vague advice / recommendation / planning queries.  The caller wants the
    most contextually relevant *prior episode*, not a specific fact.  Best
    served by the numeric episode relevance scorer.

bm25
    Specific fact lookups: decisions, current state, named open loops,
    temporal updates.  The caller knows what they're looking for; BM25 excels
    at retrieving the session that contains that specific text.

preference_rerank
    Recommendation/preference queries where the answer should come from prior
    user-authored taste/preference evidence, not generic topical overlap.

atom_fusion
    Explicit preference / constraint / goal queries with clear intent verbs.
    Atom confidence fusion adds atom-type signal on top of BM25.

multi_session_rerank
    Evidence-coverage queries: the query asks for planning/background/advice
    where multiple memory facets may be spread across sessions.

episode_bm25_fusion
    Multi-session synthesis hints: fallback RRF fuse episode scorer + BM25.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Compiled patterns -- all domain-neutral
# ---------------------------------------------------------------------------

# Strong vague-advice indicators
_VAGUE_ADVICE_RE = re.compile(
    r"\b("
    r"what\s+should\s+i\b"
    r"|what\s+would\s+(?:you\s+)?(?:suggest|recommend)\b"
    r"|what\s+(?:do\s+you\s+)?(?:think\s+i\s+should|would\s+be\s+good\s+for\s+me)\b"
    r"|(?:can\s+you\s+|please\s+)?(?:suggest|recommend)\b"
    r"|help\s+me\s+(?:plan|prepare|structure|organize|choose|pick|decide|pack|get\s+ready|figure\s+out|think\s+through)\b"
    r"|what(?:'s|\s+is)\s+(?:a\s+good|the\s+best)\s+\w+\s+for\s+me\b"
    r"|what\s+should\s+(?:i|we)\s+(?:do|make|cook|get|buy|watch|read|bring|wear|try|work\s+on)\b"
    r"|(?:any\s+)?(?:ideas?|suggestions?|tips?|advice)\s+(?:for|on|about)\b"
    r"|how\s+should\s+i\b"
    r"|what\s+type\s+of\b"
    r"|what\s+kind\s+of\b"
    r")",
    re.I,
)

# Vague open-loop queries: user asking "what did I leave undone?" without
# naming the specific topic.  Episode score handles these via open_loop atom
# alignment -- better than BM25 on generic reminders/todos with no query token.
_OPEN_LOOP_VAGUE_RE = re.compile(
    r"\b("
    r"is\s+there\s+(?:something|anything)\s+(?:i\s+)?(?:need\s+to|was\s+supposed\s+to|should|have\s+to)\b"
    r"|(?:do|did)\s+i\s+have\s+any\s+(?:outstanding|pending|open|unfinished)\b"
    r"|what\s+(?:outstanding|pending|open|unfinished)\b"
    r"|what\s+(?:am|was|were)\s+i\s+supposed\s+to\b"
    r"|what\s+(?:have\s+i\s+)?(?:left\s+undone|left\s+open|not\s+done|forgotten|been\s+meaning\s+to)\b"
    r"|anything\s+i\s+(?:haven'?t|still\s+need\s+to|was\s+supposed\s+to)\b"
    r"|what\s+pending\s+\w+\s+did\s+i\b"
    r"|what\s+did\s+i\s+(?:need\s+to|want\s+to|mean\s+to)\s+(?:do|follow|send|schedule|book|fix|finish|complete)\b"
    r")\b",
    re.I,
)

# Specific fact-lookup indicators: the caller knows the answer exists and
# wants it retrieved -- decisions, current state, named loops, temporal.
# Generic "pending"/"outstanding"/"todo" are in _OPEN_LOOP_VAGUE_RE instead.
_SPECIFIC_FACT_RE = re.compile(
    r"\b("
    r"what\s+did\s+i\s+(?:decide|choose|pick|select|go\s+with|end\s+up)\b"
    r"|which\s+(?:\w+\s+){0,4}did\s+i\s+(?:decide|choose|pick|select|go\s+with|end\s+up)\b"
    r"|what(?:'s|\s+is)\s+(?:my\s+)?(?:current|latest|new)\b"
    r"|am\s+i\s+(?:currently|still|now)\b"
    r"|what\s+(?:am|was)\s+i\s+(?:using|doing|working\s+on|eating|taking|wearing|reading|watching)\b"
    r"|(?:do\s+i|did\s+i|have\s+i)\s+(?:still|already|ever)\b"
    r"|remind\s+me\s+(?:what|about|to)\b"
    r"|which\s+(?:database|framework|tool|vendor|platform|language|provider|library)\b"
    r")",
    re.I,
)

# Temporal "current state" queries
_TEMPORAL_RE = re.compile(
    r"\b("
    r"(?:what|which)\s+(?:is|are)\s+(?:my\s+)?(?:current|latest|new|now)\b"
    r"|(?:currently|these\s+days|right\s+now|at\s+the\s+moment)\b"
    r"|(?:have\s+i\s+)?(?:switched|changed|updated|moved|migrated|stopped|quit|started)\b"
    r"|no\s+longer\b"
    r")",
    re.I,
)

# Multi-session hints
_MULTI_SESSION_RE = re.compile(
    r"\b("
    r"help\s+me\s+(?:get\s+ready|prepare)\b"
    r"|what\s+(?:do\s+i|should\s+i)\s+(?:focus\s+on|prioritise?|work\s+on)\s+(?:this\s+)?(?:week|month|quarter|year)\b"
    r"|make\s+(?:some\s+)?(?:progress|plans?)\b"
    r"|(?:this|next|upcoming)\s+(?:week|weekend|month|season|event|trip|vacation|project)\b.*(?:suggest|prioriti|focus|consider|plan)\b"
    r")",
    re.I,
)

_EVIDENCE_COVERAGE_RE = re.compile(
    r"\b("
    r"what\s+(?:factors|aspects|considerations|background|context|information|details|insights)\b"
    r"|what\s+(?:should|do)\s+i\s+(?:consider|keep\s+in\s+mind|focus\s+on)\b"
    r"|what\s+(?:are|were)\s+the\s+(?:key\s+)?(?:factors|considerations|details)\b"
    r"|can\s+you\s+(?:guide\s+me|provide\s+(?:some\s+)?details|give\s+me\s+guidance)\b"
    r"|help\s+me\s+(?:plan|prepare|organize|structure|come\s+up\s+with|figure\s+out)\b"
    r"|what\s+(?:different\s+)?aspects\s+do\s+i\s+need\b"
    r")",
    re.I,
)

# Personal aggregate-memory questions usually need evidence from more than one
# prior episode. Keep explicit relative-date math out of this route.
_PERSONAL_AGGREGATE_RE = re.compile(
    r"\b("
    r"(?:how\s+many|number\s+of|total\s+number\s+of|count\s+of)\b"
    r"|what\s+is\s+the\s+total\b"
    r"|what\s+are\s+all\s+(?:the\s+)?\w+"
    r")",
    re.I,
)

_PERSONAL_PRONOUN_RE = re.compile(r"\b(?:i|me|my|mine|we|us|our|ours)\b", re.I)

_RELATIVE_DATE_MATH_RE = re.compile(
    r"\b(?:how\s+many\s+)?(?:minutes?|hours?|days?|weeks?|months?|years?)\s+ago\b",
    re.I,
)

# Explicit preference / constraint verbs that atom fusion handles well
_ATOM_STRONG_RE = re.compile(
    r"\b("
    r"i\s+(?:like|love|prefer|enjoy|hate|dislike|avoid|always|usually|generally)\b"
    r"|my\s+(?:favorite|favourite|go-?to|preference)\b"
    r"|i\s+(?:can'?t|cannot|must|have\s+to|need\s+to|shouldn'?t)\b"
    r"|my\s+(?:budget|allergy|deadline|constraint|requirement|limitation)\b"
    r")",
    re.I,
)

# Recommendation/preference queries where a past taste statement should beat
# generic topical sessions.  This is intentionally query-shape based, not a
# domain dictionary: it identifies "use my prior taste/context to recommend".
_PREFERENCE_RECOMMEND_RE = re.compile(
    r"\b("
    r"(?:can\s+you\s+|please\s+)?(?:recommend|suggest)\b"
    r"|would\s+i\s+(?:like|enjoy)\b"
    r"|do\s+you\s+think\s+i\s+would\s+(?:like|enjoy)\b"
    r"|what\s+(?:type|kind)\s+of\b"
    r"|what\s+.*\bshould\s+i\s+(?:watch|read|listen|learn|wear|bring|try|get|buy|choose|pick)\b"
    r"|what\s+.*\bwould\s+be\s+good\s+for\s+(?:me|a\s+family)\b"
    r"|what\s+.*\bwould\s+you\s+suggest\b"
    r"|for\s+me\b"
    r")",
    re.I,
)

_PREFERENCE_EXCLUDE_RE = re.compile(
    r"\b("
    r"current|currently|latest|now|these\s+days|switched|changed|no\s+longer"
    r"|decide|decided|choice|which\s+.*\s+did"
    r"|pending|outstanding|follow\s+up|todo|to-do|supposed\s+to"
    r")\b",
    re.I,
)

# Specificity signals: proper nouns / quoted terms push toward bm25
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_QUOTED_RE = re.compile(r'["\'](.+?)["\']')


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class QueryRoute:
    """Routing decision for a single query."""

    mode: str
    """One of: episode_score, bm25, preference_rerank, multi_session_rerank, atom_fusion, episode_bm25_fusion"""

    confidence: float
    """0-1. Higher means the mode is clearly the right choice."""

    signals: list[str] = field(default_factory=list)
    """Human-readable reasons for the routing decision."""


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def route_query(query: str) -> QueryRoute:
    """Route a query to the appropriate retrieval mode.

    Deterministic, zero-cost, no external calls.
    """
    q = str(query)
    signals: list[str] = []
    scores: dict[str, float] = {
        "episode_score": 0.0,
        "bm25": 0.0,
        "preference_rerank": 0.0,
        "multi_session_rerank": 0.0,
        "atom_fusion": 0.0,
        "episode_bm25_fusion": 0.0,
    }

    # --- Vague open-loop (check BEFORE specific-fact so we don't mis-route) ---
    ol_matches = _OPEN_LOOP_VAGUE_RE.findall(q)
    if ol_matches:
        scores["episode_score"] += 1.6 * len(ol_matches)
        signals.append(f"open_loop_vague({len(ol_matches)} match)")

    # --- Vague advice ---
    va_matches = _VAGUE_ADVICE_RE.findall(q)
    if va_matches:
        scores["episode_score"] += 1.5 * len(va_matches)
        signals.append(f"vague_advice({len(va_matches)} match)")

    # --- Specific fact lookup ---
    sf_matches = _SPECIFIC_FACT_RE.findall(q)
    if sf_matches:
        scores["bm25"] += 1.8 * len(sf_matches)
        signals.append(f"specific_fact({len(sf_matches)} match)")

    # --- Temporal current-state ---
    # When temporal signals co-occur with specific_fact ("am I currently using",
    # "what is my current X"), the gold session often contains a temporal_update
    # atom (switched/changed/quit) rather than the exact query vocabulary.
    # Transfer the specific_fact bm25 score into fusion so both BM25 and
    # episode_score's temporal_update alignment contribute.
    if _TEMPORAL_RE.search(q):
        if sf_matches:
            # Move the sf bm25 score over to fusion, add fusion bonus
            scores["episode_bm25_fusion"] += scores["bm25"] + 1.8
            scores["bm25"] = 0.0
            signals.append("temporal+fact->fusion")
        else:
            scores["bm25"] += 1.2
            signals.append("temporal")

    # --- Multi-session hint ---
    if _MULTI_SESSION_RE.search(q):
        scores["multi_session_rerank"] += 1.6
        signals.append("multi_session_hint")

    if _EVIDENCE_COVERAGE_RE.search(q):
        scores["multi_session_rerank"] += 1.4
        signals.append("evidence_coverage")

    if (
        _PERSONAL_AGGREGATE_RE.search(q)
        and _PERSONAL_PRONOUN_RE.search(q)
        and not _RELATIVE_DATE_MATH_RE.search(q)
    ):
        scores["multi_session_rerank"] += 1.6
        signals.append("personal_aggregate")

    # --- Atom-strong preference/constraint ---
    if _ATOM_STRONG_RE.search(q):
        scores["atom_fusion"] += 0.8
        signals.append("atom_strong")


    # --- Preference/recommendation route ---
    # Use this when the query is asking for a personalized recommendation and
    # should prefer past explicit user taste statements over generic topical
    # sessions. Exclude fact/current/open-loop queries where BM25/temporal
    # routing is safer.
    if _PREFERENCE_RECOMMEND_RE.search(q) and not _PREFERENCE_EXCLUDE_RE.search(q):
        # Recommendation-shaped evidence-coverage queries ("what should I
        # consider", "how should I set up") usually need multiple facets, not
        # just explicit taste evidence. Let coverage win when it is present.
        if scores.get("multi_session_rerank", 0.0) > 0.0:
            scores["preference_rerank"] += 0.7
        else:
            scores["preference_rerank"] += 2.2
        signals.append("preference_recommendation")

    # --- Specificity penalty on episode_score ---
    proper_nouns = _PROPER_NOUN_RE.findall(q)
    quoted = _QUOTED_RE.findall(q)
    if proper_nouns or quoted:
        penalty = 0.5 * (len(proper_nouns) + len(quoted))
        scores["episode_score"] = max(0.0, scores["episode_score"] - penalty)
        scores["bm25"] += 0.4 * (len(proper_nouns) + len(quoted))
        signals.append(f"specificity_penalty(proper={len(proper_nouns)} quoted={len(quoted)})")

    # --- Blend toward fusion when vague-advice query has rich vocabulary ---
    # A vague advice query with 3+ salient domain terms has enough lexical
    # signal that BM25 can complement episode scoring.  Transfer the
    # episode_score signal into fusion so both signals contribute.
    # Recovers cases like "plan a healthy meal for the week" where BM25
    # directly matches gold session vocabulary.
    if va_matches and not sf_matches:
        # Domain-specific nouns: detect queries with enough lexical substance
        # that BM25 will find relevant sessions directly alongside episode scoring.
        # Requires 2+ salient terms with at least one 7+ char discriminative word.
        _GENERIC = {
            # Common 4-char function/action words
            "just", "also", "make", "want", "need", "like", "plan", "week",
            "good", "best", "from", "that", "with", "some", "more", "when",
            "then", "them", "they", "have", "been", "will", "your", "this",
            "help", "give", "find", "what", "tell", "show", "talk", "know",
            "cook", "food", "gift", "time", "work", "home", "book",
            "idea", "list", "pick", "next", "back", "life", "year",
            # Common 5-6 char generics and verbs
            "should", "would", "could", "might", "about", "maybe", "think",
            "every", "really", "pretty", "quite", "perhaps", "guess",
            "today", "before", "after", "going", "doing", "being",
            "having", "great", "happy", "these", "those", "right",
            "ready", "bring", "start", "check", "there", "their",
            "choose", "decide", "dinner", "friend", "night", "today",
            # 7+ char generics
            "something", "anything", "everything", "nothing",
            "special", "tonight", "weekend", "birthday",
            "suggest", "looking",
        }
        salient = [
            w for raw in _PROPER_NOUN_RE.sub("", q).lower().split()
            for w in [raw.strip(".,!?")]
            if len(w) >= 4 and w not in _GENERIC
        ]
        # Require at least one 7+ char discriminative term (single strong domain
        # noun like "healthy" or "podcast" is sufficient signal for BM25 blend)
        has_long = any(len(w) >= 7 for w in salient)
        if salient and has_long:
            # Do not steal clearly personalized recommendation queries from
            # the preference reranker; it handles cases like podcast/music/book
            # recommendations better than generic episode/BM25 fusion.
            if scores.get("preference_rerank", 0.0) == 0.0:
                scores["episode_bm25_fusion"] += scores["episode_score"] + 0.3
                scores["episode_score"] = 0.0
                signals.append(f"rich_vocab_blend(salient={len(salient)})")

    # --- Default episode prior for short/unclassified queries ---
    if not signals or (max(scores.values()) < 0.5):
        scores["episode_score"] += 0.6
        signals.append("default_episode_prior")

    best_mode = max(scores, key=lambda m: scores[m])
    total = sum(scores.values()) or 1.0
    confidence = scores[best_mode] / total

    return QueryRoute(mode=best_mode, confidence=round(confidence, 3), signals=signals)


def describe_route(route: QueryRoute) -> str:
    return f"mode={route.mode} conf={route.confidence:.2f} [{', '.join(route.signals)}]"
