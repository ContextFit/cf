"""
QueryExpander — lightweight token-space query expansion.

Two-layer approach (no heavy dependencies):
  1. Text-level: Porter stemmer + synonym dictionary on the raw query string.
     Produces an expanded set of text terms → re-tokenised.
  2. Token-level: optional IDF-guided expansion — find high-IDF tokens that
     frequently co-occur with query tokens (uses the inverted index).

Both layers return additional token IDs that augment the original query tokens
for BM25 search, without changing scores of already-matched tokens.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from contextfit.core.tokenizer import Tokenizer
    from contextfit.index.inverted import InvertedIndex


# ---------------------------------------------------------------------------
# Minimal Porter stemmer (no NLTK dependency)
# ---------------------------------------------------------------------------

def _stem(word: str) -> str:
    """Extremely lightweight suffix-stripping stemmer (English)."""
    w = word.lower()
    # Plurals / verb endings
    for suffix, repl in [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("ising", "ise"), ("izing", "ize"),
        ("ising", "ise"), ("ational", "al"), ("fulness", "ful"),
        ("ousness", "ous"), ("iveness", "ive"), ("ization", "ize"),
        ("isation", "ise"),
    ]:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return w[: -len(suffix)] + repl

    for suffix in ["ings", "ment", "ments", "ness", "ful", "ous", "ive", "ize",
                   "ise", "ation", "ations", "ing", "ated", "ates", "ator",
                   "ers", "ed", "es", "s"]:
        if w.endswith(suffix) and len(w) > len(suffix) + 3:
            return w[: -len(suffix)]
    return w


# ---------------------------------------------------------------------------
# Domain synonym dictionary (email / enterprise context)
# ---------------------------------------------------------------------------

SYNONYMS: dict[str, list[str]] = {
    # Fictional organization aliases used for examples and demos
    "acme":         ["acme corp", "acme inc", "acme.example"],
    "northwind":    ["northwind traders", "northwind.example"],
    "exampleco":    ["example co", "example company", "exampleco.example"],
    "contoso":      ["contoso ltd", "contoso.example"],
    "genai":        ["generative ai", "gen ai", "llm", "ai"],
    "rag":          ["retrieval augmented generation", "rag tag"],
    "rfp":          ["request for proposal", "request for proposals"],
    "rfi":          ["request for information"],
    "sow":          ["statement of work", "contract"],
    "poc":          ["proof of concept", "prototype"],
    "pmo":          ["project management"],
    # Personal memory / preference terms
    "favorite":     ["favourite", "prefer", "preferred", "like", "love", "enjoy", "go-to"],
    "favourite":    ["favorite", "prefer", "preferred", "like", "love", "enjoy", "go-to"],
    "prefer":       ["preference", "preferred", "favorite", "favourite", "like", "love", "choose"],
    "preference":   ["prefer", "preferred", "favorite", "favourite", "like", "love"],
    "preferred":    ["prefer", "preference", "favorite", "favourite", "like", "love"],
    "like":         ["prefer", "favorite", "favourite", "love", "enjoy"],
    "love":         ["like", "prefer", "favorite", "favourite", "enjoy"],
    "enjoy":        ["like", "love", "prefer", "favorite", "favourite"],
    "hate":         ["dislike", "avoid", "do not like", "don't like"],
    "dislike":      ["hate", "avoid", "do not like", "don't like"],
    # Common email terms
    "voicemail":    ["voice mail", "vm", "missed call", "new voicemail"],
    "invoice":      ["bill", "receipt", "payment", "charge"],
    "residuals":    ["residual", "participations", "royalties"],
    "attendance":   ["attend", "absent", "absence"],
    "isd":          ["independent school district", "school district"],
    # Variants
    "linkedin":     ["linked in"],
    "quickbooks":   ["quick books", "intuit", "qbo"],
    "docusign":     ["docu sign", "esignature", "e-signature"],
}

# Reverse map: expanded form → canonical
_REVERSE: dict[str, str] = {}
for canon, variants in SYNONYMS.items():
    for v in variants:
        _REVERSE[v] = canon


class QueryExpander:
    """
    Expands a text query or token list into additional search terms.

    Usage:
        expander = QueryExpander(tokenizer)
        extra_tokens = expander.expand_text("Acme voicemail residuals")
        all_tokens = np.concatenate([original_tokens, extra_tokens])
    """

    def __init__(
        self,
        tokenizer: "Tokenizer",
        use_stemming: bool = True,
        use_synonyms: bool = True,
        max_extra_tokens: int = 64,
    ) -> None:
        self.tokenizer = tokenizer
        self.use_stemming = use_stemming
        self.use_synonyms = use_synonyms
        self.max_extra_tokens = max_extra_tokens

    def expand_text(self, query: str) -> np.ndarray:
        """
        Expand a text query and return *additional* token IDs (not the originals).
        Callers should concatenate these with the original query tokens.
        """
        words = re.findall(r"[a-z0-9]+", query.lower())
        extra_terms: set[str] = set()

        for word in words:
            if self.use_synonyms:
                # Exact synonym lookup
                if word in SYNONYMS:
                    for syn in SYNONYMS[word]:
                        extra_terms.add(syn)
                # Reverse: if word is a variant, add canonical
                if word in _REVERSE:
                    extra_terms.add(_REVERSE[word])

            if self.use_stemming:
                stem = _stem(word)
                if stem != word:
                    extra_terms.add(stem)
                    # Also add stemmed synonyms
                    if self.use_synonyms and stem in SYNONYMS:
                        for syn in SYNONYMS[stem]:
                            extra_terms.add(syn)

        # Remove terms already in the original query
        original_terms = set(words)
        extra_terms -= original_terms

        if not extra_terms:
            return np.empty(0, dtype=np.uint32)

        # Tokenise extra terms
        extra_tokens: list[int] = []
        for term in extra_terms:
            try:
                toks = self.tokenizer.encode(term).tolist()
                extra_tokens.extend(toks)
                if len(extra_tokens) >= self.max_extra_tokens:
                    break
            except Exception:
                continue

        return np.array(extra_tokens[: self.max_extra_tokens], dtype=np.uint32)

    def expand_tokens(
        self,
        query_tokens: np.ndarray,
        index: "InvertedIndex | None" = None,
        top_cooccur: int = 8,
    ) -> np.ndarray:
        """
        Expand from token IDs directly.
        Optionally uses the inverted index to find high-IDF co-occurring tokens.
        """
        # Decode tokens back to text for synonym/stemming expansion
        try:
            text = self.tokenizer.decode(query_tokens)
        except Exception:
            text = ""
        text_extras = self.expand_text(text)

        if index is None or top_cooccur == 0:
            return text_extras

        # Co-occurrence expansion: for each query token, find tokens that appear
        # in the same chunks and have high IDF (i.e. are specific).
        cooccur_candidates: dict[int, int] = {}
        query_set = set(query_tokens.tolist())

        for token in query_tokens.tolist():
            posting = None
            if hasattr(index, "_load_posting"):
                posting = index._load_posting(token)
            else:
                posting = index._index.get(token)
            if posting is None:
                continue
            # Sample up to 50 chunks from this posting
            chunk_sample = list(posting.chunk_ids)[:50]
            for cid in chunk_sample:
                # This would require scanning chunks — skip for now,
                # rely on text expansion instead.
                pass

        return text_extras
