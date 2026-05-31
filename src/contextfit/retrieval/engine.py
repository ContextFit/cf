"""
Retrieval engine: End-to-end query to token IDs.

Orchestrates: tokenize → search → traverse → collect → return input_ids
No detokenization until final LLM output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from contextfit.core.chunk import Chunk, ChunkStore
from contextfit.core.expander import QueryExpander
from contextfit.core.tokenizer import Tokenizer
from contextfit.index.semantic_expand import SemanticExpander
from contextfit.graph.similarity import LSHIndex, MinHasher
from contextfit.hierarchy.levels import HierarchyBuilder
from contextfit.index.bm25 import BM25Scorer
from contextfit.index.inverted import InvertedIndex, SegmentWriter, SegmentMerger
from contextfit.metadata.index import MetadataIndex
from contextfit.retrieval.evidence_atoms import rerank_sessions_by_evidence_atoms
from contextfit.retrieval.evidence_certificates import rerank_with_optional_typed_rescue
from contextfit.retrieval.memory_atoms import (
    augment_query_for_memory_atoms,
    atom_type_priors,
    episode_relevance_score,
    extract_memory_atoms,
    normalize_memory_query,
    query_memory_intents,
)
from contextfit.retrieval.query_spec import MetadataPredicate, QuerySpec
from contextfit.retrieval.query_router import QueryRoute, describe_route, route_query
from contextfit.retrieval.relationships import RelationshipIndex
from contextfit.retrieval.token_rerank import RerankTrace, TokenNativeReranker
from contextfit.sid.generator import SIDGenerator, SIDPrediction
from contextfit.sid.learned import LearnedSIDGenerator
from contextfit.sid.semantic import SemanticIDIndex


@dataclass
class RetrievalResult:
    """Result of a retrieval query."""
    
    # Retrieved chunks
    chunks: list[Chunk]
    
    # Combined token array ready for LLM input
    input_ids: np.ndarray
    
    # Metadata
    query_tokens: np.ndarray
    scores: list[float]
    method: str  # "bm25", "graph", "hierarchy"
    semantic_ids: list[tuple[int, ...]] | None = None
    sid_predictions: list[SIDPrediction] | None = None
    rerank_traces: list[RerankTrace] | None = None
    filter_trace: dict | None = None


class RetrievalEngine:
    """
    Complete retrieval engine for ContextFit.
    
    End-to-end flow:
    1. Tokenize query
    2. Search inverted index (BM25)
    3. Expand via graph traversal
    4. Navigate hierarchy
    5. Collect token arrays
    6. Return input_ids
    
    Example:
        engine = RetrievalEngine.from_path("./knowledge_base")
        
        result = engine.query("How does async work in Python?")
        
        # Feed directly to LLM
        output = model.generate(result.input_ids)
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_store: ChunkStore,
        inverted_index: InvertedIndex,
        bm25: BM25Scorer,
        hasher: MinHasher,
        lsh: LSHIndex,
        hierarchy: HierarchyBuilder | None = None,
        sid_index: SemanticIDIndex | None = None,
        sid_generator: SIDGenerator | None = None,
        learned_sid_generator: LearnedSIDGenerator | None = None,
        metadata_index: MetadataIndex | None = None,
        relationship_index: RelationshipIndex | None = None,
    ):
        self.tokenizer = tokenizer
        self.store = chunk_store
        self.inverted = inverted_index
        self.bm25 = bm25
        self.hasher = hasher
        self.lsh = lsh
        self.hierarchy = hierarchy
        self.sid_index = sid_index
        self.sid_generator = sid_generator
        self.learned_sid_generator = learned_sid_generator
        self.metadata = metadata_index or MetadataIndex()
        self.relationships = relationship_index or RelationshipIndex()
        self.expander = QueryExpander(tokenizer)
        self.token_reranker = TokenNativeReranker(tokenizer)
        self.semantic_expander: SemanticExpander | None = None  # loaded separately
        # Segment writer — set by ingest when deferred build is active
        self._segment_writer: SegmentWriter | None = None
        if self.sid_index is not None and self.sid_generator is None:
            self.sid_generator = SIDGenerator(self.sid_index, self.bm25, self.hasher, self.lsh)
    
    @classmethod
    def create(
        cls,
        path: Path | str,
        tokenizer_name: str = "cl100k_base",
        lsh_threshold: float = 0.3,
        num_perm: int = 128,
    ) -> RetrievalEngine:
        """
        Create a new RetrievalEngine with fresh indexes.
        
        Args:
            path: Base path for storage
            tokenizer_name: Tokenizer to use
            lsh_threshold: Similarity threshold for LSH
            num_perm: Number of MinHash permutations
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        tokenizer = Tokenizer.load(tokenizer_name)
        chunk_store = ChunkStore(path / "chunks")
        inverted = InvertedIndex()
        bm25 = BM25Scorer(inverted)
        hasher = MinHasher(num_perm=num_perm)
        lsh = LSHIndex(threshold=lsh_threshold, num_perm=num_perm)
        sid_index = SemanticIDIndex()
        
        return cls(
            tokenizer=tokenizer,
            chunk_store=chunk_store,
            inverted_index=inverted,
            bm25=bm25,
            hasher=hasher,
            lsh=lsh,
            sid_index=sid_index,
        )
    
    def enable_segment_writer(self, kb_path: Path, flush_every: int = 2000) -> None:
        """Enable incremental segment-based index building.

        Call this before a large ingest.  Chunks will be written to compact
        segment files instead of the in-memory inverted index.  After ingest
        call :meth:`merge_segments` to produce the final postings.bin.
        """
        self._segment_writer = SegmentWriter(
            segments_dir=kb_path / "segments",
            flush_every=flush_every,
        )

    def merge_segments(self, kb_path: Path) -> None:
        """K-way merge all segment files into postings.bin.  O(n_segments) RAM."""
        if self._segment_writer is not None:
            self._segment_writer.flush()   # flush any remaining buffer
            self._segment_writer.save_meta()
        merger = SegmentMerger(kb_path / "segments")
        merger.merge(kb_path / "inverted")
        # Reload the lazy index so queries work immediately
        from contextfit.index.inverted import LazyInvertedIndex
        self.inverted = LazyInvertedIndex.load_lazy(kb_path / "inverted")
        self.bm25 = BM25Scorer(self.inverted)

    def ingest_token_chunks(
        self,
        token_chunks: list[np.ndarray],
        metadata: dict | None = None,
        update_indexes: bool = True,
    ) -> list[Chunk]:
        """Ingest pre-tokenized chunk arrays.

        When a segment writer is active (large ingest mode), token data is
        written to segment files instead of the in-memory inverted index,
        keeping RAM use flat regardless of corpus size.
        """
        chunks = []
        for tokens in token_chunks:
            chunk = self.store.add(tokens, metadata=metadata)
            chunks.append(chunk)

            # Keep the structured metadata index in sync with the chunk store.
            # Older callers only persisted metadata inside Chunk objects, which
            # made metadata-aware ranking/aggregation unavailable unless a CLI
            # path explicitly called MetadataIndex.add_file_chunks(). Register
            # it here so every ingest path can group by fields like session_id,
            # source, author, date, etc.
            if metadata:
                self.metadata.add(chunk.chunk_id, metadata)

            try:
                self.relationships.add_text(
                    chunk.chunk_id,
                    self.tokenizer.decode(tokens.tolist()),
                    metadata=metadata,
                )
            except Exception:
                # Relationship extraction is derived/optional; ingestion should
                # never fail because a heuristic extractor could not parse text.
                pass

            if update_indexes:
                if self._segment_writer is not None:
                    # Incremental path: write to segment (O(1) RAM)
                    self._segment_writer.add_chunk(chunk.chunk_id, tokens)
                else:
                    # Classic path: add to in-memory index
                    self.inverted.add_chunk(chunk.chunk_id, tokens)
                # LSH and SID always in-memory (they're fast)
                sig = self.hasher.hash(chunk.chunk_id, tokens)
                self.lsh.add(sig)
                if self.sid_index is not None:
                    self.sid_index.assign_from_signature(chunk.chunk_id, sig)

        if update_indexes and self._segment_writer is None:
            self.bm25.clear_cache()

        return chunks

    def rank_sessions_by_episode_score(
        self,
        query: str,
        top_k: int = 10,
        session_field: str = "session_id",
    ) -> list[dict]:
        """Rank all indexed sessions by numeric episode relevance score.

        This is a domain-agnostic, index-free ranking approach: it scores every
        session by structural features (salient-term lexical overlap with the
        query, atom intent alignment, entity context coverage, episode
        specificity) without doing any BM25 or embedding retrieval.

        Best used for vague advice/recommendation/planning queries where the
        caller wants to surface the most contextually relevant prior episodes
        rather than retrieve specific fact-matching chunks.

        Returns a list of dicts: [{"session_id": ..., "score": ..., "rank": ...}]
        sorted by descending score.
        """
        import re

        TURN_RE = re.compile(r"Turn\s+\d+\s+\(([^)]+)\):\s*", re.I)

        # Group chunk texts by session_id.
        session_texts: dict[str, list[str]] = {}
        session_order: list[str] = []
        for chunk in self.store.iter_chunks():
            sid = chunk.metadata.get(session_field)
            if not sid:
                sid = self.metadata.get(chunk.chunk_id).get(session_field)
            if not sid:
                continue
            sid = str(sid)
            if sid not in session_texts:
                session_texts[sid] = []
                session_order.append(sid)
            try:
                text = self.tokenizer.decode(chunk.tokens.tolist())
            except Exception:
                continue
            session_texts[sid].append(text)

        results = []
        for sid in session_order:
            combined = "\n".join(session_texts[sid])
            # Reconstruct pseudo-turns: split on Turn N (role): markers if present,
            # otherwise treat the whole text as a single user turn.
            parts = TURN_RE.split(combined)
            if len(parts) > 1:
                # parts alternates: [pre, role1, content1, role2, content2, ...]
                turns = []
                i = 1
                while i + 1 < len(parts):
                    role = parts[i].strip().lower()
                    content = parts[i + 1].strip()
                    if content:
                        turns.append({"role": role, "content": content})
                    i += 2
            else:
                turns = [{"role": "user", "content": combined}]

            score = episode_relevance_score(query, turns)
            results.append({"session_id": sid, "score": score})

        results.sort(key=lambda x: -x["score"])
        for rank, r in enumerate(results, 1):
            r["rank"] = rank
        return results[:top_k]

    def rerank_sessions_by_structure(
        self,
        query: str,
        bm25_session_order: list[str],
        session_texts: dict[str, str],
        top_k: int = 10,
        session_field: str = "session_id",
    ) -> list[str]:
        """Token-native structural reranker for sessions retrieved by BM25.

        Takes BM25-ranked session IDs and their concatenated chunk text, then
        re-scores each session using deterministic structural features:
        - Lexical overlap and BM25 reciprocal rank
        - Episode relevance score (structural memory-marker density)
        - Intent/evidence marker alignment (decision/pref/goal/constraint/temporal/open-loop)
        - Named entity overlap (proper nouns in query vs session text)
        - Generic-distractor penalty

        All features are token-native and do not require embeddings.
        """
        import math
        import re
        from contextfit.retrieval.memory_atoms import episode_relevance_score as _ep_score

        _STOPWORDS = {
            "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
            "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "him", "his",
            "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "our", "she", "so",
            "that", "the", "their", "them", "they", "this", "to", "was", "we", "were", "what",
            "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
            "should", "help", "please", "about", "something", "anything",
        }
        _WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*")
        _PROPER_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
        _PROPER_STOPS = {
            "What", "How", "Why", "When", "Where", "Which", "Who", "Can", "Could", "Should",
            "Would", "Will", "Do", "Does", "Did", "Is", "Are", "Was", "Were", "The", "My",
            "Your", "Our", "Their", "His", "Her", "This", "That", "These", "Those", "They",
            "Have", "Has", "Had", "Been", "Just", "Some", "Any",
        }

        _DECISION_Q = re.compile(r"\b(decide|decided|choose|chose|choice|pick|picked|select|selected|which .+ did|what did .+ decide)\b", re.I)
        _DECISION_E = re.compile(r"\b(i|we)\s+(decided|chose|picked|selected|went with|will use|accepted)\b|\bdecided to\b", re.I)
        _PREF_Q = re.compile(r"\b(recommend|suggest|would i like|do you think i would like|type of|kind of|favorite|prefer)\b", re.I)
        _PREF_E = re.compile(r"\b(i|we)\s+(love|like|prefer|enjoy|hate|dislike|avoid)|favorite|go-?to|fan of|into\b", re.I)
        _GOAL_Q = re.compile(r"\b(focus|progress|next step|how should|structure|plan|automate|work on|adjust|improve)\b", re.I)
        _GOAL_E = re.compile(r"\b(i|we)\s+(want|need|hope|plan|aim|trying|looking|working|writing)|goal|trying to\b", re.I)
        _CONSTRAINT_Q = re.compile(r"\b(should i|what should|serve|bring|order|make|budget|constraint|allergy|deadline|quick|cheap)\b", re.I)
        _CONSTRAINT_E = re.compile(r"\b(can'?t|cannot|must|need to|have to|budget|deadline|allergy|limit|only have|trying to keep)\b", re.I)
        _TEMPORAL_Q = re.compile(r"\b(current|currently|now|latest|these days|switched|changed|new|no longer)\b", re.I)
        _TEMPORAL_E = re.compile(r"\b(now|currently|recently|switched|changed|updated|no longer|used to|started|stopped|quit|latest|these days)\b", re.I)
        _OPEN_Q = re.compile(r"\b(outstanding|pending|follow up|supposed to|need to do|left undone|schedule|book|todo|to-do)\b", re.I)
        _OPEN_E = re.compile(r"\b(todo|to-do|remind me|follow up|need to|haven'?t|still need|supposed to|deadline|schedule|book|renew|send|finish)\b", re.I)
        _ENTITY_Q = re.compile(r"\b(friend|wife|husband|son|daughter|client|dog|cat|team|partner|parent|child|colleague|boss|coworker|sibling|sister|brother|spouse|relative|neighbor|mentor|manager)\b", re.I)

        def _words(text: str) -> set[str]:
            return {w for w in _WORD_RE.findall(str(text).lower()) if len(w) > 2 and w not in _STOPWORDS}

        def _proper_nouns(text: str) -> set[str]:
            return {w.lower() for w in _PROPER_RE.findall(text) if w not in _PROPER_STOPS}

        def _overlap(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / math.sqrt(len(a) * len(b))

        def _idf_lexical(q_wl: list, s_wl: list, idf_map: dict) -> float:
            """IDF-cosine style lexical overlap."""
            if not q_wl or not s_wl:
                return 0.0
            q_set = set(q_wl)
            s_set = set(s_wl)
            shared = q_set & s_set
            if not shared:
                return 0.0
            numerator = sum(idf_map.get(w, 1.0) for w in shared)
            denom = (sum(idf_map.get(w, 1.0) for w in q_set) * sum(idf_map.get(w, 1.0) for w in s_set)) ** 0.5
            if denom == 0.0:
                return 0.0
            return numerator / denom

        # Question-type slot matching patterns (production copy)
        import re as _re
        _Q_WHO_P = _re.compile(r"\b(who|whose|someone)\b", _re.I)
        _E_WHO_P = _re.compile(
            r"\b([A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})*\s+(?:is|was|told|said|recommended|suggested|asked))"
            r"|\b(he|she|they)\s+(told|said|asked|recommended|suggested|explained|mentioned)\b",
            _re.I,
        )
        _Q_WHEN_P = _re.compile(r"\b(when|how long|what date|what time)\b", _re.I)
        _E_WHEN_P = _re.compile(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
            r"|\b(last\s+(week|month|year)|next\s+(week|month|year)|yesterday|tomorrow)"
            r"|\b(\d{4})|\b(\d{1,2}/\d{1,2})",
            _re.I,
        )
        _Q_WHERE_P = _re.compile(r"\b(where|what place|location)\b", _re.I)
        _E_WHERE_P = _re.compile(
            r"\b(at|in|near)\s+[A-Z][a-z]"
            r"|\b(address|location|city|town|street|avenue|boulevard|place|venue|restaurant|hotel|office|store)\b",
            _re.I,
        )
        _Q_HOW_P = _re.compile(r"\b(how\s+(do|did|should|can|to)|steps|process|procedure)\b", _re.I)
        _E_HOW_P = _re.compile(
            r"\b(step\s*\d+|first[,.]|then[,.]|next[,.]|finally[,.]|\d+\.\s+[A-Z])"
            r"|\b(follow\s+these|instructions|procedure|process)\b",
            _re.I,
        )
        _Q_RECOMMEND_P = _re.compile(r"\b(recommend|suggest|should i get|would i like|would you recommend)\b", _re.I)
        _E_RECOMMEND_P = _re.compile(
            r"\b(would\s+(love|enjoy|like)|perfect\s+for|you'?d\s+like|you'?ll\s+love|might\s+enjoy)"
            r"|\b(i\s+recommend|we\s+recommend|highly\s+recommend)\b",
            _re.I,
        )

        def _slot_match(q: str, txt: str) -> float:
            score = 0.0
            if _Q_WHO_P.search(q) and _E_WHO_P.search(txt):
                score += 1.0
            if _Q_WHEN_P.search(q) and _E_WHEN_P.search(txt):
                score += 1.0
            if _Q_WHERE_P.search(q) and _E_WHERE_P.search(txt):
                score += 1.0
            if _Q_HOW_P.search(q) and _E_HOW_P.search(txt):
                score += 1.0
            if _Q_RECOMMEND_P.search(q) and _E_RECOMMEND_P.search(txt):
                score += 1.2
            return min(score, 2.0)

        def _window_density(q_wset: set, txt: str, window_size: int = 150) -> float:
            _WRE = _re.compile(r"[a-z0-9][a-z0-9'_-]*")
            ws = [w for w in _WRE.findall(str(txt).lower()) if len(w) > 2 and w not in _STOPWORDS]
            if not ws or not q_wset:
                return 0.0
            if len(ws) <= window_size:
                return sum(1 for w in ws if w in q_wset) / window_size
            hits = sum(1 for w in ws[:window_size] if w in q_wset)
            max_hits = hits
            for i in range(1, len(ws) - window_size + 1):
                if ws[i - 1] in q_wset:
                    hits -= 1
                if ws[i + window_size - 1] in q_wset:
                    hits += 1
                if hits > max_hits:
                    max_hits = hits
            return max_hits / window_size

        normalized_query = normalize_memory_query(query)
        q_words = _words(normalized_query)
        q_entities = _proper_nouns(query)
        bm25_rank_map = {sid: i for i, sid in enumerate(bm25_session_order, 1)}

        # Build pseudo-turns from chunk text for episode scoring.
        TURN_RE = re.compile(r"Turn\s+\d+\s+\(([^)]+)\):\s*", re.I)

        def _text_to_turns(text: str) -> list[dict]:
            parts = TURN_RE.split(text)
            if len(parts) > 1:
                turns = []
                i = 1
                while i + 1 < len(parts):
                    role = parts[i].strip().lower()
                    content = parts[i + 1].strip()
                    if content:
                        turns.append({"role": role, "content": content})
                    i += 2
                return turns
            return [{"role": "user", "content": text}]

        # Also include any sessions not in BM25 candidates (they'll rank low by default)
        all_sids = list(bm25_session_order) + [s for s in session_texts if s not in bm25_rank_map]
        max_ep = 0.0001
        ep_scores: dict[str, float] = {}
        for sid in all_sids:
            text = session_texts.get(sid, "")
            turns = _text_to_turns(text)
            ep = _ep_score(query, turns)
            ep_scores[sid] = ep
            max_ep = max(max_ep, ep)

        # Build IDF table across all session texts
        idf_map: dict[str, float] = {}
        N_sess = len(all_sids)
        if N_sess > 0:
            df_map: dict[str, int] = {}
            for sid in all_sids:
                txt = session_texts.get(sid, "")
                for w in set(_words(txt)):
                    df_map[w] = df_map.get(w, 0) + 1
            idf_map = {w: math.log((N_sess + 1) / (d + 1)) + 1.0 for w, d in df_map.items()}

        q_words_list = list(q_words)

        rows = []
        for pos, sid in enumerate(all_sids, 1):
            text = session_texts.get(sid, "")
            words = _words(text)
            words_list = list(words)
            session_entities = _proper_nouns(text)
            lexical = _overlap(q_words, words)
            bm25 = 1.0 / bm25_rank_map[sid] if sid in bm25_rank_map else 0.0
            ep = ep_scores.get(sid, 0.0) / max_ep
            entity_hit = len(q_entities & session_entities) / max(len(q_entities), 1) if q_entities else 0.0
            entity_any = 1.0 if entity_hit > 0 else 0.0
            decision = 1.0 if _DECISION_Q.search(query) and _DECISION_E.search(text) else 0.0
            pref = 1.0 if _PREF_Q.search(query) and _PREF_E.search(text) else 0.0
            goal = 1.0 if _GOAL_Q.search(query) and _GOAL_E.search(text) else 0.0
            constraint = 1.0 if _CONSTRAINT_Q.search(query) and _CONSTRAINT_E.search(text) else 0.0
            temporal = 1.0 if _TEMPORAL_Q.search(query) and _TEMPORAL_E.search(text) else 0.0
            open_loop = 1.0 if _OPEN_Q.search(query) and _OPEN_E.search(text) else 0.0
            entity_role = 1.0 if _ENTITY_Q.search(query) and lexical > 0 else 0.0
            marker_sum = decision + pref + goal + constraint + temporal + open_loop + entity_any
            generic_penalty = 0.15 if marker_sum == 0 and lexical < 0.16 else 0.0
            # Feature 1: IDF-boosted lexical
            idf_lex = _idf_lexical(q_words_list, words_list, idf_map)
            # Feature 2: Question-type slot matching
            slot = _slot_match(query, text)
            # Feature 3: Evidence window density
            density = _window_density(q_words, text)
            score = (
                1.05 * bm25
                + 0.85 * ep
                + 1.10 * lexical
                + 0.95 * decision
                + 0.75 * pref
                + 0.70 * goal
                + 0.70 * constraint
                + 0.80 * temporal
                + 0.85 * open_loop
                + 0.25 * entity_role
                + 1.20 * entity_any
                + 0.60 * entity_hit
                - generic_penalty
                + 0.40 * idf_lex    # Feature 1: IDF-weighted lexical
                + 0.35 * slot       # Feature 2: question-type slot match
                + 0.45 * density    # Feature 3: evidence window density
            )
            rows.append((score, -pos, sid))

        rows.sort(reverse=True)
        return [sid for _score, _pos, sid in rows[:top_k] if _score > 0]

    def rerank_sessions_by_preference_facets(
        self,
        query: str,
        bm25_session_order: list[str],
        session_texts: dict[str, str],
        top_k: int = 10,
    ) -> list[str]:
        """Token-native preference/recommendation reranker.

        This is deliberately gated by the query router.  For personalized
        recommendation queries, it promotes sessions with explicit user-authored
        preference evidence ("I love", "I enjoy", "favorite", "prefer") and
        scores query overlap inside local preference windows.  It stays local and
        deterministic: no embeddings, no LLM, no topic dictionary.
        """
        import math
        import re
        from contextfit.retrieval.memory_atoms import episode_relevance_score as _ep_score

        _WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*")
        _TURN_RE = re.compile(r"Turn\s+\d+\s+\(([^)]+)\):\s*", re.I)
        _PREF_MARK = re.compile(
            r"\b("
            r"love|loved|like|liked|enjoy|enjoying|prefer|preferred|favorite|favourite|go-?to|fan of"
            r"|always loved|really love|really enjoy|usually like|usually prefer"
            r")\b",
            re.I,
        )
        _STOP = {
            "what", "should", "could", "would", "can", "you", "your", "recommend", "suggest",
            "good", "some", "type", "kind", "for", "my", "me", "the", "this", "that",
            "tonight", "next", "try", "start", "bring", "think", "like", "get", "make",
            "focus", "improve", "with", "about", "from", "into", "have", "has", "had",
            "and", "are", "was", "were", "will", "best", "help", "please",
        }

        def _stem(word: str) -> str:
            word = word.lower()
            if len(word) > 4 and word.endswith("ies"):
                return word[:-3] + "y"
            if len(word) > 4 and word.endswith("ing"):
                return word[:-3]
            if len(word) > 3 and word.endswith("s"):
                return word[:-1]
            return word

        def _words(text: str) -> list[str]:
            out: list[str] = []
            for raw in _WORD_RE.findall(str(text).lower()):
                if len(raw) <= 2:
                    continue
                w = _stem(raw)
                if w not in _STOP:
                    out.append(w)
            return out

        def _overlap(a: list[str], b: list[str]) -> float:
            aset, bset = set(a), set(b)
            if not aset or not bset:
                return 0.0
            return len(aset & bset) / math.sqrt(len(aset) * len(bset))

        def _user_text(text: str) -> str:
            parts = _TURN_RE.split(text)
            if len(parts) <= 1:
                return text
            lines: list[str] = []
            i = 1
            while i + 1 < len(parts):
                role = parts[i].strip().lower()
                content = parts[i + 1].strip()
                if role in {"user", "human", "client", "customer"} and content:
                    lines.append(content)
                i += 2
            return "\n".join(lines) or text

        def _text_to_turns(text: str) -> list[dict]:
            parts = _TURN_RE.split(text)
            if len(parts) <= 1:
                return [{"role": "user", "content": text}]
            turns: list[dict] = []
            i = 1
            while i + 1 < len(parts):
                role = parts[i].strip().lower()
                content = parts[i + 1].strip()
                if content:
                    turns.append({"role": role, "content": content})
                i += 2
            return turns

        def _preference_window(text: str) -> str:
            ws = _WORD_RE.findall(str(text).lower())
            if not ws:
                return text
            window_words: list[str] = []
            for i in range(len(ws)):
                local = " ".join(ws[i : i + 4])
                if _PREF_MARK.search(local):
                    window_words.extend(ws[max(0, i - 4) : min(len(ws), i + 18)])
            return " ".join(window_words) if window_words else text

        normalized_query = normalize_memory_query(query)
        q_words = _words(normalized_query)
        bm25_rank = {sid: i for i, sid in enumerate(bm25_session_order, 1)}
        all_sids = list(bm25_session_order) + [sid for sid in session_texts if sid not in bm25_rank]

        ep_scores: dict[str, float] = {}
        max_ep = 0.0001
        for sid in all_sids:
            ep = _ep_score(normalized_query, _text_to_turns(session_texts.get(sid, "")))
            ep_scores[sid] = ep
            max_ep = max(max_ep, ep)

        rows: list[tuple[float, int, str]] = []
        for pos, sid in enumerate(all_sids, 1):
            raw_text = session_texts.get(sid, "")
            user_text = _user_text(raw_text)
            has_pref = 1.0 if _PREF_MARK.search(user_text) else 0.0
            bm25 = 1.0 / bm25_rank[sid] if sid in bm25_rank else 0.0
            ep = ep_scores.get(sid, 0.0) / max_ep
            lexical = _overlap(q_words, _words(user_text))
            pref_window_overlap = _overlap(q_words, _words(_preference_window(user_text)))
            non_pref_penalty = 0.25 if not has_pref else 0.0
            # Weights from a held-out-style exploratory run over the generated
            # agent-memory eval.  BM25/episode remain meaningful; preference
            # evidence is decisive only for this router-gated query class.
            score = (
                1.05 * bm25
                + 0.85 * ep
                + 1.10 * lexical
                + 1.50 * has_pref
                + 2.00 * pref_window_overlap
                - 1.60 * non_pref_penalty
            )
            rows.append((score, -pos, sid))

        rows.sort(reverse=True)
        return [sid for _score, _pos, sid in rows[:top_k]]

    def rerank_sessions_by_evidence_coverage(
        self,
        query: str,
        bm25_session_order: list[str],
        session_texts: dict[str, str],
        top_k: int = 10,
    ) -> list[str]:
        """Token-native multi-session/evidence-coverage reranker.

        Multi-session advice/background queries often need complementary memory
        facets (goal + constraint, preference + event, context + resource).  The
        first ranking still returns individual sessions, but scoring rewards
        sessions that carry *one strong facet* and then applies a small diversity
        pass so top-K covers different evidence types instead of near-duplicates.
        """
        import math
        import re
        from contextfit.retrieval.memory_atoms import episode_relevance_score as _ep_score

        _WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*")
        _TURN_RE = re.compile(r"Turn\s+\d+\s+\(([^)]+)\):\s*", re.I)
        _STOP = {
            "what", "should", "could", "would", "can", "you", "your", "some", "any",
            "help", "guide", "provide", "give", "tell", "about", "with", "from", "into",
            "have", "has", "had", "the", "this", "that", "these", "those", "for", "and",
            "are", "was", "were", "will", "need", "keep", "mind", "consider", "factors",
            "aspects", "information", "background", "details", "insights", "upcoming",
            "next", "week", "weekend", "month", "good", "best", "make", "sure",
        }
        _MARKERS = {
            "preference": re.compile(r"\b(love|like|enjoy|prefer|favorite|favourite|go-?to|hate|avoid)\b", re.I),
            "constraint": re.compile(r"\b(can'?t|cannot|must|need\s+to|have\s+to|budget|allergy|deadline|limit|constraint|under\s+\$?\d+)\b", re.I),
            "goal": re.compile(r"\b(want\s+to|trying\s+to|goal|aim|improve|progress|working\s+on|planning\s+to|hope\s+to|achieve)\b", re.I),
            "temporal": re.compile(r"\b(current|currently|latest|recent|recently|upcoming|next|this\s+week|this\s+weekend|changed|switched|started)\b", re.I),
            "entity": re.compile(r"\b(my|our)\s+(home|family|kids|child|dog|pet|project|trip|event|garden|kitchen|room|team|work|friend|birthday|vacation|wedding)\b", re.I),
            "open_loop": re.compile(r"\b(todo|to-do|follow\s+up|remind|pending|outstanding|need\s+to\s+(?:do|buy|bring|send|finish))\b", re.I),
        }

        def _stem(w: str) -> str:
            w = w.lower()
            if len(w) > 4 and w.endswith("ies"):
                return w[:-3] + "y"
            if len(w) > 4 and w.endswith("ing"):
                return w[:-3]
            if len(w) > 3 and w.endswith("s"):
                return w[:-1]
            return w

        def _words(text: str) -> list[str]:
            out: list[str] = []
            for raw in _WORD_RE.findall(str(text).lower()):
                if len(raw) <= 2:
                    continue
                w = _stem(raw)
                if w not in _STOP:
                    out.append(w)
            return out

        def _overlap(a: list[str], b: list[str]) -> float:
            aset, bset = set(a), set(b)
            if not aset or not bset:
                return 0.0
            return len(aset & bset) / math.sqrt(len(aset) * len(bset))

        def _user_text(text: str) -> str:
            parts = _TURN_RE.split(text)
            if len(parts) <= 1:
                return text
            lines: list[str] = []
            i = 1
            while i + 1 < len(parts):
                role = parts[i].strip().lower()
                content = parts[i + 1].strip()
                if role in {"user", "human", "client", "customer"} and content:
                    lines.append(content)
                i += 2
            return "\n".join(lines) or text

        def _turns(text: str) -> list[dict]:
            parts = _TURN_RE.split(text)
            if len(parts) <= 1:
                return [{"role": "user", "content": text}]
            turns: list[dict] = []
            i = 1
            while i + 1 < len(parts):
                role = parts[i].strip().lower()
                content = parts[i + 1].strip()
                if content:
                    turns.append({"role": role, "content": content})
                i += 2
            return turns

        q_words = _words(query)
        bm25_rank = {sid: i for i, sid in enumerate(bm25_session_order, 1)}
        all_sids = list(bm25_session_order) + [sid for sid in session_texts if sid not in bm25_rank]
        ep_raw: dict[str, float] = {}
        max_ep = 0.0001
        for sid in all_sids:
            ep = _ep_score(query, _turns(session_texts.get(sid, "")))
            ep_raw[sid] = ep
            max_ep = max(max_ep, ep)

        rows: list[tuple[float, int, str, set[str], set[str]]] = []
        for pos, sid in enumerate(all_sids, 1):
            user_text = _user_text(session_texts.get(sid, ""))
            words = _words(user_text)
            lexical = _overlap(q_words, words)
            bm25 = 1.0 / bm25_rank[sid] if sid in bm25_rank else 0.0
            ep = ep_raw.get(sid, 0.0) / max_ep
            facets = {name for name, pat in _MARKERS.items() if pat.search(user_text)}
            q_facets = {name for name, pat in _MARKERS.items() if pat.search(query)}
            # Advice/background queries often do not name the exact facet; any
            # personal facet is useful, aligned facets are better.
            facet_strength = min(len(facets), 3) / 3.0
            aligned = len(facets & q_facets) / max(1, len(q_facets)) if q_facets else 0.0
            generic_penalty = 0.30 if not facets and lexical < 0.16 else 0.0
            # Short focused user memories should beat generic assistant-heavy text.
            specificity = min(1.0, len(set(words)) / 18.0) if words else 0.0
            score = (
                0.85 * bm25
                + 0.95 * ep
                + 1.25 * lexical
                + 0.95 * facet_strength
                + 0.70 * aligned
                + 0.20 * specificity
                - generic_penalty
            )
            rows.append((score, -pos, sid, facets, set(words)))

        rows.sort(reverse=True)
        # Diversity pass: keep high score dominant, but avoid filling top-K with
        # sessions carrying identical evidence facets when complementary sessions
        # are close behind.
        selected: list[tuple[float, int, str, set[str], set[str]]] = []
        remaining = rows[:]
        while remaining and len(selected) < top_k:
            if not selected:
                selected.append(remaining.pop(0))
                continue
            covered = set().union(*(r[3] for r in selected)) if selected else set()
            best_i = 0
            best_score = -1e9
            for i, row in enumerate(remaining[: min(20, len(remaining))]):
                base, neg_pos, sid, facets, words = row
                novelty = len(facets - covered) * 0.18
                redundancy = 0.0
                for _b, _p, _sid, _fac, sel_words in selected:
                    redundancy = max(redundancy, _overlap(list(words), list(sel_words)))
                adjusted = base + novelty - 0.10 * redundancy
                if adjusted > best_score:
                    best_score = adjusted
                    best_i = i
            selected.append(remaining.pop(best_i))
        return [sid for _score, _pos, sid, _facets, _words in selected]

    def rerank_sessions_by_evidence_atoms(
        self,
        query: str,
        bm25_session_order: list[str],
        session_texts: dict[str, str],
        top_k: int = 10,
        candidate_k: int = 40,
    ) -> list[str]:
        """Token-native evidence-atom/facet selector for multi-session queries."""
        return rerank_sessions_by_evidence_atoms(
            query,
            bm25_session_order,
            session_texts,
            top_k=top_k,
            candidate_k=candidate_k,
        )

    def query_two_stage_sessions(
        self,
        query: str,
        top_k: int = 10,
        broad_k: int = 50,
        precise_k: int = 6,
        method: str = "hybrid",
        max_tokens: int = 200_000,
        session_field: str = "session_id",
        broad_filter_field: tuple[str, str] | None = ("kind", "session"),
    ) -> dict:
        """Broad parent/session discovery followed by precise in-session search.

        This is intentionally opt-in.  It is meant for cross-thread/count/list
        questions where the first pass should identify candidate parent sessions
        and the second pass should search only inside those discovered paths.
        """
        import math
        import re

        _WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*")
        _STOPWORDS = {
            "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
            "can", "could", "did", "do", "does", "for", "from", "had", "has",
            "have", "he", "her", "him", "his", "how", "i", "if", "in", "is",
            "it", "me", "my", "of", "on", "or", "our", "she", "so", "that",
            "the", "their", "them", "they", "this", "to", "was", "we", "were",
            "what", "when", "where", "which", "who", "why", "will", "with",
            "would", "you", "your", "should", "help", "about", "anything",
            "something", "list", "count", "many",
        }

        def _words(text: str) -> set[str]:
            return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}

        def _overlap(a: set[str], b: set[str]) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / math.sqrt(len(a) * len(b))

        broad_result = self.query(
            query,
            top_k=broad_k,
            method=method,
            max_tokens=max_tokens,
            filter_field=broad_filter_field,
        )
        broad_order: list[str] = []
        broad_rank: dict[str, int] = {}
        broad_scores: dict[str, float] = {}
        broad_texts: dict[str, list[str]] = {}
        for rank, (chunk, score) in enumerate(zip(broad_result.chunks, broad_result.scores), start=1):
            sid = str(chunk.metadata.get(session_field) or self.metadata.get(chunk.chunk_id).get(session_field) or "")
            if not sid:
                continue
            if sid not in broad_rank:
                broad_rank[sid] = rank
                broad_order.append(sid)
                broad_scores[sid] = score
            else:
                broad_scores[sid] = max(broad_scores[sid], score)
            try:
                txt = self.tokenizer.decode(chunk.tokens.tolist())
            except Exception:
                txt = ""
            if txt:
                broad_texts.setdefault(sid, []).append(txt)

        q_words = _words(query)
        rows: list[dict] = []
        for sid in broad_order:
            precise = self.query(
                query,
                top_k=precise_k,
                method="bm25",
                max_tokens=max_tokens,
                filter_field=(session_field, sid),
            )
            precise_scores = [score for score in precise.scores if score > 0]
            precise_text_parts: list[str] = []
            for chunk in precise.chunks:
                try:
                    txt = self.tokenizer.decode(chunk.tokens.tolist())
                except Exception:
                    txt = ""
                if txt:
                    precise_text_parts.append(txt)
            evidence_text = "\n".join(precise_text_parts or broad_texts.get(sid, []))
            lexical = _overlap(q_words, _words(evidence_text))
            rank_bonus = 1.0 / (60 + broad_rank[sid])
            precise_score = max(precise_scores, default=0.0)
            # Keep broad path discovery dominant, then let in-session precise
            # evidence break ties and demote broad-but-unsubstantiated parents.
            score = 1.00 * rank_bonus + 0.12 * math.log1p(precise_score) + 0.35 * lexical
            rows.append(
                {
                    "session_id": sid,
                    "score": score,
                    "broad_rank": broad_rank[sid],
                    "broad_score": broad_scores.get(sid, 0.0),
                    "precise_score": precise_score,
                    "lexical_overlap": lexical,
                    "precise_hits": len(precise_scores),
                }
            )

        rows.sort(key=lambda r: (-r["score"], r["broad_rank"]))
        session_ids = [str(r["session_id"]) for r in rows[:top_k]]
        return {
            "route": "two_stage_sessions",
            "session_ids": session_ids,
            "details": {
                "broad_k": broad_k,
                "precise_k": precise_k,
                "method": method,
                "candidates": rows[: min(len(rows), max(top_k, 10))],
            },
        }

    def query_auto(
        self,
        query: str,
        top_k: int = 10,
        retrieval_k: int = 50,
        method: str = "hybrid",
        max_tokens: int = 200_000,
        session_field: str = "session_id",
        evidence_atom_rerank: bool = False,
        evidence_certificate_rerank: bool = False,
        typed_rescue: bool = False,
        evidence_certificate_candidate_k: int = 80,
    ) -> dict:
        """Auto-route a query to the best retrieval mode and return results.

        Uses the deterministic query router to select between:
        - episode_score: vague advice / recommendation / planning
        - bm25: specific fact lookup, decisions, current state, open loops
        - preference_rerank: personalized recommendation/preference queries
        - multi_session_rerank: evidence-coverage planning/background queries
        - atom_fusion: preference / constraint / goal queries with strong verbs
        - episode_bm25_fusion: fallback multi-session synthesis fusion

        `evidence_certificate_rerank` enables the productionized v4
        certificate promotion layer. `typed_rescue` additionally enables the
        v5 rescue-only preference/temporal stage; it has no effect unless
        certificate reranking is enabled.

        Returns a dict with:
            route: QueryRoute
            session_ids: list[str]  ranked session IDs
            details: mode-specific metadata
        """
        route = route_query(query)
        session_ids: list[str] = []
        details: dict = {"route": describe_route(route)}

        if route.mode == "episode_score":
            ranked = self.rank_sessions_by_episode_score(query, top_k=top_k, session_field=session_field)
            session_ids = [r["session_id"] for r in ranked if r["score"] > 0]
            details["episode_scores"] = {r["session_id"]: round(r["score"], 4) for r in ranked}

        elif route.mode == "bm25":
            # BM25 retrieval + token-native structural reranking
            result = self.query(
                query, top_k=retrieval_k, method=method, max_tokens=max_tokens,
            )
            # Collect BM25 session order and per-session chunk text
            bm25_order: list[str] = []
            bm25_seen: set[str] = set()
            sess_texts: dict[str, list[str]] = {}
            for chunk in result.chunks:
                sid = str(chunk.metadata.get(session_field) or self.metadata.get(chunk.chunk_id).get(session_field) or "")
                if not sid:
                    continue
                if sid not in bm25_seen:
                    bm25_seen.add(sid)
                    bm25_order.append(sid)
                try:
                    txt = self.tokenizer.decode(chunk.tokens.tolist())
                except Exception:
                    txt = ""
                sess_texts.setdefault(sid, []).append(txt)
            flat_texts = {sid: "\n".join(parts) for sid, parts in sess_texts.items()}
            # Only apply structural reranking when BM25 confidence is low.
            # High BM25 confidence = the top session's chunk scores dominate clearly;
            # in that case trust BM25 order (e.g. specific-fact / LongMemEval queries).
            # Low BM25 confidence = scores are bunched and structural features can break ties.
            bm25_scores = [s for s in result.scores]
            if len(bm25_scores) >= 2 and bm25_scores[0] > 0:
                _confidence_ratio = bm25_scores[0] / (bm25_scores[1] if bm25_scores[1] > 0 else 0.001)
            else:
                _confidence_ratio = 1.0
            if _confidence_ratio >= 2.5:
                # BM25 is decisive — trust it
                session_ids = bm25_order[:top_k]
            else:
                session_ids = self.rerank_sessions_by_structure(
                    query, bm25_order, flat_texts, top_k=top_k, session_field=session_field
                )

        elif route.mode == "preference_rerank":
            preference_query = normalize_memory_query(query)
            pref_groups = self.query_groups(
                preference_query,
                group_by=session_field,
                top_k_groups=max(top_k, min(retrieval_k, 50)),
                retrieval_k=retrieval_k,
                method=method,
                max_tokens=max_tokens,
            )
            p_order = [str(group["value"]) for group in pref_groups if group.get("value")]
            p_seen: set[str] = set(p_order)
            p_texts: dict[str, list[str]] = {}
            atom_result = self.query(
                augment_query_for_memory_atoms(preference_query),
                top_k=retrieval_k,
                method=method,
                max_tokens=max_tokens,
                filter_field=("kind", "memory_atoms"),
            )
            atom_order: list[str] = []
            for chunk in atom_result.chunks:
                sid = str(
                    chunk.metadata.get(session_field)
                    or chunk.metadata.get("source_id")
                    or self.metadata.get(chunk.chunk_id).get(session_field)
                    or self.metadata.get(chunk.chunk_id).get("source_id")
                    or ""
                )
                if not sid:
                    continue
                if sid not in p_seen:
                    p_seen.add(sid)
                    p_order.append(sid)
                if sid not in atom_order:
                    atom_order.append(sid)
            # Preference recommendation often needs a prior taste session that
            # shares few query tokens (e.g. "language" -> "French").  If BM25
            # missed such a session entirely, include all indexed session chunks
            # as low-prior candidates so preference evidence can surface them.
            for chunk in self.store.iter_chunks(level=0):
                meta = chunk.metadata or self.metadata.get(chunk.chunk_id)
                if str(meta.get("kind", "session")) != "session":
                    continue
                sid = str(meta.get(session_field) or "")
                if not sid or sid in p_texts:
                    continue
                try:
                    txt = self.tokenizer.decode(chunk.tokens.tolist())
                except Exception:
                    txt = ""
                p_texts.setdefault(sid, []).append(txt)
            flat_p_texts = {sid: "\n".join(parts) for sid, parts in p_texts.items()}
            pref_ranked = self.rerank_sessions_by_preference_facets(
                preference_query, p_order, flat_p_texts, top_k=top_k
            )
            if top_k >= 10 and p_order:
                session_ids = p_order[:top_k]
                protected = session_ids[: min(5, len(session_ids))]
                if len(session_ids) < top_k:
                    session_ids.extend(
                        sid for sid in atom_order + pref_ranked if sid not in session_ids
                    )
                session_ids = session_ids[:top_k]
                details["preference_protected_token_top_k"] = len(protected)
            else:
                session_ids = pref_ranked
            if atom_order:
                details["preference_atom_candidate_sessions"] = len(atom_order)


        elif route.mode == "multi_session_rerank":
            cov_result = self.query(
                query, top_k=retrieval_k, method=method, max_tokens=max_tokens,
                filter_field=("kind", "session"),
            )
            c_seen: set[str] = set()
            c_order: list[str] = []
            c_texts: dict[str, list[str]] = {}
            for chunk in cov_result.chunks:
                sid = str(chunk.metadata.get(session_field) or self.metadata.get(chunk.chunk_id).get(session_field) or "")
                if not sid:
                    continue
                if sid not in c_seen:
                    c_seen.add(sid)
                    c_order.append(sid)
                try:
                    txt = self.tokenizer.decode(chunk.tokens.tolist())
                except Exception:
                    txt = ""
                c_texts.setdefault(sid, []).append(txt)
            for chunk in self.store.iter_chunks(level=0):
                meta = chunk.metadata or self.metadata.get(chunk.chunk_id)
                if str(meta.get("kind", "session")) != "session":
                    continue
                sid = str(meta.get(session_field) or "")
                if not sid or sid in c_texts:
                    continue
                try:
                    txt = self.tokenizer.decode(chunk.tokens.tolist())
                except Exception:
                    txt = ""
                c_texts.setdefault(sid, []).append(txt)
            flat_c_texts = {sid: "\n".join(parts) for sid, parts in c_texts.items()}
            if evidence_atom_rerank:
                session_ids = self.rerank_sessions_by_evidence_atoms(
                    query, c_order, flat_c_texts, top_k=top_k, candidate_k=retrieval_k
                )
                details["evidence_atom_rerank"] = True
            else:
                session_ids = self.rerank_sessions_by_evidence_coverage(
                    query, c_order, flat_c_texts, top_k=top_k
                )

        elif route.mode == "atom_fusion":
            # BM25 session retrieval + structural reranking (replaces atom-type fusion)
            sess_result = self.query(
                query, top_k=retrieval_k, method=method, max_tokens=max_tokens,
                filter_field=("kind", "session"),
            )
            s_seen: set[str] = set()
            s_order: list[str] = []
            s_texts: dict[str, list[str]] = {}
            for chunk in sess_result.chunks:
                sid = str(chunk.metadata.get(session_field) or self.metadata.get(chunk.chunk_id).get(session_field) or "")
                if not sid:
                    continue
                if sid not in s_seen:
                    s_seen.add(sid)
                    s_order.append(sid)
                try:
                    txt = self.tokenizer.decode(chunk.tokens.tolist())
                except Exception:
                    txt = ""
                s_texts.setdefault(sid, []).append(txt)
            flat_s_texts = {sid: "\n".join(parts) for sid, parts in s_texts.items()}
            session_ids = self.rerank_sessions_by_structure(
                query, s_order, flat_s_texts, top_k=top_k, session_field=session_field
            )

        elif route.mode == "episode_bm25_fusion":
            ep_ranked = self.rank_sessions_by_episode_score(query, top_k=top_k, session_field=session_field)
            ep_sessions = [r["session_id"] for r in ep_ranked if r["score"] > 0]
            bm25_result = self.query(query, top_k=retrieval_k, method=method, max_tokens=max_tokens)
            bm25_sessions: list[str] = []
            b_seen: set[str] = set()
            for chunk in bm25_result.chunks:
                sid = str(chunk.metadata.get(session_field) or self.metadata.get(chunk.chunk_id).get(session_field) or "")
                if sid and sid not in b_seen:
                    b_seen.add(sid)
                    bm25_sessions.append(sid)
            # RRF
            rrf_scores: dict[str, float] = {}
            rrf_best: dict[str, int] = {}
            for ranking in [ep_sessions, bm25_sessions]:
                for rank, sid in enumerate(ranking, 1):
                    rrf_scores[sid] = rrf_scores.get(sid, 0.0) + 1.0 / (60 + rank)
                    rrf_best[sid] = min(rrf_best.get(sid, rank), rank)
            session_ids = sorted(rrf_scores, key=lambda s: (-rrf_scores[s], rrf_best[s]))[:top_k]

        else:
            # Fallback: BM25
            result = self.query(query, top_k=retrieval_k, method=method, max_tokens=max_tokens)
            seen2: set[str] = set()
            for chunk in result.chunks:
                sid = str(chunk.metadata.get(session_field) or self.metadata.get(chunk.chunk_id).get(session_field) or "")
                if sid and sid not in seen2:
                    seen2.add(sid)
                    session_ids.append(str(sid))
                    if len(session_ids) >= top_k:
                        break

        if evidence_certificate_rerank and session_ids:
            cert_result = self.rerank_sessions_by_evidence_certificates(
                query,
                session_ids,
                route_mode=route.mode,
                top_k=top_k,
                retrieval_k=retrieval_k,
                method=method,
                max_tokens=max_tokens,
                session_field=session_field,
                candidate_k=evidence_certificate_candidate_k,
                typed_rescue=typed_rescue,
            )
            session_ids = cert_result["session_ids"]
            if cert_result["certificates"]:
                details["evidence_certificates"] = cert_result["certificates"]
            details["evidence_certificate_rerank"] = {
                "enabled": True,
                "typed_rescue": typed_rescue,
                "candidate_k": evidence_certificate_candidate_k,
            }

        return {"route": route, "session_ids": session_ids, "details": details}

    def rerank_sessions_by_evidence_certificates(
        self,
        query: str,
        current_session_ids: list[str],
        *,
        route_mode: str | None = None,
        top_k: int = 10,
        retrieval_k: int = 50,
        method: str = "hybrid",
        max_tokens: int = 200_000,
        session_field: str = "session_id",
        candidate_k: int = 80,
        typed_rescue: bool = False,
    ) -> dict:
        """Apply auditable evidence certificates to session-level results.

        This is the production hook for the v4/v5 certificate work.  It
        preserves the current top sessions unless a broader candidate has a
        reusable certificate strong enough to justify a tail promotion.
        """
        base_order: list[str] = []
        seen: set[str] = set()
        source_texts: dict[str, list[str]] = {}
        source_dates: dict[str, str] = {}

        def add_session(session_id: str, text: str = "", metadata: dict | None = None) -> None:
            sid = str(session_id)
            if not sid:
                return
            if sid not in seen:
                seen.add(sid)
                base_order.append(sid)
            if text:
                source_texts.setdefault(sid, []).append(text)
            if metadata:
                for field in ("date", "timestamp", "created_at", "updated_at"):
                    value = metadata.get(field)
                    if value and sid not in source_dates:
                        source_dates[sid] = str(value)
                        break

        for sid in current_session_ids:
            add_session(str(sid))

        broad = self.query(
            query,
            top_k=max(retrieval_k, candidate_k, top_k),
            method=method,
            max_tokens=max_tokens,
        )
        for chunk in broad.chunks:
            meta = chunk.metadata or self.metadata.get(chunk.chunk_id)
            sid = str(meta.get(session_field) or meta.get("source_id") or "")
            if not sid:
                continue
            try:
                text = self.tokenizer.decode(chunk.tokens.tolist())
            except Exception:
                text = ""
            add_session(sid, text, meta)

        # Fill missing texts/dates from indexed session chunks. This keeps the
        # post-processor usable when the current route came from episode scores
        # or grouped retrieval rather than direct chunks.
        need_all_sessions = typed_rescue
        wanted = set(base_order[: max(top_k, candidate_k)])
        for chunk in self.store.iter_chunks(level=0):
            meta = chunk.metadata or self.metadata.get(chunk.chunk_id)
            sid = str(meta.get(session_field) or meta.get("source_id") or "")
            if not sid or (not need_all_sessions and sid not in wanted):
                continue
            try:
                text = self.tokenizer.decode(chunk.tokens.tolist())
            except Exception:
                text = ""
            add_session(sid, text, meta)

        flat_texts = {sid: "\n".join(parts) for sid, parts in source_texts.items()}
        result = rerank_with_optional_typed_rescue(
            query,
            flat_texts,
            base_order,
            top_k,
            candidate_k=candidate_k,
            source_dates=source_dates,
            route_mode=route_mode,
            typed_rescue=typed_rescue,
        )
        return {
            "session_ids": result.session_ids,
            "certificates": result.traces(),
        }

    def query_groups(
        self,
        query: str,
        group_by: str,
        top_k_groups: int = 10,
        retrieval_k: int = 50,
        method: str = "hybrid",
        max_tokens: int = 200_000,
        max_score_weight: float = 1.0,
        sum_score_weight: float = 0.10,
        count_weight: float = 0.01,
        relationship_boost: float = 1.0,
        filters: list[MetadataPredicate | dict] | dict | None = None,
        filter_mode: str = "and",
        min_filter_matches: int = 1,
        filter_pushdown_threshold: float = 0.50,
    ) -> list[dict]:
        """Retrieve chunks, then rank grouped evidence by metadata field.

        This is useful when the natural unit of evidence is larger than a
        chunk: sessions, documents, emails, tickets, etc. It prevents a single
        high-scoring chunk from hiding the fact that several relevant chunks
        belong to the same source, and it returns source-level ranks for
        benchmarks such as LongMemEval.
        """
        result = self.query(
            query,
            top_k=retrieval_k,
            method=method,
            max_tokens=max_tokens,
            relationship_boost=relationship_boost,
            filters=filters,
            filter_mode=filter_mode,
            min_filter_matches=min_filter_matches,
            filter_pushdown_threshold=filter_pushdown_threshold,
        )
        groups: dict[str, dict] = {}
        for rank, (chunk, score) in enumerate(zip(result.chunks, result.scores), start=1):
            value = chunk.metadata.get(group_by) or self.metadata.get(chunk.chunk_id).get(group_by)
            if not value:
                continue
            group = groups.setdefault(
                str(value),
                {
                    "value": str(value),
                    "score": 0.0,
                    "max_score": float(score),
                    "sum_score": 0.0,
                    "count": 0,
                    "best_chunk_rank": rank,
                    "chunk_ids": [],
                },
            )
            group["max_score"] = max(group["max_score"], float(score))
            group["sum_score"] += float(score)
            group["count"] += 1
            group["best_chunk_rank"] = min(group["best_chunk_rank"], rank)
            group["chunk_ids"].append(chunk.chunk_id)

        for group in groups.values():
            group["score"] = (
                max_score_weight * group["max_score"]
                + sum_score_weight * group["sum_score"]
                + count_weight * group["count"]
            )

        ranked = sorted(groups.values(), key=lambda g: (-g["score"], g["best_chunk_rank"]))
        return ranked[:top_k_groups]

    def ingest_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 64,
        metadata: dict | None = None,
        update_indexes: bool = True,
    ) -> list[Chunk]:
        """
        Ingest text into the knowledge base.
        """
        token_chunks = self.tokenizer.chunk_text(text, chunk_size, overlap)
        return self.ingest_token_chunks(
            token_chunks,
            metadata=metadata,
            update_indexes=update_indexes,
        )

    def ingest_conversation(
        self,
        session_id: str,
        date: str,
        turns: list[dict],
        chunk_size: int = 512,
        overlap: int = 64,
        metadata: dict | None = None,
        update_indexes: bool = True,
        include_answer_marker: bool = False,
        include_parent: bool = False,
    ) -> list[Chunk]:
        """Ingest a conversation/session using turn-aware chunk boundaries.

        When ``include_parent`` is true, also index one full-session parent
        chunk. Child turn-range chunks provide precision while the parent node
        preserves whole-session lexical evidence for parent/child ranking.
        """
        from contextfit.extractors.conversation import chunk_conversation, conversation_to_text

        text_chunks = chunk_conversation(
            session_id,
            date,
            turns,
            chunk_size=chunk_size,
            overlap=overlap,
            base_metadata=metadata,
            include_answer_marker=include_answer_marker,
        )

        if include_parent:
            parent_meta = dict(metadata or {})
            parent_meta.setdefault("session_id", session_id)
            parent_meta.setdefault("date", date)
            parent_meta.setdefault("kind", "session")
            parent_meta.update({
                "chunk_type": "conversation_session_parent",
                "parent_id": session_id,
                "child_chunk_count": str(len(text_chunks)),
            })
            text_chunks.append({
                "text": conversation_to_text(
                    session_id,
                    date,
                    turns,
                    include_answer_marker=include_answer_marker,
                ),
                "metadata": parent_meta,
            })

        chunks: list[Chunk] = []
        for item in text_chunks:
            chunks.extend(
                self.ingest_token_chunks(
                    [self.tokenizer.encode(item["text"])],
                    metadata=item.get("metadata") or metadata,
                    update_indexes=update_indexes,
                )
            )
        return chunks

    def ingest_file(
        self,
        path: Path | str,
        chunk_size: int = 512,
        overlap: int = 64,
        update_indexes: bool = True,
    ) -> list[Chunk]:
        """Ingest a text file, preserving known document structure when possible.

        File-ingested chunks need a stable session_id so session-oriented
        retrieval paths can group them by source. Conversation ingestion already
        sets this; file ingestion fills it from the source path unless the
        extractor supplied a more specific value.
        """
        from contextfit.extractors import auto as auto_extractor
        from contextfit.extractors import calendar as calendar_extractor
        from contextfit.extractors import code as code_extractor
        from contextfit.extractors import document as document_extractor
        from contextfit.extractors import email as email_extractor
        from contextfit.extractors import smd as smd_extractor
        from contextfit.extractors import structured as structured_extractor
        from contextfit.extractors import tmd as tmd_extractor

        path = Path(path)
        text = path.read_text()
        suffix = path.suffix.lower()
        default_session_id = path.as_posix()

        if suffix == ".tmd":
            text_chunks = tmd_extractor.chunk_tmd(path, text, chunk_size=chunk_size, overlap=overlap)
        elif suffix == ".smd":
            text_chunks = smd_extractor.chunk_smd(path, text, chunk_size=chunk_size, overlap=overlap)
        elif suffix == ".md":
            text_chunks = document_extractor.chunk_markdown(path, text, chunk_size=chunk_size, overlap=overlap)
        elif suffix == ".txt":
            text_chunks = document_extractor.chunk_text(path, text, chunk_size=chunk_size, overlap=overlap)
        elif suffix == ".json":
            text_chunks = structured_extractor.chunk_json(path, text, chunk_size=chunk_size, overlap=overlap)
        elif suffix == ".jsonl":
            text_chunks = structured_extractor.chunk_jsonl(path, text, chunk_size=chunk_size, overlap=overlap)
        elif suffix == ".csv":
            text_chunks = structured_extractor.chunk_delimited(path, text, chunk_size=chunk_size, overlap=overlap, delimiter=",")
        elif suffix == ".tsv":
            text_chunks = structured_extractor.chunk_delimited(path, text, chunk_size=chunk_size, overlap=overlap, delimiter="\t")
        elif suffix == ".eml":
            text_chunks = email_extractor.chunk_email(path, text, chunk_size=chunk_size, overlap=overlap)
        elif suffix == ".ics":
            text_chunks = calendar_extractor.chunk_ics(path, text, chunk_size=chunk_size, overlap=overlap)
        elif code_extractor.is_code_path(path):
            text_chunks = code_extractor.chunk_code(path, text, chunk_size=chunk_size, overlap=overlap)
        else:
            meta = auto_extractor.extract(path, text)
            text_chunks = [{"text": self.tokenizer.decode(chunk), "metadata": meta} for chunk in self.tokenizer.chunk_text(text, chunk_size, overlap)]

        chunks: list[Chunk] = []
        for item in text_chunks:
            tokens = self.tokenizer.encode(item["text"])
            meta = dict(item.get("metadata") or auto_extractor.extract(path, item["text"]))
            meta.setdefault("session_id", default_session_id)
            chunks.extend(self.ingest_token_chunks(
                [tokens],
                metadata=meta,
                update_indexes=update_indexes,
            ))
        return chunks
    
    def rebuild_indexes(self, include_semantic_ids: bool = True) -> None:
        """Rebuild inverted, BM25, LSH, and optional semantic IDs from stored chunks."""
        self.inverted = InvertedIndex()
        self.bm25 = BM25Scorer(self.inverted)
        self.hasher = MinHasher()
        self.lsh = LSHIndex()
        if include_semantic_ids:
            self.sid_index = SemanticIDIndex()
        else:
            self.sid_index = None

        for chunk in self.store.iter_chunks(level=0):
            self.inverted.add_chunk(chunk.chunk_id, chunk.tokens)
            sig = self.hasher.hash(chunk.chunk_id, chunk.tokens)
            self.lsh.add(sig)
            if self.sid_index is not None:
                self.sid_index.assign_from_signature(chunk.chunk_id, sig)

        self.bm25.clear_cache()
        if self.sid_index is not None:
            self.sid_generator = SIDGenerator(self.sid_index, self.bm25, self.hasher, self.lsh)
        else:
            self.sid_generator = None

    def build_hierarchy(self, max_levels: int = 3) -> None:
        """Build the multi-level hierarchy."""
        self.hierarchy = HierarchyBuilder(
            chunk_store=self.store,
            hasher=self.hasher,
            lsh=self.lsh,
        )
        self.hierarchy.build(max_levels=max_levels)

    def build_semantic_ids(
        self,
        depth: int = 4,
        branching_factor: int = 256,
    ) -> SemanticIDIndex:
        """
        Rebuild Semantic IDs for all level-0 chunks.

        This is useful after bulk ingestion or when tuning SID depth/branching.
        """
        self.sid_index = SemanticIDIndex(depth=depth, branching_factor=branching_factor)
        for chunk in self.store.iter_chunks(level=0):
            sig = self.lsh.get_signature(chunk.chunk_id)
            if sig is None:
                sig = self.hasher.hash(chunk.chunk_id, chunk.tokens)
                self.lsh.add(sig)
            self.sid_index.assign_from_signature(chunk.chunk_id, sig)
        self.sid_generator = SIDGenerator(self.sid_index, self.bm25, self.hasher, self.lsh)
        return self.sid_index

    def train_learned_sid_generator(self) -> LearnedSIDGenerator:
        """Train the lightweight token→SID generator from stored chunks."""
        if self.sid_index is None:
            self.build_semantic_ids()

        assert self.sid_index is not None
        chunk_tokens = {
            chunk.chunk_id: chunk.tokens
            for chunk in self.store.iter_chunks(level=0)
        }
        self.learned_sid_generator = LearnedSIDGenerator(self.sid_index).fit(chunk_tokens)
        return self.learned_sid_generator
    
    def query(
        self,
        query: str | list[int] | np.ndarray,
        top_k: int = 5,
        method: str = "hybrid",
        use_hierarchy: bool = True,
        expand_graph: bool = True,
        max_tokens: int = 4096,
        # --- new: metadata pre-filter ---
        filter_domain: str | None = None,
        filter_field: tuple[str, str] | None = None,
        filters: list[MetadataPredicate | dict] | dict | None = None,
        filter_mode: str = "and",
        min_filter_matches: int = 0,
        filter_pushdown_threshold: float = 0.50,
        query_spec: QuerySpec | dict | None = None,
        # --- new: hybrid scoring ---
        metadata_boost: float = 1.0,
        relationship_boost: float = 1.0,
        # --- new: query expansion ---
        expand_query: bool = True,
        token_rerank: bool = False,
    ) -> RetrievalResult:
        """
        Query the knowledge base.

        Args:
            query: Query text or token IDs
            top_k: Number of chunks to retrieve
            method: "exact", "bm25", "sid", "graph", "hierarchy", "hybrid",
                    or "hybrid_rrf" for rank-fusion ablations
            use_hierarchy: Navigate hierarchy if available
            expand_graph: Expand results via graph neighbors
            max_tokens: Maximum total tokens in result
            filter_domain: Pre-filter to chunks from this sender domain
                           e.g. filter_domain="acme.example"
            filter_field: Pre-filter tuple (field, value)
                          e.g. filter_field=("subject", "Acme")
            filters: Structured metadata predicates from an agent/MCP query
                     spec. Supported operators include exact, contains, in,
                     gt/gte/lt/lte, after/before, and exists.
            filter_mode: Combine structured filters with "and" or "or".
            min_filter_matches: If structured filters match fewer chunks than
                                this threshold, broaden by ignoring those
                                filters. This keeps agent/planner mistakes
                                from silently over-filtering evidence.
            filter_pushdown_threshold: If filters match more than this fraction
                                       of indexed metadata, do normal candidate
                                       generation and post-filter instead of
                                       scoring a very broad explicit candidate
                                       list. Set to 1.0 to always push down.
            query_spec: Optional structured request containing query, filters,
                        filter_mode, and min_filter_matches.
            metadata_boost: Score multiplier when query terms appear in
                            metadata fields. Default 1.0 because broad fields
                            like dates/session ids can otherwise swamp text
                            relevance; callers can opt in for email headers.
            relationship_boost: Score multiplier for chunks connected by
                                derived entity/relationship backlinks. Default
                                1.0 keeps existing ranking behavior unchanged.
            expand_query: Expand query with synonyms and stems before BM25.
            token_rerank: Rerank candidate chunks using token-native phrase,
                          local-window, preference, and temporal structure.
        """
        if query_spec is not None:
            spec = QuerySpec.from_obj(query_spec)
            query = spec.query
            filters = spec.filters
            filter_mode = spec.filter_mode
            min_filter_matches = spec.min_filter_matches
            filter_pushdown_threshold = spec.filter_pushdown_threshold

        use_hybrid_rrf = method == "hybrid_rrf"
        is_hybrid = method in ("hybrid", "hybrid_rrf")

        # Tokenize query if needed
        query_text: str = query if isinstance(query, str) else ""
        if isinstance(query, str):
            query_tokens = self.tokenizer.encode(query)
        elif isinstance(query, list):
            query_tokens = np.array(query, dtype=np.uint32)
        else:
            query_tokens = query

        # --- Query expansion ---
        if expand_query and query_text:
            extra_tokens = self.expander.expand_text(query_text)
            if len(extra_tokens) > 0:
                query_tokens_expanded = np.concatenate([query_tokens, extra_tokens])
            else:
                query_tokens_expanded = query_tokens

            # Semantic (embedding-based) expansion — zero query-time API cost
            if self.semantic_expander is not None:
                sem_pairs = self.semantic_expander.expand(query_tokens.tolist())
                if sem_pairs:
                    sem_tokens = np.array([t for t, _ in sem_pairs], dtype=np.uint32)
                    query_tokens_expanded = np.concatenate([query_tokens_expanded, sem_tokens])
        else:
            query_tokens_expanded = query_tokens

        # --- Metadata pre-filter: compute allowed chunk_id set ---
        filter_set: set[int] | None = None
        post_filter_set: set[int] | None = None
        filter_trace: dict | None = None
        if filter_domain:
            filter_set = self.metadata.filter_sender_domain(filter_domain)
        if filter_field:
            field_matches = self.metadata.filter(filter_field[0], filter_field[1])
            filter_set = field_matches if filter_set is None else filter_set & field_matches
        if filters:
            filter_items = filters
            if isinstance(filter_items, dict):
                filter_items = QuerySpec.from_obj(
                    {"query": query_text or "__query__", "filters": filter_items}
                ).filters
            predicates = [MetadataPredicate.from_obj(item) for item in filter_items]
            structured_matches = self.metadata.filter_predicates(
                predicates,
                mode=filter_mode,
            )
            filtered_count = len(structured_matches)
            broadened = min_filter_matches > 0 and filtered_count < min_filter_matches
            filter_trace = {
                "mode": filter_mode,
                "predicates": [
                    {
                        "field": p.field,
                        "op": p.op,
                        "value": p.value,
                        "values": p.values,
                    }
                    for p in predicates
                ],
                "matched_chunks": filtered_count,
                "min_filter_matches": min_filter_matches,
                "broadened": broadened,
            }
            if not broadened:
                filter_set = (
                    structured_matches
                    if filter_set is None
                    else filter_set & structured_matches
                )

        if filter_set is not None and len(self.metadata) > 0:
            filter_ratio = len(filter_set) / len(self.metadata)
            pushdown = filter_ratio <= filter_pushdown_threshold
            if filter_trace is not None:
                filter_trace["filter_ratio"] = round(filter_ratio, 6)
                filter_trace["pushdown"] = pushdown
                filter_trace["pushdown_threshold"] = filter_pushdown_threshold
            if not pushdown:
                post_filter_set = filter_set
                filter_set = None

        candidates: list[tuple[int, float]] = []
        sid_predictions: list[SIDPrediction] | None = None
        exact_set: set[int] = set()
        source_ranks: dict[str, list[int]] = {}

        # Exact/token search.  This gives ContextFit grep-like behavior while
        # staying in token space: exact phrase hits rank highest, then chunks
        # containing every query token.  Hybrid mode includes these candidates
        # before semantic/graph expansion so literal matches are not drowned out.
        if method == "exact" or is_hybrid:
            exact_results: list[tuple[int, float]] = []
            query_token_list = query_tokens.tolist()
            if query_token_list:
                phrase_hits: set[int] = set()
                phrase_variants = [query_token_list]
                # BPE tokenizers often encode a phrase differently after a
                # leading space (e.g. "Brooks" vs " Brooks").  Try both so
                # phrase search behaves like literal text search in practice.
                if query_text and not query_text.startswith(" "):
                    spaced = self.tokenizer.encode(" " + query_text).tolist()
                    if spaced != query_token_list:
                        phrase_variants.append(spaced)
                for variant in phrase_variants:
                    phrase_hits |= self.inverted.search_phrase(variant)
                if filter_set is not None:
                    phrase_hits &= filter_set
                exact_results.extend((cid, 1_000_000.0) for cid in phrase_hits)

                token_hits = self.inverted.search(query_token_list, mode="and")
                if filter_set is not None:
                    token_hits &= filter_set
                # Use BM25 as a tie-breaker but keep exact-token results above
                # non-exact semantic results.
                exact_results.extend(
                    (cid, 500_000.0 + self.bm25.score_chunk(query_token_list, cid))
                    for cid in token_hits - phrase_hits
                )

            candidates.extend(exact_results[:top_k * 4])
            if use_hybrid_rrf:
                exact_set = {cid for cid, _score in exact_results}

        # BM25 search (with expanded tokens)
        if method == "bm25" or is_hybrid:
            if filter_set is not None:
                bm25_results = self.bm25.top_k(
                    query_tokens_expanded,
                    k=top_k * 4,
                    chunk_ids=list(filter_set),
                )
            else:
                bm25_results = self.bm25.top_k(query_tokens_expanded, k=top_k * 4)
            candidates.extend(bm25_results[:top_k * 2])
            if use_hybrid_rrf:
                source_ranks["bm25"] = [cid for cid, _score in bm25_results]

        # Semantic ID prefix search
        if self.sid_generator is not None and (method == "sid" or is_hybrid):
            if self.learned_sid_generator is not None and self.learned_sid_generator.trained_chunks:
                sid_predictions = self.learned_sid_generator.predict(
                    query_tokens,
                    top_k=max(top_k, 5),
                    beam_width=max(16, top_k * 4),
                )
                sid_results = self.learned_sid_generator.retrieve(
                    query_tokens,
                    top_k=top_k * 2,
                    prediction_k=max(top_k, 5),
                    beam_width=max(16, top_k * 4),
                )
            else:
                sid_predictions = self.sid_generator.predict(
                    query_tokens,
                    top_k=max(top_k, 5),
                    candidate_k=top_k * 8,
                )
                sid_results = self.sid_generator.retrieve(
                    query_tokens,
                    top_k=top_k * 2,
                    prediction_k=max(top_k, 5),
                    candidate_k=top_k * 8,
                )
            if filter_set is not None:
                sid_results = [(cid, s) for cid, s in sid_results if cid in filter_set]
            candidates.extend(sid_results)
            if use_hybrid_rrf:
                source_ranks["sid"] = [cid for cid, _score in sid_results]

            # Natural-language queries may not produce a known SID prefix yet.
            # Until a trained SID generator exists, keep `method="sid"` usable by
            # falling back to lexical candidates while still returning their SIDs.
            if method == "sid" and not candidates:
                candidates.extend(
                    self.bm25.top_k(
                        query_tokens,
                        k=top_k * 2,
                        chunk_ids=list(filter_set) if filter_set is not None else None,
                    )
                )
        
        # Hierarchy navigation
        if use_hierarchy and self.hierarchy and (method == "hierarchy" or is_hybrid):
            hier_chunks = self.hierarchy.navigate(
                query_tokens,
                top_k=top_k,
            )
            for chunk in hier_chunks:
                if filter_set is not None and chunk.chunk_id not in filter_set:
                    continue
                # Score with BM25
                score = self.bm25.score_chunk(query_tokens.tolist(), chunk.chunk_id)
                candidates.append((chunk.chunk_id, score))
            if use_hybrid_rrf:
                source_ranks["hierarchy"] = [chunk.chunk_id for chunk in hier_chunks]
        
        # Graph expansion
        if expand_graph and (method == "graph" or is_hybrid):
            # Expand top candidates via LSH
            chunk_ids_to_expand = [cid for cid, _ in candidates[:top_k]]
            graph_neighbors: list[int] = []
            for cid in chunk_ids_to_expand:
                sig = self.lsh.get_signature(cid)
                if sig:
                    neighbors = self.lsh.query(sig)
                    for neighbor_id in neighbors:
                        if filter_set is not None and neighbor_id not in filter_set:
                            continue
                        score = self.bm25.score_chunk(query_tokens.tolist(), neighbor_id)
                        candidates.append((neighbor_id, score))
                        graph_neighbors.append(neighbor_id)
            if use_hybrid_rrf:
                seen_graph: set[int] = set()
                source_ranks["graph"] = [
                    cid for cid in graph_neighbors if not (cid in seen_graph or seen_graph.add(cid))
                ]
        
        # --- Metadata hybrid boost ---
        if metadata_boost > 1.0 and len(self.metadata) > 0 and query_text:
            query_terms = query_text.lower().split()
            candidates = self.metadata.boost_scores(
                candidates, query_terms, multiplier=metadata_boost
            )

        # --- Relationship/backlink boost ---
        if relationship_boost > 1.0 and len(self.relationships) > 0 and query_text:
            candidates = self.relationships.boost_scores(
                candidates,
                query_text,
                multiplier=relationship_boost,
                append=True,
            )

        if post_filter_set is not None:
            candidates = [(cid, score) for cid, score in candidates if cid in post_filter_set]

        # Deduplicate and sort
        seen = set()
        unique_candidates = []
        for cid, score in candidates:
            if cid not in seen:
                seen.add(cid)
                unique_candidates.append((cid, score))

        use_rrf = (
            use_hybrid_rrf
            and bool(source_ranks)
            and metadata_boost == 1.0
            and relationship_boost == 1.0
        )
        if use_rrf:
            rrf_k = 60
            rank_lookups = {
                source: {cid: rank for rank, cid in enumerate(ids, start=1)}
                for source, ids in source_ranks.items()
                if ids
            }

            def rrf_score(cid: int) -> float:
                return 1000.0 * sum(
                    1.0 / (rrf_k + lookup[cid])
                    for lookup in rank_lookups.values()
                    if cid in lookup
                )

            unique_candidates = [
                (cid, raw_score if cid in exact_set else rrf_score(cid))
                for cid, raw_score in unique_candidates
            ]
        
        unique_candidates.sort(key=lambda x: -x[1])

        # Take top-k and collect chunks. With token reranking, first inspect a
        # wider candidate pool, then let token/phrase/window structure choose
        # the final order.
        candidate_limit = max(top_k * 8, 50) if token_rerank else top_k
        candidate_chunks = []
        candidate_scores = []
        for cid, score in unique_candidates[:candidate_limit]:
            chunk = self.store.get(cid)
            if chunk:
                candidate_chunks.append(chunk)
                candidate_scores.append(score)

        rerank_traces = None
        if token_rerank and query_text:
            reranked = self.token_reranker.rerank(query_text, candidate_chunks, candidate_scores)
            candidate_chunks = [chunk for chunk, _score, _trace in reranked]
            candidate_scores = [score for _chunk, score, _trace in reranked]
            rerank_traces = [trace for _chunk, _score, trace in reranked]

        chunks = []
        scores = []
        total_tokens = 0
        for chunk, score in zip(candidate_chunks, candidate_scores, strict=False):
            if len(chunks) >= top_k:
                break
            if total_tokens + chunk.token_count <= max_tokens:
                chunks.append(chunk)
                scores.append(score)
                total_tokens += chunk.token_count
        
        # Assemble input_ids
        input_ids = self._assemble_input_ids(chunks)
        semantic_ids = [
            sid.tokens
            for chunk in chunks
            if self.sid_index is not None and (sid := self.sid_index.get(chunk.chunk_id))
        ]
        
        return RetrievalResult(
            chunks=chunks,
            input_ids=input_ids,
            query_tokens=query_tokens,
            scores=scores,
            method=method,
            semantic_ids=semantic_ids or None,
            sid_predictions=sid_predictions,
            rerank_traces=rerank_traces[: len(chunks)] if rerank_traces else None,
            filter_trace=filter_trace,
        )
    
    def _assemble_input_ids(self, chunks: list[Chunk]) -> np.ndarray:
        """Combine chunk tokens into single input array."""
        if not chunks:
            return np.array([], dtype=np.uint32)
        
        # Simple concatenation with separator tokens
        # In production, might add context markers
        all_tokens = []
        for chunk in chunks:
            all_tokens.extend(chunk.tokens.tolist())
        
        return np.array(all_tokens, dtype=np.uint32)
    
    def save(self, path: Path | str) -> None:
        """Save indexes to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.store.flush(force=True)
        self.inverted.save(path / "inverted")
        if self.sid_index is not None:
            self.sid_index.save(path / "sid")
        if self.learned_sid_generator is not None:
            self.learned_sid_generator.save(path / "sid")
        self.lsh.save(path / "lsh")
        if self.metadata and len(self.metadata) > 0:
            self.metadata.save(path / "metadata")
        if self.relationships and len(self.relationships) > 0:
            self.relationships.save(path / "relationships")
    
    def load_semantic_expander(self, path: Path | str) -> "RetrievalEngine":
        """Load a pre-built SemanticExpander from <path>/expanders/."""
        exp_path = Path(path) / "expanders"
        if exp_path.exists():
            self.semantic_expander = SemanticExpander.load(exp_path)
        return self

    @classmethod
    def load(cls, path: Path | str, tokenizer_name: str = "cl100k_base") -> "RetrievalEngine":
        """Load engine from disk."""
        path = Path(path)
        
        tokenizer = Tokenizer.load(tokenizer_name)
        chunk_store = ChunkStore(path / "chunks")
        inverted_path = path / "inverted"
        inverted = InvertedIndex.load(inverted_path) if inverted_path.exists() else InvertedIndex()
        bm25 = BM25Scorer(inverted)
        hasher = MinHasher()

        # Load persisted LSH if available (avoids ~11 min rebuild over 84k chunks)
        lsh_path = path / "lsh"
        if (lsh_path / "lsh_signatures.bin").exists():
            print("Loading LSH from disk...", file=sys.stderr, flush=True)
            lsh = LSHIndex.load(lsh_path)
        else:
            print("Rebuilding LSH from chunks (first load or LSH not yet saved)...", file=sys.stderr, flush=True)
            lsh = LSHIndex()
            for chunk in chunk_store.iter_chunks(level=0):
                sig = hasher.hash(chunk.chunk_id, chunk.tokens)
                lsh.add(sig)

        sid_index = None
        if (path / "sid" / "semantic_ids.json").exists():
            sid_index = SemanticIDIndex.load(path / "sid")
        learned_sid_generator = None

        if sid_index is None:
            sid_index = SemanticIDIndex()
            for chunk in chunk_store.iter_chunks(level=0):
                sig = lsh.get_signature(chunk.chunk_id)
                if sig:
                    sid_index.assign_from_signature(chunk.chunk_id, sig)

        if (path / "sid" / "learned_sid_generator.json").exists():
            learned_sid_generator = LearnedSIDGenerator.load(path / "sid", sid_index)

        metadata = None
        if MetadataIndex.exists(path / "metadata"):
            metadata = MetadataIndex.load(path / "metadata")

        relationships = None
        if RelationshipIndex.exists(path / "relationships"):
            relationships = RelationshipIndex.load(path / "relationships")

        engine = cls(
            tokenizer=tokenizer,
            chunk_store=chunk_store,
            inverted_index=inverted,
            bm25=bm25,
            hasher=hasher,
            lsh=lsh,
            sid_index=sid_index,
            learned_sid_generator=learned_sid_generator,
            metadata_index=metadata,
            relationship_index=relationships,
        )
        engine.load_semantic_expander(path)
        return engine
    
    def aggregate(
        self,
        query: str,
        group_by: str,
        top_k: int = 20,
        filter_domain: str | None = None,
        filter_field: tuple[str, str] | None = None,
        exclude_values: list[str] | None = None,
        include_meta_fields: list[str] | None = None,
        min_count: int = 1,
    ) -> list[dict]:
        """
        Generic aggregation: "who/what/where did I … about X?"

        Works for any data type and any metadata field:

            # Emails — who sent messages about a topic
            engine.aggregate("Project Orion", group_by="from_email")

            # Documents — which authors wrote about a topic
            engine.aggregate("revenue Q4", group_by="author")

            # Notes — which sections/tags cover a topic
            engine.aggregate("machine learning", group_by="section")
            engine.aggregate("bug fix", group_by="project")

        Args:
            query:               Natural language query to scope the search.
            group_by:            Metadata field to group results by.
            top_k:               Max groups to return.
            filter_domain:       Pre-filter chunks by from_domain.
            filter_field:        Pre-filter chunks by (field, value).
            exclude_values:      Group values to skip (e.g. internal senders).
            include_meta_fields: Extra metadata fields to include per group.
            min_count:           Minimum chunk count to include a group.

        Returns:
            List of {value, count, chunk_ids, …extra_fields} sorted by count.
        """
        if len(self.metadata) == 0:
            return []

        # Build candidate set via pre-filter or BM25
        if filter_domain:
            candidate_ids = self.metadata.filter_domain(filter_domain)
        elif filter_field:
            candidate_ids = self.metadata.filter(filter_field[0], filter_field[1])
        else:
            result = self.query(
                query, top_k=top_k * 4, method="bm25",
                expand_query=True, metadata_boost=1.0,
            )
            candidate_ids = {c.chunk_id for c in result.chunks}

        # If we also have a query, intersect with BM25 hits for relevance
        if query and (filter_domain or filter_field) and candidate_ids:
            qtok = self.tokenizer.encode(query)
            extra = self.expander.expand_text(query)
            if len(extra) > 0:
                qtok = np.concatenate([qtok, extra])
            bm25_hits = {cid for cid, _ in self.bm25.top_k(qtok, k=top_k * 8)}
            refined = candidate_ids & bm25_hits
            candidate_ids = refined if refined else candidate_ids

        return self.metadata.aggregate(
            candidate_ids,
            group_by=group_by,
            top_k=top_k,
            min_count=min_count,
            exclude_values=exclude_values,
            include_meta_fields=include_meta_fields,
        )

    def stats(self) -> dict:
        """Return engine statistics."""
        return {
            "chunks": len(self.store),
            "storage": self.store.stats(),
            "index": self.inverted.stats(),
            "lsh_entries": len(self.lsh),
            "hierarchy_levels": self.hierarchy.num_levels if self.hierarchy else 0,
            "semantic_ids": self.sid_index.stats() if self.sid_index is not None else None,
            "sid_generator": self.sid_generator.stats() if self.sid_generator is not None else None,
            "learned_sid_generator": (
                self.learned_sid_generator.stats()
                if self.learned_sid_generator is not None
                else None
            ),
            "metadata": {"chunks": len(self.metadata)} if self.metadata else None,
        }
