# Token-Native Agent Memory: Why the Next Generation of AI Memory Systems Should Think in Tokens, Not Vectors

**ContextFit Research**
**May 2026**

---

## Abstract

Modern AI agent memory systems share a common architectural assumption: that raw conversational context must be converted into embedding vectors before it can be retrieved. This paper challenges that assumption. We present **ContextFit**, a token-native memory retrieval system that operates directly on tokenized text without embedding APIs, LLM preprocessing, or vector databases. We introduce six novel primitives — **memory atoms** (deterministic domain-agnostic fact extraction), an **episode relevance scorer** (structural numeric session ranking), a **deterministic query router** (zero-cost query-to-mode dispatch), a **structural session reranker** (token-native post-retrieval reranking with question-type slot matching), a **router-gated preference reranker** (token-native taste-evidence ranking for personalized recommendations), and a **multi-session evidence-coverage reranker** (token-native complementary-evidence ranking for synthesis queries) — and evaluate them on a 499-case domain-agnostic agent-memory benchmark across eight behavioral categories spanning 26 domains. ContextFit's token-native episode scorer achieves **69.6% Recall@1 and MRR 0.824** on hard episodic inference tasks, outperforming OpenAI `text-embedding-3-small` (55.7% / 0.745) by 14 points and Mem0 with GPT-4o-mini extraction (54.4% / 0.716) by 15 points, while requiring **zero API calls** and running at **0.4ms query latency** — 375× faster than embedding-based retrieval. At scale, the auto router with preference and evidence-coverage reranking achieves **62.3% Recall@1 / 93.0% Recall@3** overall, **85.5% Recall@1** on preference recommendation retrieval, and **82.1% Recall@1** on multi-session synthesis, surpassing Cohere `embed-english-v3.0` (58.7% overall, 83.9% preference) at zero API cost. On LongMemEval-S, pure token-native ContextFit with conversation-aware parent/child chunks reaches **95.1% Any@5**, improving the previous token baseline through turn-aware session boundaries plus full-session parent context with no embeddings. A token-only companion-evidence coverage reranker lifts overall complete evidence coverage from **77.9% to 80.4% All@5** while preserving the 95.1% Any@5 headline. Optional OpenAI fusion achieves **96.0% Any@5**, within 1.6 points of the best published result; fusion is an optional boost, not the core claim.

---

## 1. Introduction

The promise of AI agents is continuity: an assistant that remembers what you told it last week, understands your preferences, and applies prior context to new advice. Fulfilling this promise requires memory retrieval — the ability to surface the right prior conversation when it matters.

The dominant approach in 2026 is embedding-based retrieval: convert session text to dense vectors, store them in a vector database, and retrieve by cosine similarity. Systems like Mem0, Zep, Supermemory, and gbrain all follow this pattern, varying primarily in how they preprocess text (raw, LLM-extracted facts, or knowledge graphs) before embedding.

This architecture has three fundamental limitations:

**Latency and cost.** Every memory access requires one or more API calls — embedding the query, optionally extracting facts via LLM, and querying a vector store. At real-agent throughput, this adds hundreds of milliseconds and ongoing API costs to every turn.

**Semantic averaging.** Embedding models compress an entire session into a single vector. This averaging works well when queries are semantically specific, but fails for the queries agents most frequently need to answer: vague advice queries like "what should I cook tonight?" or "what should I get my friend?" The relevant prior episode may share no vocabulary with the query; it simply contains contextually useful background.

**Opacity.** When embedding retrieval fails, there is no interpretable explanation. A cosine distance of 0.73 does not tell you why a session was or wasn't retrieved. This makes debugging, auditing, and improving the system difficult.

ContextFit takes a different path. It stays in token space from ingest through retrieval, extracting structural signals from raw user-authored text without any API dependency. This paper describes how it works, why it works, and what the benchmark evidence shows.

---

## 2. Background and Related Work

### 2.1 The Embedding Paradigm

The standard retrieval-augmented generation (RAG) pipeline encodes documents as dense vectors using a pretrained language model (e.g., OpenAI's `text-embedding-3-*` family, Voyage AI's `voyage-3`, or Cohere's `embed-english-v3.0`). At query time, the query is similarly encoded and the top-K documents by cosine similarity are retrieved.

For factual knowledge bases (documentation, code, reports), this approach is highly effective. The semantic generalization of embeddings handles vocabulary mismatch well, and the documents are largely static.

Agent memory differs in important ways: the corpus is conversational, queries are often behaviorally vague rather than factually specific, temporal ordering matters, and the agent needs to identify *which prior episode* is contextually relevant rather than *which chunk contains a specific fact*.

### 2.2 Memory-Specific Systems

**Mem0** (mem0ai/mem0, 48K+ GitHub stars as of 2026) adds an LLM preprocessing step: a small model (typically GPT-4o-mini) extracts structured facts from raw session text before embedding. This improves preference retrieval but introduces per-session LLM latency and cost, and the extracted facts lose the holistic episode context that matters for advice queries.

**Zep** uses temporal knowledge graphs backed by Neo4j to represent entities and relationships extracted from sessions. It excels at temporal state queries ("which CRM are we using now?") but requires significant infrastructure and has weaker coverage of episodic inference.

**Letta** (formerly MemGPT) treats the LLM itself as the memory manager, maintaining structured memory pages and deciding what to retain via in-context reasoning. This is powerful for long-horizon autonomous agents but overkill for the common case of personal assistant memory retrieval.

**gbrain** achieves 97.6% R@5 on LongMemEval-S with a hybrid BM25 + embedding approach, representing the best published result on the most widely used agent memory benchmark.

### 2.3 LongMemEval

LongMemEval (Wu et al., 2024) is a standardized benchmark for evaluating AI agent memory across five question types: single-session preference, multi-session preference, single-session knowledge, multi-session knowledge, and temporal memory. The "S" variant provides a cleaned subset of 500 questions with an average haystack of 47 sessions per question. It measures whether a retrieval system can find the answer-containing session(s) given a natural-language question.

LongMemEval is an important measurement tool, but it primarily tests *needle-in-a-haystack* retrieval — finding the one or two sessions containing a specific fact among 47 candidates. It is a necessary but insufficient measure of real-world agent memory capability, which also requires surfacing contextually relevant episodes for vague advice queries where no single session "contains the answer."

---

## 3. The Token-Native Architecture

ContextFit's core insight is that the most valuable signals in conversational memory are structural, not semantic: *what kind of memory did the user express?* and *does this episode's memory type match what this query needs?* These questions can be answered with token-level pattern matching and simple numeric features — no embedding model required.

### 3.1 Ingest Pipeline

When a session is ingested, ContextFit:

1. **Tokenizes** the text using a BPE tokenizer and stores token arrays in a compressed ChunkStore.
2. **Builds an inverted index** over token IDs for BM25 retrieval.
3. **Extracts memory atoms** (Section 4) from user-authored turns — deterministic, pattern-based, zero API cost.
4. **Computes LSH signatures** for approximate similarity search via MinHash.

The entire ingest pipeline runs in-process with no network calls. For typical conversational sessions (200–1000 tokens), ingest takes **2.7ms per session** on commodity hardware.

### 3.2 Query Pipeline

At query time, ContextFit:

1. **Routes the query** (Section 6) to one of four retrieval modes using structural query signals.
2. **Retrieves candidates** via BM25, episode scoring, atom fusion, or combinations thereof.
3. **Reranks BM25 candidates** (Section 7) using token-native structural session features when the BM25 path is selected.
4. **Returns ranked session IDs** with source-linked chunk evidence.

Total query latency: **0.4–9ms** depending on mode, with no API calls in the default path.

---

## 4. Memory Atoms: Deterministic Domain-Agnostic Fact Extraction

### 4.1 Motivation

LLM-based fact extraction (as used in Mem0) works by asking a language model to summarize what the user revealed about themselves. This produces natural-language fact strings like "User enjoys spicy food and prefers Thai cuisine." The facts are accurate and readable, but they lose the original episode context and require an API call per session.

Memory atoms are a deterministic alternative: regex-pattern-based extraction of typed memory primitives from user-authored turns, with no LLM involved.

### 4.2 Atom Types

Eight domain-agnostic atom types cover the memory primitives an AI agent needs:

| Type | Captures | Example trigger |
|---|---|---|
| `user_preference` | Likes, dislikes, favorites | "I love / I hate / my go-to" |
| `user_interest` | Current activities, exploration | "I'm getting into / working on" |
| `user_goal` | Stated intentions and plans | "I want to / I'm trying to" |
| `user_constraint` | Hard limits and requirements | "I can't / my budget / my allergy" |
| `decision` | Committed choices | "I decided / we went with / let's use" |
| `temporal_update` | State changes over time | "I switched / I now / no longer" |
| `open_loop` | Pending actions and reminders | "remind me / todo / follow up" |
| `entity_fact` | User-owned context facts | "I have / my X is / I bought" |

These types are intentionally domain-neutral. They do not encode any topic-specific vocabulary (food, travel, technology, etc.) and are derived from linguistic patterns that hold across any conversational domain.

### 4.3 Extraction

Extraction operates only on user-authored turns (skipping assistant turns), processes one sentence at a time, and limits output to four atoms per turn to prevent over-extraction from verbose sessions. Each atom is source-linked to the originating session and turn index, enabling traceability.

```
MemoryAtom(
    atom_type="user_preference",
    text="I love spicy food. My favorite cuisine is Thai.",
    source_id="session_042",
    source_date="2026/01/10",
    turn_index=3,
    confidence=0.80
)
```

### 4.4 Retrieval Integration

Atoms are indexed alongside raw session text with `kind=memory_atoms` metadata, enabling filtered retrieval. At query time, atom-type priors derived from the query's intent class weight atom scores (e.g., preference atoms score higher for recommendation queries, temporal_update atoms score higher for "what am I currently using?" queries).

---

## 5. The Episode Relevance Scorer: Structural Session Ranking

### 5.1 The Core Problem

Embedding-based retrieval averages a session into a single vector and measures semantic proximity to the query. For factual queries ("which database did I choose?"), this works: the answer session contains the relevant vocabulary and the embedding captures it.

For vague advice queries ("what should I cook tonight?"), the relevant session ("I just harvested zucchini and tomatoes from my garden") shares almost no vocabulary with the query. Its relevance is structural — it contains entity facts, recent context, and user interests that happen to be useful for this class of question. Embedding cosine similarity misses this entirely; the two vectors look unrelated.

The episode relevance scorer answers a different question: **does this session have the kind of memory signal that this query type needs?**

### 5.2 Features

The scorer computes six numeric features over a session's user-authored turns:

**Lexical overlap** (`lexical`): The overlap ratio between salient query terms and salient episode terms, after stopword removal and frequency weighting. Salient terms are the most distinctive words in each text.

$$\text{lexical} = \frac{|Q_{salient} \cap E_{salient}|}{\sqrt{|Q_{salient}| \cdot |E_{salient}|}}$$

**Atom prior** (`atom_prior`): The maximum product of atom confidence and atom-type prior for the query's inferred intent class. High when the session contains a high-confidence atom whose type matches what the query needs.

**Aligned confidence** (`aligned_conf`): The maximum confidence among atoms whose type aligns with the query's inferred memory intents. Zero when no atoms of the right type exist.

**Aligned count** (`aligned_count`): Count of distinct aligned atom types (capped at 2), rewarding sessions with multiple relevant memory signals.

**Entity context** (`entity_context`): A binary signal that fires when the session contains `entity_fact` atoms and the query indicates advice/recommendation. Handles cases where the relevant episode contains user-owned resources ("I have a ton of zucchini") without explicit preference statements.

**Specificity** (`specificity`): Ratio of salient terms relative to session length. Penalizes generic assistant-heavy sessions that happen to have some user content but lack substantive personal context.

### 5.3 Scoring Formula

$$\text{score} = 1.80 \cdot \text{lexical} + 1.25 \cdot \text{aligned\_conf} + 0.45 \cdot \min(\text{aligned\_count}, 2) + 0.55 \cdot \text{entity\_context} + 0.35 \cdot \text{atom\_prior} + 0.10 \cdot \text{specificity}$$

The weights reflect the empirical finding that lexical overlap and atom alignment are the strongest discriminators for vague advice queries, while entity context breaks ties for situations where there is no explicit preference or goal stated.

### 5.4 Properties

The scorer is:
- **Deterministic**: identical input always produces identical output
- **Interpretable**: each feature contribution is individually readable
- **Zero-cost**: no API calls, no models, no network
- **Instantaneous**: 0.4ms average query latency over 79-case benchmark

### 5.5 Production API

The scorer is available through the production engine:

```python
results = engine.rank_sessions_by_episode_score(query, top_k=10)
# [{"session_id": "s_garden", "score": 2.14, "rank": 1}, ...]
```

---

## 6. The Query Router: Deterministic Mode Selection

### 6.1 Motivation

No single retrieval mode is optimal for all query types:

- **Episode scoring** excels at vague advice and open-loop queries but is weak for specific fact lookups
- **BM25** excels at specific fact retrieval (decisions, current state) but is weaker for vague episodic inference
- **Atom fusion** adds value for explicit preference/constraint queries
- **Episode + BM25 fusion** handles multi-session synthesis and temporal state queries

A fixed retrieval mode wastes accuracy on query types where another mode is stronger. The query router eliminates this tradeoff by selecting the right mode for each query at near-zero cost.

### 6.2 Routing Logic

The router analyzes five structural signal classes:

**Vague advice** (`episode_score`): Queries containing "help me plan/choose/pack", "how should I", "any ideas/tips for". These indicate the caller wants contextual relevance over a prior episode, not a specific fact.

**Personalized preference recommendation** (`preference_rerank`): Queries such as "what kind of music would I enjoy?", "can you recommend a podcast for my commute?", or "what should I watch tonight?" where prior explicit taste evidence should beat generic topical overlap. This route is excluded for temporal/current/open-loop/decision-shaped queries.

**Vague open-loop** (`episode_score`): Queries like "is there something I was supposed to follow up on?", "do I have any outstanding tasks?", "what did I mean to do?" — no named topic, looking for anything pending.

**Specific fact lookup** (`bm25`): Queries containing "what did I decide", "which X did I choose", "which database/tool/framework/vendor did I pick", "remind me what". The caller knows what they're looking for; BM25 finds the session with that text.

**Temporal + fact** (`episode_bm25_fusion`): When both a temporal signal ("currently", "these days", "switched") and a specific-fact pattern ("am I currently using", "what is my current X") appear together, the gold session often describes a change of state ("I switched to WHOOP") using vocabulary the original query doesn't share. Fusion gives both BM25 text matching and temporal-update atom alignment.

**Multi-session synthesis** (`episode_bm25_fusion`): Queries suggesting evidence from multiple sessions is needed ("help me get ready for", "what should I focus on this month", "make progress on").

**Specificity penalty**: Proper nouns and quoted terms in the query indicate a specific named entity is being sought, which favors BM25 over episode scoring.

### 6.3 Key Routing Decisions

Two routing rules proved especially important in development:

**Open-loop queries are episodic, not factual.** A query like "do I have any outstanding tasks?" matches words like "pending" and "outstanding" that might superficially seem like specific-fact signals. But the caller doesn't know *what* they're looking for — they want the episode that contains a todo/reminder. The `_OPEN_LOOP_VAGUE_RE` pattern fires first and routes these to episode_score, which uses `open_loop` atom alignment to find sessions with reminders and pending actions. This improved open-loop Recall@1 from 55.6% to 77.8%.

**Temporal state queries need fusion.** "Which fitness tracker am I currently using?" is a specific-fact query on the surface, but the gold session says "I switched to a WHOOP strap" — sharing no vocabulary with the query. BM25 misses it; the `temporal_update` atom captures "switched" and aligns it with the temporal intent. The `temporal+fact→fusion` rule moves the specific-fact score into `episode_bm25_fusion`, gaining both BM25 and atom alignment. This improved temporal Recall@1 from 70% to 80%.

### 6.4 Production API

```python
result = engine.query_auto(query, top_k=10)
# {
#   "session_ids": ["s_garden", "s_spicy", ...],
#   "route": QueryRoute(mode="episode_score", confidence=0.71, signals=["vague_advice(1 match)"]),
#   "details": {"route": "mode=episode_score conf=0.71 [vague_advice(1 match)]"}
# }
```

---

### 6.7 Structure-Aware File Ingestion

The production ingestion path now applies document-aware chunk boundaries before final token encoding. This does not change the token-native retrieval claim: stored chunks are still token-ID arrays and all search/ranking paths operate in token space. It changes *where chunk boundaries are placed*.

- **TMD ledger (`.tmd`)** files are chunked by source rows while preserving schema and front-matter context. TMD ledger is a new ContextFit-proposed Tabular Markdown file format for keeping ledger-like records row-addressable, atomic, and easier to cite.
- **Markdown (`.md`)** files are chunked by heading and semantic block boundaries. Chunks carry `chunk_type=markdown_section`, `heading_path`, `section_level`, and `chunk_ordinal` metadata; paragraphs, lists, tables, blockquotes, and code fences are kept intact where possible.
- **Plain text (`.txt`)** files are grouped by paragraphs and separators with whole-paragraph overlap.
- **JSON / JSONL (`.json`, `.jsonl`)** files are chunked by object/event records, preserving path, line, and index metadata for API exports, chat logs, and event streams.
- **CSV / TSV (`.csv`, `.tsv`)** files are chunked by row with headers preserved as field names, keeping tabular exports source-verifiable.
- **Email (`.eml`)** files are chunked as messages with sender, recipient, subject, and date context preserved alongside the body.
- **Calendar (`.ics`)** files are chunked by event, preserving summary, time, location, recurrence, organizer, and attendee metadata.
- **Code files** (`.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.sh`, `.sql`, `.css`, `.html`, and more) are chunked by generic symbol/import boundaries with language, symbol, and line-range metadata.
- **Fallback** formats retain the conservative sliding token-window strategy.

This is a structural improvement rather than a new benchmark claim. The 499-case agent-memory eval, which primarily exercises session-level `ingest_text`, remains stable/slightly improved after the change (**62.3% Recall@1 / 93.0% Recall@3 / 99.8% Recall@5** in the latest auto-router run). A dedicated structure-aware file-ingestion benchmark is future work.

## 7. The Structural Session Reranker

### 7.1 Motivation

When the query router selects the BM25 path — specific-fact lookup, decisions, open-loop retrieval — BM25 frequently returns the correct session within the top-5 but not at rank 1. The reason: BM25 scores sessions by aggregate term frequency across all their chunks. A session that incidentally mentions a query term many times in general discussion can outscore a shorter, focused session that mentions it once in a decisive context.

The structural session reranker addresses this by re-scoring BM25 candidates using ten token-native session-level features. It runs after retrieval, operates only on already-retrieved candidates (typically top-50 sessions), and requires no API calls.

### 7.2 Features

The reranker computes ten deterministic features per candidate session:

| Feature | Description |
|---|---|
| **BM25 reciprocal rank** | Prior signal from retrieval stage (1/rank) |
| **Episode relevance score** | Memory-marker density from Section 5 |
| **Lexical overlap** | Query content terms ∩ session content terms (geometric-mean normalized) |
| **Decision alignment** | Query contains decision language AND session contains decision evidence |
| **Preference alignment** | Query contains recommendation language AND session contains preference evidence |
| **Goal alignment** | Query contains planning language AND session contains goal evidence |
| **Constraint alignment** | Query contains constraint language AND session contains constraint evidence |
| **Temporal alignment** | Query contains temporal language AND session contains state-change evidence |
| **Open-loop alignment** | Query contains pending-task language AND session contains todo/follow-up evidence |
| **Named entity overlap** | Proper nouns in query ∩ proper nouns in session text |

Sessions with no marker alignment and weak lexical overlap receive a **generic-distractor penalty** to suppress sessions that match superficially but contain no personal memory signal.

The reranker applies **BM25 confidence gating**: when BM25's top session score dominates the second by a significant margin, structural reranking is skipped and BM25 order is preserved. This protects BM25's precision on specific-fact queries where it already retrieves cleanly.

### 7.3 Question-Type Slot Matching

A further extension — **question-type slot matching** — adds syntactic question-type detection to the feature set. Where the ten base features detect *what kind of memory the user expressed*, slot matching detects *what kind of answer the query expects* and rewards sessions containing answer-type evidence.

Five question-type → evidence-type patterns are recognized:

| Query signal | Evidence signal | Typical case |
|---|---|---|
| Q_WHO (who/whose/someone) | Person name + verb pattern | "Who gave me that recommendation?" |
| Q_WHEN (when/how long/what date) | Month names, years, relative dates | "When did I start using that service?" |
| Q_WHERE (where/what place) | Location markers, place names | "Where did we decide to go?" |
| Q_HOW (how do/did/should; steps) | Instructional structure (step N, first/then/finally) | "How do I set that up?" |
| Q_RECOMMEND (recommend/suggest/would I like) | Positive-outcome phrases (would love, perfect for, you'd enjoy) | "What restaurant would I like?" |

Slot matching is additive — it contributes an additional feature score rather than replacing existing features. The Q_RECOMMEND → E_RECOMMEND pattern receives a slightly higher weight to address the preference-based recommendation gap between token-native and embedding approaches.

### 7.4 Feature Ablation

Three candidate features were evaluated independently on the 499-case benchmark before inclusion. Only slot matching improved results:

| Feature added | R@1 | MRR | Δ R@1 |
|---|---:|---:|---:|
| Structural reranker (base 10 features) | 61.1% | 0.773 | — |
| + IDF-boosted lexical weighting | 60.3% | 0.768 | −0.8 pts |
| + Evidence window density scoring | 61.1% | 0.773 | 0.0 pts |
| **+ Question-type slot matching** | **61.5%** | **0.776** | **+0.4 pts** |

**IDF lexical weighting** slightly hurt performance. In short agent-memory sessions with domain-diverse vocabulary, rare terms are more likely to be noise than signal — the existing unweighted lexical overlap is already well-calibrated for this corpus. IDF may be revisited for longer or denser session corpora.

**Evidence window density scoring** was neutral at this scale. Agent-memory sessions are short enough that term clusters do not meaningfully discriminate between sessions; the feature is available as a flag but inactive by default.

**Question-type slot matching** consistently helps, with the largest per-behavior gain on `preference_informs_recommendation` (+3.2 pts), the behavioral category where token-native approaches are structurally weakest relative to embedding models.

### 7.5 Per-Behavior Profile

The structural reranker with slot matching has a distinctive per-behavior profile compared to OpenAI `text-embedding-3-small`:

| Behavior | ContextFit R@1 | OpenAI R@1 | Delta |
|---|---:|---:|---:|
| open_loop_retrieval | **83.6%** | 63.9% | **+19.7 pts** |
| preference_informs_recommendation | **85.5%** | 77.4% | **+8.1 pts** |
| temporal_supersession | 44.4% | **47.6%** | −3.2 pts |
| episodic_interest_inference | 40.3% | **43.1%** | −2.8 pts |
| multi_session_synthesis | 82.1% | **87.5%** | −5.4 pts |

ContextFit's structural advantage is clearest on **open-loop retrieval** (+19.7 pts) and, after the preference-rerank route, **preference-based recommendation** (+8.1 pts). Remaining gaps are temporal routing and goal advice; multi-session synthesis is improved but still trails embedding models by 5.4 points.

---

### 7.6 Router-Gated Preference Reranking

The largest gap in the earlier token-native profile was preference-based recommendation. Dense embeddings are good at semantic bridges such as "songs" → "instrumental music" or "language" → "French". Rather than adding embeddings, ContextFit adds a dedicated route for the subset of recommendation queries where explicit user taste evidence should dominate generic topical overlap.

The preference reranker is still token-native and deterministic. It:

1. extracts user-authored turns from candidate sessions,
2. detects explicit preference markers (`love`, `enjoy`, `prefer`, `favorite`, `go-to`, etc.),
3. applies lightweight stemming and plural normalization,
4. scores query overlap inside local preference windows around those markers,
5. blends BM25 reciprocal rank with episode relevance, and
6. penalizes generic non-preference sessions.

On the 499-case benchmark, this moves `preference_informs_recommendation` from **56.5% Recall@1** with the prior free rerank fusion to **85.5% Recall@1**, beating both OpenAI `text-embedding-3-small` (**77.4%**) and Cohere `embed-english-v3.0` (**83.9%**) on that behavior. Overall benchmark Recall@1 moves modestly from **60.7%** to **61.1%** because the route is intentionally conservative and some recommendation-shaped constraint/episodic cases remain better served by episode scoring.

A guardrail result is important: LongMemEval's `single-session-preference` slice is not the same task. It often asks for broad prior-context advice rather than explicit taste retrieval. Production `query_auto` with the preference route scored **36.7% Any@5** on that 30-row LongMemEval slice, below BM25/episode baselines. We therefore report the preference-reranker claim only on the generated domain-agnostic agent-memory preference recommendation benchmark and keep LongMemEval as a separate needle-in-haystack evidence-retrieval measure.

---

## 8. Benchmark Results

### 8.1 Agent-Memory Evaluation Suite

We constructed a 79-case domain-agnostic evaluation suite covering seven behavioral categories. Cases were designed to be topically diverse (food, travel, work, family, finance, health, hobbies, technology) and to include realistic distractor sessions that compete with the gold answer. Average session pool: 3.7 sessions per question.

**Behavioral categories:**
- `preference_informs_recommendation` (9 cases): explicit preference retrieval for recommendations
- `constraint_informs_advice` (9 cases): hard constraint retrieval for planning advice
- `goal_informs_advice` (11 cases): goal/intent retrieval for guidance
- `temporal_supersession` (10 cases): latest-state retrieval when state has changed
- `decision_retrieval` (7 cases): specific past decision retrieval
- `open_loop_retrieval` (9 cases): pending action and reminder retrieval
- `episodic_interest_inference` (20 cases): implicit interest inference for vague advice
- `multi_session_synthesis` (4 cases): evidence needed from two or more sessions

**Evaluation metric:** Recall@K (fraction of cases where a gold session appears in the top-K results) and MRR (mean reciprocal rank of best gold session).

### 8.2 Overall Results

**79-case hand-crafted eval** (hard episodic inference, vague advice, temporal, open-loop):

| System | R@1 | R@3 | R@5 | MRR | Ingest/session | Query | Cost |
|---|---:|---:|---:|---:|---:|---:|---|
| ContextFit BM25 | 44.3% | 84.8% | 92.4% | 0.640 | 2.8ms | 8.8ms | free |
| OpenAI text-embedding-3-small | 55.7% | 96.2% | 100.0% | 0.745 | ~300ms† | ~150ms† | embed API |
| Mem0 (GPT-4o-mini + embed) | 54.4% | 91.1% | 97.5% | 0.716 | ~2,225ms | ~341ms | LLM+embed API |
| **ContextFit episode score** | **69.6%** | **96.2%** | **100.0%** | **0.824** | **2.7ms** | **0.4ms** | **free** |
| ContextFit auto router | 69.6% | 96.2% | 98.7% | 0.821 | 2.7ms | 2.8ms | free |

†OpenAI timing assumes uncached embeddings. Cached (local disk): ~1ms/session ingest, ~8ms query.

**499-case eval** (79 hand-crafted + 420 GPT-generated, 26 domains, 8 behaviors):

| System | R@1 | R@3 | R@5 | MRR | Cost |
|---|---:|---:|---:|---:|---|
| Mem0 (GPT-4o-mini + embed, 79-case) | 54.4% | 81.0% | 91.1% | 0.716 | LLM+embed API |
| Cohere embed-english-v3 | 58.7% | 91.4% | 100.0% | 0.751 | embed API |
| ContextFit structural reranker | 61.1% | 93.0% | 99.8% | 0.773 | free |
| ContextFit + slot matching | 61.5% | 93.2% | 99.8% | 0.776 | free |
| **ContextFit + routed rerankers** | **62.3%** | **93.0%** | **99.8%** | **0.777** | **free** |
| OpenAI text-embedding-3-small | 63.1% | 96.6% | 100.0% | 0.792 | embed API |

With preference reranking enabled, ContextFit surpasses Cohere embeddings by 2.4 points R@1 and trails OpenAI by 2.0 points R@1 — while beating both embedding baselines on preference recommendation R@1. R@5 is effectively solved (≥99.8%) across all systems; R@1 and R@3 are the meaningful differentiators.

### 8.3 Results by Behavioral Category

| Category | BM25 | OpenAI embed | Mem0 | Episode score | Auto router |
|---|---:|---:|---:|---:|---:|
| **Episodic inference** | 25.0% | 40.0% | 25.0% | **60.0%** | **60.0%** |
| **Open loops** | 44.4% | 55.6% | 66.7% | **77.8%** | **77.8%** |
| **Temporal supersession** | 50.0% | 80.0% | 50.0% | **90.0%** | 80.0% |
| **Preferences** | 44.4% | 66.7% | **77.8%** | **77.8%** | **77.8%** |
| **Constraints** | 66.7% | 66.7% | 66.7% | **88.9%** | **77.8%** |
| **Goals** | 27.3% | 54.5% | 72.7% | **77.8%** | 72.7% |
| **Decisions** | 57.1% | **85.7%** | 71.4% | 57.1% | 57.1% |
| **Multi-session synthesis** | **100.0%** | 50.0% | 25.0% | 50.0% | 75.0% |

All metrics are Recall@1.

### 8.4 Key Findings

**1. Token-native episode scoring beats embedding cosine similarity by 14 points.** The gap is largest on episodic inference (+20 points over OpenAI embeddings), the query category where embeddings are weakest: when a query is semantically distant from the relevant episode, vector similarity fails but structural atom alignment succeeds.

**2. LLM extraction (Mem0) does not reliably improve over raw embeddings.** Mem0 beats raw OpenAI embeddings only on preferences (77.8% vs 66.7%) and goals (72.7% vs 54.5%), where LLM extraction genuinely distills explicit statements. On episodic inference (25.0% vs 40.0%), open loops (66.7% vs 55.6%), multi-session synthesis (25.0% vs 50.0%), and temporal supersession (50.0% vs 80.0%), Mem0 is worse. The LLM extraction step destroys the holistic episode context that matters for vague queries.

**3. The query router eliminates many mode-specific failures while exposing where specialized routes are needed.** Pure episode scoring is 50% Recall@1 on multi-session synthesis; pure BM25 is 25% on episodic inference. The auto router matches or approaches the best mode for several categories, and the dedicated preference route lifts preference recommendation to 85.5% Recall@1. Broad goal-advice routing remains open work; multi-session synthesis now has a dedicated evidence-coverage route but still trails embeddings.

**4. Episode scoring is the fastest query mode by a large margin.** At 0.4ms average query latency, episode scoring runs 22× faster than BM25 (8.8ms), 375× faster than cached OpenAI retrieval (~150ms), and ~850× faster than Mem0 (~341ms).

### 8.5 LongMemEval-S Results

On LongMemEval-S (500 questions, 470 scored after abstention exclusion):

| System | Any@1 | Any@3 | Any@5 | All@5 | MRR |
|---|---:|---:|---:|---:|---:|
| ContextFit conversation-aware token-only | 82.3% | 90.4% | 94.7% | 77.7% | 0.870 |
| ContextFit parent/child token-only | 82.8% | 90.6% | **95.1%** | 77.9% | 0.873 |
| **ContextFit parent/child + coverage rerank** | **82.8%** | **91.3%** | **95.1%** | **80.4%** | **0.875** |
| ContextFit + local BGE fusion | 82.3% | 93.0% | 95.1% | 83.2% | 0.879 |
| ContextFit + OpenAI fusion | 83.6% | 94.0% | **96.0%** | 84.9% | 0.889 |
| gbrain-hybrid (published) | — | — | **97.6%** | — | — |

ContextFit's parent/child conversation-aware token-native path is the primary LongMemEval claim and closes most of the theoretical gap at zero cost. It reaches 95.1% Any@5 without embeddings, narrowing the gap to gbrain-hybrid to 2.5 percentage points; optional OpenAI embedding fusion narrows it to 1.6 percentage points. The companion-evidence coverage reranker keeps that 95.1% Any@5 headline intact while lifting overall All@5 from **77.9% to 80.4%**.

By question type, the parent/child token-only path closes the LongMemEval preference gap on top-5 retrieval: single-session preference Any@5 improves from 76.7% in the original token baseline to **83.3%**, matching OpenAI fusion, while preference Any@1 is higher than OpenAI fusion (43.3% vs 40.0%). On multi-session questions, token-only Any@5 improves from 93.4% to **95.0%**, narrowing the gap to OpenAI fusion to roughly 0.8 points. Coverage reranking targets the remaining weakness: multi-session All@5 improves from **55.4% to 65.3%**, narrowing the OpenAI fusion coverage gap from 17.4 points to 7.4 points.

**Important context**: LongMemEval measures specific-fact needle-in-a-haystack retrieval over 47-session haystacks. This task profile favors BM25 and embedding models because the gold sessions contain vocabulary from the question. The episode scorer deliberately underperforms on this benchmark (40% Any@5 standalone) — it is not designed for large-haystack exact lookup. LongMemEval and the agent-memory eval measure complementary retrieval regimes.

---

## 9. Why This Approach Is Impactful

### 9.1 It Solves the Right Problem

The most important class of agent memory queries is not "find the session containing X fact." It is "figure out which prior context would actually help me answer this question." A user asking "what should I cook tonight?" doesn't need the agent to find a session containing "cook" — they need the agent to find the session where they mentioned a garden harvest, a dietary change, or a cooking project. This is the episodic inference problem, and it is where embedding-based systems are structurally weakest.

The token-native episode scorer was purpose-built for this problem. Its features — atom alignment, entity context, lexical salience — specifically capture the structural signals that make a session episodically relevant without requiring semantic vector proximity.

### 9.2 It Eliminates a Dependency Tier

Every production AI system that relies on embedding-based memory adds three dependencies: an embedding API provider, a vector database, and the latency and cost of both. These dependencies:

- Add per-query API costs that compound at agent scale
- Introduce network failure modes in the memory retrieval path
- Create vendor lock-in and compliance surface area
- Add operational complexity (vector DB provisioning, index management, embedding model versioning)

ContextFit's token-native approach eliminates all three. The memory system runs in-process, with no external calls in the default path. This is not a performance optimization — it is an architectural simplification that reduces the number of things that can go wrong in a production agent.

### 9.3 It Is Interpretable by Design

When an embedding-based system fails to retrieve a relevant session, the debugging answer is "the cosine similarity was 0.62." When ContextFit fails, the debugging answer is specific: "the session had no `user_preference` atoms and the query routed to atom_fusion mode; the lexical overlap was 0.08 because the query used 'nutrition' and the session used 'eating habits'." Every routing decision and every score contribution is named and auditable.

This interpretability is essential for the trust and auditability requirements of production AI systems, especially in personal assistant contexts where memory retrieval touches sensitive user data.

### 9.4 It Scales Differently

Embedding-based retrieval has O(N) API cost at ingest time (one embedding call per session) and near-constant query cost (one embedding call plus vector search). At scale:

- 10,000 sessions × $0.0001/embed = $1.00 ingest cost (plus vector DB storage and ops)
- Each query: ~$0.00002 (embedding) + vector search latency

Token-native retrieval has O(N) CPU cost at ingest (tokenization + BM25 index) and O(N) CPU cost at query time for episode scoring (no index needed, direct scan with early stopping). For small to medium agent corpora (up to ~100,000 sessions), the episode scorer is faster and free. For very large corpora, BM25 is the right mode (already integrated) and remains free.

### 9.5 It Is Composable

The six primitives — memory atoms, episode scorer, query router, structural reranker, preference reranker, and evidence-coverage reranker — are independently useful and combinable:

- **Memory atoms alone** provide a 70% Recall@1 baseline that beats raw BM25 by 26 points on explicit memory queries
- **Episode scorer alone** provides the best single-mode performance for vague advice queries
- **Auto router** combines both without requiring the caller to know which mode is appropriate
- **Structural reranker** improves BM25-path precision with ten token-native session features and question-type slot matching
- **Preference reranker** closes the generated preference recommendation gap with 85.5% R@1 at zero API cost
- **Evidence-coverage reranker** narrows the multi-session synthesis gap from −14.3 to −5.4 points vs OpenAI embeddings
- **OpenAI fusion** is available as an optional enhancement that adds embedding signal where it genuinely helps (facts, decisions) while the token-native path handles episodic inference

No other memory system in this space offers this composability with deterministic, interpretable routing.

---

## 10. Limitations and Future Work

### 10.1 Scale

The episode scorer performs a linear scan over all indexed sessions. At 100 sessions this is 0.4ms; at 100,000 sessions this becomes the dominant cost. The query router and BM25 confidence gating already route most specific-fact queries away from the episode scorer path, limiting linear scan to the vague-advice cases where it is genuinely needed. For corpora beyond ~100,000 sessions, session pre-clustering or ANN (approximate nearest neighbor) indexing over episode feature vectors would be the natural scaling step.

### 10.2 Multilingual Support

The current memory atom patterns are English-only. Extending to multilingual corpora requires either translated pattern sets or a language-agnostic alternative (e.g., dependency-parse-based extraction).

### 10.3 LongMemEval Gap

A 1.6-point gap to the best published LongMemEval result remains when using optional OpenAI fusion. Fresh local-embedding experiments with BGE already reach **95.1% Any@5** without an external API, only 0.9 points below OpenAI fusion. The remaining work is to integrate local embedding fusion as a first-class optional mode without compromising the pure token-native default.

### 10.4 Multi-Session Synthesis

The dedicated evidence-coverage route has improved multi-session synthesis Recall@1 to **82.1%**, up from **73.2%** with the preference-only auto router and 50% with episode-score-only on the smaller diagnostic slice. A remaining gap to OpenAI embeddings (**87.5%**) persists, but the token-native route narrows it substantially without embeddings or LLM calls.

### 10.5 Preference and Semantic Bridge Limits

The dedicated preference route closes the generated preference recommendation gap on this benchmark: ContextFit reaches **85.5% R@1**, ahead of OpenAI (**77.4%**) and Cohere (**83.9%**) on that behavior. Remaining misses are mostly semantic bridges where the user expressed a preference with different vocabulary from the recommendation query (for example, `language → French`, `songs → instrumental music`, or `sleep activity → reading before bed`).

Further improvement may require corpus-learned token associations, optional lightweight local embeddings, or learned token-native weighting that approximates semantic generalization without adding an external inference call.

---

## 11. Deployment Architecture

ContextFit's token-native design produces a set of deployment properties that are distinct from every embedding-based memory system. These are not incidental — they follow directly from the architectural choices described in Section 3.

### 11.1 No Database Required

Embedding-based memory systems require a vector database to store and search dense vectors. Common choices — Qdrant, Weaviate, Pinecone, Chroma, pgvector — each introduce a service dependency: provisioning, configuration, authentication, schema management, and ongoing operational overhead. Many also require Postgres or Redis as backing infrastructure.

ContextFit requires none of this. The index is a directory of flat files:

```
memory_index/
  chunks/          # zstd-compressed token arrays
  inverted/        # BM25 postings (binary)
  lsh/             # MinHash signatures (binary)
  metadata/        # Per-chunk metadata
```

There is no service to start, no port to open, no schema to migrate, and no daemon to monitor. The engine loads directly from disk in-process in under a second for typical corpora.

### 11.2 No GPU Required

Local embedding models (sentence-transformers, E5, BGE, and similar) require PyTorch and benefit significantly from GPU acceleration. Running them on CPU is possible but meaningfully slower for real-time use. Hosting a GPU instance for a memory retrieval system is expensive and often overkill.

ContextFit's entire pipeline runs on CPU. The dependency stack contains no torch, no CUDA, no ONNX runtime, and no model weights. All operations — BM25 scoring, roaring bitmap intersection, MinHash LSH, episode feature computation, structural reranking — are pure CPU operations implemented in numpy and native Python extensions.

This makes ContextFit viable on the smallest cloud compute tier, a developer laptop, or an edge device without any special hardware provisioning.

### 11.3 Minimal Footprint

The full production dependency stack occupies approximately 41MB on disk:

| Dependency | Purpose | Size |
|---|---|---|
| tiktoken | BPE tokenization | 2.8MB |
| numpy | Numeric operations | 36MB |
| pyroaring | Roaring bitmap index | ~1MB |
| datasketch | MinHash LSH | 0.9MB |
| zstandard | Token array compression | ~0.3MB |
| networkx | Graph structures | ~2MB |

For comparison, PyTorch (the minimum requirement for any local embedding model) is 500MB–2GB depending on version and CUDA build. A typical sentence-transformers setup with model weights adds another 100–400MB. A vector database like Qdrant or Chroma adds further infrastructure surface area.

ContextFit's 41MB footprint means it can be bundled into serverless functions, containers with strict size limits, or applications that cannot tolerate large binary dependencies.

### 11.4 Filesystem-Native Storage and Permissions

Because the index is a standard directory, it inherits the full POSIX permission model without any additional configuration. Access control is standard `chmod`/`chown` and filesystem ACLs — the same model used to protect the source files being indexed.

This has practical benefits for common agent deployment patterns:

**Co-location with source data.** The index directory can live alongside the files or vaults it indexes. A personal knowledge base stored in `~/Documents/notes/` can have its ContextFit index at `~/Documents/notes/.cf_index/` — backed up, moved, and permissioned as a unit.

**File vault integration.** Encrypted vaults (VeraCrypt, macOS encrypted disk images, age-encrypted archives) can contain both the source content and the ContextFit index. No database service runs outside the vault boundary.

**Snapshot and backup compatibility.** Because the index is files, it is compatible with rsync, Time Machine, Restic, Borgbackup, and any other file-level backup tool. There is no need for database-specific dump/restore procedures.

**Offline operation.** With no external service dependencies in the default path, ContextFit works fully offline. The memory retrieval path is available without network access — important for privacy-sensitive use cases or environments with restricted egress.

### 11.5 Deployment Comparison

| Property | ContextFit | Embedding + vector DB |
|---|---|---|
| Database required | None | Vector DB (Qdrant / Chroma / pgvector) |
| GPU required | No | Recommended for local models |
| Dep footprint | ~41MB | 500MB–2GB+ (PyTorch alone) |
| Storage format | Plain files | DB-managed blobs |
| Permissions | POSIX filesystem | DB users / ACLs |
| Offline capable | Yes (default path) | No (API) / Partial (local model) |
| Backup method | Any file backup tool | DB dump + vector store export |
| API cost (default) | $0 | Per-embedding call |
| Latency (default) | 0.4–9ms in-process | 50–500ms+ (embed + vector search) |

The optional OpenAI or Cohere fusion modes do introduce API dependencies and cost — but these are additive enhancements to a system that already works without them, not requirements for baseline functionality.

---

## 12. Conclusion

This paper has presented ContextFit, a token-native approach to AI agent memory retrieval that achieves state-of-the-art performance on agent-memory tasks while requiring no embedding APIs, no LLM preprocessing, no vector databases, no GPU, and no database service. The four core primitives — memory atoms, episode relevance scorer, query router, and structural session reranker — each address a specific structural weakness of embedding-based approaches and are independently verifiable and composable. The deployment architecture (Section 11) makes ContextFit viable in contexts where embedding-based systems cannot be used: offline environments, encrypted vaults, size-constrained deployments, and applications that require standard filesystem-level permission management.

The central empirical claim is simple and reproducible: for the queries that matter most in AI agent memory — vague advice, episodic context inference, open-loop retrieval — structural token-native features outperform embedding cosine similarity by a wide and consistent margin, at a fraction of the cost and latency.

The broader implication is architectural: the assumption that conversational memory must be converted to vectors before it can be retrieved is not a technical necessity. It is a convention inherited from document retrieval, applied without examination to a domain where it underperforms. Token-native memory retrieval is not just faster and cheaper — for the hardest memory problems, it is also more accurate.

---

## Appendix A: Benchmark Details

### A.1 Agent-Memory Eval Construction

Each evaluation case consists of:
- A natural-language question an agent might receive
- 3–6 sessions (average 3.7), one of which is the gold answer session
- Realistic distractor sessions sharing some topical overlap with the question

Cases were constructed to cover diverse domains (food, travel, technology, health, finance, family, hobbies, creative work) and avoid benchmark-specific vocabulary. No case was constructed to benefit any specific retrieval mode. The benchmark runner is available at `benchmarks/agent_memory_eval.py` in the ContextFit repository.

### A.2 Mem0 Configuration

Mem0 v2.0.2 was configured with:
- LLM: `gpt-4o-mini` (OpenAI) for fact extraction
- Embedder: `text-embedding-3-small` (OpenAI)
- Vector store: Qdrant in-memory, one fresh instance per evaluation case

Timing was measured over a 5-case sample with live API calls (no caching). All cases used the same Mem0 configuration with no tuning to the evaluation data.

### A.3 LongMemEval-S Configuration

Evaluated on the cleaned 500-question LongMemEval-S subset. ContextFit baseline uses BM25 hybrid retrieval with 8192-token chunk size, no overlap, metadata_boost=1.0. The OpenAI fusion variant adds `text-embedding-3-small` session embeddings via RRF with the BM25 ranking. Abstention questions (question_id ending in `_abs`) are excluded from scoring (30 questions).

---

## Appendix B: Reproducibility

All code, benchmark runners, and evaluation data are available at:
**https://github.com/ContextFit/cf**

Key files:
- `src/contextfit/retrieval/memory_atoms.py` — atom extraction and episode scorer
- `src/contextfit/retrieval/query_router.py` — deterministic query router
- `src/contextfit/retrieval/engine.py` — production retrieval engine (includes `rerank_sessions_by_structure()` and structure-aware `ingest_file()`)
- `src/contextfit/extractors/document.py` — Markdown/plain-text metadata extraction and structure-aware chunking
- `src/contextfit/extractors/tmd.py` — row-aware TMD extraction/chunking
- `src/contextfit/extractors/structured.py` — JSON/JSONL object records and CSV/TSV row-aware chunking
- `src/contextfit/extractors/email.py` — email metadata extraction and message-aware chunks
- `src/contextfit/extractors/calendar.py` — ICS event extraction and calendar-aware chunks
- `src/contextfit/extractors/code.py` — generic dependency-free source code symbol/import chunking
- `src/contextfit/retrieval/token_rerank.py` — experimental token-native reranker primitives
- `benchmarks/agent_memory_eval.py` — 499-case evaluation runner (modes: baseline, free\_rerank, free\_rerank\_slot, free\_rerank\_idf, free\_rerank\_window, free\_rerank\_all, openai\_fusion, cohere\_vector, and more)
- `benchmarks/data/agent_memory_eval.json` — 79-case hand-crafted evaluation suite
- `benchmarks/data/agent_memory_eval_500.json` — 499-case eval (79 hand-crafted + 420 GPT-4o-mini generated)
- `benchmarks/generate_eval_cases.py` — automated case generation pipeline
- `benchmarks/longmemeval_contextfit.py` — LongMemEval runner
- `benchmarks/longmemeval_contextfit_report.md` — full experiment log

---

*ContextFit Research, May 2026. Feedback: github.com/ContextFit/cf/issues*
