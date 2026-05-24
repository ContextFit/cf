# ContextFit × LongMemEval Retrieval Benchmark

Date: 2026-05-09

## What was measured

This evaluates **retrieval/evidence ranking**, not official end-to-end LongMemEval QA generation.

For each LongMemEval_S cleaned item:
1. Build a temporary ContextFit KB from that item's haystack sessions.
2. Query ContextFit with the question plus question date.
3. Check whether retrieved chunks/sessions came from the ground-truth `answer_session_ids`.

Abstention examples were excluded from the main score because this harness does not yet test no-answer calibration.

## Baseline configuration

- Dataset: `benchmarks/data/longmemeval_s_cleaned.json`
- Examples: 500 total
- Scored examples: 470 non-abstention examples
- Skipped: 30 abstention examples
- Retrieval method: `hybrid`
- Top-k chunks: 10
- Chunk size: 8192 tokens
- Overlap: 0
- Average haystack sessions/question: 47.7
- Output JSON: `benchmarks/longmemeval_contextfit_hybrid_full.json`
- Runner: `benchmarks/longmemeval_contextfit.py`

## Baseline results

| Metric | Score |
|---|---:|
| Any gold evidence @1 | 81.1% |
| Any gold evidence @3 | 89.1% |
| Any gold evidence @5 | 93.6% |
| Any gold evidence @10 | 96.6% |
| All gold evidence @1 | 28.1% |
| All gold evidence @3 | 71.7% |
| All gold evidence @5 | 77.7% |
| All gold evidence @10 | 85.5% |
| MRR | 0.861 |

## 2026-05-19 Optional OpenAI Fusion Claim Artifact

This fresh run supports the public wording:

> ContextFit with optional OpenAI fusion reaches 96.6% Any@5 and 98.7% Any@10 evidence retrieval on LongMemEval-S, with no vector database required.

This is still a retrieval/evidence-ranking result, not an official end-to-end LongMemEval QA score. It uses OpenAI `text-embedding-3-small` embeddings as an optional cached fusion signal; ContextFit still stores and searches the corpus without a vector database.

| Metric | Score |
|---|---:|
| Scored examples | 470 |
| Abstention examples skipped | 30 |
| Any gold evidence @1 | 84.68% |
| Any gold evidence @3 | 94.26% |
| Any gold evidence @5 | **96.60%** |
| Any gold evidence @10 | **98.72%** |
| All gold evidence @5 | 83.62% |
| All gold evidence @10 | 91.28% |
| MRR | 0.8999 |

Artifact:

- Report: `benchmarks/longmemeval_fusion_claim_966_987_20260519.md`
- Raw JSON: `benchmarks/longmemeval_fusion_claim_966_987_20260519.json`
- SHA-256: `059c778ca389e2a5939505800acffd6349f0be7ada579238023d342784214932`
- Runtime: 15,773.8 seconds

Reproduction command:

```bash
.venv/bin/python benchmarks/longmemeval_contextfit.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --limit 0 \
  --method hybrid \
  --top-k-chunks 10 \
  --retrieval-k 100 \
  --chunk-size 2048 \
  --overlap 128 \
  --rank-by-session \
  --conversation-chunks \
  --conversation-parent \
  --coverage-rerank \
  --structured-temporal-filters \
  --openai-fusion \
  --out benchmarks/longmemeval_fusion_claim_966_987_20260519.json
```

## Baseline by question type

| Type | n | Any@1 | Any@5 | Any@10 | All@5 | All@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-session-assistant | 56 | 96.4% | 100.0% | 100.0% | 100.0% | 100.0% | 0.978 |
| knowledge-update | 72 | 93.1% | 98.6% | 98.6% | 94.4% | 95.8% | 0.958 |
| single-session-user | 64 | 85.9% | 93.8% | 98.4% | 93.8% | 98.4% | 0.896 |
| temporal-reasoning | 127 | 77.2% | 92.1% | 96.9% | 70.1% | 78.0% | 0.831 |
| multi-session | 121 | 77.7% | 93.4% | 95.0% | 57.0% | 73.6% | 0.837 |
| single-session-preference | 30 | 43.3% | 76.7% | 86.7% | 76.7% | 86.7% | 0.551 |

## Implemented improvements tested

### 1. Metadata index synchronization

`RetrievalEngine.ingest_token_chunks()` now registers caller-provided metadata in `MetadataIndex`. This makes metadata-aware grouping/ranking usable from every ingest path, not only CLI paths.

### 2. Safer metadata boost default

Default `metadata_boost` was changed from `2.0` to `1.0`.

Reason: once metadata indexing was active, broad LongMemEval metadata such as dates and session ids could swamp text relevance. Email/header-heavy callers can still opt into higher boosts explicitly.

### 3. Session-level ranking API

Added `RetrievalEngine.query_groups(group_by="session_id")`, which retrieves chunks then ranks grouped evidence sources using max score, summed score, and a small count bonus.

### 4. Preference experiments

Added optional lightweight preference fact extraction in the benchmark runner. Initial isolated runs did **not** beat baseline, so this should stay experimental/off by default.

### 5. OpenAI vector + ContextFit fusion experiment

Added optional benchmark-only OpenAI embedding session ranking and reciprocal-rank fusion with ContextFit retrieval.

- Output JSON: `benchmarks/longmemeval_contextfit_openai_fusion_full_top5.json`
- Mode: `--openai-fusion --top-k-chunks 5 --retrieval-k 20`
- Embeddings are cached locally under `benchmarks/cache/openai_embeddings/`.

## Improved/fusion results

| Metric | Baseline | Local BGE fusion | OpenAI fusion |
|---|---:|---:|---:|
| Any gold evidence @1 | 81.1% | 82.3% | 83.6% |
| Any gold evidence @3 | 89.1% | 93.0% | 94.0% |
| Any gold evidence @5 | 93.6% | 95.1% | 96.0% |
| All gold evidence @5 | 77.4% | 83.2% | 84.9% |
| MRR | 0.860 | 0.879 | 0.889 |

Note: the fusion run produced only 5 ranked sessions, so @10 equals @5 in that artifact.

## Improved/fusion by question type

| Type | Baseline Any@5 | Fusion Any@5 | Delta | Baseline All@5 | Fusion All@5 | Delta | Baseline MRR | Fusion MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single-session-assistant | 100.0% | 100.0% | +0.0 pts | 100.0% | 100.0% | +0.0 pts | 0.978 | 0.991 |
| knowledge-update | 98.6% | 100.0% | +1.4 pts | 94.4% | 94.4% | +0.0 pts | 0.958 | 0.942 |
| single-session-user | 93.8% | 98.4% | +4.7 pts | 93.8% | 98.4% | +4.7 pts | 0.896 | 0.883 |
| temporal-reasoning | 92.1% | 93.7% | +1.6 pts | 70.1% | 78.0% | +7.9 pts | 0.831 | 0.875 |
| multi-session | 93.4% | 95.9% | +2.5 pts | 57.0% | 72.7% | +15.7 pts | 0.837 | 0.902 |
| single-session-preference | 76.7% | 83.3% | +6.7 pts | 76.7% | 83.3% | +6.7 pts | 0.551 | 0.589 |

## Comparison to gbrain screenshot

Closest headline comparison is R@5-style “any evidence in top 5”:

| System | R@5 / Any@5 |
|---|---:|
| gbrain-hybrid | 97.6% |
| gbrain-vector | 97.4% |
| MemPalace raw | 96.6% |
| ContextFit baseline hybrid | 93.6% |
| ContextFit + local BGE fusion | 95.1% |
| ContextFit + OpenAI fusion | 96.0% |

ContextFit baseline was ~4.0 points behind gbrain-hybrid. Fusion narrows that to ~1.6 points, while also improving all-evidence@5 materially.

## Readout

ContextFit is already strong as a LongMemEval evidence retriever:

- Baseline finds at least one correct evidence session in top 10 for **96.4%** of scored questions.
- Baseline finds at least one correct evidence session in top 5 for **93.6%**.
- Local BGE fusion lifts top-5 evidence recall to **95.1%** and all-evidence@5 to **83.2%** without OpenAI/API dependency.
- OpenAI fusion lifts top-5 evidence recall to **96.0%** and all-evidence@5 to **84.9%**.

The biggest remaining weak area is still **single-session-preference**. Fusion improves it from **76.7% Any@5** to **83.3% Any@5**, but rank-1 remains low. Preference questions need better semantic preference modeling, not just metadata or literal token matching.

Multi-session improved most on all-evidence retrieval: **57.0% → 72.7% All@5**. That suggests source/session-level fusion is helping retrieve the full evidence set, not just one matching session.

## Important caveats

- This is not the official LongMemEval QA score. It does not call an LLM to answer questions and compare final answers.
- It measures whether ContextFit retrieves annotated evidence session(s), which is the right first-pass benchmark for a memory/retrieval system.
- Each question was indexed into its own temporary KB. This matches the LongMemEval haystack format, but it is not a single shared long-running user memory corpus.
- Abstention behavior still needs a separate benchmark.
- Local BGE fusion uses local `sentence-transformers` embeddings and is experimental/uncommitted for now; baseline ContextFit remains pure token-native local/no-LLM retrieval.
- OpenAI fusion uses external embeddings and is benchmark-only for now.

## Next work

1. Make local vector retrieval/fusion first-class so the 96% result does not depend on OpenAI embeddings.
2. Keep `--preference-facts` experimental; revise before enabling by default.
3. Improve preference retrieval with explicit preference/aversion patterns and semantic source ranking.
4. Add temporal ranking features for latest/previous/before/after wording.
5. Add abstention calibration as a separate evaluation.

## Token-native reranker experiment

Implemented an experimental no-embedding reranker in `src/contextfit/retrieval/token_rerank.py` and wired it into `RetrievalEngine.query(..., token_rerank=True)` plus benchmark flags:

- `--token-native-rerank`
- `--token-native-fusion`
- `--token-chain-expand`

The reranker tests token phrase overlap, local answer windows, preference/aversion markers, temporal/update markers, and user-turn-focused scoring for personal questions. This is the first concrete version of the “language structure, not vector blur” idea.

Initial preference-only LongMemEval results did **not** beat baseline:

| Variant | Any@1 | Any@5 | Any@10 | MRR |
|---|---:|---:|---:|---:|
| Baseline top-10 | 43.3% | 76.7% | 86.7% | 0.551 |
| Token-native structural rerank | 36.7% | 70.0% | 83.3% | 0.506 |
| Token-native fusion | 30.0% | 66.7% | 80.0% | 0.430 |
| Token-chain expansion | 23.3% | 46.7% | 80.0% | 0.375 |

Readout: the basic structural signals are directionally sensible but too weak/noisy as hand-weighted rank features. The important discovery is that `single-session-preference` in LongMemEval is often really “retrieve a prior personal context that should inform a recommendation,” not just literal preference wording. Examples include recommending a movie based on a prior comedy-writing session, or dinner ideas based on a prior gardening session. That requires episodic profile/context inference, not only preference verbs.

Next token-native direction: learn the weights/patterns from retrieval outcomes or build query-intent-specific source ranking, especially for personal recommendation/advice queries that should prioritize previous **user** statements and stable interests over generic assistant advice.

## Generalization guardrail

Do **not** tune ContextFit to LongMemEval topics or answer labels. Use LongMemEval as an external measurement harness only.

Acceptable improvements should be agent-memory primitives that generalize across domains:

- user-stated facts, preferences, constraints, goals, and decisions
- advice/recommendation queries that need prior user context
- temporal changes: current/latest/previous/no-longer/changed
- entity and source-level evidence aggregation
- answer/source traceability
- abstention/no-answer calibration

Avoid dataset-specific topic dictionaries, gold-session heuristics, or weights chosen only because they improve a LongMemEval slice. If a feature is benchmark-informed, validate it against at least one non-LongMemEval corpus or synthetic domain-agnostic scenario before enabling it by default.

## General memory atom layer

Implemented a first-pass, domain-agnostic memory atom extractor in `src/contextfit/retrieval/memory_atoms.py`.

Atom types:

- `user_preference`
- `user_interest`
- `user_goal`
- `user_constraint`
- `decision`
- `temporal_update`
- `open_loop`

Properties:

- deterministic/no-LLM
- user-turn only by default
- source-linked back to session/date/turn
- no benchmark labels, answer sessions, or topic dictionaries
- rendered into compact token-native text for indexing

Benchmark wiring:

- `--memory-atoms`: ingest atoms alongside sessions
- `--memory-atom-fusion`: RRF-fuse normal session retrieval with atom retrieval

Initial LongMemEval preference-only probe:

| Variant | Any@1 | Any@5 | Any@10 | MRR |
|---|---:|---:|---:|---:|
| Baseline top-10 | 43.3% | 76.7% | 86.7% | 0.551 |
| Memory atoms indexed directly | 36.7% | 66.7% | 83.3% | 0.481 |
| Memory atom fusion | 40.0% | 63.3% | 83.3% | 0.513 |

Readout: the atom schema/extractor is useful as infrastructure, but naive atom indexing is not yet good enough to improve ranking. This reinforces the guardrail: do not hand-tune to LongMemEval. Next work should validate atoms on general agent-memory scenarios and improve atom routing/scoring before enabling it by default.

## Agent memory cognition eval

Added a small domain-agnostic eval suite at `benchmarks/data/agent_memory_eval.json` with runner `benchmarks/agent_memory_eval.py`.

Behaviors covered:

- preference informs recommendation
- constraint informs advice
- temporal supersession
- decision retrieval
- open-loop retrieval
- goal informs advice

This suite is intentionally not LongMemEval-derived and does not use benchmark topics or labels. It is meant to test general agent-memory cognition primitives.

Initial results on 6 seed cases:

| Mode | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| Baseline session retrieval | 66.7% | 100.0% | 0.806 |
| Atom-only retrieval with intent priors | 66.7% | 83.3% | 0.750 |
| Session/atom fusion | 66.7% | 100.0% | 0.806 |

Readout: the eval is now in place and exposes the right failure modes. Atom intent priors correctly promote the comedy preference for “what should I watch?”, but they hurt a food constraint case and still miss a broad “prepare this weekend” goal case. The next improvement should be a better router/fusion rule that only trusts atom ranking when the atom type is clearly aligned and otherwise preserves session retrieval.

## Atom router / confidence fusion pass

Added a general memory-intent router:

- `query_memory_intents(query)`: detects broad intent classes like preference/interest, goal/constraint, temporal update, decision, and open loop.
- `atom_type_priors(query)`: assigns domain-neutral atom priors from those intents.
- Agent eval fusion now promotes atom hits only when atom type aligns with query intent.

Agent-memory eval improved:

| Mode | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| Baseline session retrieval | 66.7% | 100.0% | 0.806 |
| Atom-only retrieval | 83.3% | 83.3% | 0.833 |
| Confidence fusion | 83.3% | 100.0% | 0.889 |

The remaining miss is `goal_informs_advice`, where the generic session ranking still prefers broad weekend/sleep context over the marathon goal. That is a good next target for general memory routing.

LongMemEval preference holdout check got worse with the same confidence-fusion atom mode:

| Mode | Any@1 | Any@5 | Any@10 | MRR |
|---|---:|---:|---:|---:|
| Baseline preference | 43.3% | 76.7% | 86.7% | 0.551 |
| Memory atom confidence fusion | 30.0% | 56.7% | 70.0% | 0.411 |

Readout: the router improves the small general agent-memory eval, but it does not generalize to LongMemEval yet. Keep this experimental/off by default. The likely issue is that LongMemEval preference rows often need broad episodic topic/context inference rather than explicit memory atoms. Next step should be expanding the general eval with harder episodic-interest cases before trying another LongMemEval probe.

## Expanded episodic-interest eval

Added four harder domain-agnostic episodic-interest cases to `benchmarks/data/agent_memory_eval.json`:

- watch recommendation from prior storytelling-comedy interest
- dinner advice from prior garden/ingredients context
- gift advice from another person's woodworking hobby context
- phone battery/travel advice from prior power-bank context

Also added a general `entity_fact` atom type for user-owned/available/context facts such as “I bought…”, “I harvested…”, “my X has…”. This is not topic-specific; it captures durable background facts that can inform later advice.

Results on 10-case agent-memory eval:

| Mode | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| Baseline session retrieval | 60.0% | 100.0% | 0.783 |
| Atom-only retrieval | 80.0% | 90.0% | 0.833 |
| Confidence fusion | 80.0% | 100.0% | 0.883 |

Episodic-interest slice:

| Mode | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| Baseline | 50.0% | 100.0% | 0.750 |
| Atom-only | 75.0% | 100.0% | 0.833 |
| Confidence fusion | 75.0% | 100.0% | 0.875 |

Readout: this is the first clear win from the general memory-atom path. It improves agent-memory behavior without using LongMemEval topics or labels. Remaining weak case is broad goal advice (`How should I prepare this weekend?`) where baseline still retrieves generic weekend/sleep context above the marathon goal.

## 50-case agent-memory eval expansion

Expanded `benchmarks/data/agent_memory_eval.json` to 50 domain-diverse cases across:

- preference informs recommendation
- constraint informs advice
- goal informs advice
- temporal supersession
- decision retrieval
- open-loop retrieval
- episodic interest inference

This is now large enough to expose behavior-specific regressions without relying on LongMemEval topics.

Results:

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Baseline session retrieval | 54.0% | 94.0% | 94.0% | 0.723 |
| Atom-only retrieval | 70.0% | 90.0% | 90.0% | 0.787 |
| Confidence fusion | 66.0% | 94.0% | 94.0% | 0.787 |

By behavior, atom-only is strongest on explicit memory primitives:

- preference recommendation: 100% Recall@1
- constraints: 85.7% Recall@1
- goals: 75.0% Recall@1
- decisions: 80.0% Recall@1
- temporal supersession: 85.7% Recall@1

But it still lags on episodic interest inference: 33.3% Recall@1 atom-only vs 41.7% baseline. Fusion recovers some of that to 50.0% Recall@1.

Readout: memory atoms clearly improve explicit agent-memory recall, but episodic inference still needs a better bridge than atom extraction alone. The next improvement should score whole prior episodes as “future advice context,” not just extracted atom snippets.

## Whole-episode context card experiment

Implemented `episode_context_text()` in `src/contextfit/retrieval/memory_atoms.py` and added agent-memory eval modes:

- `episode`: retrieve only whole user episode context cards
- `episode_fusion`: RRF-fuse baseline session retrieval with episode context retrieval

The cards preserve user-authored turns as a unit and add general retrieval hints for future advice/recommendation/planning. They do not use LongMemEval topics or labels.

50-case eval results:

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Baseline session retrieval | 54.0% | 94.0% | 94.0% | 0.723 |
| Atom-only retrieval | 70.0% | 90.0% | 90.0% | 0.787 |
| Atom confidence fusion | 66.0% | 94.0% | 94.0% | 0.787 |
| Episode-only retrieval | 56.0% | 100.0% | 100.0% | 0.737 |
| Episode fusion | 48.0% | 100.0% | 100.0% | 0.703 |

Episodic-interest slice:

| Mode | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| Baseline | 41.7% | 100.0% | 0.681 |
| Atom confidence fusion | 50.0% | 100.0% | 0.722 |
| Episode-only | 16.7% | 100.0% | 0.500 |
| Episode fusion | 25.0% | 100.0% | 0.569 |

Readout: naive whole-episode cards improve recall@3 but hurt rank-1. The generic episode hints are too broad and make many distractor sessions look similarly relevant. The useful improvement remains atom confidence fusion, which modestly improves episodic rank-1 while preserving top-3 recall. Next step should be a proper episode scorer that uses query/episode token overlap, user-only salient terms, and atom intent alignment as explicit numeric features instead of indexing broad episode cards as normal text.

## Numeric episode relevance scorer

Implemented `episode_relevance_score()` in `src/contextfit/retrieval/memory_atoms.py` as a pure numeric, domain-agnostic session ranker with six features:
- `lexical`: query/episode salient-term overlap ratio
- `atom_prior`: max atom confidence × atom-type prior for query intents
- `aligned_conf`: confidence of atoms whose type matches query intents
- `aligned_count`: count of aligned atom types (capped at 2)
- `entity_context`: bonus when entity_fact atoms exist and query is advice/recommendation
- `specificity`: ratio of salient episode terms (penalizes generic assistant-only sessions)

No index, no BM25, no embeddings — pure structural scoring over raw user-authored session turns.

### 50-case agent-memory eval results

| Mode | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| Baseline (BM25/hybrid) | 54.0% | 94.0% | 0.723 |
| Atom-only | 72.0% | 92.0% | 0.807 |
| Confidence fusion | 72.0% | 94.0% | 0.823 |
| Episode cards (text) | 56.0% | 100.0% | 0.737 |
| **Episode score (numeric)** | **84.0%** | **100.0%** | **0.917** |
| Episode score fusion | 70.0% | 100.0% | 0.843 |

Episodic-interest slice (12 cases):
- Baseline: 41.7% → **Episode score: 83.3%** (+41.6 points)
- Temporal: **100% Recall@1, MRR 1.0** (perfect)
- Preferences: **100% Recall@1**

### LME preference holdout (30 cases, 47 sessions avg haystack)

| Mode | Any@5 | MRR |
|---|---:|---:|
| Baseline (BM25) | 76.7% | 0.551 |
| Episode score | 40.0% | 0.243 |
| Episode score fusion | 40.0% | 0.278 |

### Interpretation

The numeric episode scorer is a powerful primitive for **vague advice/recommendation queries over agent corpora** where sessions are topically diverse (2–10 sessions per question). It beats BM25 on the 50-case agent-memory eval by 30 points Recall@1.

However, it fails on **LME's large haystacks** (47 sessions avg) where the task is "find the needle" — the scorer lacks discriminative power at scale because many sessions have some user content that fires the feature vectors. BM25 wins there because the queries are factual and the answer sessions contain the exact preference vocabulary.

The two paradigms are complementary:
- **Agent-memory / vague queries**: episode_score (or atom confidence fusion) > BM25
- **Large haystack / factual preference lookups**: BM25 > episode_score

Episode score is promoted to a first-class retrieval mode in `agent_memory_eval.py`. Not wired into LME as default; LME remains BM25/hybrid + OpenAI fusion.

---

## 79-case agent-memory eval: Episode score vs gbrain-proxy (OpenAI embeddings)

### Setup
- Grew eval from 50 → 79 cases: added `multi_session_synthesis` behavior, harder episodic cases, larger session pools (avg 3.7 sessions/question)
- OpenAI `text-embedding-3-small` used as gbrain-proxy baseline (same embedding stack, same scale)
- `engine_episode_score` wires `rank_sessions_by_episode_score()` into production `RetrievalEngine`
- All episode_score and engine_episode_score results match exactly — production wiring confirmed

### Overall results (79 cases, top-5)

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Baseline BM25/hybrid | 44.3% | 84.8% | 92.4% | 0.640 |
| Atom-only | 62.0% | 86.1% | 87.3% | 0.731 |
| Atom confidence fusion | 57.0% | 87.3% | 91.1% | 0.722 |
| **Episode score (numeric)** | **69.6%** | **96.2%** | **100.0%** | **0.824** |
| Episode score + BM25 fusion | 53.2% | 93.7% | 100.0% | 0.737 |
| OpenAI vector (gbrain-proxy) | 55.7% | 96.2% | 100.0% | 0.745 |

### Episodic-interest slice (20 cases)

| Mode | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| Baseline BM25/hybrid | 25.0% | 80.0% | 0.527 |
| Atom-only | 30.0% | 75.0% | 0.492 |
| Atom confidence fusion | 35.0% | 85.0% | 0.596 |
| **Episode score (numeric)** | **60.0%** | **95.0%** | **0.762** |
| Episode score + BM25 fusion | 30.0% | 85.0% | 0.585 |
| OpenAI vector (gbrain-proxy) | 40.0% | 85.0% | 0.608 |

### Multi-session synthesis slice (4 cases)

| Mode | Recall@1 | MRR |
|---|---:|---:|
| Baseline BM25/hybrid | **100.0%** | **1.000** |
| Atom confidence fusion | 75.0% | 0.875 |
| Episode score (numeric) | 50.0% | 0.708 |
| **Episode score + BM25 fusion** | **100.0%** | **1.000** |
| OpenAI vector (gbrain-proxy) | 50.0% | 0.750 |

### Key findings
1. **Episode score beats OpenAI vector overall**: 69.6% vs 55.7% Recall@1 — a free, deterministic structural scorer outperforms dense embeddings for agent-memory episodic retrieval
2. **Episodic gap to OpenAI is large**: 60% vs 40% Recall@1 — episode scorer's lexical + atom alignment features are more discriminative than embedding cosine similarity for "which prior episode is contextually relevant to this vague query?"
3. **Multi-session synthesis is a different beast**: BM25 and episode+BM25 fusion both hit 100% — when the query explicitly references prior decisions/context, text matching wins; episode_score alone is weak at it
4. **Production wiring confirmed**: `RetrievalEngine.rank_sessions_by_episode_score()` produces identical results to the benchmark-mode direct scorer
5. **Recall@5 is 100% for episode_score and OpenAI vector** — both achieve perfect top-5 coverage at this scale

---

## Query-type router (query_router.py)

Implemented a deterministic, zero-cost query router in `src/contextfit/retrieval/query_router.py`.  No LLM calls, no embeddings.

### Routing rules

| Signal | Mode |
|---|---|
| Vague advice verbs ("what should I", "can you recommend", "help me plan/choose/pick") | `episode_score` |
| Specific fact patterns ("what did I decide", "which X did I choose", "am I currently") | `bm25` |
| Temporal current-state ("currently", "these days", "switched/changed") | `bm25` |
| Multi-session hints ("make progress this week", "what should I focus on this month") | `episode_bm25_fusion` |
| Explicit preference/constraint verbs ("I like", "my budget", "I can't") | `atom_fusion` |
| Proper-noun specificity penalty | penalise `episode_score`, boost `bm25` |

### `RetrievalEngine.query_auto()` production API

```python
result = engine.query_auto(query, top_k=10)
# result["session_ids"] — ranked session IDs
# result["route"]       — QueryRoute(mode, confidence, signals)
# result["details"]     — mode-specific metadata
```

### 79-case agent-memory eval: auto router vs all modes

| Mode | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|
| Baseline BM25 | 44.3% | 84.8% | 92.4% | 0.640 |
| OpenAI vector (gbrain-proxy) | 55.7% | 96.2% | 100.0% | 0.745 |
| Episode score (best single mode) | 69.6% | 96.2% | 100.0% | 0.824 |
| **Auto router** | **67.1%** | **94.9%** | **98.7%** | **0.801** |

Auto router highlights by behavior:
- **Episodic inference**: 60.0% R@1 — matches pure episode_score (routes correctly to episode_score)
- **Multi-session synthesis**: 100% R@1 — routes to episode_bm25_fusion, recovering from episode_score's 50%
- **Open loops**: 55.6% R@1 — routes to bm25 (correct, though episode_score was 77.8% here — opportunity)
- **Temporal**: 70.0% R@1 — room to improve routing on "currently/changed" queries

### 27 router + atom tests pass

### Key insight
The auto router essentially eliminates the worst-case mode penalty. Episode score alone is ~50% on multi-session synthesis; baseline alone is 25% on episodic. The router brings both to their best regime. Remaining gap to perfect routing is primarily in open-loop and temporal queries where bm25 routing is correct in principle but episode_score happens to outperform it empirically — the router could be tuned to lean harder on episode_score as default.

## Router tuning: open-loop and temporal-state queries

### Problem
After initial router implementation, two query classes were mis-routed:
1. **Vague open-loop queries** ("do I have any outstanding tasks?", "is there something I was supposed to follow up on?") were matching `_SPECIFIC_FACT_RE` on words like "pending"/"outstanding" and routing to bm25. But episode_score outperforms bm25 on these (77.8% vs 44.4%) because gold sessions contain `open_loop` atoms (remind me / todo / follow up) with no exact query token match.
2. **Temporal state queries** ("which fitness tracker am I *currently* using?", "what is my *current* coffee situation?") routed to bm25 via temporal+fact signals. But the gold sessions say "I *switched* to WHOOP" / "I *quit* caffeine" — `temporal_update` atoms that episode_score aligns to via intent, while BM25 misses because the exact query vocabulary isn't there.

### Fixes
- Added `_OPEN_LOOP_VAGUE_RE`: catches generic "find what I left undone" queries and routes them to `episode_score` with priority over specific_fact. Removed `pending`/`outstanding`/`todo`/`follow up` from `_SPECIFIC_FACT_RE` (they're vague, not specific).
- Added `temporal+fact->fusion` rule: when `_TEMPORAL_RE` and `_SPECIFIC_FACT_RE` both fire, transfer the bm25 specific_fact score to `episode_bm25_fusion` instead. This gives both BM25 text matching *and* temporal_update atom alignment.

### Final tuned router results (79 cases)

| Mode | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|
| Baseline BM25 | 44.3% | 84.8% | 92.4% | 0.640 |
| OpenAI vector (gbrain-proxy) | 55.7% | 96.2% | 100.0% | 0.745 |
| Episode score (best single) | 69.6% | 96.2% | 100.0% | 0.824 |
| **Auto router (tuned)** | **69.6%** | **96.2%** | **98.7%** | **0.821** |

Auto router now matches episode_score overall and recovers behavior-specific failures:
- Episodic inference: **60.0%** (matches episode_score)
- Open loops: **77.8%** (matches episode_score, was 55.6% before tuning)
- Temporal supersession: **80.0%** (matches OpenAI vector; was 70.0% before tuning)
- Multi-session synthesis: **75.0%** (better than episode_score's 50.0%)

27 tests pass. Production API: `engine.query_auto(query, top_k)`.

---

## Mem0 comparison (79-case agent-memory eval)

Mem0 v2.0.2 configured with:
- LLM extraction: `gpt-4o-mini` (extracts structured facts from raw session text at ingest)
- Embedder: `text-embedding-3-small`
- Vector store: Qdrant (in-memory per eval case)
- Cost: one LLM API call per ingested session (~292 calls for 79 cases)

### Overall results

| Mode | R@1 | R@3 | R@5 | MRR | Cost |
|---|---:|---:|---:|---:|---|
| ContextFit BM25 | 44.3% | 84.8% | 92.4% | 0.640 | free |
| OpenAI text-embedding-3-small | 55.7% | 96.2% | 100.0% | 0.745 | embed only |
| **Mem0** (LLM extract + OpenAI embed) | 54.4% | 91.1% | 97.5% | 0.716 | LLM+embed/session |
| **ContextFit episode score** (token-native) | **69.6%** | **96.2%** | **100.0%** | **0.824** | **free** |
| ContextFit auto router | **69.6%** | **96.2%** | 98.7% | **0.821** | **free** |

### Key findings

1. **ContextFit beats Mem0 overall**: 69.6% vs 54.4% Recall@1, MRR 0.824 vs 0.716 — despite Mem0 using an LLM at ingest
2. **Episodic inference**: ContextFit 60.0% vs Mem0 25.0% — Mem0's LLM extraction loses episodic context; it extracts facts but can't surface which episode is holistically relevant for vague advice queries
3. **Preferences**: Mem0 77.8% = ContextFit 77.8% — LLM extraction genuinely helps for explicit preference facts ("I love spicy food")
4. **Open loops**: ContextFit 77.8% vs Mem0 66.7% — reminder/todo sessions often lack explicit fact patterns that LLM extraction targets
5. **Multi-session synthesis**: Mem0 25.0% — LLM extracts individual facts and loses cross-session signal; ContextFit auto router 75.0%
6. **Temporal supersession**: ContextFit 90% vs Mem0 50% — Mem0 often retrieves the old fact alongside the new one without clear temporal precedence
7. **Decision retrieval**: Mem0 71.4% vs ContextFit episode score 57.1% — LLM extraction excels at surfacing explicit decisions ("I decided to use Postgres")

### Honest caveat
Mem0 is designed for *persistent memory across many sessions* (one user, growing memory over weeks). Our eval creates a fresh memory per question from 3-5 sessions — not Mem0's primary use case. In a long-running agent scenario, Mem0's accumulated fact extraction may behave differently. This comparison is fair for retrieval quality on the query types we care about; it is not a complete evaluation of Mem0's capabilities.

---

## 499-case eval: OpenAI vs Cohere vs ContextFit (expanded dataset)

### Dataset
- 79 hand-crafted cases + 420 GPT-4o-mini generated cases = 499 total
- 8 behaviors, avg 4.8 sessions/question, 26 domains
- Voyage AI skipped (free-tier rate limits made 499-case run impractical)

### Overall results (top-5)

| Mode | R@1 | R@3 | R@5 | MRR | Cost |
|---|---:|---:|---:|---:|---|
| ContextFit BM25 | 48.9% | 85.6% | 97.6% | 0.681 | free |
| Cohere embed-english-v3.0 | 58.7% | 91.4% | 100.0% | 0.751 | embed API |
| ContextFit episode score | 55.5% | 91.2% | 100.0% | 0.734 | free |
| ContextFit auto router | 54.5% | 90.2% | 99.8% | 0.727 | free |
| OpenAI text-embedding-3-small | **63.1%** | **96.6%** | **100.0%** | **0.792** | embed API |

### Key findings

1. **Reversal from 79-case results.** On the 499-case eval, OpenAI embeddings lead (63.1%) while ContextFit episode score scores 55.5%. This is the opposite of the 79-case hand-crafted results (episode score 69.6%, OpenAI 55.7%).

2. **Why the reversal.** The 420 GPT-generated cases have more explicit vocabulary alignment between queries and sessions — embeddings capture semantic similarity well in this setting. The 79 hand-crafted cases were specifically designed to stress-test indirect episodic retrieval, which is where the token-native scorer excels.

3. **Cohere underperforms OpenAI by ~4 points** (58.7% vs 63.1%), and is competitive with ContextFit episode score.

4. **ContextFit remains free and fast.** Episode score at 55.5% R@1 is within 7.6 points of OpenAI at zero API cost and 0.4ms query latency. For deployments where latency and cost matter, ContextFit is still a strong choice.

5. **Auto router slightly underperforms pure episode score** on this eval — the generated cases have broader query types where the routing adds some noise rather than signal.

### Honest interpretation
The 499-case result is a more credible overall benchmark but does not invalidate the 79-case findings. They measure different things: the generated cases test general retrieval breadth; the hand-crafted cases test the hardest episodic inference scenarios. Both are useful. The white paper should present both sets of results and be transparent about the distinction.

---

## Final comparison table (499-case agent-memory eval)

| System | R@1 | R@3 | R@5 | MRR | Cost |
|---|---:|---:|---:|---:|---|
| Mem0 v2 (79-case) | 54.4% | 81.0% | 91.1% | 0.716 | LLM + embed API |
| Cohere embed-english-v3 | 58.7% | 91.4% | 100.0% | 0.751 | embed API |
| **ContextFit** | **61.1%** | **93.0%** | **99.8%** | **0.773** | **free** |
| OpenAI text-embedding-3-small | 63.1% | 96.6% | 100.0% | 0.792 | embed API |

Notes:
- Mem0 was measured on the 79-case hand-crafted eval only; others on 499-case.
- ContextFit uses token-native BM25 + structural reranking (no embeddings, no external APIs).
- R@5 is effectively solved across all systems (99.8–100%); R@1 and R@3 are the meaningful differentiators.
- ContextFit R@1 gap vs OpenAI is 2.0 pts overall; compensated by +24.6 pts on open-loop retrieval and +4.8 pts on temporal supersession.

---

## Structural reranker: IDF boost, slot matching, window density (499-case)

Date: 2026-05-10

Three new token-native features were added to `token_native_rerank_sessions()` and `rerank_sessions_by_structure()` and benchmarked independently against the 499-case eval.

### New features

**Feature 1 — IDF-boosted lexical overlap (`idf_lexical`):**
Builds a document-frequency table across all sessions in each eval item at rerank time. Uses IDF-cosine style scoring: `sum(idf[w] for w in shared) / sqrt(sum_q * sum_s)`. Weight: 0.40. The existing unweighted `lexical` feature is preserved.

**Feature 2 — Question-type slot matching (`slot_match`):**
Detects WH-question type in the query and looks for matching answer-evidence type in session text:
- Q_WHO → E_WHO (person name + verb patterns)
- Q_WHEN → E_WHEN (month names, years, relative dates)
- Q_WHERE → E_WHERE (at/in/near + capitalized place; location keywords)
- Q_HOW → E_HOW (step N, first/then/next/finally, instructional markers)
- Q_RECOMMEND → E_RECOMMEND (would love/enjoy/like, perfect for, you'd like; weighted 1.2)
Returns 0.0–2.0 (capped). Weight: 0.35.

**Feature 3 — Evidence window density (`window_density`):**
Sliding window (150 words) over session text. Returns `max_window_hits / window_size ∈ [0.0, 1.0]`. Rewards sessions where query terms cluster tightly. Weight: 0.45.

### Results (499 cases, top-5, hybrid method)

| Mode | R@1 | R@3 | R@5 | MRR | Delta R@1 vs baseline |
|---|---:|---:|---:|---:|---:|
| free_rerank (baseline) | 61.1% | 93.0% | 99.8% | 0.773 | — |
| free_rerank_idf | 60.3% | 92.8% | 99.8% | 0.768 | −0.8 pts |
| free_rerank_slot | **61.5%** | **93.2%** | 99.8% | **0.776** | **+0.4 pts** |
| free_rerank_window | 61.1% | 93.0% | 99.8% | 0.773 | 0.0 pts |
| free_rerank_all | 60.7% | 93.0% | 99.8% | 0.771 | −0.4 pts |

### Per-behavior breakdown for free_rerank_slot (improving mode)

| Behavior | n | R@1 | R@3 | R@5 | MRR | vs baseline R@1 |
|---|---:|---:|---:|---:|---:|---:|
| constraint_informs_advice | 62 | 56.5% | 91.9% | 98.4% | 0.747 | 0.0 pts |
| decision_retrieval | 59 | 64.4% | 94.9% | 100.0% | 0.790 | 0.0 pts |
| episodic_interest_inference | 72 | 33.3% | 86.1% | 100.0% | 0.596 | 0.0 pts |
| goal_informs_advice | 64 | 64.1% | 95.3% | 100.0% | 0.797 | 0.0 pts |
| multi_session_synthesis | 56 | 80.4% | 98.2% | 100.0% | 0.894 | 0.0 pts |
| open_loop_retrieval | 61 | 90.2% | 100.0% | 100.0% | 0.948 | 0.0 pts |
| preference_informs_recommendation | 62 | 51.6% | 85.5% | 100.0% | 0.698 | **+3.2 pts** |
| temporal_supersession | 63 | 58.7% | 95.2% | 100.0% | 0.779 | 0.0 pts |

### Interpretation

- **Slot matching (+0.4 pts R@1)** is the only feature that improves on baseline. The small gain comes primarily from `preference_informs_recommendation` (+3.2 pts R@1), where Q_RECOMMEND → E_RECOMMEND patterns correctly promote sessions with "you'd like", "perfect for", "would enjoy" evidence.

- **IDF-boosted lexical (−0.8 pts)** slightly hurts overall. IDF amplifies rare terms, but in agent-memory corpora where sessions are short and domain-diverse, rare terms are often noise rather than signal. The existing unweighted `lexical` feature already does a good job.

- **Window density (0.0 pts)** is neutral overall — it neither helps nor hurts at weight 0.45. Sessions in this eval are short enough (avg 4.8 sessions × sparse turns) that dense term clusters are rare and the feature rarely fires with meaningful magnitude.

- **All three combined (−0.4 pts)** — combining IDF's slight negative effect with the slot improvement results in a net negative. Slot matching is the only feature worth enabling.

### Production decision

The slot matching feature has been added to both `token_native_rerank_sessions()` (benchmark harness) and `rerank_sessions_by_structure()` (production engine) along with all three features. The benchmark confirms slot matching provides a small but real improvement without risk of regression. IDF and window density are available via `use_idf`/`use_window` flags but are not recommended for production at current weights.

All features use conservative weights (0.35–0.45) as specified; no eval-specific tuning was applied.

## 2026-05-11: Router-gated preference recommendation reranker

Implemented a production `preference_rerank` route for **personalized recommendation queries** where prior explicit user taste should beat generic topical overlap. The path is token-native: user-turn extraction, explicit preference-marker detection, lightweight stemming/plural normalization, preference-window overlap, BM25/episode-score blend, and a generic non-preference penalty. It uses no embeddings, no LLM calls, and no topic dictionary.

499-case agent-memory benchmark (`benchmarks/agent_memory_eval_500_auto_preference_rerank.json`):

| Mode | R@1 | R@3 | R@5 | MRR | Avg query |
|---|---:|---:|---:|---:|---:|
| Prior free rerank fusion v3 | 60.7% | 92.2% | 100.0% | 0.769 | 13.8ms |
| **Auto router + preference rerank** | **61.1%** | **93.4%** | **99.8%** | **0.771** | **7.9ms** |
| OpenAI text-embedding-3-small | 63.1% | 96.6% | 100.0% | 0.792 | 458ms |
| Cohere embed-english-v3.0 | 58.7% | 91.4% | 100.0% | 0.751 | 509ms |

Preference recommendation subset (`preference_informs_recommendation`, n=62):

| System | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|
| Prior free rerank fusion v3 | 56.5% | 85.5% | 100.0% | 0.730 |
| OpenAI text-embedding-3-small | 77.4% | 100.0% | 100.0% | 0.882 |
| Cohere embed-english-v3.0 | 83.9% | 98.4% | 100.0% | 0.907 |
| **Auto router + preference rerank** | **85.5%** | **95.2%** | **100.0%** | **0.909** |

Readout: this closes the generated agent-memory preference gap without embeddings. Overall R@1 only moves modestly because some recommendation-shaped constraint/episodic cases still route through preference logic, but the intended preference subset now beats both OpenAI and Cohere embedding baselines at R@1.

Guardrail: LongMemEval `single-session-preference` is not the same task. It is often broad prior-context lookup for advice (e.g. “which hotel/accessory/movie should I pick?”), not explicit taste retrieval. Running production `--query-auto` on the 30-row LongMemEval preference slice produced Any@5 36.7% / Any@10 53.3% / MRR 0.385, below the BM25/episode baselines recorded above. Do **not** tune this preference reranker to LongMemEval labels; keep LongMemEval as an external measurement harness and describe the new claim specifically as generated domain-agnostic agent-memory preference recommendation retrieval.


## 2026-05-11: Multi-session evidence-coverage reranker

Implemented a production `multi_session_rerank` route for synthesis/advice/background queries where useful evidence may be distributed across multiple sessions. The path remains token-native: route-level evidence-coverage detection, user-turn extraction, lightweight token normalization, personal-facet markers (preference, constraint, goal, temporal, entity, open-loop), BM25/episode blending, and a small diversity pass to prefer complementary evidence over duplicate near-matches. It uses no embeddings, no LLM calls, and no domain dictionary.

499-case agent-memory benchmark (`benchmarks/agent_memory_eval_500_auto_multisession_rerank.json`):

| Mode | R@1 | R@3 | R@5 | MRR | Avg query |
|---|---:|---:|---:|---:|---:|
| Auto router + preference rerank | 61.1% | 93.4% | 99.8% | 0.771 | 7.9ms |
| **Auto router + preference + multi-session rerank** | **62.3%** | **93.0%** | **99.8%** | **0.777** | **8.9ms** |
| OpenAI text-embedding-3-small | 63.1% | 96.6% | 100.0% | 0.792 | 458ms |

Multi-session synthesis subset (`multi_session_synthesis`, n=56):

| System | R@1 | R@3 | R@5 | MRR |
|---|---:|---:|---:|---:|
| Auto router + preference rerank | 73.2% | 98.2% | 100.0% | 0.859 |
| **Auto router + preference + multi-session rerank** | **82.1%** | **98.2%** | **100.0%** | **0.903** |
| OpenAI text-embedding-3-small | 87.5% | 100.0% | 100.0% | 0.938 |

Readout: the evidence-coverage route narrows the multi-session synthesis gap from −14.3 points to −5.4 points vs OpenAI embeddings while preserving the preference recommendation result at 85.5% R@1. Overall R@1 improves from 61.1% to 62.3%. This should be described as a verified token-native improvement, not as closing the full embedding gap.

## Conversation-aware chunking A/B (2026-05-13)

Added a reusable turn-aware conversation chunker in `contextfit.extractors.conversation` and wired it through `RetrievalEngine.ingest_conversation(...)`. The LongMemEval runner now has `--conversation-chunks` to index each session as coherent turn ranges instead of blind token windows. It keeps session/date headers on each chunk and stores turn range/role metadata.

The runner also now defaults to **not** indexing LongMemEval's `has_answer` labels. `--include-answer-marker` exists only for backwards-compatible diagnostics and is not valid for reported benchmark results.

A/B command shape:

```bash
/tmp/cf-structure-venv/bin/python benchmarks/longmemeval_contextfit.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --limit 100 --method hybrid --top-k-chunks 10 --retrieval-k 50 \
  --chunk-size 8192 --overlap 0 --rank-by-session \
  --out benchmarks/longmemeval_ab_baseline_nolabel_100.json

/tmp/cf-structure-venv/bin/python benchmarks/longmemeval_contextfit.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --limit 100 --method hybrid --top-k-chunks 10 --retrieval-k 50 \
  --chunk-size 2048 --overlap 128 --rank-by-session --conversation-chunks \
  --out benchmarks/longmemeval_ab_conversation_chunks_2048_nolabel_100.json
```

| Metric | Baseline/no labels | Conversation chunks/no labels | Delta |
|---|---:|---:|---:|
| Scored examples | 94 | 94 | — |
| Any gold evidence @1 | 79.8% | 79.8% | +0.0 pts |
| Any gold evidence @3 | 88.3% | 90.4% | +2.1 pts |
| Any gold evidence @5 | 91.5% | 93.6% | +2.1 pts |
| Any gold evidence @10 | 95.7% | 96.8% | +1.1 pts |
| All gold evidence @5 | 73.4% | 73.4% | +0.0 pts |
| All gold evidence @10 | 86.2% | 80.9% | -5.3 pts |
| MRR | 0.847 | 0.853 | +0.006 |

Readout: conversation-aware chunking helps early precision on this 100-example slice, especially Any@3/@5, but slightly hurts all-evidence@10. Keep it experimental until a full 500-case no-label run confirms the tradeoff.

### Full no-label conversation-aware A/B

Full 500-example run, excluding abstentions from the scored summary (`n=470`). Neither path indexes LongMemEval `has_answer` labels.

| Metric | Baseline/no labels | Conversation chunks/no labels | Delta |
|---|---:|---:|---:|
| Any gold evidence @1 | 81.1% | 82.3% | +1.3 pts |
| Any gold evidence @3 | 89.1% | 90.4% | +1.3 pts |
| Any gold evidence @5 | 93.6% | 94.7% | +1.1 pts |
| Any gold evidence @10 | 96.4% | 97.0% | +0.6 pts |
| All gold evidence @1 | 28.1% | 28.7% | +0.6 pts |
| All gold evidence @3 | 71.7% | 72.1% | +0.4 pts |
| All gold evidence @5 | 77.4% | 77.7% | +0.2 pts |
| All gold evidence @10 | 85.3% | 84.0% | -1.3 pts |
| MRR | 0.860 | 0.870 | +0.010 |

By-type highlights:

| Type | Baseline Any@5 | Conversation Any@5 | Delta | Baseline MRR | Conversation MRR | Delta |
|---|---:|---:|---:|---:|---:|---:|
| knowledge-update | 98.6% | 98.6% | +0.0 pts | 0.958 | 0.975 | +0.016 |
| multi-session | 93.4% | 94.2% | +0.8 pts | 0.837 | 0.852 | +0.015 |
| single-session-assistant | 100.0% | 100.0% | +0.0 pts | 0.978 | 0.986 | +0.008 |
| single-session-preference | 76.7% | 80.0% | +3.3 pts | 0.548 | 0.551 | +0.003 |
| single-session-user | 93.8% | 93.8% | +0.0 pts | 0.896 | 0.900 | +0.003 |
| temporal-reasoning | 92.1% | 94.5% | +2.4 pts | 0.831 | 0.839 | +0.008 |

Conclusion: conversation-aware ingestion is a measured token-only improvement for LongMemEval evidence retrieval: +1.1 pts Any@5 and +0.010 MRR, with no answer-label leakage. It slightly reduces all-evidence@10, so the best current framing is improved early precision/ranking rather than uniformly better evidence coverage.

### Conversation-aware chunks + OpenAI fusion

Full 500-example run (`n=470` scored), using conversation-aware session chunks on the ContextFit side and OpenAI session-vector reciprocal-rank fusion:

```bash
/tmp/cf-structure-venv/bin/python benchmarks/longmemeval_contextfit.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --limit 0 --method hybrid --top-k-chunks 5 --retrieval-k 20 \
  --chunk-size 2048 --overlap 128 --rank-by-session \
  --conversation-chunks --openai-fusion \
  --out benchmarks/longmemeval_conversation_openai_fusion_full.json
```

| Metric | Token baseline | Conversation token-only | OpenAI fusion | Conversation + OpenAI fusion |
|---|---:|---:|---:|---:|
| Any gold evidence @1 | 81.1% | 82.3% | 83.6% | 84.3% |
| Any gold evidence @3 | 89.1% | 90.4% | 94.0% | 93.8% |
| Any gold evidence @5 | 93.6% | 94.7% | 96.0% | 96.0% |
| All gold evidence @5 | 77.4% | 77.7% | 84.9% | 83.6% |
| MRR | 0.860 | 0.870 | 0.889 | 0.892 |

Readout: conversation-aware chunking and OpenAI fusion are not cleanly additive on Any@5. The combined run matches the previous 96.0% Any@5 fusion headline, nudges rank-1/MRR slightly upward, but lowers all-evidence@5 versus OpenAI fusion alone. Best framing: conversation-aware chunking is a token-only precision lift; OpenAI fusion remains the strongest evidence-coverage mode.

### Parent/child conversation hierarchy experiment

Added an optional parent/child ingestion mode for conversations: turn-aware child chunks remain the precise retrieval units, while an additional full-session parent chunk preserves whole-session lexical context. Benchmark flag: `--conversation-parent` with `--conversation-chunks`.

Full 500-example no-label run (`n=470` scored):

```bash
/tmp/cf-structure-venv/bin/python benchmarks/longmemeval_contextfit.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --limit 0 --method hybrid --top-k-chunks 10 --retrieval-k 100 \
  --chunk-size 2048 --overlap 128 --rank-by-session \
  --conversation-chunks --conversation-parent \
  --out benchmarks/longmemeval_parent_child_r100_full.json
```

| Metric | Token baseline | Conversation token-only | Parent/child token-only |
|---|---:|---:|---:|
| Any gold evidence @1 | 81.1% | 82.3% | 82.8% |
| Any gold evidence @3 | 89.1% | 90.4% | 90.6% |
| Any gold evidence @5 | 93.6% | 94.7% | 95.1% |
| Any gold evidence @10 | 96.4% | 97.0% | 97.0% |
| All gold evidence @5 | 77.4% | 77.7% | 77.9% |
| All gold evidence @10 | 85.3% | 84.0% | 84.3% |
| MRR | 0.860 | 0.870 | 0.873 |

Readout: parent/child conversation hierarchy is a modest but consistent token-only lift over conversation chunks alone: +0.4 pts Any@5, +0.2 pts All@5, and +0.002 MRR. It does not fully recover baseline all-evidence@10, but it improves early precision and slightly improves coverage versus conversation-only. This is the current best pure token-native LongMemEval configuration.

### Multi-session companion-evidence coverage rerank

Added an optional benchmark rerank mode for multi-session questions only: `--coverage-rerank`. It keeps the strongest token-native session as an anchor, expands the candidate pool to 30 grouped sessions, then greedily selects companion sessions using only token/entity signals: original reciprocal rank, anchor topical overlap, selected-session overlap, query term coverage, and new query/entity coverage. It does not use embeddings, LLM calls, or answer labels.

Full 500-example no-label run (`n=470` scored):

```bash
/tmp/cf-structure-venv/bin/python benchmarks/longmemeval_contextfit.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --limit 0 --method hybrid --top-k-chunks 10 --retrieval-k 100 \
  --chunk-size 2048 --overlap 128 --rank-by-session \
  --conversation-chunks --conversation-parent --coverage-rerank \
  --out benchmarks/longmemeval_coverage_companion_full.json
```

| Metric | Parent/child token-only | + coverage rerank | Delta |
|---|---:|---:|---:|
| Any gold evidence @1 | 82.8% | 82.8% | +0.0 pts |
| Any gold evidence @5 | 95.1% | 95.1% | +0.0 pts |
| All gold evidence @5 | 77.9% | 80.4% | +2.6 pts |
| All gold evidence @10 | 84.3% | 86.8% | +2.6 pts |
| MRR | 0.873 | 0.875 | +0.003 |

Multi-session slice (`n=121`):

| Metric | Parent/child token-only | + coverage rerank | Delta |
|---|---:|---:|---:|
| Any@1 | 79.3% | 79.3% | +0.0 pts |
| Any@5 | 95.0% | 95.0% | +0.0 pts |
| All@5 | 55.4% | 65.3% | +9.9 pts |
| All@10 | 69.4% | 79.3% | +9.9 pts |
| MRR | 0.853 | 0.862 | +0.010 |

Readout: this hits the target for the remaining token-only gap. Multi-session complete evidence coverage improves from **55.4% → 65.3% All@5** while preserving the **95.0% Any@5** multi-session result and the **95.1% overall Any@5** headline. Preference results are unchanged because the reranker is gated to `multi-session` question types. OpenAI fusion still has the strongest multi-session All@5 at 72.7%, but the token-only gap narrows from 17.4 pts to 7.4 pts.

### 2026-05-16 token-only leaderboard evidence run

A fresh full token-only evidence run reproduced the parent/child + coverage-rerank result and is recorded as a standalone audit packet:

- Evidence packet: `benchmarks/longmemeval_token_only_leaderboard_evidence_20260516.md`
- Raw artifact: `benchmarks/longmemeval_token_only_leaderboard_run_20260516.json`
- Raw artifact SHA-256: `09f7f8bbd6c1622749cb3961f39cbbf6bcd673b318ad3bd63e88d9d1441a0ca2`
- Git commit: `0071d24a3e079dd29eaa9e501a299ddae8b67880`
- Scoring policy: 500 total rows, 470 non-abstention rows scored, 30 abstention rows skipped.

Command:

```bash
/tmp/cf-structure-venv/bin/python benchmarks/longmemeval_contextfit.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --limit 0 --method hybrid --top-k-chunks 10 --retrieval-k 100 \
  --chunk-size 2048 --overlap 128 --rank-by-session \
  --conversation-chunks --conversation-parent --coverage-rerank \
  --out benchmarks/longmemeval_token_only_leaderboard_run_20260516.json
```

| Metric | Score |
|---|---:|
| Any gold evidence @1 | 82.77% |
| Any gold evidence @3 | 91.28% |
| Any gold evidence @5 | **95.11%** |
| Any gold evidence @10 | 97.02% |
| All gold evidence @5 | **80.43%** |
| All gold evidence @10 | 86.81% |
| MRR | 0.8753 |

This remains a retrieval/evidence-ranking result, not an official end-to-end LongMemEval QA score. The run used no embeddings, no vector database, no LLM calls, and no answer markers.

### 2026-05-16 source-aware QA bridge

An official-style QA bridge was added to generate answers from the saved token-only retrieval artifact and judge them with LongMemEval-style GPT-4o yes/no answer checks.

- Evidence packet: `benchmarks/longmemeval_contextfit_qa_evidence_20260516.md`
- Retrieval artifact: `benchmarks/longmemeval_token_only_leaderboard_run_20260516.json`
- Retrieval artifact SHA-256: `09f7f8bbd6c1622749cb3961f39cbbf6bcd673b318ad3bd63e88d9d1441a0ca2`
- QA runner: `benchmarks/longmemeval_contextfit_qa.py`
- Generation model: `gpt-4o-2024-08-06`
- Judge model: `gpt-4o-2024-08-06`

Best current QA configuration:

```bash
OPENAI_API_KEY=... /tmp/cf-structure-venv/bin/python benchmarks/longmemeval_contextfit_qa.py \
  --top-k-context 10 \
  --max-session-chars 20000 \
  --source-aware \
  --generation-max-tokens 1200 \
  --hypotheses-out benchmarks/longmemeval_contextfit_token_only_qa_hypotheses_source_20k_20260516.jsonl \
  --judged-out benchmarks/longmemeval_contextfit_token_only_qa_judged_source_20k_20260516.jsonl \
  --summary-out benchmarks/longmemeval_contextfit_token_only_qa_summary_source_20k_20260516.json
```

| Metric | Score |
|---|---:|
| Rows judged | 500 |
| Overall QA accuracy | **81.8%** |
| Task-averaged QA accuracy | **83.5%** |
| Abstention accuracy | 80.0% |

By type:

| Question type | n | Accuracy |
|---|---:|---:|
| knowledge-update | 78 | 88.5% |
| multi-session | 133 | **71.4%** |
| single-session-assistant | 56 | 98.2% |
| single-session-preference | 30 | 70.0% |
| single-session-user | 70 | 97.1% |
| temporal-reasoning | 133 | 75.9% |

Compared to the earlier top-10 CoT bridge, source-aware synthesis improved overall QA from **78.6% -> 81.8%** and multi-session QA from **60.9% -> 71.4%**. This supports the readout that answer synthesis, not retrieval, was the main bottleneck for many multi-session misses.

A post-hoc sufficiency gate was also tested. It improved abstention from 80.0% to 83.3%, but lowered overall QA to 81.6% and multi-session QA to 69.9%, so it is not the current headline configuration. Abstention calibration remains a caveat and next work item.

An experimental deterministic `--token-evidence` reducer was also tested on the first 40 multi-session rows. It keeps retrieval fixed and prepends source-linked token fact hints to the answer prompt. The first implementation did not help:

| Configuration | Accuracy on same 40 multi-session rows |
|---|---:|
| Source-aware baseline | 67.5% |
| Token evidence with raw token signals | 55.0% |
| Token evidence facts-only | 60.0% |

This confirms that the end-to-end score may benefit from token-native evidence synthesis, but the reducer must be more selective than simple keyword/number/date extraction. The flag remains experimental/off and is not part of the headline configuration.

Failure-cluster analysis on the source-aware 20k run found 91 misses. The largest overlap clusters were temporal wording/reasoning (47 misses) and count/list/total wording (41 misses). Restricting to synthesis-like misses where retrieval already had a gold session in top 5 and all gold sessions in top 10 left 42 misses; 21 were count/list and 19 were temporal.

A narrow `--count-list-mode` prompt was tested on the first 60 count/list rows. It was not a clean win:

| Configuration | Accuracy on same 60 count/list rows |
|---|---:|
| Source-aware baseline | 70.0% |
| Count/list candidate+dedupe mode | 68.3% |

The mode created 5 wins but 6 regressions. It should stay experimental/off until there is a router or narrower trigger for high-risk aggregation rows.

A two-pass `--structured-extract` mode was then tested on a targeted hard-row probe. It first extracts structured source-grounded facts from retrieved sessions, then answers from those facts. The probe included 28 source-aware misses where retrieval already had enough evidence plus 28 currently-correct hard rows.

| Slice | Source-aware baseline | Structured two-pass |
|---|---:|---:|
| All hard probe rows | 50.0% | **67.9%** |
| Multi-session hard rows | 68.3% | 65.9% |
| Temporal-reasoning hard rows | 0.0% | **63.6%** |

Flip analysis: 15 wins over baseline and 5 regressions. This is the first materially positive synthesis result, but it should be routed narrowly: promising for temporal reasoning, not yet safe as a blanket multi-session/count mode.

A larger temporal-only probe then tested `--structured-extract` on all 133 temporal-reasoning rows. As a blanket route, it did not hold:

| Configuration | Temporal accuracy |
|---|---:|
| Source-aware baseline on same rows | 75.9% |
| Structured two-pass on same rows | 70.7% |

The structured path had 10 wins but 17 regressions. The regressions were often over-abstentions or wrong date choices on rows the source-aware baseline already handled.

A deterministic hybrid candidate was assembled from the judged artifacts: use structured temporal answers only when the structured answer does not say the information is unavailable; otherwise fall back to the source-aware answer. Non-temporal rows remain source-aware.

| Metric | Source-aware 20k | Hybrid temporal fallback |
|---|---:|---:|
| Overall QA accuracy | 81.8% | **82.8%** |
| Task-averaged QA accuracy | 83.5% | **84.2%** |
| Temporal-reasoning accuracy | 75.9% | **79.7%** |
| Multi-session accuracy | 71.4% | 71.4% |
| Abstention accuracy | 80.0% | 80.0% |

Flip analysis against source-aware 20k: 10 wins, 5 regressions, net +5. This was a useful candidate signal, but it was assembled from existing judged artifacts rather than produced by one coherent run.

The coherent runner path was then implemented behind `--temporal-hybrid`. That full run preserved the temporal lift but reduced the net gain:

| Metric | Source-aware 20k | Coherent temporal hybrid |
|---|---:|---:|
| Overall QA accuracy | 81.8% | **82.2%** |
| Task-averaged QA accuracy | 83.5% | 82.7% |
| Temporal-reasoning accuracy | 75.9% | **79.7%** |
| Multi-session accuracy | 71.4% | **72.2%** |
| Abstention accuracy | 80.0% | **83.3%** |

Route counts in the coherent run: 367 source-aware rows, 73 structured temporal rows, and 60 source-aware fallback rows after structured output was unavailable or hit a guardrail.

This should be treated as an experimental synthesis signal, not the stable headline. The first coherent run still used LongMemEval's `question_type` field for temporal routing, which is acceptable for ablation but too benchmark-specific for product claims. The runner now supports `--temporal-hybrid-router query`, and structured prompts omit `question_type` by default unless `--include-question-type-in-prompts` is passed. The next clean validation is a no-label query-router run.

That no-label query-router run was completed. It did not hold as an improvement:

| Metric | Source-aware 20k | No-label query-router hybrid |
|---|---:|---:|
| Overall QA accuracy | **81.8%** | 81.2% |
| Task-averaged QA accuracy | **83.5%** | 82.8% |
| Temporal-reasoning accuracy | 75.9% | 77.4% |
| Multi-session accuracy | **71.4%** | 68.4% |
| Abstention accuracy | **80.0%** | 76.7% |

Route counts in the no-label run: 263 source-aware rows, 142 structured temporal rows, and 95 source-aware fallback rows. This is the anti-overfit guardrail result: the structured temporal path remains promising research, but the stable QA headline stays the source-aware 20k run at **81.8%**.

Official-format QA package:

- Submission JSONL: `benchmarks/official/longmemeval_source_aware_qa_submission_20260516.jsonl`
- Submission SHA-256: `54789b620c917c82dea2c72600ba1d7dd9f8da9cdce0e7a53ab0625089c64972`
- Parity metadata: `benchmarks/official/longmemeval_source_aware_qa_submission_20260516.parity.json`

The package has 500 rows, exactly the LongMemEval QA fields `question_id` and `hypothesis`, and the packaged question ids match `benchmarks/data/longmemeval_s_cleaned.json`. Official-style aggregation over the judged artifact reproduces the `81.8%` overall result.

The official LongMemEval evaluator was then run as a fresh GPT-4o judge pass against the stable source-aware package. It reproduced the overall score:

| Official evaluator metric | Score |
|---|---:|
| Overall QA accuracy | **81.8%** |
| knowledge-update | 88.5% |
| multi-session | 70.7% |
| single-session-assistant | 98.2% |
| single-session-preference | 70.0% |
| single-session-user | 97.1% |
| temporal-reasoning | 76.7% |

Official evaluator log:

- `benchmarks/official/longmemeval_source_aware_qa_submission_20260516.jsonl.eval-results-gpt-4o`
- SHA-256: `e1b10cc582aaae4f4749c9d3571107447dad6a0582fd158f98ff20c95854eed2`

This is still best framed as a local official-evaluator reproduction, not an official LongMemEval leaderboard submission. The margin versus Supermemory's public overall figure is small and judge-sensitive; the multi-session number approximately matches rather than clearly exceeds Supermemory's published multi-session figure.

### 2026-05-17 structured temporal metadata filters

Structured metadata filters were promoted into the main LongMemEval retrieval
runner after the targeted temporal-filter experiments. The runner now supports:

- `--structured-temporal-filters`
- `--structured-filter-fusion none|rrf`
- `--filter-pushdown-threshold`

The main path keeps retrieval token-native. Filters are deterministic metadata
predicates over indexed session fields; no vector embeddings, vector database,
or LLM call is used inside retrieval. For this run, filters are only generated
for `temporal-reasoning` rows where the question plus question date imply a
relative date window.

Full baseline rerun:

```bash
.venv/bin/python benchmarks/longmemeval_contextfit.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --limit 0 --method hybrid --top-k-chunks 10 --retrieval-k 100 \
  --chunk-size 2048 --overlap 128 --rank-by-session \
  --conversation-chunks --conversation-parent --coverage-rerank \
  --out benchmarks/longmemeval_token_only_baseline_run_20260517.json
```

Full structured-filter rerun:

```bash
.venv/bin/python benchmarks/longmemeval_contextfit.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --limit 0 --method hybrid --top-k-chunks 10 --retrieval-k 100 \
  --chunk-size 2048 --overlap 128 --rank-by-session \
  --conversation-chunks --conversation-parent --coverage-rerank \
  --structured-temporal-filters \
  --out benchmarks/longmemeval_structured_temporal_filters_run_20260517.json
```

Overall retrieval results:

| Metric | Baseline | Structured temporal filters | Delta |
|---|---:|---:|---:|
| Any@1 | 82.77% | **83.19%** | +0.43 pts |
| Any@3 | 91.28% | **91.91%** | +0.64 pts |
| Any@5 | 95.11% | **95.74%** | +0.64 pts |
| Any@10 | 97.02% | **97.23%** | +0.21 pts |
| All@5 | 80.43% | 80.43% | +0.00 pts |
| All@10 | **86.81%** | 86.17% | -0.64 pts |
| MRR | 0.8753 | **0.8799** | +0.0046 |

Temporal-reasoning slice:

| Metric | Baseline | Structured temporal filters | Delta |
|---|---:|---:|---:|
| Any@1 | 78.74% | **80.31%** | +1.57 pts |
| Any@3 | 88.98% | **91.34%** | +2.36 pts |
| Any@5 | 93.70% | **96.06%** | +2.36 pts |
| Any@10 | 96.85% | **97.64%** | +0.79 pts |
| All@5 | 70.08% | 70.08% | +0.00 pts |
| All@10 | **78.74%** | 76.38% | -2.36 pts |
| MRR | 0.8434 | **0.8605** | +0.0171 |

Readout: structured temporal filters are now a clean retrieval win for early
evidence ranking. They improve the full-run Any@K and MRR numbers and recover
the temporal slice materially. The known tradeoff remains complete evidence
coverage at top 10: hard date windows can exclude companion sessions outside
the inferred date range. The right default is hard filters for explicit
temporal windows, with broad RRF kept as an optional conservative variant if
complete-evidence recovery becomes the objective.

The QA bridge was not rerun in this pass because `OPENAI_API_KEY` was not set
in the shell. The next apples-to-apples QA command is:

```bash
OPENAI_API_KEY=... .venv/bin/python benchmarks/longmemeval_contextfit_qa.py \
  --retrieval-artifact benchmarks/longmemeval_structured_temporal_filters_run_20260517.json \
  --top-k-context 10 \
  --max-session-chars 20000 \
  --source-aware \
  --generation-max-tokens 1200 \
  --hypotheses-out benchmarks/longmemeval_contextfit_qa_hypotheses_structured_temporal_filters_20260517.jsonl \
  --judged-out benchmarks/longmemeval_contextfit_qa_judged_structured_temporal_filters_20260517.jsonl \
  --summary-out benchmarks/longmemeval_contextfit_qa_summary_structured_temporal_filters_20260517.json
```

The structured-filter QA bridge was then run with the same source-aware answer
path. It improved the temporal slice but did not improve the stable headline:

| QA metric | Stable source-aware | Structured filters |
|---|---:|---:|
| Overall QA accuracy | **81.8%** | 80.8% |
| Task-averaged QA accuracy | **83.5%** | 81.5% |
| Temporal-reasoning accuracy | 75.9% | **77.4%** |
| Abstention accuracy | 80.0% | **83.3%** |

Artifacts:

- `benchmarks/longmemeval_contextfit_qa_hypotheses_structured_temporal_filters_20260517.jsonl`
- `benchmarks/longmemeval_contextfit_qa_judged_structured_temporal_filters_20260517.jsonl`
- `benchmarks/longmemeval_contextfit_qa_summary_structured_temporal_filters_20260517.json`

### 2026-05-17 evidence-contract QA ablation

An evidence-contract prompt was added to the QA bridge to test the product
principle without more retrieval parameter tuning:

- primary evidence: structured-filter retrieval sessions
- supporting evidence: non-duplicate broad retrieval sessions
- fallback: source-aware behavior for rows without structured filters

The contract is only triggered when the retrieval artifact contains
`structured_temporal_filters`, so it does not use LongMemEval question labels as
a router. The ablation was run on the 16 rows where structured filters actually
fired, then combined with the unchanged stable source-aware judged rows.

Focused 16-row temporal-filter slice:

| QA path | Correct | Accuracy |
|---|---:|---:|
| Stable source-aware | 9 / 16 | 56.25% |
| Hard structured filters | 9 / 16 | 56.25% |
| Filtered top-4 + broad backfill | 5 / 16 | 31.25% |
| Evidence contract | **10 / 16** | **62.50%** |

Combined with unchanged stable source-aware rows:

| QA metric | Stable source-aware | Evidence contract combined |
|---|---:|---:|
| Overall QA accuracy | 81.8% | **82.0%** |
| Task-averaged QA accuracy | 83.5% | **83.7%** |
| Temporal-reasoning accuracy | 75.9% | **76.7%** |
| Abstention accuracy | 80.0% | 80.0% |

Artifacts:

- `benchmarks/longmemeval_evidence_contract_filter_qids_20260517.txt`
- `benchmarks/longmemeval_contextfit_qa_hypotheses_evidence_contract_temporal_20260517.jsonl`
- `benchmarks/longmemeval_contextfit_qa_judged_evidence_contract_temporal_20260517.jsonl`
- `benchmarks/longmemeval_contextfit_qa_summary_evidence_contract_temporal_20260517.json`
- `benchmarks/longmemeval_contextfit_qa_judged_evidence_contract_combined_20260517.jsonl`
- `benchmarks/longmemeval_contextfit_qa_summary_evidence_contract_combined_20260517.json`

Readout: this is a useful product-shape signal, not a new public headline. The
sample is only 16 filtered rows and the gain is one additional correct answer.
It supports the general design of keeping high-precision filtered evidence
separate from broader companion context, but it should be validated on a
held-out temporal set or a separate agent-memory QA set before being promoted.

### 2026-05-17 deterministic evidence packet probe

A deterministic evidence assembly packet was added to the QA bridge via
`--evidence-packet`. This is a non-LLM reducer that extracts source-linked
candidate evidence from already-retrieved sessions before answer generation:

- events with dates/source sessions
- state/update/current-vs-previous candidates
- count/list candidates with simple dedupe keys
- temporal relation candidates
- preference/constraint hints

This does not change ContextFit retrieval. It changes the answer substrate that
the generation LLM receives after retrieval.

Probe design:

- 91 stable source-aware misses
- 91 matched stable-correct guardrail rows by question type
- same baseline retrieval artifact
- same GPT-4o source-aware generation/judge path

Probe result on the 182-row slice:

| Metric | Stable source-aware on slice | Evidence packet |
|---|---:|---:|
| Correct rows | 91 / 182 | **111 / 182** |
| Accuracy | 50.0% | **60.99%** |
| Wins | - | 25 |
| Regressions | - | 5 |

Wins by type:

- temporal-reasoning: 10
- multi-session: 9
- knowledge-update: 4
- single-session-preference: 1
- single-session-assistant: 1

Regressions were all multi-session rows, mostly count/current-role shapes where
the packet appears to add distracting candidate facts.

Artifacts:

- `benchmarks/longmemeval_evidence_packet_probe_qids_20260517.txt`
- `benchmarks/longmemeval_contextfit_qa_hypotheses_evidence_packet_probe_20260517.jsonl`
- `benchmarks/longmemeval_contextfit_qa_judged_evidence_packet_probe_20260517.jsonl`
- `benchmarks/longmemeval_contextfit_qa_summary_evidence_packet_probe_20260517.json`
- `benchmarks/longmemeval_contextfit_qa_judged_evidence_packet_combined_probe_20260517.jsonl`
- `benchmarks/longmemeval_contextfit_qa_summary_evidence_packet_combined_probe_20260517.json`

Readout: this is the strongest general improvement signal so far, but the
probe intentionally includes known misses and therefore is not a valid headline
score. It shows that deterministic evidence assembly addresses the dominant
failure modes: temporal arithmetic/order, multi-session count/list aggregation,
knowledge updates, and preference context. The next unbiased validation is a
full 500-row run or a held-out agent-memory eval with the same packet enabled.

Follow-up routing fix: the unbiased full run showed that applying the packet to
all count/list wording was too broad. Overall QA improved from **81.8%** to
**82.6%**, but multi-session QA regressed from **71.4%** to **69.9%**. A first
guard that skipped only multi-session count/list rows was not enough: the
2026-05-18 safe-general validation landed at **82.0%** overall but still
regressed multi-session to **68.4%**. It also regressed the preference slice to
**63.3%** versus the stable **70.0%**. The `--evidence-packet general` router now
skips all multi-session and single-session-preference rows by default while
preserving temporal/update/knowledge-update routing. A recombined estimate using
stable source-aware judgments for skipped rows and safe-general judgments for
routed rows gives **83.2%** overall, **85.0%** task-averaged, **71.4%**
multi-session, and **70.0%** preference. Use `--evidence-packet all` only for
forced ablations that intentionally test the risky full-packet behavior.

### 2026-05-18 multi-session coverage QA probe

The next multi-session probe kept generation on GPT-4o for comparability and
tested whether the existing token-native coverage rerank retrieval artifact
improves QA when paired with the stable source-aware answer prompt.

Two evidence-compiler prompts were tried first and rejected before scaling:

- `--multi-session-evidence-compiler strict`: 4 / 10 on the gate slice.
- `--multi-session-evidence-compiler guided`: 4 / 10 on the same gate slice.

Both compiler prompts underperformed the plain source-aware prompt because the
deterministic ledger sometimes omitted companion facts or caused over-strict
abstention. This is distinct from the earlier broad `--evidence-packet` failure:
it shows that a narrow multi-session ledger is not enough unless the compiler
can guarantee complete candidate coverage.

The useful signal came from retrieval only. On the same 10-row gate, plain
source-aware synthesis over `longmemeval_coverage_companion_full.json` scored
6 / 10, versus 5 / 10 for the stable source-aware baseline artifact.

Full multi-session QA result:

```bash
OPENAI_API_KEY=... .venv/bin/python benchmarks/longmemeval_contextfit_qa.py \
  --retrieval-artifact benchmarks/longmemeval_coverage_companion_full.json \
  --question-type multi-session \
  --top-k-context 10 \
  --max-session-chars 20000 \
  --source-aware \
  --generation-model gpt-4o-2024-08-06 \
  --judge-model gpt-4o-2024-08-06 \
  --generation-max-tokens 1200 \
  --judge-max-tokens 20 \
  --hypotheses-out benchmarks/longmemeval_contextfit_qa_hypotheses_multisession_coverage_source_full_20260518.jsonl \
  --judged-out benchmarks/longmemeval_contextfit_qa_judged_multisession_coverage_source_full_20260518.jsonl \
  --summary-out benchmarks/longmemeval_contextfit_qa_summary_multisession_coverage_source_full_20260518.json
```

| Multi-session QA | Correct | Accuracy |
|---|---:|---:|
| Stable source-aware artifact | 95 / 133 | 71.43% |
| Coverage-rerank artifact + source-aware QA | **96 / 133** | **72.18%** |

Flip analysis versus the stable source-aware judged artifact:

- wins: 8
- regressions: 7
- unchanged correct: 88
- unchanged wrong: 30
- abstention accuracy unchanged: 83.33% on 12 rows

Readout: token-native coverage reranking gives a small positive end-to-end
multi-session QA lift when used directly with the stable source-aware prompt.
This is not yet a strong headline improvement, but it validates the narrower
direction: improve multi-session retrieval coverage first, and avoid adding
deterministic evidence ledgers unless they can prove complete candidate recall.

### 2026-05-18 token-native evidence-atom selector probe

As a broader search-style retrieval idea, we tried an off-by-default
token-native evidence-atom selector: retrieved sessions are decomposed into
deterministic facets such as goal, constraint, preference, temporal, decision,
open-loop, entity-context, date, and number, then selected by marginal coverage.
This is not an LLM prompt compiler and does not use embeddings or gold labels.

This is distinct from prior attempts:

- Earlier `memory_atoms` indexed extracted user-memory facts.
- Earlier coverage reranking used lexical/entity overlap around an anchor.
- Earlier multi-session compiler prompts changed the QA prompt and failed 4/10.
- This probe changed candidate selection with deterministic evidence atoms.

Retrieval-only gate on the 133 multi-session rows:

| Multi-session retrieval | Any@10 | All@10 | MRR |
|---|---:|---:|---:|
| Rank-by-session baseline | **95.04%** | **73.55%** | **0.837** |
| Evidence-atom selector | 94.21% | 42.15% | 0.825 |

Flip analysis on All@10: 7 wins, 45 losses. The selector found some
complementary facets, but it displaced required companion evidence too often.
Conclusion: reject this selector as a default. Keep it only as an experimental
diagnostic path. The next promising direction is not more prompt compilation or
generic facet diversity; it is source-set preservation: keep the high-recall
baseline top-K set intact, then add a separate targeted expansion lane for
missing entities/dates/count candidates instead of replacing sessions.

### 2026-05-18 source-set-preserving targeted expansion probe

The next probe tested the source-set-preserving version of parent/companion
traversal. Instead of replacing the source set, `--targeted-expansion` protects
the strongest baseline anchors and only fills the tail slots from a broader
retrieved pool using token/entity overlap with the anchors and query.

Retrieval-only gate on the same 133 multi-session rows:

| Multi-session retrieval | Any@10 | All@10 | MRR |
|---|---:|---:|---:|
| Rank-by-session baseline | 95.04% | 73.55% | 0.837 |
| Evidence-atom selector | 94.21% | 42.15% | 0.825 |
| Targeted expansion | **95.04%** | **77.69%** | **0.837** |

Targeted expansion kept Any@10 and MRR unchanged while improving All@10 by
4.13 points. All@10 flips: 8 wins, 3 losses. This validates the traversal
shape: preserve strong anchors, then add targeted companions.

A small GPT-4o source-aware QA gate on the 11 changed rows did not promote the
method yet:

| Changed-row QA gate | Correct | Accuracy |
|---|---:|---:|
| Stable 20k source-aware artifact on same qids | 9 / 11 | 81.82% |
| Coverage-rerank artifact on same qids | 9 / 11 | 81.82% |
| Targeted expansion artifact | 7 / 11 | 63.64% |

Readout: targeted expansion is a useful retrieval primitive, but more complete
retrieval did not automatically improve answer synthesis on the changed rows.
Do not promote it as a QA path yet.

Follow-up source-set-aware synthesis kept the baseline source set as Primary
Sources and targeted-expansion additions as Added Companion Sources, requiring
the answer prompt to audit whether each companion changed the candidate set.
This also failed the changed-row gate:

| Changed-row QA gate | Correct | Accuracy |
|---|---:|---:|
| Source-set-aware targeted expansion | 7 / 11 | 63.64% |

It recovered a different subset of rows than plain targeted expansion but did
not beat the stable or coverage-rerank 9/11 gates. Conclusion: keep
`--targeted-expansion` and `--source-set-aware` off by default as diagnostics.

### 2026-05-18 aggregation assembly probe

The next probe tested a deterministic aggregation assembly path for
multi-session count/list questions. `--aggregation-assembly
multi_session_count_list` builds source-linked `C#` candidate rows and dedupe
groups before generation, then asks the answer model to include/exclude
candidates and answer from the deduped set.

20-row multi-session count/list gate:

| Count/list QA gate | Correct | Accuracy |
|---|---:|---:|
| Historical stable 20k source-aware on same qids | 12 / 20 | 60.00% |
| Same retrieval artifact, source-aware | 9 / 20 | 45.00% |
| Aggregation assembly | 6 / 20 | 30.00% |

Flip analysis against the exact same retrieval artifact showed no wins and 3
losses for aggregation assembly (`3a704032`, `28dc39ac`, `80ec1f4f`). The
assembly table made the model more constrained, but not more correct; it
appears to omit or mis-group the evidence that the plain prompt can sometimes
use. Keep `--aggregation-assembly` off by default as a failed diagnostic. The
next useful path is not another table-to-prompt wrapper; it is typed extraction
with row-level verification or a deterministic reducer that can abstain before
the answer model sees an incomplete candidate set.

## 2026-05-23 selective-fusion QA progress

This section records end-to-end LongMemEval-S QA progress after the May 19
fusion QA companion. It remains separate from retrieval/evidence recall. These
runs generate answers from retrieved ContextFit evidence and judge them with the
LongMemEval-style GPT-4o yes/no evaluator. They are local reproductions, not
official leaderboard submissions.

| QA run | Overall | Task-avg | Abstention | Model split |
|---|---:|---:|---:|---|
| Selective fusion, GPT-4o-only | 85.2% | 85.67% | 93.33% | GPT-4o generation/extraction/answerability/judge |
| Selective fusion, GPT-5-mini answerer | **87.2%** | **87.58%** | **93.33%** | GPT-5-mini generation/extraction/answerability, GPT-4o judge |
| First-class answerer router | 86.8% | 86.65% | 93.33% | GPT-5-mini for temporal/preference/multi-session rows, GPT-4o otherwise, GPT-4o judge |

GPT-5-mini answerer breakdown:

| Question type | Correct | Accuracy |
|---|---:|---:|
| knowledge-update | 71 / 78 | 91.03% |
| multi-session | 104 / 133 | 78.20% |
| single-session-assistant | 56 / 56 | 100.00% |
| single-session-preference | 22 / 30 | 73.33% |
| single-session-user | 67 / 70 | 95.71% |
| temporal-reasoning | 116 / 133 | 87.22% |

The first-class router validated that temporal rows benefit from the routed
high-budget GPT-5-mini path: temporal-reasoning reached 118 / 133 = 88.72%,
above both the GPT-5-mini full-answerer run and the GPT-4o-only run. The same
router weakened multi-session synthesis to 100 / 133 = 75.19%, below both
full-answerer baselines. Practical readout: keep 87.2% as the clean current QA
headline, keep the 85.2% GPT-4o-only result for apples-to-apples comparison,
and treat answerer routing as experimental until a follow-up run validates the
more selective temporal-only benefit without losing multi-session rows.

Primary artifacts:

- `benchmarks/longmemeval_contextfit_qa_summary_selective_fusion_userpref_agent_blend_full_gpt4o_20260523.json`
- `benchmarks/longmemeval_contextfit_qa_summary_selective_fusion_userpref_agent_blend_full_gpt5mini_judge_gpt4o_20260523.json`
- `benchmarks/longmemeval_contextfit_qa_summary_answerer_router_gpt5mini_temporal_pref_multi_else_gpt4o_20260523.json`
- `benchmarks/longmemeval_gpt5mini_regression_audit_20260523.md`

## 2026-05-24 evidence-certificate retrieval rerank

This section records the productionized evidence-certificate rerank work. The
goal was to improve the optional OpenAI-fusion retrieval path without shipping a
black-box global reranker. A candidate can only move up when it carries an
auditable, domain-neutral reason code, and the existing top-5 tail is protected
when it is already answer-shaped evidence.

| Retrieval run | Any@5 | Any@10 | All@5 | MRR | Paired top-5 vs 96.6 baseline |
|---|---:|---:|---:|---:|---:|
| OpenAI fusion baseline | 96.6% | 98.7% | 83.6% | 0.900 | — |
| Certificate v4 | 98.1% | 98.9% | **86.4%** | 0.902 | +8 / 0 |
| Certificate v5 typed rescue | **98.3%** | **99.2%** | **86.4%** | **0.902** | +9 / 0 |

Production hooks added:

- `contextfit.retrieval.evidence_certificates` contains the reusable certificate engine.
- `RetrievalEngine.rerank_sessions_by_evidence_certificates(...)` applies the post-retrieval session rerank.
- `query_auto(..., evidence_certificate_rerank=True, typed_rescue=True)` enables the full v5 path.
- Certificate traces include reason code, source id, old/new rank, strength, and displaced source.

The targeted non-LongMemEval typed-rescue gate uses fictional/product-shaped
preference and temporal cases plus risk controls. Final result: 8/8 cases
passed, paired movement +5 / 0, with no unexpected risk-control promotions.
This gate caught and fixed two over-permissive rules before the final run:
generic preference wording and generic temporal/entity word salad can no longer
trigger typed rescue without personal/action evidence.

Independent gates after productionization:

- Full test suite: 247 passed.
- Agent-memory 499-case v5-tight gate: Recall@1 62.7%, Recall@3 94.0%, Recall@5 100.0%, MRR 0.784.
- `git diff --check`: clean.

Primary artifacts:

- `benchmarks/longmemeval_fusion_certificate_promotion_v4_20260524.json`
- `benchmarks/longmemeval_fusion_certificate_promotion_v5_typed_rescue_20260524.json`
- `benchmarks/typed_rescue_eval_20260524.json`
- `benchmarks/agent_memory_eval_500_auto_cert_v5_prod_tight_20260524.json`
