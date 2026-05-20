# LongMemEval-S Optional OpenAI Fusion Artifact

Date: 2026-05-19

This artifact documents a retrieval/evidence-ranking run on LongMemEval-S. It is not an official end-to-end LongMemEval QA score.

## Claim Supported

ContextFit with optional OpenAI fusion reaches **96.6% Any@5** and **98.7% Any@10** on LongMemEval-S, with **no vector database required**.

The precise wording matters:

- This is evidence retrieval, not final answer-generation accuracy.
- This uses OpenAI `text-embedding-3-small` embeddings as an optional fusion signal.
- Embeddings are cached locally and fused with ContextFit rankings; no vector database is required.
- The core token-native ContextFit path remains usable without embeddings or API calls.

## Result

| Metric | Score |
|---|---:|
| Scored examples | 470 |
| Abstention examples skipped | 30 |
| Any evidence @1 | 84.68% |
| Any evidence @3 | 94.26% |
| Any evidence @5 | **96.60%** |
| Any evidence @10 | **98.72%** |
| All evidence @5 | 83.62% |
| All evidence @10 | 91.28% |
| MRR | 0.8999 |

## Configuration

- Dataset: `benchmarks/data/longmemeval_s_cleaned.json`
- Runner: `benchmarks/longmemeval_contextfit.py`
- Retrieval method: `hybrid`
- Top-k chunks / sessions: `10`
- Retrieval candidate pool: `100`
- Chunk size: `2048`
- Overlap: `128`
- Session ranking: enabled
- Conversation chunks: enabled
- Conversation parent/session context: enabled
- Coverage rerank: enabled
- Structured temporal filters: enabled
- OpenAI fusion: enabled
- Answer markers: disabled
- Vector database: none

## Reproduction Command

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

## Raw Artifact

- JSON: `benchmarks/longmemeval_fusion_claim_966_987_20260519.json`
- SHA-256: `059c778ca389e2a5939505800acffd6349f0be7ada579238023d342784214932`
- Runtime: 15,773.8 seconds

## Recommended Public Wording

> ContextFit with optional OpenAI fusion reaches 96.6% Any@5 and 98.7% Any@10 evidence retrieval on LongMemEval-S, with no vector database required.

Avoid saying “96.6% LongMemEval score” unless running the official end-to-end answer-generation and judging pipeline.
