# LongMemEval-S Token-Only Leaderboard Evidence

Date: 2026-05-16

This file records the audit packet for the fresh token-only LongMemEval-S evidence-retrieval run. The raw JSON artifact is the source of truth.

## Scope

This is a LongMemEval-S retrieval/evidence-ranking result, not an official end-to-end LongMemEval QA generation score.

For each LongMemEval-S item, the benchmark builds a per-question ContextFit knowledge base from the haystack sessions, queries with the question plus question date, and measures whether retrieved sessions match `answer_session_ids`.

Abstention rows are excluded from the scored retrieval summary because this harness does not evaluate no-answer calibration.

## Repository State

- Repository: `https://github.com/ContextFit/cf`
- Local path: `/Users/christophe/.openclaw/tools/cf`
- Git commit: `0071d24a3e079dd29eaa9e501a299ddae8b67880`
- Benchmark runner: `benchmarks/longmemeval_contextfit.py`
- Dataset: `benchmarks/data/longmemeval_s_cleaned.json`
- Raw artifact: `benchmarks/longmemeval_token_only_leaderboard_run_20260516.json`
- Raw artifact SHA-256: `09f7f8bbd6c1622749cb3961f39cbbf6bcd673b318ad3bd63e88d9d1441a0ca2`

## Command

```bash
/tmp/cf-structure-venv/bin/python benchmarks/longmemeval_contextfit.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --limit 0 --method hybrid --top-k-chunks 10 --retrieval-k 100 \
  --chunk-size 2048 --overlap 128 --rank-by-session \
  --conversation-chunks --conversation-parent --coverage-rerank \
  --out benchmarks/longmemeval_token_only_leaderboard_run_20260516.json
```

## Configuration

| Setting | Value |
|---|---:|
| Method | `hybrid` |
| Limit | `0` (all rows) |
| Total rows | 500 |
| Scored rows | 470 |
| Skipped abstention rows | 30 |
| Top-k chunks / sessions | 10 |
| Retrieval candidate pool | 100 |
| Chunk size | 2048 |
| Overlap | 128 |
| Rank by session | yes |
| Conversation chunks | yes |
| Conversation parent chunk | yes |
| Coverage rerank | yes |
| Include answer marker | no |
| OpenAI vector rerank | no |
| OpenAI fusion | no |
| Token-native rerank flag | no |
| Embedding API calls | no |
| Vector database | no |
| LLM calls | no |

## Overall Results

| Metric | Score |
|---|---:|
| Any gold evidence @1 | 82.77% |
| Any gold evidence @3 | 91.28% |
| Any gold evidence @5 | 95.11% |
| Any gold evidence @10 | 97.02% |
| All gold evidence @1 | 28.94% |
| All gold evidence @3 | 71.49% |
| All gold evidence @5 | 80.43% |
| All gold evidence @10 | 86.81% |
| MRR | 0.8753 |
| Average haystack sessions/question | 47.7 |

## Results By Question Type

| Question type | n | Any@1 | Any@3 | Any@5 | Any@10 | All@5 | All@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `knowledge-update` | 72 | 95.83% | 98.61% | 98.61% | 100.00% | 94.44% | 94.44% | 0.9736 |
| `multi-session` | 121 | 79.34% | 92.56% | 95.04% | 96.69% | 65.29% | 79.34% | 0.8624 |
| `single-session-assistant` | 56 | 96.43% | 98.21% | 100.00% | 100.00% | 100.00% | 100.00% | 0.9768 |
| `single-session-preference` | 30 | 43.33% | 63.33% | 83.33% | 83.33% | 83.33% | 83.33% | 0.5522 |
| `single-session-user` | 64 | 89.06% | 92.19% | 95.31% | 98.44% | 95.31% | 98.44% | 0.9151 |
| `temporal-reasoning` | 127 | 78.74% | 88.98% | 93.70% | 96.85% | 70.08% | 78.74% | 0.8434 |

## Claim Boundary

Defensible claim:

> ContextFit reaches 95.1% Any@5 and 80.4% All@5 on LongMemEval-S evidence retrieval with a pure token-only parent/child conversation-aware retriever plus coverage rerank, using no embeddings, no vector database, no LLM calls, and no answer markers.

Do not claim this as an official LongMemEval QA leaderboard score. The official QA path requires model-generated hypotheses and the LongMemEval GPT-4o judging script.

Do not claim universal state of the art. Public retrieval comparisons include higher reported R@5 numbers from hybrid BM25 + embedding systems; the core claim here is the token-only/no-embedding retrieval result and complete-evidence coverage.

