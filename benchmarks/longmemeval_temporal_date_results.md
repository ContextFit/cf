# Temporal date hint validation

Variant: `--temporal-date-rerank` on top of the current companion recipe:

```bash
/tmp/cf-structure-venv/bin/python benchmarks/longmemeval_contextfit.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --limit 0 --method hybrid --top-k-chunks 10 --retrieval-k 100 \
  --chunk-size 2048 --overlap 128 --rank-by-session \
  --conversation-chunks --conversation-parent --coverage-rerank \
  --temporal-date-rerank \
  --out benchmarks/longmemeval_temporal_date_hint_full.json
```

## LongMemEval retrieval/evidence ranking

Compared with `benchmarks/longmemeval_coverage_companion_full.json`.

| metric | baseline | temporal date hint | delta |
|---|---:|---:|---:|
| n | 470 | 470 | — |
| Any@1 | 0.8277 | 0.8234 | -0.0043 |
| Any@3 | 0.9128 | 0.9085 | -0.0043 |
| Any@5 | 0.9511 | 0.9532 | +0.0021 |
| Any@10 | 0.9702 | 0.9702 | +0.0000 |
| All evidence@5 | 0.8043 | 0.8043 | +0.0000 |
| All evidence@10 | 0.8681 | 0.8702 | +0.0021 |
| MRR | 0.8753 | 0.8727 | -0.0026 |

Changed ranks: 10 temporal items total; 4 improved, 6 worsened. Top-5 recovered: 2. Top-5 lost: 1.

Notable recoveries:
- `eac54add`: absent → 5 (`four weeks ago`)
- `4dfccbf8`: 6 → 2 (`Wednesday two months ago`)

Notable regression:
- `gpt4_68e94288`: 3 → 6, creating the one Top-5 loss
- `gpt4_fa19884d`: 7 → absent, but it was already outside Top-5

Interpretation: useful for LongMemEval Any@5, but not a clean win because @1/@3/MRR soften slightly. Keep as an optional benchmark flag unless a later scorer can preserve high-rank precision.

## Independent agent-memory gate

Command:

```bash
/tmp/cf-structure-venv/bin/python benchmarks/agent_memory_eval.py \
  benchmarks/data/agent_memory_eval_500.json \
  --mode auto --method hybrid --top-k 5 \
  --out benchmarks/agent_memory_eval_500_auto_after_temporal_date.json
```

Compared with `benchmarks/agent_memory_eval_500_auto.json`.

| metric | baseline | after temporal work | delta |
|---|---:|---:|---:|
| n | 499 | 499 | — |
| Recall@1 | 0.5451 | 0.6232 | +0.0782 |
| Recall@3 | 0.9018 | 0.9299 | +0.0281 |
| Recall@5 | 0.9980 | 0.9980 | +0.0000 |
| MRR | 0.7274 | 0.7774 | +0.0500 |
| Avg query ms | 7.6 | 8.9 | +1.3 ms |

Note: this benchmark does not use the LongMemEval-only temporal flag directly; the improvement reflects current code/runtime state of the general auto benchmark and should be treated as a gate result, not proof that the flag caused the gain.

## Test gate

```bash
/tmp/cf-structure-venv/bin/python -m pytest tests
```

Result: 81 passed in 13.83s.
