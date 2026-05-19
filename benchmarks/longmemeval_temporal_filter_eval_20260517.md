# LongMemEval Temporal Structured-Filter Eval - 2026-05-17

This is a targeted retrieval evaluation for ContextFit's structured metadata
filter interface. It checks whether agent-style date filters help prior
LongMemEval temporal misses while keeping the core retrieval path token-native.

## Setup

- Data: `benchmarks/data/longmemeval_s_cleaned.json`
- Harness: `benchmarks/longmemeval_temporal_filter_eval.py`
- Variants:
  - `baseline`: token-native retrieval, no structured filters
  - `filters`: structured `date` range filters when a relative-date window is inferable
  - `filters_temporal_rerank`: filters plus existing temporal date reranker
  - `filters_broad_fusion`: reciprocal-rank fusion of filtered retrieval and
    broader unfiltered retrieval
  - `filters_top2_broad_merge`: keep top two filtered sessions, then fill from
    broader unfiltered retrieval, then remaining filtered sessions
- Filters are derived only from question text and question date, not answer labels.

## Known Temporal Miss Slice

Artifact: `benchmarks/longmemeval_temporal_filter_eval_20260517.json`

Rows: 8 known `temporal-reasoning` miss@5 cases from
`benchmarks/longmemeval_miss_analysis.md`.

| Variant | Any@1 | Any@3 | Any@5 | Any@10 | All@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.000 | 0.000 | 0.125 | 0.500 | 0.000 | 0.077 |
| filters | 0.125 | 0.250 | 0.250 | 0.625 | 0.000 | 0.219 |
| filters + temporal rerank | 0.125 | 0.250 | 0.250 | 0.625 | 0.000 | 0.219 |
| filters + broad RRF | 0.125 | 0.250 | 0.250 | 0.625 | 0.000 | 0.221 |
| top-2 filter + broad merge | 0.125 | 0.125 | 0.125 | 0.500 | 0.000 | 0.171 |

Best-rank improvements: 5. Regressions: 0.

Recovered/improved examples:

- `4dfccbf8`: rank 4 -> 1
- `gpt4_fa19884d`: rank 7 -> 3
- `9a707b82`: rank 8 -> 6
- `eac54add`: absent -> 8
- `gpt4_e061b84g`: rank 10 -> 8

## All Temporal Slice

Artifact: `benchmarks/longmemeval_temporal_filter_eval_all_temporal_20260517.json`

Rows: 133 `temporal-reasoning` cases. Explicit relative-date windows were
inferable for 16 rows.

| Variant | Any@1 | Any@3 | Any@5 | Any@10 | All@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.759 | 0.850 | 0.917 | 0.970 | 0.767 | 0.824 |
| filters | 0.774 | 0.880 | 0.932 | 0.977 | 0.744 | 0.841 |
| filters + temporal rerank | 0.774 | 0.880 | 0.932 | 0.977 | 0.744 | 0.841 |
| filters + broad RRF | 0.774 | 0.880 | 0.932 | 0.977 | 0.752 | 0.841 |
| top-2 filter + broad merge | 0.774 | 0.865 | 0.925 | 0.970 | 0.774 | 0.838 |

Best-rank improvements: 9. Best-rank regressions: 2.

Important caveat: all-evidence@10 regressed on 3 rows. Those are multi-evidence
temporal questions where at least one gold session falls outside the inferred
relative-date window. This is expected behavior for hard prefilters and argues
for guarded usage rather than unconditional temporal routing.

The two broad-merge variants confirmed the tradeoff:

- `filters_broad_fusion` slightly improves all-evidence@10 versus hard filters
  while preserving the Any@K/MRR gains, but it does not fully recover baseline
  all-evidence.
- `filters_top2_broad_merge` recovers all-evidence@10 above baseline on the
  all-temporal slice, but weakens the known-miss slice and gives back too much
  Any@K/MRR. It is not a good default.

## Interpretation

Structured metadata filters help the exact failure cluster they were intended to
help. They are especially useful for explicit relative-date questions like
"last Friday", "10 days ago", "four weeks ago", and "Wednesday two months ago".

They should not be used as a blanket temporal strategy. For questions requiring
multiple evidence sessions, a date filter can recover the direct event while
dropping companion evidence outside the date window.

Recommended next product behavior:

- Use structured filters as an agent-query capability when the host/planner has
  high confidence in the time constraint.
- Keep `min_filter_matches` and filter traces enabled.
- Prefer soft/expansion behavior for multi-evidence temporal questions:
  retrieve inside the date window, then merge with a broader unfiltered
  token-native retrieval pass.
- Current best default from this probe remains hard structured filters for
  explicit date-window queries, with broad RRF as a possible conservative
  variant. The top-2 broad merge should be treated as a rejected tradeoff unless
  a better multi-evidence router is added.
- Benchmark claims should separate pure retrieval, structured-filter retrieval,
  agent/planner query specs, and answer-model QA.

## Verification

- `python3 -m py_compile src/contextfit/metadata/index.py benchmarks/longmemeval_contextfit.py benchmarks/longmemeval_temporal_filter_eval.py`
- `.venv/bin/python -m ruff check src/contextfit/metadata/index.py benchmarks/longmemeval_temporal_filter_eval.py`
- `.venv/bin/python -m pytest tests/test_query_spec.py tests/test_mcp.py`: 11 passed
- `.venv/bin/python -m pytest`: 120 passed in 31.32s
