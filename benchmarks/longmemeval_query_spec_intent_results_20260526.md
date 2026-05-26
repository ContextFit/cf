# LongMemEval QuerySpec Intent Retrieval Gate

Date: 2026-05-26

This gate tests whether deterministic question-to-retrieval-intent rewrites help the
misses from the single-command QA routed-policy run:

- QA source run: `benchmarks/longmemeval_contextfit_qa_judged_type_routed_policy_single_command_max16k_20260526.jsonl`
- Miss set: 52 judged QA misses, of which 49 are non-abstention retrieval-evaluable rows
- Retrieval lane: token-native grouped session retrieval with conversation chunks, coverage rerank, and structured temporal filters
- No answer labels were indexed; no freeform LLM query rewrite was used
- OpenAI-fusion replay was not run in this shell because `OPENAI_API_KEY` was unavailable

## Policies

`off` uses the existing wrapped benchmark question:

```text
Question date: ...
Question: ...
```

`query_spec_v1` builds deterministic intent variants for all question types. Example
facets include `user_preference`, `user_constraint`, `temporal_update`,
`distinct_items`, and `direct_evidence`. The variants are retrieved separately and
reciprocal-rank fused.

`query_spec_rescue_v1` is the conservative version. It applies query-intent variants
only to the weak/rescue slices observed in the QA miss set: `multi-session` and
`single-session-preference`. Stronger slices keep the original query.

## Miss-Set Results

| Policy | Any@1 | Any@5 | Any@10 | All@5 | All@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| `off` | 24/49 | 40/49 | 41/49 | 23/49 | 26/49 | 0.6067 |
| `query_spec_v1` | 26/49 | 39/49 | 43/49 | 24/49 | 29/49 | 0.6293 |
| `query_spec_rescue_v1` | 27/49 | 39/49 | 43/49 | 24/49 | 29/49 | 0.6463 |

## Row Movement

`query_spec_rescue_v1` improved 7 rows:

- `gpt4_2f8be40d` multi-session: rank 3 -> 1
- `gpt4_194be4b3` multi-session: rank 2 -> 1
- `1a8a66a6` multi-session: rank 2 -> 1
- `d24813b1` single-session-preference: absent -> 9
- `57f827a0` single-session-preference: rank 4 -> 3
- `157a136e` multi-session: rank 3 -> 2
- `c18a7dc8` multi-session: absent -> 9

It worsened 3 rows:

- `6d550036` multi-session: rank 4 -> 6
- `88432d0a` multi-session: rank 7 -> 9
- `0edc2aef` single-session-preference: rank 3 -> 4

## Interpretation

The conservative QuerySpec policy is promising but not ready as a public result.
It improves MRR and top-10 evidence recovery on the QA-miss set, especially by
recovering two previously absent rows into the top 10. It also slightly hurts
Any@5 on this slice, which matters because downstream prompts often consume the
top 10 but benchmark retrieval headlines usually report top 5.

Recommended next step: run `query_spec_rescue_v1` on a broader holdout or the full
500 token-native retrieval set, then only run QA if the full-slice damage profile
is acceptable. Do not claim an end-to-end QA gain from this gate alone.

