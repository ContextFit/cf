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

`query_spec_guarded_v2` keeps the same slice gating as `query_spec_rescue_v1`,
but it treats the original query ranking as the authority for the first five
slots. QuerySpec variants can add certified evidence only to tail slots 6-10.
This directly tests whether QuerySpec can keep the top-10 / all-evidence gains
without paying the Any@5 regression seen in the blunt RRF-fusion policies.

## Miss-Set Results

| Policy | Any@1 | Any@5 | Any@10 | All@5 | All@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| `off` | 24/49 | 40/49 | 41/49 | 23/49 | 26/49 | 0.6067 |
| `query_spec_v1` | 26/49 | 39/49 | 43/49 | 24/49 | 29/49 | 0.6293 |
| `query_spec_rescue_v1` | 27/49 | 39/49 | 43/49 | 24/49 | 29/49 | 0.6463 |
| `query_spec_guarded_v2` | 24/49 | 40/49 | 42/49 | 23/49 | 28/49 | 0.6105 |

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

## Full-500 Retrieval Check

Because the miss-set gate can overstate gains, `query_spec_guarded_v2` was also
run against the full LongMemEval-S 500-row retrieval set using the same
token-native retrieval lane. Abstention rows are skipped for retrieval metrics,
so the effective `n` is 470.

| Policy | Any@1 | Any@5 | Any@10 | All@5 | All@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| `off` | 83.19% | 95.74% | 97.23% | 80.43% | 85.96% | 0.8803 |
| `query_spec_guarded_v2` | 83.19% | 95.74% | 97.45% | 80.43% | 86.17% | 0.8807 |

The full-set movement is intentionally small:

- Any@5 and All@5 are unchanged.
- Any@10 improves by one row.
- All@10 improves by one row.
- The only slice with measurable movement is `multi-session`: Any@10 and
  All@10 both move from 96.69% / 79.34% to 97.52% / 80.17%.
- All stronger slices remain unchanged because guarded v2 only applies to the
  configured rescue slices and protects original-query top-5 evidence.

## Interpretation

The blunt QuerySpec policies are promising but not ready as public QA-improvement
claims. They improve MRR and top-10 evidence recovery on the QA-miss set,
especially by recovering two previously absent rows into the top 10, but they
slightly hurt Any@5 on that slice.

`query_spec_guarded_v2` is safer. It recovers a small amount of tail evidence on
both the miss set and full set while preserving top-5 retrieval. This is worth
keeping as an experimental guarded retrieval policy, but the full-set gain is not
large enough to justify an expensive end-to-end QA run by itself. The next useful
step is to make QuerySpec certificates stronger and more semantic, then rerun the
same full-500 retrieval gate before spending on QA.
