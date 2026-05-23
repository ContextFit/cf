# LongMemEval GPT-5-mini Regression Audit

Date: 2026-05-23

## Baselines

- GPT-4o selective-fusion answerer: `426/500 = 85.2%`
- GPT-5-mini selective-fusion answerer, GPT-4o judge: `436/500 = 87.2%`
- Paired flips from GPT-4o to GPT-5-mini: `+27/-17`, net `+10`

## Regression Shape

The 17 GPT-5-mini regressions are not one broad retrieval failure.

- `multi-session`: 11 regressions
- `single-session-preference`: 2 regressions
- `knowledge-update`: 2 regressions
- `single-session-user`: 1 regression
- `temporal-reasoning`: 1 regression

Only 3 of the 17 regressions used count-list ledger routes. The rest used `source_aware`.

## Main Failure Modes

- Over-literal interpretation where GPT-4o followed benchmark intent, e.g. album copies vs poster copies.
- Over-strict exclusion in count/list questions, e.g. excluding snake plant or vegan cuisine.
- Over-inclusion in aggregation rows, e.g. charity total includes an April benefit concert outside the intended set.
- Temporal anchoring error, e.g. calculating weeks ago from wall-clock date instead of benchmark session date.
- Abstention/inference boundary errors, e.g. inferring Harvard as the poster-presentation university.
- Some regressions look like judge/gold quirks rather than obviously worse answers.

## Routing Analysis

Post-hoc non-oracle routing by LongMemEval `question_type` gives the best current candidate:

- Use GPT-5-mini for `temporal-reasoning`, `single-session-preference`, and `multi-session`.
- Use GPT-4o for `knowledge-update`, `single-session-user`, and `single-session-assistant`.

This composite scores `438/500 = 87.6%`, task-avg `88.03%`, with paired flips vs GPT-4o selective-fusion of `+26/-14`.

Artifacts:

- `benchmarks/longmemeval_contextfit_qa_summary_composite_gpt5mini_temporal_pref_multi_else_gpt4o_20260523.json`
- `benchmarks/longmemeval_contextfit_qa_judged_composite_gpt5mini_temporal_pref_multi_else_gpt4o_20260523.jsonl`
- `benchmarks/longmemeval_contextfit_qa_hypotheses_composite_gpt5mini_temporal_pref_multi_else_gpt4o_20260523.jsonl`

## Recommendation

Do not patch prompts broadly yet. The safest next candidate is model routing:

1. Treat `87.2%` as the clean full GPT-5-mini answerer result.
2. Treat `87.6%` as a routing-analysis candidate, not a fresh full run.
3. If publishing or reporting, label the routing candidate clearly as a composite from existing judged runs unless rerun through a first-class routing harness.
4. Next implementation step: add a first-class answerer-router option to the QA harness and rerun once, preserving GPT-4o judge.
