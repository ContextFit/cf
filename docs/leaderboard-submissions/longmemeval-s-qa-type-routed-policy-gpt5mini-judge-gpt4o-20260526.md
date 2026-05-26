# LongMemEval-S QA Type-Routed Policy Packet

Date: 2026-05-26

This packet records ContextFit's single-command question-type-routed
LongMemEval-S QA run. It is a local LongMemEval-style evaluation, not an
official leaderboard submission.

## Result

ContextFit question-type-routed QA with `gpt-5-mini` answering and
`gpt-4o-2024-08-06` judging reaches:

- Overall QA accuracy: **89.6%** (`448/500`)
- Task-averaged QA accuracy: **88.3%**
- Abstention accuracy: **90.0%** (`27/30`)

## Metric Boundary

This is **end-to-end QA accuracy**, not retrieval recall.

- Dataset: `longmemeval_s`
- Rows: 500
- Answer model: `gpt-5-mini`
- Judge model: `gpt-4o-2024-08-06`
- Judge output: yes/no answer correctness
- Routing: LongMemEval `question_type` labels

Because this uses dataset-provided `question_type` labels, public wording should
call it **question-type-routed** or **oracle type-routed**. It should not be
presented as a production query-router result.

## Routing Policy

The run used one benchmark command with `--qa-policy conservative_type_routed_20260526`.

| Route | Question types | Rows |
|---|---|---:|
| `source_set_aware` | multi-session, knowledge-update, single-session-user | 281 |
| `v5_source_aware` | temporal-reasoning | 133 |
| `baseline_source_aware` | single-session-preference, single-session-assistant | 86 |

## By Type

| Question type | n | Correct | Accuracy |
|---|---:|---:|---:|
| knowledge-update | 78 | 72 | 92.3% |
| multi-session | 133 | 111 | 83.5% |
| single-session-assistant | 56 | 56 | 100.0% |
| single-session-preference | 30 | 20 | 66.7% |
| single-session-user | 70 | 67 | 95.7% |
| temporal-reasoning | 133 | 122 | 91.7% |

## Artifacts

- Machine-readable packet: `benchmarks/submissions/contextfit_longmemeval_s_qa_type_routed_policy_gpt5mini_judge_gpt4o_20260526.json`
- Summary artifact: `benchmarks/longmemeval_contextfit_qa_summary_type_routed_policy_single_command_max16k_20260526.json`
- Summary SHA-256: `85e404807e15cc91f522803b563a708c13a0e5c74e211723224d60fdea3d2458`
- Hypotheses artifact: `benchmarks/longmemeval_contextfit_qa_hypotheses_type_routed_policy_single_command_max16k_20260526.jsonl`
- Hypotheses SHA-256: `76a1a74414e30d9977bc9fdfdd48b6caf28a997ebfd5ed8ed2c5b378347f5d23`
- Judged artifact: `benchmarks/longmemeval_contextfit_qa_judged_type_routed_policy_single_command_max16k_20260526.jsonl`
- Judged SHA-256: `546e09650698d1c0940c78ab1b1522ab422de005a1d6513f2e98caa45d49ea24`
- V5 retrieval artifact: `benchmarks/longmemeval_fusion_certificate_promotion_v5_typed_rescue_20260524.json`
- V5 retrieval SHA-256: `2b7043391dab26aa58ec4dc4f17ceb4b6d5f708c34b8e2a220fa4b9d580f9de2`
- Baseline/primary retrieval artifact: `benchmarks/longmemeval_selective_fusion_userpref_token_base_20260523.json`
- Baseline/primary retrieval SHA-256: `a4aab490d380e421ac309fa3050f3b3ea91729f411bd0ae99c1cefa649052df6`

## Reproduction Command

```bash
.venv/bin/python benchmarks/longmemeval_contextfit_qa.py \
  --qa-policy conservative_type_routed_20260526 \
  --generation-model gpt-5-mini \
  --extraction-model gpt-5-mini \
  --judge-model gpt-4o-2024-08-06 \
  --generation-seed 20260526 \
  --judge-seed 20260522 \
  --completion-cache-dir benchmarks/.completion_cache/type_routed_policy_single_command_20260526 \
  --hypotheses-out benchmarks/longmemeval_contextfit_qa_hypotheses_type_routed_policy_single_command_max16k_20260526.jsonl \
  --judged-out benchmarks/longmemeval_contextfit_qa_judged_type_routed_policy_single_command_max16k_20260526.jsonl \
  --extract-out benchmarks/longmemeval_contextfit_qa_extract_type_routed_policy_single_command_max16k_20260526.jsonl \
  --answerability-out benchmarks/longmemeval_contextfit_qa_answerability_type_routed_policy_single_command_max16k_20260526.jsonl \
  --summary-out benchmarks/longmemeval_contextfit_qa_summary_type_routed_policy_single_command_max16k_20260526.json
```

## Comparison Wording

ContextFit reports **89.6% end-to-end QA accuracy** on LongMemEval-S with a
single-command question-type-routed policy. This is below the earlier
post-hoc stitched projection of 90.2%, but it is the cleaner referenceable
artifact because generation and judging were produced by one run and one
manifest.
