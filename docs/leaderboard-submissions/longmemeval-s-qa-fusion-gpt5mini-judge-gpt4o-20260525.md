# LongMemEval-S QA Comparison Packet

Date: 2026-05-25

This packet records ContextFit's current fusion end-to-end QA comparison lane.
It is a local LongMemEval-style evaluation, not an official leaderboard
submission.

## Result

ContextFit + OpenAI fusion with `gpt-5-mini` answering/extraction and
`gpt-4o-2024-08-06` judging reaches:

- Overall QA accuracy: **87.2%** (`436/500`)
- Task-averaged QA accuracy: **87.6%**
- Abstention accuracy: **93.3%** (`28/30`)

## Metric Boundary

This is **end-to-end QA accuracy**, not retrieval recall.

- Dataset: `longmemeval_s`
- Rows: 500
- Retrieval: ContextFit selective OpenAI fusion
- Answer/extraction model: `gpt-5-mini`
- Judge model: `gpt-4o-2024-08-06`
- Judge output: yes/no answer correctness

## By Type

| Question type | n | Accuracy |
|---|---:|---:|
| knowledge-update | 78 | 91.0% |
| multi-session | 133 | 78.2% |
| single-session-assistant | 56 | 100.0% |
| single-session-preference | 30 | 73.3% |
| single-session-user | 70 | 95.7% |
| temporal-reasoning | 133 | 87.2% |

## Artifacts

- Machine-readable packet: `benchmarks/submissions/contextfit_longmemeval_s_qa_fusion_gpt5mini_judge_gpt4o_20260525.json`
- Summary artifact: `benchmarks/longmemeval_contextfit_qa_summary_selective_fusion_userpref_agent_blend_full_gpt5mini_judge_gpt4o_20260523.json`
- Summary SHA-256: `c9cf70a052f927f1fca7ed234f8ad66dea476a1335de0fcb30372383dcf3609e`
- Hypotheses artifact: `benchmarks/longmemeval_contextfit_qa_hypotheses_selective_fusion_userpref_agent_blend_full_gpt5mini_judge_gpt4o_20260523.jsonl`
- Hypotheses SHA-256: `89921e7853e72d87c390823b1d5b25ea887cfc33d29f098596c296a1ebdb14f1`
- Judged artifact: `benchmarks/longmemeval_contextfit_qa_judged_selective_fusion_userpref_agent_blend_full_gpt5mini_judge_gpt4o_20260523.jsonl`
- Judged SHA-256: `92b050139ce8daeee3f5b249437f158bd7fae118657b973269debd3251c7f661`
- Retrieval artifact: `benchmarks/longmemeval_selective_fusion_userpref_token_base_20260523.json`
- Retrieval SHA-256: `a4aab490d380e421ac309fa3050f3b3ea91729f411bd0ae99c1cefa649052df6`

## Comparison Wording

ContextFit + OpenAI fusion reports 87.2% end-to-end QA accuracy on
LongMemEval-S using `gpt-5-mini` as the answer/extraction model and
`gpt-4o-2024-08-06` as the judge. This is a QA score, not the separate
99.00% LongMemEval-S session retrieval R@5 result.
