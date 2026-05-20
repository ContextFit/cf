# LongMemEval-S Fusion Source-Aware QA Run

Date: 2026-05-19

This artifact records the end-to-end QA run using the May 19 ContextFit fusion retrieval artifact and GPT-4o source-aware answer generation/judging.

## Inputs

- Dataset: `benchmarks/data/longmemeval_s_cleaned.json`
- Retrieval artifact: `benchmarks/longmemeval_fusion_claim_966_987_20260519.json`
- Retrieval artifact SHA-256: `059c778ca389e2a5939505800acffd6349f0be7ada579238023d342784214932`
- Generation model: `gpt-4o-2024-08-06`
- Judge model: `gpt-4o-2024-08-06`
- Route: `source_aware`
- Top-k context: `10`
- Max session chars: `20000`

## QA Results

- Overall accuracy: `84.8%`
- Task-averaged accuracy: `86.81%`
- Abstention accuracy: `86.67%`
- Total questions: `500`

## By Type

| Type | N | Accuracy |
| --- | ---: | ---: |
| knowledge-update | 78 | 88.46% |
| multi-session | 133 | 74.44% |
| single-session-assistant | 56 | 98.21% |
| single-session-preference | 30 | 80.00% |
| single-session-user | 70 | 98.57% |
| temporal-reasoning | 133 | 81.20% |

## Official-Style Submission Package

- Submission JSONL: `benchmarks/official/longmemeval_fusion_source_aware_qa_submission_20260519.jsonl`
- Parity JSON: `benchmarks/official/longmemeval_fusion_source_aware_qa_submission_20260519.parity.json`
- Submission rows: `500`
- Schema: `question_id`, `hypothesis`
- Submission SHA-256: `3a6406516ec033dedf412a3a38df7a050ac49cebe78568c62ab5363d042fa8c5`

## Positioning

This result is an end-to-end GPT-4o LongMemEval-S QA artifact: ContextFit retrieves source evidence, GPT-4o produces the answer from retrieved context, and GPT-4o judges answer correctness. It is distinct from the retrieval-only evidence recall artifact, which reported `96.60%` Any@5 and `98.72%` Any@10 evidence recall.
