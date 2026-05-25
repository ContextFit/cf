# LongMemEval-S Retrieval Submission

Date: 2026-05-25

This is the public submission packet for ContextFit's LongMemEval-S session
retrieval result.

## Headline

ContextFit + OpenAI fusion reaches **99.00% R@5** on LongMemEval-S session
retrieval over all 500 questions.

ContextFit token-native reaches **96.20% R@5** on the same metric, with no
embeddings, no vector database, and no LLM in the retrieval loop.

## Metric Boundary

This is **session retrieval recall**, not end-to-end QA accuracy.

- Dataset: `longmemeval_s`
- Rows: 500
- Target: `answer_session_ids`
- Hit: at least one answer session appears in the top K retrieved sessions
- Primary metric: R@5
- Judge model: none
- Answer-generation model: none

## Results

| System | R@1 | R@3 | R@5 | R@10 | Hits@5 | Embeddings | Vector store |
|---|---:|---:|---:|---:|---:|---|---|
| ContextFit token-native | 81.80% | 90.40% | 96.20% | 97.80% | 481/500 | no | no |
| ContextFit + OpenAI fusion | 84.60% | 95.20% | 99.00% | 99.60% | 495/500 | yes | no |
| gbrain-hybrid published reference | - | - | 97.60% | - | 488/500 | yes | local |
| MemPalace raw published reference | - | - | 96.60% | - | 483/500 | yes | local |

## Artifacts

- Machine-readable packet: `benchmarks/submissions/contextfit_longmemeval_s_session_retrieval_r5_20260525.json`
- Public report: <https://www.context.fit/longmemeval_gbrain_style_contextfit_20260525.html>
- Token-native artifact: `benchmarks/longmemeval_token_native_certificate_promotion_v5_typed_rescue_tight_20260524.json`
- Token-native SHA-256: `c0e7ebc5d925549e1e3058b6100ab0786654c4d8c1bd4a99fe920c57f3ff2ea6`
- Fusion artifact: `benchmarks/longmemeval_fusion_selective_chunk_promotion_v5_typed_rescue_20260524.json`
- Fusion SHA-256: `ababf7387cb18c9310e82c35a57594aeef3cd40a2b60d9a1f336f69b65d3dcd2`

## Submission Text

ContextFit + OpenAI fusion reports 99.00% R@5 (495/500) on LongMemEval-S
session retrieval. ContextFit token-native reports 96.20% R@5 (481/500) on
the same metric with no embeddings, no vector database, and no LLM in the
retrieval loop. The metric is session retrieval recall against
`answer_session_ids`, not end-to-end QA accuracy or an LLM-judge score.

Public report:
<https://www.context.fit/longmemeval_gbrain_style_contextfit_20260525.html>

Repository:
<https://github.com/ContextFit/cf>

Commit:
`0ddf3a725615011f3ff76146874039455b1409ab`
