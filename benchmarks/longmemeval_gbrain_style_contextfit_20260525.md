# LongMemEval-S gbrain-style Session Retrieval

Date: 2026-05-25

This report scores existing full-run ContextFit LongMemEval-S artifacts with the
same headline metric used by `garrytan/gbrain-evals`: session retrieval R@5
over all 500 questions.

## Metric

- Dataset: `benchmarks/data/longmemeval_s_cleaned.json`
- Rows: 500
- Scoring target: `answer_session_ids`
- Hit: at least one answer session appears in the top K retrieved sessions
- Primary metric: R@5 over all 500 rows
- This is retrieval recall, not answer generation accuracy and not an LLM-judge
  score.

## Results

| System | R@1 | R@3 | R@5 | R@10 | Hits@5 | Embeddings | Vector store |
|---|---:|---:|---:|---:|---:|---|---|
| gbrain-hybrid | - | - | 97.60% | - | 488/500 | yes | local |
| MemPalace raw | - | - | 96.60% | - | 483/500 | yes | local |
| ContextFit token-native | 81.80% | 90.40% | 96.20% | 97.80% | 481/500 | no | no |
| ContextFit + OpenAI fusion | 84.60% | 95.20% | 99.00% | 99.60% | 495/500 | yes | no |

The gbrain row is the published `gbrain-evals` headline for LongMemEval `_s`.
The MemPalace row is the published raw/zero-API baseline cited by both
`gbrain-evals` and MemPalace's benchmark documentation.
The ContextFit rows are computed from local full-run artifacts listed below.

## ContextFit Artifacts

Token-native:

- Artifact: `benchmarks/longmemeval_token_native_certificate_promotion_v5_typed_rescue_tight_20260524.json`
- SHA-256: `c0e7ebc5d925549e1e3058b6100ab0786654c4d8c1bd4a99fe920c57f3ff2ea6`
- Config: hybrid retrieval, conversation chunks, parent/session context,
  coverage rerank, evidence certificates, typed rescue
- No embeddings, no vector DB, no LLM in retrieval

Optional fusion:

- Artifact: `benchmarks/longmemeval_fusion_selective_chunk_promotion_v5_typed_rescue_20260524.json`
- SHA-256: `ababf7387cb18c9310e82c35a57594aeef3cd40a2b60d9a1f336f69b65d3dcd2`
- Config: token-native ContextFit plus route-gated OpenAI chunk/full-session
  vector fusion, evidence certificates, typed rescue
- Uses OpenAI embeddings as a fusion signal; no vector DB and no LLM in retrieval

## Per-Type R@5

| Question type | Token-native | Fusion |
|---|---:|---:|
| knowledge-update | 98.7% | 100.0% |
| multi-session | 95.5% | 100.0% |
| single-session-assistant | 100.0% | 100.0% |
| single-session-preference | 90.0% | 86.7% |
| single-session-user | 95.7% | 100.0% |
| temporal-reasoning | 95.5% | 99.2% |

## Scoring Command

```bash
python3 - <<'PY'
import json
from pathlib import Path

files = [
    ("ContextFit token-native", "benchmarks/longmemeval_token_native_certificate_promotion_v5_typed_rescue_tight_20260524.json"),
    ("ContextFit fusion", "benchmarks/longmemeval_fusion_selective_chunk_promotion_v5_typed_rescue_20260524.json"),
]

for name, path in files:
    rows = json.loads(Path(path).read_text())["rows"]
    print(name)
    for k in (1, 3, 5, 10):
        hits = sum(r["best_rank"] is not None and r["best_rank"] <= k for r in rows)
        print(f"R@{k}: {hits}/{len(rows)} = {hits / len(rows) * 100:.2f}%")
PY
```

## Website Wording

Recommended concise wording:

> On LongMemEval-S session retrieval, ContextFit reaches 96.20% R@5 with no
> embeddings or vector database. With optional OpenAI fusion, ContextFit reaches
> 99.00% R@5, above gbrain-hybrid's published 97.60% R@5 and MemPalace raw's
> published 96.60% R@5 on the same split and top-5 retrieval metric.

Keep the qualifier "session retrieval R@5" attached to the claim. Do not call
this a LongMemEval QA score.
