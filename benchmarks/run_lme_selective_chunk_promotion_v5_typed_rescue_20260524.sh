#!/usr/bin/env zsh
set -euo pipefail

cd /Users/christophe/.openclaw/tools/cf

export OPENAI_API_KEY="$(
  node -e 'const fs=require("fs"); const p=JSON.parse(fs.readFileSync(process.env.HOME+"/.openclaw/openclaw.json","utf8")); const key=p?.models?.providers?.openai?.apiKey || p?.providers?.openai?.apiKey || ""; if(!key) process.exit(2); process.stdout.write(key);'
)"

BASE="benchmarks/longmemeval_fusion_certificate_promotion_v5_typed_rescue_20260524.json"
CAND="benchmarks/longmemeval_fusion_selective_chunk_promotion_v5_typed_rescue_20260524.json"

echo "[selective-chunk-v5] start $(date)"
echo "[selective-chunk-v5] baseline artifact ${BASE}"
echo "[selective-chunk-v5] candidate -> ${CAND}"
.venv/bin/python benchmarks/longmemeval_contextfit.py \
  benchmarks/data/longmemeval_s_cleaned.json \
  --limit 0 \
  --method hybrid \
  --top-k-chunks 10 \
  --retrieval-k 100 \
  --chunk-size 2048 \
  --overlap 128 \
  --rank-by-session \
  --conversation-chunks \
  --conversation-parent \
  --coverage-rerank \
  --structured-temporal-filters \
  --openai-fusion \
  --openai-chunk-fusion selective \
  --fusion-certificate-promotion \
  --fusion-typed-rescue \
  --fusion-final-candidate-k 80 \
  --out "${CAND}"

.venv/bin/python - <<'PY'
import json
from pathlib import Path

base = json.loads(Path("benchmarks/longmemeval_fusion_certificate_promotion_v5_typed_rescue_20260524.json").read_text())
cand = json.loads(Path("benchmarks/longmemeval_fusion_selective_chunk_promotion_v5_typed_rescue_20260524.json").read_text())

for label, payload in [("full-session-v5", base), ("selective-chunk-v5", cand)]:
    overall = payload["summary"]["overall"]
    print(
        f"[selective-chunk-v5] {label}: n={overall['n']} "
        f"any@5={overall['any_recall@5']:.6f} "
        f"any@10={overall['any_recall@10']:.6f} "
        f"all@5={overall['all_evidence@5']:.6f} "
        f"all@10={overall['all_evidence@10']:.6f} "
        f"mrr={overall['mrr']:.6f}"
    )

base_rows = {r["question_id"]: r for r in base["rows"]}
mode_counts = {}
any_wins = []
any_losses = []
all_wins = []
all_losses = []

def hit(row, key):
    return row.get(key) is not None and row[key] <= 5

for row in cand["rows"]:
    old = base_rows[row["question_id"]]
    mode = row.get("openai_vector_mode", "unknown")
    mode_counts[mode] = mode_counts.get(mode, 0) + 1
    old_any = hit(old, "best_rank")
    new_any = hit(row, "best_rank")
    if not old_any and new_any:
        any_wins.append((row["question_id"], row["question_type"], old["best_rank"], row["best_rank"], mode))
    elif old_any and not new_any:
        any_losses.append((row["question_id"], row["question_type"], old["best_rank"], row["best_rank"], mode))
    old_all = hit(old, "all_evidence_rank")
    new_all = hit(row, "all_evidence_rank")
    if not old_all and new_all:
        all_wins.append((row["question_id"], row["question_type"], old["all_evidence_rank"], row["all_evidence_rank"], mode))
    elif old_all and not new_all:
        all_losses.append((row["question_id"], row["question_type"], old["all_evidence_rank"], row["all_evidence_rank"], mode))

print(f"[selective-chunk-v5] vector_modes={mode_counts}")
print(f"[selective-chunk-v5] paired_any5 wins={len(any_wins)} losses={len(any_losses)}")
print(f"[selective-chunk-v5] paired_all5 wins={len(all_wins)} losses={len(all_losses)}")
print(f"[selective-chunk-v5] any5_wins={any_wins}")
print(f"[selective-chunk-v5] any5_losses={any_losses}")
print(f"[selective-chunk-v5] all5_wins={all_wins}")
print(f"[selective-chunk-v5] all5_losses={all_losses}")
print("[selective-chunk-v5] done")
PY

echo "[selective-chunk-v5] done $(date)"
