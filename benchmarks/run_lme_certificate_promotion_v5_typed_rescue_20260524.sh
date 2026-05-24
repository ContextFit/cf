#!/usr/bin/env zsh
set -euo pipefail

cd /Users/christophe/.openclaw/tools/cf

export OPENAI_API_KEY="$(
  node -e 'const fs=require("fs"); const p=JSON.parse(fs.readFileSync(process.env.HOME+"/.openclaw/openclaw.json","utf8")); const key=p?.models?.providers?.openai?.apiKey || p?.providers?.openai?.apiKey || ""; if(!key) process.exit(2); process.stdout.write(key);'
)"

BASE="benchmarks/longmemeval_fusion_ab_baseline_20260524.json"
CAND="benchmarks/longmemeval_fusion_certificate_promotion_v5_typed_rescue_20260524.json"

echo "[cert-v5-rescue] start $(date)"
echo "[cert-v5-rescue] baseline artifact ${BASE}"
echo "[cert-v5-rescue] candidate -> ${CAND}"
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
  --fusion-certificate-promotion \
  --fusion-typed-rescue \
  --fusion-final-candidate-k 80 \
  --out "${CAND}"

.venv/bin/python - <<'PY'
import json
from pathlib import Path

for label, path in [
    ("baseline", Path("benchmarks/longmemeval_fusion_ab_baseline_20260524.json")),
    ("certificate-v4", Path("benchmarks/longmemeval_fusion_certificate_promotion_v4_20260524.json")),
    ("certificate-v5-rescue", Path("benchmarks/longmemeval_fusion_certificate_promotion_v5_typed_rescue_20260524.json")),
]:
    payload = json.loads(path.read_text())
    overall = payload["summary"]["overall"]
    print(
        f"[cert-v5-rescue] {label}: n={overall['n']} "
        f"any@5={overall['any_recall@5']:.6f} "
        f"any@10={overall['any_recall@10']:.6f} "
        f"all@5={overall['all_evidence@5']:.6f} "
        f"mrr={overall['mrr']:.6f}"
    )

base = json.loads(Path("benchmarks/longmemeval_fusion_ab_baseline_20260524.json").read_text())
v4 = json.loads(Path("benchmarks/longmemeval_fusion_certificate_promotion_v4_20260524.json").read_text())
cand = json.loads(Path("benchmarks/longmemeval_fusion_certificate_promotion_v5_typed_rescue_20260524.json").read_text())
base_rows = {r["question_id"]: r for r in base["rows"]}
v4_rows = {r["question_id"]: r for r in v4["rows"]}

def hit(row, key="best_rank"):
    return row[key] is not None and row[key] <= 5

wins = []
losses = []
v4_wins = []
v4_losses = []
cert_counts = {}
for row in cand["rows"]:
    old = base_rows[row["question_id"]]
    prev = v4_rows[row["question_id"]]
    old_hit = hit(old)
    new_hit = hit(row)
    prev_hit = hit(prev)
    if not old_hit and new_hit:
        wins.append((row["question_id"], old["best_rank"], row["best_rank"], row["question_type"], row.get("retrieval_certificates")))
    elif old_hit and not new_hit:
        losses.append((row["question_id"], old["best_rank"], row["best_rank"], row["question_type"], row.get("retrieval_certificates")))
    if not prev_hit and new_hit:
        v4_wins.append((row["question_id"], prev["best_rank"], row["best_rank"], row["question_type"], row.get("retrieval_certificates")))
    elif prev_hit and not new_hit:
        v4_losses.append((row["question_id"], prev["best_rank"], row["best_rank"], row["question_type"], row.get("retrieval_certificates")))
    for cert in row.get("retrieval_certificates") or []:
        cert_counts[cert.get("certificate", "unknown")] = cert_counts.get(cert.get("certificate", "unknown"), 0) + 1
print(f"[cert-v5-rescue] paired vs baseline top5 wins={len(wins)} losses={len(losses)}")
print(f"[cert-v5-rescue] paired vs v4 top5 wins={len(v4_wins)} losses={len(v4_losses)}")
print(f"[cert-v5-rescue] certificate_counts={cert_counts}")
print(f"[cert-v5-rescue] wins_vs_baseline={wins}")
print(f"[cert-v5-rescue] losses_vs_baseline={losses}")
print(f"[cert-v5-rescue] wins_vs_v4={v4_wins}")
print(f"[cert-v5-rescue] losses_vs_v4={v4_losses}")
print("[cert-v5-rescue] done")
PY

echo "[cert-v5-rescue] done $(date)"
