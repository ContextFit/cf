#!/usr/bin/env zsh
set -euo pipefail

cd /Users/christophe/.openclaw/tools/cf

export OPENAI_API_KEY="$(
  node -e 'const fs=require("fs"); const p=JSON.parse(fs.readFileSync(process.env.HOME+"/.openclaw/openclaw.json","utf8")); const key=p?.models?.providers?.openai?.apiKey || p?.providers?.openai?.apiKey || ""; if(!key) process.exit(2); process.stdout.write(key);'
)"

BASE="benchmarks/longmemeval_fusion_ab_baseline_20260524.json"
CAND="benchmarks/longmemeval_fusion_certificate_promotion_v4_20260524.json"

echo "[cert-v4] start $(date)"
echo "[cert-v4] baseline artifact ${BASE}"
echo "[cert-v4] candidate -> ${CAND}"
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
  --fusion-final-candidate-k 20 \
  --out "${CAND}"

.venv/bin/python - <<'PY'
import json
from pathlib import Path

for label, path in [
    ("baseline", Path("benchmarks/longmemeval_fusion_ab_baseline_20260524.json")),
    ("certificate-v4", Path("benchmarks/longmemeval_fusion_certificate_promotion_v4_20260524.json")),
]:
    payload = json.loads(path.read_text())
    overall = payload["summary"]["overall"]
    print(
        f"[cert-v4] {label}: n={overall['n']} "
        f"any@5={overall['any_recall@5']:.6f} "
        f"any@10={overall['any_recall@10']:.6f} "
        f"all@5={overall['all_evidence@5']:.6f} "
        f"mrr={overall['mrr']:.6f}"
    )

base = json.loads(Path("benchmarks/longmemeval_fusion_ab_baseline_20260524.json").read_text())
cand = json.loads(Path("benchmarks/longmemeval_fusion_certificate_promotion_v4_20260524.json").read_text())
base_rows = {r["question_id"]: r for r in base["rows"]}
wins = []
losses = []
all_wins = []
all_losses = []
cert_counts = {}
for row in cand["rows"]:
    old = base_rows[row["question_id"]]
    old_hit = old["best_rank"] is not None and old["best_rank"] <= 5
    new_hit = row["best_rank"] is not None and row["best_rank"] <= 5
    old_all = old["all_found_at"] is not None and old["all_found_at"] <= 5
    new_all = row["all_found_at"] is not None and row["all_found_at"] <= 5
    if not old_hit and new_hit:
        wins.append((row["question_id"], old["best_rank"], row["best_rank"], row["question_type"], row.get("retrieval_certificates")))
    elif old_hit and not new_hit:
        losses.append((row["question_id"], old["best_rank"], row["best_rank"], row["question_type"], row.get("retrieval_certificates")))
    if not old_all and new_all:
        all_wins.append((row["question_id"], old["all_found_at"], row["all_found_at"], row["question_type"], row.get("retrieval_certificates")))
    elif old_all and not new_all:
        all_losses.append((row["question_id"], old["all_found_at"], row["all_found_at"], row["question_type"], row.get("retrieval_certificates")))
    for cert in row.get("retrieval_certificates") or []:
        cert_counts[cert.get("certificate", "unknown")] = cert_counts.get(cert.get("certificate", "unknown"), 0) + 1
print(f"[cert-v4] paired top5 wins={len(wins)} losses={len(losses)}")
print(f"[cert-v4] paired all5 wins={len(all_wins)} losses={len(all_losses)}")
print(f"[cert-v4] certificate_counts={cert_counts}")
print(f"[cert-v4] wins={wins}")
print(f"[cert-v4] losses={losses}")
print(f"[cert-v4] all_wins={all_wins}")
print(f"[cert-v4] all_losses={all_losses}")
print("[cert-v4] done")
PY

echo "[cert-v4] done $(date)"
