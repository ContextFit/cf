#!/usr/bin/env python3
"""Sample 5 eval cases to get accurate Mem0 timing."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from agent_memory_eval import eval_mem0

data = json.loads(Path("benchmarks/data/agent_memory_eval.json").read_text())
sample = data[:5]

total_ingest_ms = 0.0
total_query_ms = 0.0
total_sessions = 0

for item in sample:
    retrieved, ingest_ms, query_ms = eval_mem0(item, item["question"], 5)
    n = len(item["sessions"])
    total_ingest_ms += ingest_ms
    total_query_ms += query_ms
    total_sessions += n
    print(f"{item['id']}: ingest={ingest_ms:.0f}ms ({ingest_ms/n:.0f}ms/sess) query={query_ms:.0f}ms retrieved={retrieved[:3]}")

print(f"\nAverages over {len(sample)} cases / {total_sessions} sessions:")
print(f"  Ingest: {total_ingest_ms/len(sample):.0f}ms per case / {total_ingest_ms/total_sessions:.0f}ms per session")
print(f"  Query:  {total_query_ms/len(sample):.0f}ms")
