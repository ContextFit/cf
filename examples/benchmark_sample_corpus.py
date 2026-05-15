#!/usr/bin/env python3
"""Run a deterministic ContextFit sample-corpus benchmark.

This is a lightweight local benchmark, not a formal benchmark harness. It is
useful for comparing storage/index changes on a laptop or OpenClaw node.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from contextfit import RetrievalEngine

TOPICS = {
    "python_async": ["python", "async", "await", "coroutine", "asyncio", "event loop"],
    "rust_memory": ["rust", "ownership", "borrowing", "lifetimes", "compiler", "memory safety"],
    "drone_mapping": ["drone", "orthomosaic", "photogrammetry", "gps", "tileset", "mesh"],
    "agent_memory": [
        "agent",
        "memory",
        "semantic id",
        "retrieval",
        "knowledge graph",
        "token store",
    ],
}

QUERIES = [
    ("python asyncio coroutine task", "python_async"),
    ("rust ownership lifetimes compiler", "rust_memory"),
    ("drone orthomosaic photogrammetry tileset", "drone_mapping"),
    ("agent memory semantic id retrieval", "agent_memory"),
]


def make_document(topic: str, i: int, repeats: int = 8) -> str:
    terms = TOPICS[topic]
    lines = []
    for j in range(repeats):
        rotated = terms[j % len(terms) :] + terms[: j % len(terms)]
        lines.append(
            f"Document {i} section {j} covers {topic}. "
            f"Signals: {', '.join(rotated)}. "
            f"The relationship links {rotated[0]}, {rotated[1]}, and {rotated[2]}."
        )
    return "\n".join(lines)


def run(kb: Path, docs_per_topic: int) -> dict:
    engine = RetrievalEngine.create(kb)

    ingest_start = time.perf_counter()
    for topic in TOPICS:
        for i in range(docs_per_topic):
            engine.ingest_text(
                make_document(topic, i),
                chunk_size=128,
                overlap=24,
                metadata={"topic": topic},
            )
    learned = engine.train_learned_sid_generator()
    engine.save(kb)
    ingest_seconds = time.perf_counter() - ingest_start

    load_start = time.perf_counter()
    loaded = RetrievalEngine.load(kb)
    load_seconds = time.perf_counter() - load_start

    query_metrics = []
    correct = 0
    query_start = time.perf_counter()
    for query, expected_topic in QUERIES:
        one_start = time.perf_counter()
        result = loaded.query(query, method="hybrid", top_k=5)
        elapsed = time.perf_counter() - one_start
        top_topic = result.chunks[0].metadata.get("topic") if result.chunks else None
        correct += int(top_topic == expected_topic)
        query_metrics.append(
            {
                "query": query,
                "expected_topic": expected_topic,
                "top_topic": top_topic,
                "seconds": elapsed,
                "chunks": len(result.chunks),
                "input_tokens": len(result.input_ids),
                "sid_predictions": len(result.sid_predictions or []),
            }
        )
    total_query_seconds = time.perf_counter() - query_start

    stats = loaded.stats()
    postings_path = kb / "inverted" / "postings.bin"
    return {
        "docs_per_topic": docs_per_topic,
        "topics": len(TOPICS),
        "chunks": stats["chunks"],
        "accuracy_at_1": correct / len(QUERIES),
        "ingest_seconds": ingest_seconds,
        "load_seconds": load_seconds,
        "total_query_seconds": total_query_seconds,
        "avg_query_ms": (total_query_seconds / len(QUERIES)) * 1000,
        "postings_bytes": postings_path.stat().st_size if postings_path.exists() else None,
        "storage_bytes": stats["storage"]["data_size_bytes"],
        "vocab_size": stats["index"]["vocab_size"],
        "learned_sid_trained_chunks": learned.trained_chunks,
        "queries": query_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ContextFit on a sample corpus")
    parser.add_argument("--kb", type=Path, help="KB path. Defaults to a temporary directory.")
    parser.add_argument("--docs-per-topic", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    if args.kb:
        args.kb.mkdir(parents=True, exist_ok=True)
        result = run(args.kb, args.docs_per_topic)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run(Path(tmpdir) / "kb", args.docs_per_topic)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("ContextFit sample benchmark")
        print("=" * 32)
        for key, value in result.items():
            if key != "queries":
                print(f"{key}: {value}")
        print("\nqueries:")
        for item in result["queries"]:
            print(
                f"- {item['query']} -> {item['top_topic']} "
                f"({item['seconds'] * 1000:.2f} ms)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
