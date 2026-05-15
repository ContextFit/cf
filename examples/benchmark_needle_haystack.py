#!/usr/bin/env python3
"""Needle-in-a-Haystack benchmark for ContextFit.

This is a retrieval-focused NIAH/RULER-style synthetic benchmark. It hides exact
answer facts inside many distractor documents, then measures whether ContextFit
retrieves the chunk containing the answer.

It is designed for local comparisons against other retrieval stores:
- ContextFit hybrid/SID/BM25
- grep/Markdown scan baselines
- vector stores, once adapters exist
"""

from __future__ import annotations

import argparse
import json
import random
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from contextfit import RetrievalEngine

WORDS = [
    "satellite",
    "orchard",
    "quantum",
    "ledger",
    "harbor",
    "compiler",
    "lantern",
    "canyon",
    "velocity",
    "notebook",
    "reservoir",
    "signal",
    "archive",
    "matrix",
    "garden",
    "circuit",
    "voyage",
    "texture",
    "compass",
    "library",
]


def make_filler(rng: random.Random, sentences: int = 10) -> str:
    lines = []
    for i in range(sentences):
        terms = rng.sample(WORDS, 6)
        lines.append(
            f"Filler section {i}: {terms[0]} {terms[1]} observations describe "
            f"{terms[2]} patterns near {terms[3]}, with {terms[4]} and {terms[5]} context."
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class NeedleCase:
    needle_id: str
    answer: str
    query: str
    document: str
    insertion_percent: float


ANSWER_RE = re.compile(r"passcode for (NEEDLE-\d{4}) is ([A-Z0-9-]+)")


def make_needle_case(index: int, insertion_percent: float, rng: random.Random) -> NeedleCase:
    needle_id = f"NEEDLE-{index:04d}"
    answer = f"CODE-{rng.randrange(100000, 999999)}"
    query = f"What is the passcode for {needle_id}?"
    needle = (
        f"IMPORTANT NEEDLE FACT: The passcode for {needle_id} is {answer}. "
        f"Only this sentence contains the correct answer for {needle_id}."
    )

    filler_before = make_filler(rng, sentences=12)
    filler_after = make_filler(rng, sentences=12)
    before_lines = filler_before.splitlines()
    after_lines = filler_after.splitlines()
    all_filler = before_lines + after_lines
    insert_at = int(len(all_filler) * insertion_percent)
    lines = all_filler[:insert_at] + [needle] + all_filler[insert_at:]
    document = "\n".join(lines)

    return NeedleCase(
        needle_id=needle_id,
        answer=answer,
        query=query,
        document=document,
        insertion_percent=insertion_percent,
    )


def build_benchmark_kb(
    kb: Path,
    needle_count: int,
    distractor_count: int,
    seed: int,
) -> tuple[RetrievalEngine, list[NeedleCase], float]:
    rng = random.Random(seed)
    engine = RetrievalEngine.create(kb)
    cases = []

    start = time.perf_counter()
    for i in range(distractor_count):
        engine.ingest_text(
            make_filler(rng, sentences=24),
            chunk_size=192,
            overlap=32,
            metadata={"kind": "distractor", "doc_id": f"distractor-{i:04d}"},
        )

    for i in range(needle_count):
        insertion_percent = i / max(1, needle_count - 1)
        case = make_needle_case(i, insertion_percent, rng)
        cases.append(case)
        engine.ingest_text(
            case.document,
            chunk_size=192,
            overlap=32,
            metadata={
                "kind": "needle",
                "needle_id": case.needle_id,
                "answer": case.answer,
                "insertion_percent": insertion_percent,
            },
        )

    engine.train_learned_sid_generator()
    engine.save(kb)
    return engine, cases, time.perf_counter() - start


def chunk_contains_answer(engine: RetrievalEngine, chunk, case: NeedleCase) -> bool:
    text = engine.tokenizer.decode(chunk.tokens)
    return case.needle_id in text and case.answer in text


def run_contextfit_trials(
    engine: RetrievalEngine,
    cases: list[NeedleCase],
    method: str,
    top_k: int,
) -> dict:
    latencies = []
    hits = 0
    answer_hits = 0
    by_depth = []

    for case in cases:
        start = time.perf_counter()
        result = engine.query(case.query, method=method, top_k=top_k)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

        metadata_hit = any(
            chunk.metadata.get("needle_id") == case.needle_id
            for chunk in result.chunks
        )
        answer_hit = any(chunk_contains_answer(engine, chunk, case) for chunk in result.chunks)
        hits += int(metadata_hit)
        answer_hits += int(answer_hit)
        by_depth.append(
            {
                "needle_id": case.needle_id,
                "insertion_percent": case.insertion_percent,
                "hit": metadata_hit,
                "answer_hit": answer_hit,
                "latency_ms": elapsed * 1000,
                "retrieved_chunk_ids": [chunk.chunk_id for chunk in result.chunks],
            }
        )

    return {
        "method": method,
        "top_k": top_k,
        "recall_at_k": hits / len(cases),
        "answer_recall_at_k": answer_hits / len(cases),
        "avg_query_ms": (sum(latencies) / len(latencies)) * 1000,
        "max_query_ms": max(latencies) * 1000,
        "trials": by_depth,
    }


def run_grep_baseline(engine: RetrievalEngine, cases: list[NeedleCase]) -> dict:
    """Simple Markdown/source-like scan baseline over stored chunks."""
    latencies = []
    hits = 0

    chunks = list(engine.store.iter_chunks(level=0))
    decoded = [(chunk, engine.tokenizer.decode(chunk.tokens)) for chunk in chunks]

    for case in cases:
        start = time.perf_counter()
        found = any(case.needle_id in text and case.answer in text for _, text in decoded)
        latencies.append(time.perf_counter() - start)
        hits += int(found)

    return {
        "method": "linear_text_scan",
        "recall_at_1": hits / len(cases),
        "avg_query_ms": (sum(latencies) / len(latencies)) * 1000,
        "max_query_ms": max(latencies) * 1000,
    }


def run(args: argparse.Namespace) -> dict:
    if args.kb:
        kb = args.kb
        kb.mkdir(parents=True, exist_ok=True)
        engine, cases, build_seconds = build_benchmark_kb(
            kb,
            args.needles,
            args.distractors,
            args.seed,
        )
    else:
        tmp = tempfile.TemporaryDirectory()
        kb = Path(tmp.name) / "kb"
        engine, cases, build_seconds = build_benchmark_kb(
            kb,
            args.needles,
            args.distractors,
            args.seed,
        )

    load_start = time.perf_counter()
    loaded = RetrievalEngine.load(kb)
    load_seconds = time.perf_counter() - load_start

    methods = args.methods.split(",")
    results = [
        run_contextfit_trials(loaded, cases, method=method, top_k=args.top_k)
        for method in methods
    ]

    baseline = run_grep_baseline(loaded, cases) if args.grep_baseline else None
    stats = loaded.stats()

    return {
        "benchmark": "needle_in_haystack_synthetic",
        "needles": args.needles,
        "distractors": args.distractors,
        "top_k": args.top_k,
        "seed": args.seed,
        "chunks": stats["chunks"],
        "vocab_size": stats["index"]["vocab_size"],
        "postings_bytes": (kb / "inverted" / "postings.bin").stat().st_size,
        "storage_bytes": stats["storage"]["data_size_bytes"],
        "build_seconds": build_seconds,
        "load_seconds": load_seconds,
        "contextfit": results,
        "baseline": baseline,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ContextFit needle-in-haystack benchmark")
    parser.add_argument("--kb", type=Path, help="Optional persistent KB path")
    parser.add_argument("--needles", type=int, default=20)
    parser.add_argument("--distractors", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--methods", default="bm25,sid,hybrid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grep-baseline", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run(args)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("ContextFit Needle-in-a-Haystack Benchmark")
        print("=" * 44)
        print(f"needles: {result['needles']}  distractors: {result['distractors']}")
        print(f"chunks: {result['chunks']}  vocab: {result['vocab_size']}")
        print(f"build: {result['build_seconds']:.3f}s  load: {result['load_seconds']:.3f}s")
        for item in result["contextfit"]:
            print(
                f"{item['method']}: recall@{item['top_k']}={item['recall_at_k']:.3f} "
                f"answer_recall@{item['top_k']}={item['answer_recall_at_k']:.3f} "
                f"avg={item['avg_query_ms']:.2f}ms"
            )
        if result["baseline"]:
            b = result["baseline"]
            print(f"{b['method']}: recall={b['recall_at_1']:.3f} avg={b['avg_query_ms']:.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
