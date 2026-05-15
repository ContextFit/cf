"""Needle-in-a-Haystack benchmark smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from examples.benchmark_needle_haystack import run


def test_needle_haystack_benchmark_recalls_exact_needles(tmp_path: Path):
    args = argparse.Namespace(
        kb=tmp_path / "kb",
        needles=8,
        distractors=40,
        top_k=5,
        methods="bm25,sid,hybrid",
        seed=7,
        grep_baseline=True,
        json=True,
    )

    result = run(args)

    assert result["benchmark"] == "needle_in_haystack_synthetic"
    assert result["needles"] == 8
    assert result["chunks"] > result["needles"]
    assert result["postings_bytes"] > 0
    assert result["baseline"]["recall_at_1"] == 1.0

    by_method = {item["method"]: item for item in result["contextfit"]}
    assert by_method["bm25"]["answer_recall_at_k"] == 1.0
    assert by_method["hybrid"]["answer_recall_at_k"] == 1.0
    # SID should at least be usable and fast on exact-identifier queries.
    assert by_method["sid"]["answer_recall_at_k"] >= 0.75

    for item in result["contextfit"]:
        assert item["avg_query_ms"] < 100
        assert len(item["trials"]) == 8
