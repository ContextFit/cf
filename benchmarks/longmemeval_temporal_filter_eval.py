#!/usr/bin/env python3
"""Targeted LongMemEval temporal retrieval check for structured filters.

This is intentionally small and local. It compares:
  A. baseline token-native retrieval
  B. baseline + structured metadata date filters when a relative-date window is
     inferable from the question
  C. B + existing temporal date reranker over a wider candidate pool

No answer labels are used to build filters. Gold labels are used only for
evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.longmemeval_contextfit import (
    _relative_date_window,
    reciprocal_rank_fusion,
    session_to_text,
    temporal_date_rerank_sessions,
    unique_sessions_from_chunks,
)
from contextfit.retrieval.engine import RetrievalEngine

TEMPORAL_MISS_QIDS = [
    "4dfccbf8",
    "gpt4_fa19884d",
    "gpt4_e061b84g",
    "9a707b82",
    "gpt4_468eb063",
    "gpt4_468eb064",
    "eac54add",
    "gpt4_8279ba03",
]


def _query(item: dict[str, Any]) -> str:
    return f"Question date: {item.get('question_date','')}\nQuestion: {item['question']}"


def _window_filters(query: str) -> list[dict[str, str]]:
    window = _relative_date_window(query)
    if window is None:
        return []
    start, end = window
    return [
        {"field": "date", "op": "on_or_after", "value": start.isoformat()},
        {"field": "date", "op": "on_or_before", "value": end.isoformat()},
    ]


def _build_engine(item: dict[str, Any], chunk_size: int, overlap: int) -> RetrievalEngine:
    tmp = Path(tempfile.mkdtemp(prefix="cf-lme-temporal-filter-"))
    engine = RetrievalEngine.create(tmp)
    # Attach temp path for cleanup without extending the public engine API.
    engine._temporal_filter_tmp = tmp  # type: ignore[attr-defined]
    for sid, date, sess in zip(
        item["haystack_session_ids"],
        item["haystack_dates"],
        item["haystack_sessions"],
        strict=True,
    ):
        engine.ingest_text(
            session_to_text(sid, date, sess),
            chunk_size=chunk_size,
            overlap=overlap,
            metadata={
                "session_id": sid,
                "date": date,
                "question_id": item["question_id"],
                "kind": "session",
            },
            update_indexes=True,
        )
    return engine


def _cleanup_engine(engine: RetrievalEngine) -> None:
    tmp = getattr(engine, "_temporal_filter_tmp", None)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


def _retrieve(
    engine: RetrievalEngine,
    query: str,
    item: dict[str, Any],
    variant: str,
    top_k: int,
    retrieval_k: int,
    method: str,
    filter_pushdown_threshold: float,
) -> tuple[list[str], dict[str, Any] | None]:
    filters = _window_filters(query)
    use_filters = (
        variant in {
            "filters",
            "filters_temporal_rerank",
            "filters_broad_fusion",
            "filters_top2_broad_merge",
        }
        and bool(filters)
    )
    pool_k = max(retrieval_k, top_k)
    if variant in {"filters_temporal_rerank", "filters_broad_fusion"}:
        pool_k = max(pool_k, 100)

    result = engine.query(
        query,
        top_k=pool_k,
        method=method,
        max_tokens=200_000,
        filters=filters if use_filters else None,
        min_filter_matches=1,
        filter_pushdown_threshold=filter_pushdown_threshold,
    )
    sessions = unique_sessions_from_chunks(result.chunks)
    if variant == "filters_broad_fusion" and filters:
        broad_result = engine.query(
            query,
            top_k=pool_k,
            method=method,
            max_tokens=200_000,
        )
        broad_sessions = unique_sessions_from_chunks(broad_result.chunks)
        sessions = reciprocal_rank_fusion([sessions, broad_sessions])[:top_k]
    elif variant == "filters_top2_broad_merge" and filters:
        broad_result = engine.query(
            query,
            top_k=pool_k,
            method=method,
            max_tokens=200_000,
        )
        broad_sessions = unique_sessions_from_chunks(broad_result.chunks)
        sessions = _unique_sessions(
            sessions[:2] + broad_sessions + sessions[2:],
            top_k=top_k,
        )
    elif variant == "filters_temporal_rerank":
        session_dates = dict(zip(
            item["haystack_session_ids"],
            item["haystack_dates"],
            strict=True,
        ))
        sessions = temporal_date_rerank_sessions(
            query,
            session_dates,
            sessions,
            top_k=top_k,
        )
    else:
        sessions = sessions[:top_k]
    return sessions, result.filter_trace


def _unique_sessions(values: list[str], top_k: int) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
        if len(out) >= top_k:
            break
    return out


def _score(item: dict[str, Any], sessions: list[str]) -> dict[str, Any]:
    gold = set(item.get("answer_session_ids") or [])
    ranks = [sessions.index(g) + 1 for g in gold if g in sessions]
    return {
        "retrieved_sessions": sessions,
        "gold_sessions": sorted(gold),
        "best_rank": min(ranks) if ranks else None,
        "all_found_at": max(ranks) if len(ranks) == len(gold) and ranks else None,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    def metrics(sub: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(sub)
        out: dict[str, Any] = {"n": n}
        if not n:
            return out
        for k in (1, 3, 5, 10):
            out[f"any@{k}"] = sum(
                r["best_rank"] is not None and r["best_rank"] <= k
                for r in sub
            ) / n
            out[f"all@{k}"] = sum(
                r["all_found_at"] is not None and r["all_found_at"] <= k
                for r in sub
            ) / n
        out["mrr"] = sum(
            (1 / r["best_rank"]) if r["best_rank"] else 0
            for r in sub
        ) / n
        out["with_window"] = sum(bool(r["filters"]) for r in sub)
        return out

    return {variant: metrics(sub) for variant, sub in sorted(by_variant.items())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data",
        type=Path,
        default=Path("benchmarks/data/longmemeval_s_cleaned.json"),
        nargs="?",
    )
    parser.add_argument("--all-temporal", action="store_true")
    parser.add_argument("--qids", nargs="*", default=None)
    parser.add_argument("--method", choices=["exact", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieval-k", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--filter-pushdown-threshold", type=float, default=0.5)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/longmemeval_temporal_filter_eval_20260517.json"),
    )
    args = parser.parse_args()

    data = json.loads(args.data.read_text())
    if args.all_temporal:
        selected = [
            item for item in data
            if str(item.get("question_type") or "").startswith("temporal-reasoning")
        ]
    else:
        qids = set(args.qids or TEMPORAL_MISS_QIDS)
        selected = [item for item in data if item["question_id"] in qids]

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i, item in enumerate(selected, 1):
        query = _query(item)
        filters = _window_filters(query)
        engine = _build_engine(item, args.chunk_size, args.overlap)
        try:
            for variant in (
                "baseline",
                "filters",
                "filters_temporal_rerank",
                "filters_broad_fusion",
                "filters_top2_broad_merge",
            ):
                st = time.time()
                sessions, trace = _retrieve(
                    engine,
                    query,
                    item,
                    variant,
                    top_k=args.top_k,
                    retrieval_k=args.retrieval_k,
                    method=args.method,
                    filter_pushdown_threshold=args.filter_pushdown_threshold,
                )
                scored = _score(item, sessions)
                row = {
                    "question_id": item["question_id"],
                    "question_type": item.get("question_type"),
                    "question": item["question"],
                    "question_date": item.get("question_date"),
                    "variant": variant,
                    "filters": filters,
                    "filter_trace": trace,
                    "seconds": time.time() - st,
                    **scored,
                }
                rows.append(row)
            print(
                f"[{i}/{len(selected)}] {item['question_id']} "
                f"window={bool(filters)}",
                flush=True,
            )
        finally:
            _cleanup_engine(engine)

    payload = {
        "benchmark": "LongMemEval temporal structured-filter slice",
        "method": args.method,
        "top_k": args.top_k,
        "retrieval_k": args.retrieval_k,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "filter_pushdown_threshold": args.filter_pushdown_threshold,
        "all_temporal": args.all_temporal,
        "qids": [item["question_id"] for item in selected],
        "elapsed_seconds": time.time() - t0,
        "summary": _summarize(rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
