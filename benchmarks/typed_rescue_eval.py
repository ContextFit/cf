#!/usr/bin/env python3
"""Targeted non-LongMemEval gate for typed-rescue certificates.

This benchmark intentionally evaluates the rescue decision layer directly with
fictional, product-shaped memory cases.  The goal is not broad retrieval score;
it is to prove that typed rescue fires on preference/temporal cases where it
should and stays silent on generic-overlap or protected-tail cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contextfit.retrieval.evidence_certificates import apply_typed_rescue


def evaluate_case(case: dict[str, Any], *, top_k: int, candidate_k: int) -> dict[str, Any]:
    before = list(case["initial_order"][:top_k])
    result = apply_typed_rescue(
        case["query"],
        case["source_texts"],
        case["base_order"],
        before,
        top_k,
        candidate_k=candidate_k,
        route_mode=case.get("route_mode"),
    )
    after = result.session_ids[:top_k]
    traces = result.traces()
    certs = [trace.get("certificate") for trace in traces if trace.get("action") == "promote"]
    expected = case.get("expected_certificate")
    gold = case["gold_session_id"]
    before_hit = gold in before
    after_hit = gold in after

    if expected:
        passed = after_hit and expected in certs
    else:
        passed = after == before and not certs

    return {
        "id": case["id"],
        "behavior": case["behavior"],
        "passed": passed,
        "before_hit": before_hit,
        "after_hit": after_hit,
        "before": before,
        "after": after,
        "gold_session_id": gold,
        "expected_certificate": expected,
        "certificates": traces,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    promotion_rows = [r for r in rows if r["expected_certificate"]]
    risk_rows = [r for r in rows if not r["expected_certificate"]]
    wins = sum((not r["before_hit"]) and r["after_hit"] for r in rows)
    losses = sum(r["before_hit"] and not r["after_hit"] for r in rows)
    unexpected_promotions = [
        r["id"]
        for r in risk_rows
        if any(t.get("action") == "promote" for t in r["certificates"])
    ]
    return {
        "overall": {
            "n": len(rows),
            "passed": sum(r["passed"] for r in rows),
            "pass_rate": (sum(r["passed"] for r in rows) / len(rows)) if rows else 0.0,
            "paired_wins": wins,
            "paired_losses": losses,
        },
        "promotion_cases": {
            "n": len(promotion_rows),
            "passed": sum(r["passed"] for r in promotion_rows),
        },
        "risk_controls": {
            "n": len(risk_rows),
            "passed": sum(r["passed"] for r in risk_rows),
            "unexpected_promotions": unexpected_promotions,
        },
        "certificate_counts": {
            cert: sum(
                1
                for r in rows
                for t in r["certificates"]
                if t.get("action") == "promote" and t.get("certificate") == cert
            )
            for cert in sorted({
                t.get("certificate")
                for r in rows
                for t in r["certificates"]
                if t.get("action") == "promote"
            })
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data",
        type=Path,
        default=Path("benchmarks/data/typed_rescue_eval.json"),
        nargs="?",
    )
    parser.add_argument("--out", type=Path, default=Path("benchmarks/typed_rescue_eval_20260524.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    args = parser.parse_args()

    cases = json.loads(args.data.read_text())
    rows = [evaluate_case(case, top_k=args.top_k, candidate_k=args.candidate_k) for case in cases]
    summary = {
        "data": str(args.data),
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "summary": summarize(rows),
        "rows": rows,
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary["summary"], indent=2))
    if summary["summary"]["overall"]["passed"] != summary["summary"]["overall"]["n"]:
        return 1
    if summary["summary"]["overall"]["paired_losses"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
