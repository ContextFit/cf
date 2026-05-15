#!/usr/bin/env python3
"""
Before/after benchmark: semantic expansion vs baseline BM25.

For each query, shows:
  - tokens in original query
  - tokens added by semantic expansion
  - top-5 results with and without expansion
  - recall delta (new hits, dropped hits)

Usage:
    python scripts/benchmark_semantic_expansion.py <kb_path> [--queries-file queries.txt]
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextfit.retrieval.engine import RetrievalEngine
from contextfit.index.semantic_expand import SemanticExpander


# ---------------------------------------------------------------------------
# Default test queries per KB type
# ---------------------------------------------------------------------------

EMAIL_QUERIES = [
    "Acme contract proposal",
    "voicemail missed call",
    "invoice payment receipt",
    "residuals participations royalties",
    "flight booking travel",
    "school attendance absence",
    "job opportunity recruiter",
    "streaming video platform OTT",
]

TAX_QUERIES = [
    "rental property deductions",
    "retirement account contributions",
    "estimated tax payments",
    "K-1 partnership income",
    "home office expenses",
    "depreciation schedule",
    "state residency Texas California",
    "extension filing deadline",
]


def decode_tokens(engine, toks: list[int]) -> list[str]:
    return [engine.tokenizer.decode([t]).strip() for t in toks]


def run_query(engine, query: str, use_expansion: bool, top_k: int = 5):
    """Run query with or without semantic expansion; return chunk_ids + scores."""
    if use_expansion:
        result = engine.query(query, top_k=top_k, method="bm25", expand_query=True)
    else:
        # Temporarily remove semantic expander
        saved = engine.semantic_expander
        engine.semantic_expander = None
        result = engine.query(query, top_k=top_k, method="bm25", expand_query=True)
        engine.semantic_expander = saved
    return result


def chunk_preview(chunk, engine, max_chars: int = 120) -> str:
    try:
        text = engine.tokenizer.decode(chunk.tokens.tolist())
    except Exception:
        text = f"<chunk {chunk.chunk_id}>"
    text = " ".join(text.split())
    return textwrap.shorten(text, max_chars, placeholder="…")


def format_result_list(result, engine) -> list[str]:
    lines = []
    for i, (chunk, score) in enumerate(zip(result.chunks, result.scores)):
        src = ""
        if engine.metadata and len(engine.metadata) > 0:
            m = engine.metadata.get(chunk.chunk_id)
            if m:
                src = m.get("subject") or m.get("source", "")
                src = f"  [{src[:60]}]" if src else ""
        preview = chunk_preview(chunk, engine)
        lines.append(f"  {i+1}. [{score:.3f}] {preview}{src}")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("kb_path", help="Path to knowledge base")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--queries", nargs="+", help="Override queries")
    args = p.parse_args()

    kb = Path(args.kb_path).resolve()
    print(f"Loading {kb} ...", flush=True)
    engine = RetrievalEngine.load(kb)

    has_exp = engine.semantic_expander is not None
    emb_ready = has_exp and engine.semantic_expander.embeddings.stats().get("ready", False)
    print(f"Semantic expander: {'✓ embedding' if emb_ready else '✗ not loaded'}")
    print(f"Chunks: {len(engine.store)}  |  Metadata: {len(engine.metadata) > 0}")
    print()

    if not has_exp:
        print("No semantic expander found — run build_embedding_expander.py first.")
        sys.exit(1)

    # Pick queries
    if args.queries:
        queries = args.queries
    else:
        # Auto-detect KB type by chunk count
        queries = TAX_QUERIES if len(engine.store) < 1000 else EMAIL_QUERIES

    sep = "─" * 72

    for query in queries:
        print(sep)
        print(f"QUERY: {repr(query)}")

        # Show query tokens
        q_toks = engine.tokenizer.encode(query).tolist()
        q_strings = decode_tokens(engine, q_toks)
        print(f"  Query tokens ({len(q_toks)}): {q_strings}")

        # Show what semantic expansion adds
        sem_pairs = engine.semantic_expander.expand(q_toks)
        if sem_pairs:
            sem_tokens = [(engine.tokenizer.decode([t]).strip(), f"{w:.2f}")
                          for t, w in sem_pairs[:12]]
            print(f"  Semantic adds ({len(sem_pairs)}): {sem_tokens}")
        else:
            print("  Semantic adds: (none)")

        # Run both
        res_base = run_query(engine, query, use_expansion=False, top_k=args.top_k)
        res_exp  = run_query(engine, query, use_expansion=True,  top_k=args.top_k)

        base_ids = {c.chunk_id for c in res_base.chunks}
        exp_ids  = {c.chunk_id for c in res_exp.chunks}
        new_hits  = exp_ids - base_ids
        lost_hits = base_ids - exp_ids

        print(f"\n  BASELINE (no semantic):")
        for line in format_result_list(res_base, engine):
            print(line)

        print(f"\n  WITH SEMANTIC EXPANSION:")
        for line in format_result_list(res_exp, engine):
            print(line)

        if new_hits:
            print(f"\n  ✓ NEW hits gained:  {len(new_hits)} chunk(s)")
        if lost_hits:
            print(f"  ✗ hits displaced:   {len(lost_hits)} chunk(s)")
        if not new_hits and not lost_hits:
            print(f"\n  = Same top-{args.top_k} result set")

    print(sep)
    print("Done.")


if __name__ == "__main__":
    main()
