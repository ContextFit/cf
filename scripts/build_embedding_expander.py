#!/usr/bin/env python3
"""
Build and save an EmbeddingExpander for a ContextFit knowledge base.

Usage:
    python scripts/build_embedding_expander.py <kb_path> [--dry-run]

Embeds only lexical BPE tokens (alphabetic, min-length 3) to keep cost low
and signal quality high. Saves embedding_expander.npz to <kb_path>/expanders/.

Progress is logged to stdout so you can tail it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure src/ is on the path when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextfit.retrieval.engine import RetrievalEngine
from contextfit.index.semantic_expand import EmbeddingExpander


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("kb_path", help="Path to knowledge base (e.g. ../tmp/contextfit-email-kb)")
    p.add_argument("--dry-run", action="store_true", help="Print token stats then exit")
    p.add_argument("--no-filter", action="store_true", help="Embed ALL tokens (not recommended)")
    args = p.parse_args()

    kb = Path(args.kb_path).resolve()
    if not kb.exists():
        print(f"ERROR: KB not found at {kb}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    print(f"Loading engine from {kb} ...", flush=True)
    engine = RetrievalEngine.load(kb)
    print(f"  Loaded in {time.time()-t0:.1f}s | chunks={len(engine.store)}", flush=True)

    # Collect all unique token IDs from L0 chunks
    print("Scanning vocab ...", flush=True)
    t1 = time.time()
    seen: set[int] = set()
    for chunk in engine.store.iter_chunks(level=0):
        toks = chunk.tokens.tolist() if hasattr(chunk.tokens, "tolist") else list(chunk.tokens)
        seen.update(toks)
    print(f"  {len(seen)} unique tokens in {time.time()-t1:.1f}s", flush=True)

    if args.dry_run:
        from contextfit.index.semantic_expand import _is_lexical_token
        lexical = [(t, engine.tokenizer.decode([t])) for t in sorted(seen)]
        lexical = [(t, s) for t, s in lexical if _is_lexical_token(s)]
        print(f"Dry run: would embed {len(lexical)} / {len(seen)} tokens")
        cost_est = len(lexical) / 1_000_000 * 0.02   # text-embedding-3-small pricing
        print(f"Estimated cost: ~${cost_est:.4f} USD")
        print("Sample lexical tokens:")
        for t, s in lexical[100:130]:
            print(f"  {t:6d}  {repr(s)}")
        return

    exp_path = kb / "expanders"
    exp_path.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding EmbeddingExpander (filter_lexical={not args.no_filter}) ...", flush=True)
    t2 = time.time()
    emb = EmbeddingExpander()
    emb.build(
        corpus_token_ids=seen,
        tokenizer=engine.tokenizer,
        cache_path=exp_path,
        filter_lexical=not args.no_filter,
    )
    elapsed = time.time() - t2
    stats = emb.stats()
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Embedded {stats['tokens_embedded']} tokens | dim={stats['vector_dim']} | size={stats['size_mb']} MB")
    print(f"Saved to {exp_path / 'embedding_expander.npz'}")


if __name__ == "__main__":
    main()
