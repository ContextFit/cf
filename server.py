#!/usr/bin/env python3
"""
ContextFit HTTP query server.

Keeps one or more RetrievalEngine instances warm in memory.
Exposes a simple REST API for querying and incremental ingest.

Default port: 8765
Bind:         127.0.0.1 (localhost only)

Endpoints:
  GET  /status              — health check + loaded KB stats
  POST /query               — search a KB
  POST /query_auto          — auto-routed search (episode_score/bm25/fusion)
  POST /ingest              — add files to a KB (incremental)
  POST /rebuild-expanders   — rebuild embedding expander for a KB

Usage:
  python server.py --kb memory:/path/to/memory-kb --kb email:/path/to/email-kb
  python server.py --kb memory:/path/to/memory-kb --port 8765
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent / "src"))

from contextfit.retrieval.engine import RetrievalEngine
from contextfit.retrieval.extractive import default_reference_expiry, extract_evidence, format_evidence_compact
from contextfit.core.tokenizer import Tokenizer

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    kb: str = "memory"
    top_k: int = 10
    method: str = "hybrid"
    expand_query: bool = True
    filter_domain: str | None = None
    filter_field: list[str] | None = None   # [field, value]
    include_text: bool = True
    max_text_chars: int = 400
    return_spans: bool = True
    max_spans: int = 5
    extractive: str = "none"  # none | spans | rows | bullets | auto
    max_evidence_chars: int = 1200
    response_format: str = "json"  # json | compact
    citation_mode: str = "inline"  # inline | handles
    reference_ttl_seconds: int = 3600


class IngestRequest(BaseModel):
    kb: str = "memory"
    paths: list[str]
    chunk_size: int = 512
    overlap: int = 64


class QueryAutoRequest(BaseModel):
    query: str
    kb: str = "memory"
    top_k: int = 10
    retrieval_k: int = 50
    method: str = "hybrid"
    include_text: bool = True
    max_text_chars: int = 400
    return_spans: bool = True
    max_spans: int = 5
    extractive: str = "none"  # none | spans | rows | bullets | auto
    max_evidence_chars: int = 1200
    response_format: str = "json"  # json | compact
    citation_mode: str = "inline"  # inline | handles
    reference_ttl_seconds: int = 3600


class RebuildExpandersRequest(BaseModel):
    kb: str = "memory"


# ---------------------------------------------------------------------------
# App + state
# ---------------------------------------------------------------------------

app = FastAPI(title="ContextFit", version="0.1.1")

# kb_name -> {engine, path, loaded_at}
_engines: dict[str, dict[str, Any]] = {}
_kb_paths: dict[str, Path] = {}


def _load_engine(name: str, path: Path) -> RetrievalEngine:
    print(f"[cf-server] Loading KB '{name}' from {path} ...", flush=True)
    t0 = time.time()
    engine = RetrievalEngine.load(path)
    elapsed = time.time() - t0
    stats = engine.stats()
    print(f"[cf-server] '{name}' ready: {stats['chunks']} chunks in {elapsed:.1f}s", flush=True)
    _engines[name] = {"engine": engine, "path": path, "loaded_at": time.time()}
    return engine


def _get_engine(name: str) -> RetrievalEngine:
    if name not in _engines:
        if name in _kb_paths:
            return _load_engine(name, _kb_paths[name])
        raise HTTPException(status_code=404, detail=f"KB '{name}' not found. Loaded KBs: {list(_kb_paths)}")
    return _engines[name]["engine"]


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def _find_text_spans(text: str, query: str, max_spans: int = 5) -> list[dict[str, Any]]:
    """Return proof-grade literal spans for a query inside decoded chunk text."""
    spans: list[dict[str, Any]] = []
    if not query.strip():
        return spans

    lower_text = text.lower()
    lower_query = query.lower()

    # Highest confidence: exact full phrase.
    start = 0
    while len(spans) < max_spans:
        idx = lower_text.find(lower_query, start)
        if idx < 0:
            break
        end = idx + len(query)
        spans.append({
            "kind": "phrase",
            "start_char": idx,
            "end_char": end,
            "line": _line_for_offset(text, idx),
            "match": text[idx:end],
        })
        start = end

    # Then token/word matches, useful for hybrid validation when the full
    # phrase is not contiguous.  Keep this simple and deterministic.
    if len(spans) < max_spans:
        terms = []
        seen = set()
        for term in re.findall(r"[\w.-]+", query):
            key = term.lower()
            if key not in seen:
                seen.add(key)
                terms.append(term)
        for term in terms:
            if len(spans) >= max_spans:
                break
            for m in re.finditer(re.escape(term), text, flags=re.IGNORECASE):
                spans.append({
                    "kind": "term",
                    "start_char": m.start(),
                    "end_char": m.end(),
                    "line": _line_for_offset(text, m.start()),
                    "match": text[m.start():m.end()],
                })
                break

    return spans


def _find_tmd_rows(text: str, query: str, max_rows: int = 5) -> list[dict[str, Any]]:
    """Extract matching TMD rows from a chunk, if any."""
    terms = [t.lower() for t in re.findall(r"[\w.-]+", query)]
    if not terms:
        return []
    rows: list[dict[str, Any]] = []
    row_re = re.compile(r"^(?P<table>\w+)\[(?P<id>[^\]]*)\]:\s*(?P<body>.+)$")
    for line_no, line in enumerate(text.splitlines(), start=1):
        m = row_re.match(line.strip())
        if not m:
            continue
        low = line.lower()
        if any(term in low for term in terms):
            rows.append({
                "table": m.group("table"),
                "row_id": m.group("id"),
                "line": line_no,
                "raw": line.strip(),
            })
            if len(rows) >= max_rows:
                break
    return rows


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/status")
def status():
    result = {"status": "ok", "kbs": {}}
    for name, info in _engines.items():
        stats = info["engine"].stats()
        exp = info["engine"].semantic_expander
        result["kbs"][name] = {
            "chunks": stats["chunks"],
            "path": str(info["path"]),
            "loaded_at": info["loaded_at"],
            "semantic_expander": exp.embeddings.stats() if exp else None,
        }
    # List unloaded KBs too
    for name, path in _kb_paths.items():
        if name not in result["kbs"]:
            result["kbs"][name] = {"path": str(path), "loaded": False}
    return result


@app.post("/query")
def query(req: QueryRequest):
    engine = _get_engine(req.kb)

    filter_field = tuple(req.filter_field) if req.filter_field and len(req.filter_field) == 2 else None

    t0 = time.time()
    result = engine.query(
        req.query,
        top_k=req.top_k,
        method=req.method,
        expand_query=req.expand_query,
        filter_domain=req.filter_domain,
        filter_field=filter_field,
    )
    elapsed_ms = (time.time() - t0) * 1000

    chunks_out = []
    compact_blocks: list[str] = []
    references: dict[str, Any] = {}
    evidence_index = 1
    compact_response = req.response_format == "compact"
    references_expire_at = default_reference_expiry(req.reference_ttl_seconds)
    for chunk, score in zip(result.chunks, result.scores):
        item: dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "score": round(score, 4),
        }
        if not compact_response:
            item["token_count"] = chunk.token_count
            item["metadata"] = chunk.metadata
        decoded_text: str | None = None
        if req.include_text or req.return_spans or req.extractive != "none":
            try:
                decoded_text = engine.tokenizer.decode(chunk.tokens.tolist())
            except Exception:
                decoded_text = None

        if req.include_text:
            if decoded_text is not None:
                item["text"] = decoded_text[:req.max_text_chars]
                if len(decoded_text) > req.max_text_chars:
                    item["text"] += "…"
            else:
                item["text"] = None

        if req.return_spans and decoded_text is not None:
            item["matches"] = _find_text_spans(decoded_text, req.query, max_spans=req.max_spans)
            source = str(chunk.metadata.get("source", ""))
            if chunk.metadata.get("domain") == "tmd" or source.endswith(".tmd"):
                item["tmd_rows"] = _find_tmd_rows(decoded_text, req.query, max_rows=req.max_spans)
        if req.extractive != "none" and decoded_text is not None:
            evidence = extract_evidence(
                decoded_text,
                req.query,
                mode=req.extractive,  # type: ignore[arg-type]
                max_items=req.max_spans,
                max_chars=req.max_evidence_chars,
                metadata=chunk.metadata,
            )
            if compact_response:
                formatted = format_evidence_compact(
                    evidence,
                    metadata=chunk.metadata,
                    chunk_id=chunk.chunk_id,
                    chunk_score=round(float(score), 4),
                    start_index=evidence_index,
                    citation_mode=req.citation_mode,  # type: ignore[arg-type]
                    expires_at=references_expire_at,
                )
                block: str
                if isinstance(formatted, tuple):
                    block, refs = formatted
                    references.update(refs)
                else:
                    block = formatted
                if block:
                    item["evidence"] = block
                    compact_blocks.append(block)
                    evidence_index += len(evidence)
            else:
                item["evidence"] = [entry.to_json() for entry in evidence]
        chunks_out.append(item)

    # Show what tokens were added by expansion
    q_tokens = engine.tokenizer.encode(req.query).tolist()
    expanded: list[str] = []
    if engine.semantic_expander and req.expand_query:
        pairs = engine.semantic_expander.expand(q_tokens)
        expanded = [engine.tokenizer.decode([t]).strip() for t, _ in pairs[:8]]

    response = {
        "query": req.query,
        "kb": req.kb,
        "method": req.method,
        "elapsed_ms": round(elapsed_ms, 1),
        "chunks": chunks_out,
        "expansion": expanded,
    }
    if compact_response:
        response["compact_context"] = "\n".join(compact_blocks)
        if req.citation_mode == "handles":
            response["references"] = references
            response["references_expire_at"] = references_expire_at
    return response


@app.post("/query_auto")
def query_auto(req: QueryAutoRequest):
    """Route a query to the best retrieval mode automatically.

    Uses the deterministic query router to pick between episode_score, bm25,
    atom_fusion, and episode_bm25_fusion.  Returns ranked session_ids plus
    the routing decision and top chunks with text/spans.
    """
    from contextfit.retrieval.query_router import describe_route

    engine = _get_engine(req.kb)
    t0 = time.time()

    auto_result = engine.query_auto(
        req.query,
        top_k=req.top_k,
        retrieval_k=req.retrieval_k,
        method=req.method,
    )
    elapsed_ms = (time.time() - t0) * 1000

    route = auto_result["route"]
    session_ids = auto_result["session_ids"]

    # Return top chunks from BM25 for text/span evidence even in episode_score
    # mode — the session ranking is the primary signal, but callers often want
    # chunk-level proof snippets too.
    try:
        chunk_result = engine.query(
            req.query,
            top_k=req.top_k,
            method=req.method,
            max_tokens=200_000,
        )
        chunks_out = []
        compact_blocks: list[str] = []
        references: dict[str, Any] = {}
        evidence_index = 1
        compact_response = req.response_format == "compact"
        references_expire_at = default_reference_expiry(req.reference_ttl_seconds)
        for chunk, score in zip(chunk_result.chunks, chunk_result.scores):
            item: dict[str, Any] = {
                "chunk_id": chunk.chunk_id,
                "score": round(float(score), 4),
            }
            if not compact_response:
                item["token_count"] = chunk.token_count
                item["metadata"] = chunk.metadata
            decoded_text: str | None = None
            if req.include_text or req.return_spans or req.extractive != "none":
                try:
                    decoded_text = engine.tokenizer.decode(chunk.tokens.tolist())
                except Exception:
                    decoded_text = None
            if req.include_text and decoded_text is not None:
                item["text"] = decoded_text[:req.max_text_chars]
                if len(decoded_text) > req.max_text_chars:
                    item["text"] += "…"
            if req.return_spans and decoded_text is not None:
                item["matches"] = _find_text_spans(decoded_text, req.query, max_spans=req.max_spans)
                source = str(chunk.metadata.get("source", ""))
                if chunk.metadata.get("domain") == "tmd" or source.endswith(".tmd"):
                    item["tmd_rows"] = _find_tmd_rows(decoded_text, req.query, max_rows=req.max_spans)
            if req.extractive != "none" and decoded_text is not None:
                evidence = extract_evidence(
                    decoded_text,
                    req.query,
                    mode=req.extractive,  # type: ignore[arg-type]
                    max_items=req.max_spans,
                    max_chars=req.max_evidence_chars,
                    metadata=chunk.metadata,
                )
                if compact_response:
                    formatted = format_evidence_compact(
                        evidence,
                        metadata=chunk.metadata,
                        chunk_id=chunk.chunk_id,
                        chunk_score=round(float(score), 4),
                        start_index=evidence_index,
                        citation_mode=req.citation_mode,  # type: ignore[arg-type]
                        expires_at=references_expire_at,
                    )
                    block: str
                    if isinstance(formatted, tuple):
                        block, refs = formatted
                        references.update(refs)
                    else:
                        block = formatted
                    if block:
                        item["evidence"] = block
                        compact_blocks.append(block)
                        evidence_index += len(evidence)
                else:
                    item["evidence"] = [entry.to_json() for entry in evidence]
            chunks_out.append(item)
    except Exception:
        chunks_out = []

    response = {
        "query": req.query,
        "kb": req.kb,
        "method": req.method,
        "route": describe_route(route),
        "route_mode": route.mode,
        "route_confidence": route.confidence,
        "route_signals": route.signals,
        "session_ids": session_ids,
        "elapsed_ms": round(elapsed_ms, 1),
        "chunks": chunks_out,
    }
    if req.response_format == "compact":
        response["compact_context"] = "\n".join(compact_blocks) if "compact_blocks" in locals() else ""
        if req.citation_mode == "handles":
            response["references"] = references if "references" in locals() else {}
            response["references_expire_at"] = references_expire_at if "references_expire_at" in locals() else None
    return response


@app.post("/ingest")
def ingest(req: IngestRequest):
    if req.kb not in _kb_paths:
        raise HTTPException(status_code=404, detail=f"KB '{req.kb}' not configured")

    kb_path = _kb_paths[req.kb]
    results = []
    errors = []

    # Use CLI ingest logic via subprocess for isolation.
    # Ingest each real path directly so chunk metadata preserves the original
    # source path (instead of a temp staging path).  This is slower for large
    # backfills but perfect for incremental memory/session deltas.
    import subprocess
    venv_python = Path(__file__).parent / ".venv" / "bin" / "python3"
    cli = str(venv_python) if venv_python.exists() else sys.executable

    cli_env = os.environ.copy()
    src_path = str(Path(__file__).parent / "src")
    cli_env["PYTHONPATH"] = src_path + (os.pathsep + cli_env["PYTHONPATH"] if cli_env.get("PYTHONPATH") else "")

    for raw_path in req.paths:
        src = Path(raw_path)
        if not src.exists():
            errors.append(f"missing: {src}")
            continue

        cmd = [
            cli, "-m", "contextfit.cli",
            "--kb", str(kb_path),
            "ingest",
            "--chunk-size", str(req.chunk_size),
            "--overlap", str(req.overlap),
            "--workers", "1",
            "--resume",
            str(src),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=cli_env)
        if proc.returncode == 0:
            results.append({"status": "ok", "path": str(src), "output": proc.stdout[-200:]})
        else:
            errors.append(f"{src}: {proc.stderr[-300:]}")

    # If ingest succeeded, rebuild indexes so newly added chunks are searchable.
    # This is cheap for the memory KB and avoids a stale BM25/LSH view after
    # incremental file additions.
    if results and not errors:
        rebuild_cmd = [
            cli, "-m", "contextfit.cli",
            "--kb", str(kb_path),
            "build-index",
        ]
        rebuild = subprocess.run(rebuild_cmd, capture_output=True, text=True, timeout=300, env=cli_env)
        if rebuild.returncode != 0:
            errors.append(f"build-index failed: {rebuild.stderr[-300:]}")

    # Reload engine to pick up new chunks/indexes
    if req.kb in _engines:
        _load_engine(req.kb, kb_path)

    return {"kb": req.kb, "results": results, "errors": errors}


@app.post("/rebuild-expanders")
def rebuild_expanders(req: RebuildExpandersRequest):
    if req.kb not in _kb_paths:
        raise HTTPException(status_code=404, detail=f"KB '{req.kb}' not configured")

    kb_path = _kb_paths[req.kb]
    import subprocess
    venv_python = Path(__file__).parent / ".venv" / "bin" / "python3"
    cli = str(venv_python) if venv_python.exists() else sys.executable
    script = Path(__file__).parent / "scripts" / "build_embedding_expander.py"

    proc = subprocess.run(
        [cli, str(script), str(kb_path)],
        capture_output=True, text=True, timeout=600,
        env={**os.environ},
    )

    if proc.returncode == 0:
        # Reload to pick up new expander
        if req.kb in _engines:
            _load_engine(req.kb, kb_path)
        return {"status": "ok", "output": proc.stdout[-1000:]}
    else:
        raise HTTPException(status_code=500, detail=proc.stderr[-500:])


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--kb", action="append", metavar="NAME:PATH",
        help="KB to serve. Format: name:/path/to/kb. Repeat for multiple KBs.",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--lazy", action="store_true", help="Don't pre-load KBs; load on first query")
    args = parser.parse_args()

    if not args.kb:
        parser.error("At least one --kb NAME:PATH is required")

    for spec in args.kb:
        if ":" not in spec:
            parser.error(f"Invalid --kb spec '{spec}'. Use NAME:/path/to/kb")
        name, _, path_str = spec.partition(":")
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            print(f"[cf-server] WARNING: KB path does not exist yet: {path}", flush=True)
            path.mkdir(parents=True, exist_ok=True)
        _kb_paths[name] = path

    if not args.lazy:
        for name, path in _kb_paths.items():
            if path.exists() and any(path.iterdir()):
                _load_engine(name, path)
            else:
                print(f"[cf-server] KB '{name}' path empty — will create on first ingest", flush=True)

    print(f"[cf-server] Listening on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
