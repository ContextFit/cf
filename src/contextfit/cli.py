"""
ContextFit CLI.

Usage:
    contextfit ingest ./documents --tokenizer tiktoken
    contextfit build-index
    contextfit train-sid
    contextfit query "What is ContextFit?"
    contextfit stats
"""

from __future__ import annotations

import argparse
import base64
import email
import email.policy
import html
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

from contextfit.core.tokenizer import Tokenizer
from contextfit.retrieval.engine import RetrievalEngine
from contextfit.retrieval.extractive import default_reference_expiry, extract_evidence, format_evidence_compact
from contextfit.extractors import auto as auto_extractor
from contextfit.extractors import calendar as calendar_extractor
from contextfit.extractors import code as code_extractor
from contextfit.extractors import document as document_extractor
from contextfit.extractors import structured as structured_extractor
from contextfit.extractors import tmd as tmd_extractor


def _chunk_to_json(
    engine: RetrievalEngine,
    chunk,
    score: float,
    rank: int,
    query: str | None = None,
    extractive: str = "none",
    max_evidence_chars: int = 1200,
    compact: bool = False,
    citation_mode: str = "inline",
    references: dict[str, object] | None = None,
    reference_expires_at: str | None = None,
) -> dict:
    sid = engine.sid_index.get(chunk.chunk_id) if engine.sid_index is not None else None
    text = engine.tokenizer.decode(chunk.tokens.tolist())
    payload = {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "score": score,
    }
    if not compact:
        payload.update(
            {
                "level": chunk.level,
                "parent_id": chunk.parent_id,
                "token_count": chunk.token_count,
                "semantic_id": list(sid.tokens) if sid else None,
                "metadata": chunk.metadata,
                "preview": text[:400],
                "tokens": chunk.tokens.tolist(),
            }
        )
    if query and extractive != "none":
        evidence = extract_evidence(
            text,
            query,
            mode=extractive,  # type: ignore[arg-type]
            max_chars=max_evidence_chars,
            metadata=chunk.metadata,
        )
        if compact:
            formatted = format_evidence_compact(
                evidence,
                metadata=chunk.metadata,
                chunk_id=chunk.chunk_id,
                chunk_score=float(score),
                citation_mode=citation_mode,  # type: ignore[arg-type]
                expires_at=reference_expires_at,
            )
            if isinstance(formatted, tuple):
                evidence_text, refs = formatted
                payload["evidence"] = evidence_text
                if references is not None:
                    references.update(refs)
            else:
                payload["evidence"] = formatted
        else:
            payload["evidence"] = [entry.to_json() for entry in evidence]
    return payload


def _query_to_json(
    engine: RetrievalEngine,
    query: str,
    result,
    extractive: str = "none",
    max_evidence_chars: int = 1200,
    compact: bool = False,
    citation_mode: str = "inline",
    reference_ttl_seconds: int = 3600,
) -> dict:
    references: dict[str, object] = {}
    reference_expires_at = default_reference_expiry(reference_ttl_seconds)
    payload = {
        "query": query,
        "method": result.method,
        "query_tokens": result.query_tokens.tolist(),
        "retrieved_chunks": len(result.chunks),
        "input_token_count": len(result.input_ids),
        "input_ids": result.input_ids.tolist(),
        "sid_predictions": [
            {
                "prefix": list(prediction.prefix),
                "score": prediction.score,
                "depth": prediction.depth,
                "support": prediction.support,
                "candidate_chunks": list(prediction.candidate_chunks),
            }
            for prediction in (result.sid_predictions or [])
        ],
        "chunks": [
            _chunk_to_json(
                engine,
                chunk,
                score,
                rank=i + 1,
                query=query,
                extractive=extractive,
                max_evidence_chars=max_evidence_chars,
                compact=compact,
                citation_mode=citation_mode,
                references=references,
                reference_expires_at=reference_expires_at,
            )
            for i, (chunk, score) in enumerate(zip(result.chunks, result.scores, strict=False))
        ],
    }
    if compact:
        payload["compact_context"] = "\n".join(
            chunk.get("evidence", "") for chunk in payload["chunks"] if chunk.get("evidence")
        )
        if citation_mode == "handles":
            payload["references"] = references
            payload["references_expire_at"] = reference_expires_at
    return payload


def _format_size(bytes_val: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _manifest_path(kb_path: Path) -> Path:
    return kb_path / "ingest_manifest.json"


def _load_manifest(kb_path: Path) -> dict[str, Any] | None:
    path = _manifest_path(kb_path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _save_manifest(kb_path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = time.time()
    _atomic_write_json(_manifest_path(kb_path), manifest)


def _new_manifest(args: argparse.Namespace, kb_path: Path, source: Path, total_files: int, total_bytes: int) -> dict[str, Any]:
    now = time.time()
    return {
        "version": 1,
        "status": "running",
        "source": str(source),
        "kb_path": str(kb_path),
        "tokenizer": args.tokenizer,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "defer_index_build": args.defer_index_build,
        "started_at": now,
        "updated_at": now,
        "processed_files": [],
        "skipped_files": [],
        "failed_files": [],
        "counts": {
            "discovered_files": total_files,
            "processed_files": 0,
            "skipped_files": 0,
            "failed_files": 0,
            "chunks": 0,
            "tokens": 0,
            "source_bytes": total_bytes,
        },
        "phases": {
            "ingest_complete": False,
            "index_built": False,
            "sid_trained": False,
        },
        "phase_metrics": {},
        "settings": {
            "workers": args.workers,
            "checkpoint_every": args.checkpoint_every,
            "artifact_checkpoint_every": args.artifact_checkpoint_every,
            "max_file_bytes": args.max_file_bytes,
            "max_file_tokens": args.max_file_tokens,
        },
    }


def _update_phase_metric(manifest: dict[str, Any], key: str, seconds: float) -> None:
    manifest.setdefault("phase_metrics", {})[key] = round(seconds, 6)


def _record_issue(manifest: dict[str, Any], bucket: str, path: Path, reason: str) -> None:
    manifest[bucket].append({"path": str(path), "reason": reason})
    count_key = {
        "skipped_files": "skipped_files",
        "failed_files": "failed_files",
    }[bucket]
    manifest["counts"][count_key] += 1


# Zero-width / invisible characters used in email anti-preview tricks
_INVISIBLE_CHARS = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u00ad\ufeff\u2060]+")
# Runs of non-breaking spaces (often used as padding in HTML emails)
_NBSP_RUN = re.compile(r"[\u00a0\xa0]{2,}")


def _clean_plain_text(text: str) -> str:
    """Post-process plain text: unescape HTML entities, strip anti-preview junk."""
    text = html.unescape(text)                    # &nbsp; → NBSP, &zwnj; → U+200C, etc.
    text = _INVISIBLE_CHARS.sub("", text)         # strip zero-width chars
    text = _NBSP_RUN.sub(" ", text)              # collapse NBSP runs to single space
    text = text.replace("\u00a0", " ")           # remaining NBSPs → regular space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Drop lines that are >80% punctuation/symbols (residual garbage)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        alnum = sum(c.isalnum() or c.isspace() for c in stripped)
        if len(stripped) < 4 or alnum / len(stripped) > 0.2:
            lines.append(line)
    return "\n".join(lines).strip()


def _strip_html(text: str) -> str:
    """Strip HTML tags, decode entities, and clean invisible chars."""
    # Remove style/script blocks entirely
    text = re.sub(r"<(style|script)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    return _clean_plain_text(text)


def _parse_markdown_email(raw: str) -> tuple[dict, str]:
    """Parse markdown-formatted email files (# Email: ... **From:** ... ## Content ...).

    Returns (headers_dict, body_text).
    """
    headers: dict[str, str] = {}
    body = ""

    # Extract markdown headers: **Key:** value
    for hdr in ("From", "To", "CC", "Subject", "Date"):
        m = re.search(rf"\*\*{hdr}:\*\*\s*(.+)", raw, re.IGNORECASE)
        if m:
            headers[hdr.lower()] = m.group(1).strip()

    # Also try subject from # Email: ... title line
    if "subject" not in headers:
        m = re.search(r"^#\s+Email:\s*(.+)", raw, re.MULTILINE)
        if m:
            headers["subject"] = m.group(1).strip()

    # Extract body after ## Content
    content_match = re.search(r"^##\s+Content\s*\n", raw, re.MULTILINE)
    if content_match:
        body_raw = raw[content_match.end():]

        # Detect and decode raw base64 bodies (no MIME headers, just base64 lines)
        stripped_body = body_raw.strip()
        b64_lines = [l for l in stripped_body.splitlines() if l.strip()]
        if b64_lines and all(re.match(r'^[A-Za-z0-9+/=]{20,}$', l.strip()) for l in b64_lines[:6]):
            try:
                # Only take the consecutive base64 block (stop at first non-b64 line)
                b64_block = []
                for line in b64_lines:
                    if re.match(r'^[A-Za-z0-9+/=]{20,}$', line.strip()):
                        b64_block.append(line)
                    else:
                        break
                decoded = base64.decodebytes('\n'.join(b64_block).encode()).decode('utf-8', errors='ignore')
                body = _strip_html(decoded) if '<' in decoded else _clean_plain_text(decoded)
                return headers, body.strip()
            except Exception:
                pass

        # Try to parse as MIME if it looks like it; otherwise strip HTML
        if re.search(r"Content-Type:", body_raw, re.IGNORECASE):
            # Detect multipart boundary from first '--...' line
            boundary_match = re.match(r"^--(\S+)", body_raw.lstrip())
            boundary = boundary_match.group(1) if boundary_match else None

            # Reconstruct minimal headers for the email parser
            fake_mime = ""
            if "from" in headers:
                fake_mime += f"From: {headers['from']}\n"
            if "to" in headers:
                fake_mime += f"To: {headers['to']}\n"
            if "subject" in headers:
                fake_mime += f"Subject: {headers['subject']}\n"
            if boundary:
                fake_mime += f'Content-Type: multipart/alternative; boundary="{boundary}"\n'
            fake_mime += "\n" + body_raw
            try:
                msg = email.message_from_string(fake_mime, policy=email.policy.compat32)
                plain_parts, html_parts = [], []
                for part in msg.walk():
                    ct = part.get_content_type()
                    if "attachment" in str(part.get("Content-Disposition", "")):
                        continue
                    try:
                        payload = part.get_payload(decode=True)
                        if payload is None:
                            continue
                        charset = part.get_content_charset() or "utf-8"
                        text = payload.decode(charset, errors="ignore")
                    except Exception:
                        continue
                    if ct == "text/plain":
                        plain_parts.append(_clean_plain_text(text))
                    elif ct == "text/html":
                        html_parts.append(_strip_html(text))
                body = ("\n\n".join(plain_parts).strip()
                        or "\n\n".join(html_parts).strip()
                        or _strip_html(body_raw))
            except Exception:
                body = _strip_html(body_raw)
        else:
            body = _strip_html(body_raw) if "<" in body_raw else body_raw
    else:
        body = raw

    return headers, body.strip()


def _extract_email_text(raw: str, path: Path) -> tuple[str, dict]:
    """Parse email file, returning (clean_text, metadata_dict).

    Handles two formats:
    - Markdown email files (# Email / **From:** / ## Content)
    - Raw MIME email files (From: / Content-Type: multipart/...)

    Metadata includes source path + From/To/CC/Subject/Date.
    All metadata is prepended to body text so every chunk carries contact context.
    """
    meta: dict[str, str] = {"source": str(path)}

    try:
        # Detect format
        is_markdown_email = bool(re.match(r"#\s+Email:", raw.lstrip()) or re.search(r"\*\*From:\*\*", raw[:500]))

        if is_markdown_email:
            headers, body = _parse_markdown_email(raw)
            meta.update(headers)
        else:
            # Raw MIME path
            msg = email.message_from_string(raw, policy=email.policy.compat32)
            for hdr in ("From", "To", "CC", "Subject", "Date"):
                val = msg.get(hdr, "").strip()
                if val:
                    meta[hdr.lower()] = val

            plain_parts: list[str] = []
            html_parts: list[str] = []
            if msg.is_multipart():
                for part in msg.walk():
                    ct = part.get_content_type()
                    if "attachment" in str(part.get("Content-Disposition", "")):
                        continue
                    try:
                        payload = part.get_payload(decode=True)
                        if payload is None:
                            continue
                        charset = part.get_content_charset() or "utf-8"
                        text = payload.decode(charset, errors="ignore")
                    except Exception:
                        continue
                    if ct == "text/plain":
                        plain_parts.append(_clean_plain_text(text))
                    elif ct == "text/html":
                        html_parts.append(_strip_html(text))
            else:
                payload = msg.get_payload(decode=True)
                if payload is not None:
                    charset = msg.get_content_charset() or "utf-8"
                    text = payload.decode(charset, errors="ignore")
                    ct = msg.get_content_type()
                    if ct == "text/html":
                        html_parts.append(_strip_html(text))
                    else:
                        plain_parts.append(_clean_plain_text(text))

            body = ("\n\n".join(plain_parts).strip()
                    or "\n\n".join(html_parts).strip()
                    or _strip_html(raw))

        # Build header preamble — prepended so contact info appears in every chunk
        preamble_lines = []
        for hdr in ("From", "To", "CC", "Subject", "Date"):
            if hdr.lower() in meta:
                preamble_lines.append(f"{hdr}: {meta[hdr.lower()]}")
        preamble = "\n".join(preamble_lines)

        full_text = (preamble + "\n\n" + body).strip() if preamble else body
        return full_text, meta

    except Exception:
        # Fallback: strip HTML if needed
        text = _strip_html(raw) if "<html" in raw.lower() else raw
        return text, {"source": str(path)}


def _discover_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    patterns = ("*.txt", "*.md", "*.tmd", "*.eml", "*.ics", "*.json", "*.jsonl", "*.csv", "*.tsv") + tuple(f"*{ext}" for ext in sorted(code_extractor.CODE_EXTENSIONS))
    files: list[Path] = []
    for pattern in patterns:
        files.extend(source.rglob(pattern))
    return sorted(files)


def _preprocess_file(
    path: Path,
    tokenizer_name: str,
    chunk_size: int,
    overlap: int,
    max_file_bytes: int,
    max_file_tokens: int,
) -> dict[str, Any]:
    started = time.time()
    size = path.stat().st_size
    if max_file_bytes and size > max_file_bytes:
        return {"path": str(path), "status": "skipped", "reason": f"file_too_large:{size}", "bytes": size, "seconds": time.time() - started}

    raw = path.read_text(errors="ignore")
    text, email_meta = _extract_email_text(raw, path)
    tokenizer = Tokenizer.load(tokenizer_name)

    # Structure-aware pre-chunking by file type.  The tokenizer remains the
    # source of truth for final token IDs, but boundaries now follow human units
    # of meaning where possible: TMD rows, Markdown sections, text paragraphs.
    is_email = any(k in email_meta for k in ("from", "to", "subject", "date"))
    if path.suffix == ".tmd":
        text_chunks = tmd_extractor.chunk_tmd(path, text, chunk_size=chunk_size, overlap=overlap)
    elif path.suffix == ".md" and not is_email:
        text_chunks = document_extractor.chunk_markdown(path, text, chunk_size=chunk_size, overlap=overlap)
    elif path.suffix == ".txt" and not is_email:
        text_chunks = document_extractor.chunk_text(path, text, chunk_size=chunk_size, overlap=overlap)
    elif path.suffix == ".json":
        text_chunks = structured_extractor.chunk_json(path, text, chunk_size=chunk_size, overlap=overlap)
    elif path.suffix == ".jsonl":
        text_chunks = structured_extractor.chunk_jsonl(path, text, chunk_size=chunk_size, overlap=overlap)
    elif path.suffix == ".csv":
        text_chunks = structured_extractor.chunk_delimited(path, text, chunk_size=chunk_size, overlap=overlap, delimiter=",")
    elif path.suffix == ".tsv":
        text_chunks = structured_extractor.chunk_delimited(path, text, chunk_size=chunk_size, overlap=overlap, delimiter="\t")
    elif path.suffix == ".ics":
        text_chunks = calendar_extractor.chunk_ics(path, text, chunk_size=chunk_size, overlap=overlap)
    elif code_extractor.is_code_path(path):
        text_chunks = code_extractor.chunk_code(path, text, chunk_size=chunk_size, overlap=overlap)
    else:
        # EML/email and unknown text keep the tokenizer's conservative sliding
        # window, but still get auto metadata.
        meta = email_meta or auto_extractor.extract(path, text)
        text_chunks = [{"text": tokenizer.decode(chunk), "metadata": meta} for chunk in tokenizer.chunk_text(text, chunk_size=chunk_size, overlap=overlap)]

    token_items = []
    token_count = 0
    for item in text_chunks:
        tokens = tokenizer.encode(item["text"])
        token_items.append({"tokens": tokens, "metadata": item.get("metadata") or auto_extractor.extract(path, item["text"])})
        token_count += len(tokens)
    if max_file_tokens and token_count > max_file_tokens:
        return {
            "path": str(path),
            "status": "skipped",
            "reason": f"token_count_exceeded:{token_count}",
            "bytes": size,
            "seconds": time.time() - started,
        }

    return {
        "path": str(path),
        "status": "ok",
        "bytes": size,
        "token_items": token_items,
        "token_count": token_count,
        "file_meta": email_meta or auto_extractor.extract(path, text),
        "seconds": time.time() - started,
    }


def _checkpoint_artifacts(engine: RetrievalEngine, kb_path: Path, manifest: dict[str, Any], include_indexes: bool) -> None:
    engine.store.flush(force=True)
    if include_indexes:
        engine.save(kb_path)
    _save_manifest(kb_path, manifest)


def cmd_ingest(args: argparse.Namespace) -> int:
    source = Path(args.source)
    kb_path = Path(args.kb or "./contextfit_kb")

    print("ContextFit Ingestion")
    print("=" * 50)
    print(f"Source: {source}")
    print(f"Knowledge base: {kb_path}")
    print(f"Chunk size: {args.chunk_size} tokens, overlap: {args.overlap}")
    print(f"Workers: {args.workers}")
    print(f"Deferred index build: {args.defer_index_build}")
    print()

    start_discovery = time.time()
    files = _discover_files(source)
    discovery_seconds = time.time() - start_discovery
    total_bytes = sum(f.stat().st_size for f in files)

    if not files:
        print(f"⚠️  No supported text/data files found in {source}")
        return 1

    manifest = _load_manifest(kb_path)
    processed_paths: set[str] = set()
    if manifest:
        processed_paths = set(manifest.get("processed_files", []))
        processed_paths |= {entry["path"] for entry in manifest.get("skipped_files", [])}

    if manifest and not args.resume:
        print(f"Existing manifest found at {_manifest_path(kb_path)}")
        print("Use --resume to continue, or delete the knowledge base to start over.")
        return 1

    if manifest is None:
        manifest = _new_manifest(args, kb_path, source, len(files), total_bytes)
    remaining_files = [f for f in files if str(f) not in processed_paths]

    print(f"Discovering files... found {len(files)} files")
    print(f"Remaining this run: {len(remaining_files)} files")
    print(f"Total size: {_format_size(total_bytes)}")
    print()

    start_init = time.time()
    if args.resume and kb_path.exists() and (kb_path / "chunks").exists():
        engine = RetrievalEngine.load(kb_path, tokenizer_name=args.tokenizer)
    else:
        engine = RetrievalEngine.create(kb_path, tokenizer_name=args.tokenizer)

    # Enable segment writer for large ingests (deferred index build)
    if args.defer_index_build:
        engine.enable_segment_writer(kb_path, flush_every=args.segment_flush_every)
    init_seconds = time.time() - start_init

    _update_phase_metric(manifest, "discovery_seconds", discovery_seconds)
    _update_phase_metric(manifest, "init_seconds", init_seconds)
    _save_manifest(kb_path, manifest)

    total_chunks = manifest["counts"]["chunks"]
    total_tokens = manifest["counts"]["tokens"]
    checkpoint_every = max(1, args.checkpoint_every)
    artifact_checkpoint_every = max(0, args.artifact_checkpoint_every)
    preprocess_wall_start = time.time()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = pool.map(
            lambda file_path: _preprocess_file(
                file_path,
                tokenizer_name=args.tokenizer,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                max_file_bytes=args.max_file_bytes,
                max_file_tokens=args.max_file_tokens,
            ),
            remaining_files,
        )

        start_ingest = time.time()
        with tqdm(
            total=len(remaining_files),
            desc="Ingesting",
            unit="file",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for i, result in enumerate(results, start=1):
                path = Path(result["path"])
                pbar.set_postfix_str(f"{path.name[:30]}...")

                if result["status"] == "skipped":
                    _record_issue(manifest, "skipped_files", path, result["reason"])
                else:
                    try:
                        chunks = []
                        default_session_id = path.as_posix()
                        for item in result["token_items"]:
                            meta = dict(item.get("metadata") or result.get("file_meta", {"source": str(path)}))
                            meta.setdefault("session_id", default_session_id)
                            chunks.extend(engine.ingest_token_chunks(
                                [item["tokens"]],
                                metadata=meta,
                                update_indexes=not args.defer_index_build,
                            ))
                        chunk_count = len(chunks)
                        token_count = result["token_count"]
                        total_chunks += chunk_count
                        total_tokens += token_count
                        manifest["counts"]["chunks"] = total_chunks
                        manifest["counts"]["tokens"] = total_tokens
                        manifest["processed_files"].append(str(path))
                        manifest["counts"]["processed_files"] = len(manifest["processed_files"])
                        if args.verbose:
                            tqdm.write(f"  ✓ {path.name}: {chunk_count} chunks, {token_count:,} tokens")
                    except Exception as exc:
                        _record_issue(manifest, "failed_files", path, str(exc))
                        if args.verbose:
                            tqdm.write(f"  ✗ {path.name}: {exc}")

                if i % checkpoint_every == 0:
                    _checkpoint_artifacts(
                        engine,
                        kb_path,
                        manifest,
                        include_indexes=(not args.defer_index_build and artifact_checkpoint_every == 0),
                    )
                if artifact_checkpoint_every and i % artifact_checkpoint_every == 0:
                    _checkpoint_artifacts(
                        engine,
                        kb_path,
                        manifest,
                        include_indexes=not args.defer_index_build,
                    )
                pbar.update(1)

        ingest_seconds = time.time() - start_ingest

    preprocess_wall_seconds = time.time() - preprocess_wall_start
    engine.store.flush(force=True)

    build_index_seconds = 0.0
    if args.defer_index_build:
        print("\nIndex build deferred. Run `contextfit build-index` next.")
    elif args.rebuild_index_after_ingest:
        print("\nRebuilding indexes in batch...", flush=True)
        start_rebuild = time.time()
        engine.rebuild_indexes(include_semantic_ids=True)
        build_index_seconds = time.time() - start_rebuild
        print(f"  ✓ Rebuilt indexes in {_format_time(build_index_seconds)}")

    train_sid_seconds = 0.0
    if args.train_sid_generator:
        if args.defer_index_build:
            print("\nRebuilding indexes before SID training...", flush=True)
            start_rebuild = time.time()
            engine.rebuild_indexes(include_semantic_ids=True)
            build_index_seconds += time.time() - start_rebuild
        print("\nTraining learned SID generator...", flush=True)
        start_sid = time.time()
        learned = engine.train_learned_sid_generator()
        train_sid_seconds = time.time() - start_sid
        print(f"  ✓ Trained on {learned.trained_chunks} chunks in {_format_time(train_sid_seconds)}")
        manifest["phases"]["sid_trained"] = True

    if args.hierarchy and not args.defer_index_build:
        print(f"\nBuilding hierarchy ({args.hierarchy_levels} levels)...", flush=True)
        start_hierarchy = time.time()
        engine.build_hierarchy(max_levels=args.hierarchy_levels)
        _update_phase_metric(manifest, "hierarchy_seconds", time.time() - start_hierarchy)

    print(f"\nSaving to {kb_path}...", end=" ", flush=True)
    start_save = time.time()
    engine.save(kb_path)
    save_seconds = time.time() - start_save
    print(f"done ({_format_time(save_seconds)})")

    elapsed_total = preprocess_wall_seconds
    _update_phase_metric(manifest, "preprocess_wall_seconds", preprocess_wall_seconds)
    _update_phase_metric(manifest, "ingest_seconds", ingest_seconds)
    _update_phase_metric(manifest, "build_index_seconds", build_index_seconds)
    _update_phase_metric(manifest, "train_sid_seconds", train_sid_seconds)
    _update_phase_metric(manifest, "save_seconds", save_seconds)
    manifest["status"] = "completed"
    manifest["phases"]["ingest_complete"] = True
    manifest["phases"]["index_built"] = not args.defer_index_build
    _save_manifest(kb_path, manifest)

    stats = engine.stats()
    print()
    print("=" * 50)
    print("Ingestion Complete")
    print("=" * 50)
    print(f"Files processed: {manifest['counts']['processed_files']}/{len(files)}")
    print(f"Skipped: {manifest['counts']['skipped_files']} | Failed: {manifest['counts']['failed_files']}")
    print(f"Total chunks: {total_chunks:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Time: {_format_time(elapsed_total)}")
    if ingest_seconds > 0:
        print(f"Rate: {total_tokens / max(ingest_seconds, 1e-9):,.0f} tokens/sec")
    print(f"Knowledge base size: {_format_size(stats['storage']['data_size_bytes'])}")
    print(f"Index vocabulary: {stats['index']['vocab_size']:,} tokens")
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    kb_path = Path(args.kb or "./contextfit_kb")
    if not (kb_path / "chunks").exists():
        print(f"Knowledge base chunks not found at {kb_path}")
        return 1

    from contextfit.index.inverted import SegmentMerger
    segments_dir = kb_path / "segments"
    segments_exist = SegmentMerger.segments_exist(segments_dir)

    start = time.time()

    if segments_exist and not args.no_segments:
        # --- Fast path: k-way segment merge (O(n_segments) RAM) ---
        print("Building indexes via segment merge (incremental, low-memory)...", flush=True)
        merger = SegmentMerger(segments_dir)
        merger.merge(kb_path / "inverted")
        elapsed = time.time() - start
        manifest = _load_manifest(kb_path) or {"phase_metrics": {}, "phases": {}}
        manifest.setdefault("phases", {})["index_built"] = True
        manifest.setdefault("phases", {})["index_method"] = "segment_merge"
        _update_phase_metric(manifest, "build_index_seconds", elapsed)
        _save_manifest(kb_path, manifest)
        print(f"\u2713 Built indexes via segment merge in {_format_time(elapsed)}")
    else:
        # --- Classic path: full in-memory rebuild (small corpora only) ---
        print("Building indexes from stored chunks (in-memory)...", flush=True)
        engine = RetrievalEngine.load(kb_path, tokenizer_name=args.tokenizer)
        engine.rebuild_indexes(include_semantic_ids=not args.no_semantic_ids)
        if args.hierarchy:
            engine.build_hierarchy(max_levels=args.hierarchy_levels)
        engine.save(kb_path)
        elapsed = time.time() - start
        manifest = _load_manifest(kb_path) or {"phase_metrics": {}, "phases": {}}
        manifest.setdefault("phases", {})["index_built"] = True
        manifest.setdefault("phases", {})["index_method"] = "in_memory"
        _update_phase_metric(manifest, "build_index_seconds", elapsed)
        _save_manifest(kb_path, manifest)
        print(f"\u2713 Built indexes in {_format_time(elapsed)}")
    return 0


def cmd_train_sid(args: argparse.Namespace) -> int:
    kb_path = Path(args.kb or "./contextfit_kb")
    if not kb_path.exists():
        print(f"Knowledge base not found at {kb_path}")
        return 1

    print("Training learned SID generator...", flush=True)
    start = time.time()
    engine = RetrievalEngine.load(kb_path, tokenizer_name=args.tokenizer)
    learned = engine.train_learned_sid_generator()
    engine.save(kb_path)
    elapsed = time.time() - start
    manifest = _load_manifest(kb_path) or {"phase_metrics": {}, "phases": {}}
    manifest.setdefault("phases", {})["sid_trained"] = True
    _update_phase_metric(manifest, "train_sid_seconds", elapsed)
    _save_manifest(kb_path, manifest)
    print(f"✓ Trained on {learned.trained_chunks} chunks in {_format_time(elapsed)}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    kb_path = Path(args.kb or "./contextfit_kb")
    if not kb_path.exists():
        print(f"Knowledge base not found at {kb_path}")
        print("Run 'contextfit ingest' first.")
        return 1

    engine = RetrievalEngine.load(kb_path, tokenizer_name=args.tokenizer)
    result = engine.query(args.query, top_k=args.top_k, method=args.method)

    if args.json:
        print(json.dumps(_query_to_json(engine, args.query, result), indent=2))
        return 0

    print(f"Query: {args.query}")
    print(f"Method: {result.method}")
    print(f"Query tokens: {len(result.query_tokens)}")
    print(f"Retrieved: {len(result.chunks)} chunks, {len(result.input_ids)} tokens")
    if result.sid_predictions:
        print("SID predictions:")
        for prediction in result.sid_predictions[:3]:
            print(
                f"  depth={prediction.depth} score={prediction.score:.4f} "
                f"support={prediction.support} prefix={list(prediction.prefix)}"
            )
    print()
    for i, (chunk, score) in enumerate(zip(result.chunks, result.scores, strict=False)):
        print(f"--- Chunk {i+1} (id={chunk.chunk_id}, score={score:.4f}) ---")
        text = engine.tokenizer.decode(chunk.tokens[:100])
        if len(chunk.tokens) > 100:
            text += "..."
        print(text)
        print()
    return 0



def _load_cli_vaults(args: argparse.Namespace) -> dict[str, Any]:
    """Load the local vault registry for CLI commands.

    Falls back to a single `default` vault from --kb when no registry exists.
    """
    from contextfit.mcp import VaultConfig, _load_vault_registry

    registry = Path(getattr(args, "vault_registry", "~/.contextfit/vaults.json")).expanduser()
    if registry.exists():
        return _load_vault_registry(registry)
    kb_path = Path(args.kb or "./contextfit_kb").expanduser().resolve()
    return {
        "default": VaultConfig(
            name="default",
            kb_path=kb_path,
            description="Default ContextFit knowledge base from --kb.",
        )
    }


def _resolve_cli_vault(args: argparse.Namespace, vault_name: str | None) -> Any:
    vaults = _load_cli_vaults(args)
    name = vault_name or "default"
    if name not in vaults:
        available = ", ".join(sorted(vaults)) or "none"
        raise SystemExit(f"Unknown vault '{name}'. Available vaults: {available}")
    return vaults[name]


def cmd_vaults(args: argparse.Namespace) -> int:
    vaults = _load_cli_vaults(args)
    data = [
        {"name": v.name, "kb_path": str(v.kb_path), "description": v.description}
        for v in vaults.values()
    ]
    if args.json:
        print(json.dumps({"vaults": data}, indent=2))
        return 0
    print("ContextFit vaults")
    for vault in data:
        desc = f" — {vault['description']}" if vault["description"] else ""
        print(f"- {vault['name']}: {vault['kb_path']}{desc}")
    return 0


def _print_search_result(
    engine: RetrievalEngine,
    query: str,
    result: Any,
    json_output: bool,
    vault_name: str | None = None,
    extractive: str = "none",
    max_evidence_chars: int = 1200,
    compact: bool = False,
    citation_mode: str = "inline",
    reference_ttl_seconds: int = 3600,
) -> None:
    if json_output:
        payload = _query_to_json(
            engine,
            query,
            result,
            extractive=extractive,
            max_evidence_chars=max_evidence_chars,
            compact=compact,
            citation_mode=citation_mode,
            reference_ttl_seconds=reference_ttl_seconds,
        )
        if vault_name:
            payload["vault"] = vault_name
        print(json.dumps(payload, indent=2))
        return

    prefix = f"Vault: {vault_name}\n" if vault_name else ""
    print(f"{prefix}Query: {query}")
    print(f"Method: {result.method}")
    print(f"Retrieved: {len(result.chunks)} chunks, {len(result.input_ids)} tokens")
    print()
    for i, (chunk, score) in enumerate(zip(result.chunks, result.scores, strict=False)):
        print(f"--- Chunk {i+1} (id={chunk.chunk_id}, score={score:.4f}) ---")
        print(f"source: {chunk.metadata.get('source', 'unknown')}")
        text = engine.tokenizer.decode(chunk.tokens[:100])
        if len(chunk.tokens) > 100:
            text += "..."
        print(text)
        print()


def cmd_search(args: argparse.Namespace) -> int:
    vault = _resolve_cli_vault(args, args.vault)
    if not vault.kb_path.exists():
        print(f"Knowledge base not found at {vault.kb_path}")
        return 1
    engine = RetrievalEngine.load(vault.kb_path, tokenizer_name=args.tokenizer)
    result = engine.query(args.query, top_k=args.top_k, method=args.method)
    _print_search_result(
        engine,
        args.query,
        result,
        args.json,
        vault.name,
        extractive=args.extractive,
        max_evidence_chars=args.max_evidence_chars,
        compact=args.compact,
        citation_mode=args.citation_mode,
        reference_ttl_seconds=args.reference_ttl_seconds,
    )
    return 0


def cmd_search_all(args: argparse.Namespace) -> int:
    vaults = _load_cli_vaults(args)
    payload = {"query": args.query, "vaults": []}
    for vault in vaults.values():
        if not vault.kb_path.exists():
            payload["vaults"].append(
                {"vault": vault.name, "kb_path": str(vault.kb_path), "error": "knowledge base not found"}
            )
            continue
        engine = RetrievalEngine.load(vault.kb_path, tokenizer_name=args.tokenizer)
        result = engine.query(args.query, top_k=args.top_k, method=args.method)
        item = _query_to_json(
            engine,
            args.query,
            result,
            extractive=getattr(args, "extractive", "none"),
            max_evidence_chars=getattr(args, "max_evidence_chars", 1200),
            compact=getattr(args, "compact", False),
            citation_mode=getattr(args, "citation_mode", "inline"),
            reference_ttl_seconds=getattr(args, "reference_ttl_seconds", 3600),
        )
        item["vault"] = vault.name
        item["kb_path"] = str(vault.kb_path)
        payload["vaults"].append(item)

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    for item in payload["vaults"]:
        print(f"=== Vault: {item['vault']} ===")
        if "error" in item:
            print(item["error"])
            print()
            continue
        for chunk in item["chunks"]:
            print(f"--- Chunk {chunk['rank']} (id={chunk['chunk_id']}, score={chunk['score']:.4f}) ---")
            print(f"source: {chunk['metadata'].get('source', 'unknown')}")
            print(chunk["preview"])
            print()
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    vault = _resolve_cli_vault(args, args.vault)
    if not vault.kb_path.exists():
        print(f"Knowledge base not found at {vault.kb_path}")
        return 1
    engine = RetrievalEngine.load(vault.kb_path, tokenizer_name=args.tokenizer)
    chunk = engine.store.get(args.chunk_id)
    if chunk is None:
        print(f"Chunk not found: {args.chunk_id}")
        return 1
    text = engine.tokenizer.decode(chunk.tokens.tolist())
    payload = {
        "vault": vault.name,
        "kb_path": str(vault.kb_path),
        "chunk_id": chunk.chunk_id,
        "token_count": chunk.token_count,
        "metadata": chunk.metadata,
        "text": text,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"Vault: {vault.name}")
    print(f"Chunk: {chunk.chunk_id}")
    print(f"Tokens: {chunk.token_count}")
    print(f"Metadata: {json.dumps(chunk.metadata, ensure_ascii=False)}")
    print()
    print(text)
    return 0

def cmd_stats(args: argparse.Namespace) -> int:
    kb_path = Path(args.kb or "./contextfit_kb")
    if not kb_path.exists():
        print(f"Knowledge base not found at {kb_path}")
        return 1

    engine = RetrievalEngine.load(kb_path, tokenizer_name=args.tokenizer)
    stats = engine.stats()
    manifest = _load_manifest(kb_path)
    if manifest is not None:
        stats["manifest"] = {
            "status": manifest.get("status"),
            "counts": manifest.get("counts"),
            "phases": manifest.get("phases"),
            "phase_metrics": manifest.get("phase_metrics"),
        }

    if args.json:
        print(json.dumps(stats, indent=2))
        return 0

    print("ContextFit Knowledge Base Statistics")
    print("=" * 40)
    print(f"Chunks: {stats['chunks']}")
    print(f"Storage: {stats['storage']['data_size_bytes'] / 1024:.1f} KB")
    print(f"  Compressed: {stats['storage']['compressed']}")
    print("Index:")
    print(f"  Vocabulary: {stats['index']['vocab_size']} tokens")
    print(f"  Avg posting size: {stats['index']['avg_posting_size']:.1f}")
    print(f"  Avg chunk length: {stats['index']['avg_chunk_length']:.1f}")
    print(f"LSH entries: {stats['lsh_entries']}")
    print(f"Hierarchy levels: {stats['hierarchy_levels']}")
    if stats.get("semantic_ids"):
        sid = stats["semantic_ids"]
        print(f"Semantic IDs: {sid['sid_count']} chunks")
        print(f"  Depth: {sid['depth']}, branching: {sid['branching_factor']}")
        print(f"  Prefixes: {sid['prefix_count']}")
    if stats.get("learned_sid_generator"):
        learned = stats["learned_sid_generator"]
        print("Learned SID generator:")
        print(f"  trained chunks: {learned['trained_chunks']}")
        print(f"  token features: {learned['token_features']}")
        print(f"  prefixes: {learned['prefixes']}")
    if manifest is not None:
        print("Manifest:")
        print(f"  Status: {manifest.get('status')}")
        for key, value in manifest.get("phase_metrics", {}).items():
            print(f"  {key}: {_format_time(float(value))}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="contextfit", description="Token-native knowledge base for LLM scale")
    parser.add_argument("--kb", "-k", help="Knowledge base path (default: ./contextfit_kb)")
    parser.add_argument("--tokenizer", "-t", default="cl100k_base", help="Tokenizer name (default: cl100k_base)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    ingest = subparsers.add_parser("ingest", help="Ingest documents")
    ingest.add_argument("source", help="File or directory to ingest")
    ingest.add_argument("--verbose", "-v", action="store_true", help="Show each file as it's processed")
    ingest.add_argument("--chunk-size", "-c", type=int, default=512, help="Tokens per chunk (default: 512)")
    ingest.add_argument("--overlap", "-o", type=int, default=64, help="Token overlap between chunks (default: 64)")
    ingest.add_argument("--hierarchy", action="store_true", help="Build hierarchy after index build")
    ingest.add_argument("--hierarchy-levels", type=int, default=3, help="Number of hierarchy levels (default: 3)")
    ingest.add_argument("--train-sid-generator", action="store_true", help="Train the learned token-to-SID generator after ingestion")
    ingest.add_argument("--checkpoint-every", type=int, default=25, help="Force an on-disk chunk checkpoint every N files (default: 25)")
    ingest.add_argument("--artifact-checkpoint-every", type=int, default=250, help="Save index/SID artifacts every N files; 0 disables (default: 250)")
    ingest.add_argument("--workers", type=int, default=4, help="Parallel file read/tokenize workers (default: 4)")
    ingest.add_argument("--resume", action="store_true", help="Resume from an existing ingest manifest")
    ingest.add_argument("--defer-index-build", action="store_true", help="Only store chunks during ingest; build indexes later")
    ingest.add_argument("--rebuild-index-after-ingest", action="store_true", help="Rebuild indexes in one batch after ingest completes")
    ingest.add_argument("--max-file-bytes", type=int, default=0, help="Skip files larger than this many bytes (0 disables)")
    ingest.add_argument("--max-file-tokens", type=int, default=0, help="Skip files whose chunked token count exceeds this value (0 disables)")
    ingest.add_argument("--segment-flush-every", type=int, default=2000,
                        help="Flush a segment file every N chunks during deferred ingest (default: 2000)")

    build_index = subparsers.add_parser("build-index", help="Rebuild indexes from stored chunks")
    build_index.add_argument("--no-semantic-ids", action="store_true", help="Skip semantic ID rebuild")
    build_index.add_argument("--hierarchy", action="store_true", help="Build hierarchy after index rebuild")
    build_index.add_argument("--hierarchy-levels", type=int, default=3, help="Number of hierarchy levels (default: 3)")
    build_index.add_argument("--no-segments", action="store_true",
                             help="Force in-memory rebuild even if segment files exist")

    train_sid = subparsers.add_parser("train-sid", help="Train the learned SID generator from stored chunks")

    query = subparsers.add_parser("query", help="Query the default knowledge base")
    query.add_argument("query", help="Query text")
    query.add_argument("--top-k", "-k", type=int, default=5, help="Number of results (default: 5)")
    query.add_argument("--method", "-m", choices=["exact", "bm25", "sid", "graph", "hierarchy", "hybrid", "hybrid_rrf"], default="hybrid", help="Retrieval method (default: hybrid)")
    query.add_argument("--json", action="store_true", help="Emit machine-readable JSON including input_ids and chunk metadata")

    vaults = subparsers.add_parser("vaults", help="List registered local vaults")
    vaults.add_argument("action", choices=["list"], help="Vault action")
    vaults.add_argument("--vault-registry", default="~/.contextfit/vaults.json", help="Vault registry path")
    vaults.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    search = subparsers.add_parser("search", help="Search a named vault with JSON-friendly output")
    search.add_argument("query", help="Query text")
    search.add_argument("--vault", default="default", help="Vault name (default: default)")
    search.add_argument("--vault-registry", default="~/.contextfit/vaults.json", help="Vault registry path")
    search.add_argument("--top-k", "-k", type=int, default=5, help="Number of results (default: 5)")
    search.add_argument("--method", "-m", choices=["exact", "bm25", "sid", "graph", "hierarchy", "hybrid", "hybrid_rrf"], default="hybrid", help="Retrieval method (default: hybrid)")
    search.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    search.add_argument("--extractive", choices=["none", "spans", "rows", "bullets", "auto"], default="none", help="Add deterministic query-focused evidence to JSON output")
    search.add_argument("--max-evidence-chars", type=int, default=1200, help="Maximum evidence characters per chunk (default: 1200)")
    search.add_argument("--compact", action="store_true", help="With --json and --extractive, emit compact prompt-oriented evidence instead of full JSON evidence metadata")
    search.add_argument("--citation-mode", choices=["inline", "handles"], default="inline", help="Compact citation style: inline provenance or tiny @rN handles with a references sidecar")
    search.add_argument("--reference-ttl-seconds", type=int, default=3600, help="Expiry TTL for handle references (default: 3600)")

    search_all = subparsers.add_parser("search-all", help="Search all registered vaults")
    search_all.add_argument("query", help="Query text")
    search_all.add_argument("--vault-registry", default="~/.contextfit/vaults.json", help="Vault registry path")
    search_all.add_argument("--top-k", "-k", type=int, default=5, help="Number of results per vault (default: 5)")
    search_all.add_argument("--method", "-m", choices=["exact", "bm25", "sid", "graph", "hierarchy", "hybrid", "hybrid_rrf"], default="hybrid", help="Retrieval method (default: hybrid)")
    search_all.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    search_all.add_argument("--extractive", choices=["none", "spans", "rows", "bullets", "auto"], default="none", help="Add deterministic query-focused evidence to JSON output")
    search_all.add_argument("--max-evidence-chars", type=int, default=1200, help="Maximum evidence characters per chunk (default: 1200)")
    search_all.add_argument("--compact", action="store_true", help="With --json and --extractive, emit compact prompt-oriented evidence instead of full JSON evidence metadata")
    search_all.add_argument("--citation-mode", choices=["inline", "handles"], default="inline", help="Compact citation style: inline provenance or tiny @rN handles with a references sidecar")
    search_all.add_argument("--reference-ttl-seconds", type=int, default=3600, help="Expiry TTL for handle references (default: 3600)")

    chunk = subparsers.add_parser("chunk", help="Fetch a chunk from a named vault")
    chunk.add_argument("chunk_id", type=int, help="Chunk id")
    chunk.add_argument("--vault", default="default", help="Vault name (default: default)")
    chunk.add_argument("--vault-registry", default="~/.contextfit/vaults.json", help="Vault registry path")
    chunk.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    stats = subparsers.add_parser("stats", help="Show knowledge base statistics")
    stats.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    mcp = subparsers.add_parser(
        "mcp",
        help="Run a Claude Desktop-compatible MCP stdio server",
    )
    mcp.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Default search result count (default: 5)",
    )
    mcp.add_argument(
        "--method",
        choices=["exact", "bm25", "sid", "graph", "hierarchy", "hybrid", "hybrid_rrf"],
        default="hybrid",
        help="Default retrieval method (default: hybrid)",
    )
    mcp.add_argument(
        "--max-preview-chars",
        type=int,
        default=1200,
        help="Maximum characters per search preview (default: 1200)",
    )
    mcp.add_argument(
        "--vault-registry",
        default="~/.contextfit/vaults.json",
        help="JSON registry of named vaults (default: ~/.contextfit/vaults.json)",
    )

    args = parser.parse_args()
    if args.command == "ingest":
        return cmd_ingest(args)
    if args.command == "build-index":
        return cmd_build_index(args)
    if args.command == "train-sid":
        return cmd_train_sid(args)
    if args.command == "query":
        return cmd_query(args)
    if args.command == "vaults":
        return cmd_vaults(args)
    if args.command == "search":
        return cmd_search(args)
    if args.command == "search-all":
        return cmd_search_all(args)
    if args.command == "chunk":
        return cmd_chunk(args)
    if args.command == "stats":
        return cmd_stats(args)
    if args.command == "mcp":
        from contextfit.mcp import ContextFitMCPServer, MCPServerConfig

        kb_path = Path(args.kb or "./contextfit_kb").expanduser().resolve()
        vault_registry = Path(args.vault_registry).expanduser()
        server = ContextFitMCPServer(
            MCPServerConfig(
                kb_path=kb_path,
                tokenizer=args.tokenizer,
                default_top_k=args.top_k,
                default_method=args.method,
                max_preview_chars=args.max_preview_chars,
                vault_registry=vault_registry,
            )
        )
        server.serve()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
