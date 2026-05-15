#!/usr/bin/env python3
"""
Incremental memory indexer for ContextFit.

Scans the OpenClaw workspace memory files and session transcripts,
ingests anything new or modified since the last run.

Designed to be called from the OpenClaw heartbeat or a cron job.

Usage:
    python scripts/index_memory.py [--workspace ~/.openclaw/workspace] [--server http://127.0.0.1:8765]
    python scripts/index_memory.py --dry-run   # show what would be indexed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import urllib.request
import urllib.error


DEFAULT_WORKSPACE = Path.home() / ".openclaw" / "workspace"
DEFAULT_SERVER    = "http://127.0.0.1:8765"
STATE_FILE        = Path.home() / ".openclaw" / "contextfit-index-state.json"

# Files/dirs to index from the workspace
MEMORY_PATTERNS = [
    "MEMORY.md",
    "memory/*.md",
    "memory/*.txt",
    "tax/**/*.md",
    "tax/**/*.txt",
    "financials/**/*.md",
    "**/*.tmd",
]

# Session transcripts
SESSION_PATTERNS = [
    "memory/.dreams/session-corpus/*.txt",
    "memory/.dreams/session-corpus/**/*.txt",
]

# Always skip these files/directories.  TMD files are allowed anywhere in
# user-facing workspace content, but avoid indexing tool repos, generated
# caches, plugins, virtualenvs, and VCS internals.
SKIP_PATTERNS = {
    "HEARTBEAT.md",
    "AGENTS.md",
    "SOUL.md",
    "IDENTITY.md",
    "TOOLS.md",
    "USER.md",
}

SKIP_DIRS = {
    ".git",
    ".obsidian",
    ".venv",
    "node_modules",
    "tmp",
    "cf",
    "ContextFit",
    "tmd",
}


def file_sig(path: Path) -> str:
    """Cheap change-detection: mtime + size."""
    st = path.stat()
    return f"{st.st_mtime:.0f}:{st.st_size}"


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"indexed": {}}  # path -> sig


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)


def should_skip(path: Path, workspace: Path) -> bool:
    if path.name in SKIP_PATTERNS:
        return True
    try:
        rel = path.relative_to(workspace)
    except ValueError:
        rel = path
    return any(part in SKIP_DIRS for part in rel.parts[:-1])


def collect_files(workspace: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in MEMORY_PATTERNS + SESSION_PATTERNS:
        for p in workspace.glob(pattern):
            if p.is_file() and not should_skip(p, workspace):
                files.append(p)
    return sorted(set(files))


def server_ingest(server: str, kb: str, paths: list[Path]) -> dict:
    url = f"{server}/ingest"
    payload = json.dumps({"kb": kb, "paths": [str(p) for p in paths]}).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def server_status(server: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{server}/status", timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    p.add_argument("--server",    default=DEFAULT_SERVER)
    p.add_argument("--kb",        default="memory")
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--force",     action="store_true", help="Re-index all files regardless of state")
    args = p.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()

    # Check server
    status = server_status(args.server)
    if status is None:
        print(f"ContextFit server not reachable at {args.server} — skipping indexing", flush=True)
        sys.exit(0)

    state = load_state()
    all_files = collect_files(workspace)

    # Find new or changed files
    to_index: list[Path] = []
    for f in all_files:
        sig = file_sig(f)
        key = str(f)
        if args.force or state["indexed"].get(key) != sig:
            to_index.append(f)

    print(f"Files found: {len(all_files)}  |  New/changed: {len(to_index)}", flush=True)

    if not to_index:
        print("Nothing to index.", flush=True)
        return

    if args.dry_run:
        print("Dry run — would index:")
        for f in to_index[:30]:
            print(f"  {f.relative_to(workspace)}")
        if len(to_index) > 30:
            print(f"  ... and {len(to_index) - 30} more")
        return

    # Ingest in batches of 50
    batch_size = 50
    indexed_count = 0
    for i in range(0, len(to_index), batch_size):
        batch = to_index[i:i + batch_size]
        print(f"Ingesting batch {i // batch_size + 1} ({len(batch)} files) ...", flush=True)
        try:
            result = server_ingest(args.server, args.kb, batch)
            errors = result.get("errors", [])
            if errors:
                print(f"  Errors: {errors}", flush=True)
                # Do not mark errored batches indexed; they'll retry next run.
                continue
            # Update state only after a clean ingest response.
            for f in batch:
                state["indexed"][str(f)] = file_sig(f)
            indexed_count += len(batch)
        except Exception as e:
            print(f"  Ingest error: {e}", flush=True)

    save_state(state)
    print(f"Done. Indexed {indexed_count} file(s).", flush=True)


if __name__ == "__main__":
    main()
