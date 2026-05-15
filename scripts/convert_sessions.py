#!/usr/bin/env python3
"""
Convert OpenClaw session JSONL files to markdown for ContextFit ingestion.

Extracts user/assistant messages and saves as searchable markdown files.

Usage:
    python scripts/convert_sessions.py --output ./session-corpus/
    python scripts/convert_sessions.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


SESSION_DIRS = [
    Path.home() / ".openclaw" / "agents" / "main" / "sessions",
    Path.home() / ".openclaw" / "agents" / "voice" / "sessions",
    Path.home() / ".openclaw" / "agents" / "codex" / "sessions",
]

DEFAULT_OUTPUT = Path.home() / ".openclaw" / "workspace-main" / "session-corpus"


def parse_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL session file."""
    messages = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # Handle OpenClaw session format: type=message with nested message
                if record.get("type") == "message" and "message" in record:
                    msg = record["message"]
                    msg["_timestamp"] = record.get("timestamp")
                    messages.append(msg)
                # Handle plain message format
                elif "role" in record:
                    messages.append(record)
            except json.JSONDecodeError:
                continue
    return messages


def session_to_markdown(messages: list[dict], session_id: str) -> str:
    """Convert session messages to markdown."""
    lines = [f"# Session: {session_id}\n"]
    
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        # Skip system messages and empty content
        if role == "system" or not content:
            continue
        
        # Skip tool calls/results (they're noisy for search)
        if role == "tool" or msg.get("tool_calls"):
            continue
            
        # Handle content that might be a list (multimodal)
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts)
        
        if not content or not content.strip():
            continue
            
        # Format as markdown
        speaker = "**User:**" if role == "user" else "**Assistant:**"
        lines.append(f"\n{speaker}\n{content.strip()}\n")
    
    return "\n".join(lines)


def get_session_date(messages: list[dict]) -> str | None:
    """Try to extract date from session messages."""
    for msg in messages:
        # Look for timestamp in metadata or content
        ts = msg.get("timestamp") or msg.get("created_at")
        if ts:
            try:
                if isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts)
                    return dt.strftime("%Y-%m-%d")
            except:
                pass
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--min-messages", type=int, default=4, help="Skip sessions with fewer messages")
    args = p.parse_args()
    
    output_dir = Path(args.output).expanduser().resolve()
    
    # Find all session JSONL files
    sessions: list[tuple[Path, str]] = []
    for session_dir in SESSION_DIRS:
        if not session_dir.exists():
            continue
        for f in session_dir.glob("*.jsonl"):
            session_id = f.stem
            sessions.append((f, session_id))
    
    print(f"Found {len(sessions)} session files")
    
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    converted = 0
    skipped = 0
    
    for path, session_id in sessions:
        messages = parse_jsonl(path)
        
        # Filter to user/assistant only
        relevant = [m for m in messages if m.get("role") in ("user", "assistant")]
        
        if len(relevant) < args.min_messages:
            skipped += 1
            continue
        
        md = session_to_markdown(messages, session_id)
        
        # Get date for filename
        date = get_session_date(messages) or "unknown"
        out_name = f"{date}_{session_id[:8]}.md"
        out_path = output_dir / out_name
        
        if args.dry_run:
            print(f"Would write: {out_name} ({len(relevant)} messages)")
        else:
            out_path.write_text(md)
            converted += 1
    
    print(f"Converted: {converted} | Skipped (< {args.min_messages} messages): {skipped}")


if __name__ == "__main__":
    main()
