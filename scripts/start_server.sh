#!/bin/bash
# Start the ContextFit query server.
# Sourced by the LaunchAgent plist.

set -euo pipefail

CF_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$CF_DIR/.venv"
LOG_DIR="$HOME/Library/Logs/contextfit"
MEMORY_KB="$HOME/.openclaw/tools/tmp/contextfit-memory-kb"

mkdir -p "$LOG_DIR"
mkdir -p "$MEMORY_KB"

exec "$VENV/bin/python3" "$CF_DIR/server.py" \
    --kb "memory:$MEMORY_KB" \
    --kb "email:$HOME/.openclaw/tools/tmp/contextfit-email-kb" \
    --kb "tax:$HOME/.openclaw/tools/tmp/contextfit-tax-kb" \
    --port 8765 \
    --host 127.0.0.1 \
    >> "$LOG_DIR/server.log" 2>&1
