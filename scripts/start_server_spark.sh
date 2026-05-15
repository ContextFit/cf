#!/bin/bash
# Start the ContextFit query server on DGX Spark.

set -euo pipefail

CF_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$CF_DIR/.venv"
LOG_DIR="$HOME/.openclaw/logs/contextfit"
WORKSPACE="$HOME/.openclaw/workspace-main"

mkdir -p "$LOG_DIR"

exec "$VENV/bin/python3" "$CF_DIR/server.py" \
    --kb "memory:$WORKSPACE/contextfit-memory-kb" \
    --kb "sessions:$WORKSPACE/contextfit-sessions-kb" \
    --port 8765 \
    --host 127.0.0.1 \
    >> "$LOG_DIR/server.log" 2>&1
