#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGIN_DIR="$ROOT/integrations/openclaw/contextfit-search"

if ! command -v openclaw >/dev/null 2>&1; then
  echo "openclaw CLI not found on PATH" >&2
  exit 1
fi

openclaw plugins install "$PLUGIN_DIR" --force
cat <<MSG

Installed ContextFit OpenClaw plugin from:
  $PLUGIN_DIR

Next: select the context engine in ~/.openclaw/openclaw.json:

  plugins.slots.contextEngine = "contextfit"
  plugins.entries.contextfit-search.enabled = true

See docs/OPENCLAW_INTEGRATION.md for the full config block.
MSG
