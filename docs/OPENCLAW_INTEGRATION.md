# OpenClaw Integration

ContextFit ships an OpenClaw plugin at:

```text
integrations/openclaw/contextfit-search
```

The plugin provides two OpenClaw surfaces:

1. `contextfit_search` — an agent-callable tool for explicit KB searches.
2. `contextfit` / `contextfit-search` — context engine aliases that perform fail-open ContextFit auto-recall during context assembly.

The context engine currently delegates compaction back to OpenClaw, so it does not replace OpenClaw's compaction lifecycle. If the ContextFit server is unavailable, auto-recall is skipped and the turn continues.

## Prerequisites

Start a ContextFit server, for example:

```bash
python server.py \
  --kb memory:/path/to/contextfit-memory-kb \
  --port 8765 \
  --host 127.0.0.1
```

The plugin defaults to `http://127.0.0.1:8765` and KB `memory`.

## Install into OpenClaw

From the ContextFit repo root:

```bash
openclaw plugins install ./integrations/openclaw/contextfit-search --force
```

Then enable/select it in `~/.openclaw/openclaw.json`:

```json5
{
  plugins: {
    slots: {
      contextEngine: "contextfit-search"
    },
    entries: {
      "contextfit-search": {
        enabled: true,
        config: {
          baseUrl: "http://127.0.0.1:8765",
          defaultKb: "memory",
          engine: {
            autoRecall: true,
            method: "hybrid",
            topK: 5,
            includeText: false,
            returnSpans: true,
            maxPromptChars: 2000,
            maxAdditionChars: 4000
          }
        }
      }
    }
  }
}
```

Restart OpenClaw Gateway:

```bash
openclaw gateway restart
```

## Verify

```bash
openclaw plugins inspect contextfit-search --runtime --json
```

Expected runtime surfaces:

- `toolNames` includes `contextfit_search`
- `contextEngineIds` includes `contextfit` and `contextfit-search`
- diagnostics are empty

Smoke-test ContextFit directly:

```bash
curl -sS -X POST http://127.0.0.1:8765/query \
  -H 'content-type: application/json' \
  -d '{"kb":"memory","query":"Brooks Glycerin 22","method":"hybrid","top_k":1,"return_spans":true,"include_text":false}'
```

## Design notes

- The manual tool is source-verifiable: results include source paths, spans/matches, and `.tmd` rows when available.
- The context engine is deliberately conservative: it injects compact recall handles into `systemPromptAddition` rather than large text blobs.
- Text materialization should remain lazy. Prefer token/span/row handles until final evidence or user-facing output is needed.
- Compaction is delegated to OpenClaw via `delegateCompactionToRuntime(...)`.
