# Deploy ContextFit CLI on a MacBook

ContextFit ships a Python CLI named `contextfit`.

Commands currently available:

```bash
contextfit ingest <file-or-directory>
contextfit query "your question"
contextfit stats
```

## Requirements

- macOS with Python 3.10+
- A local copy of the `contextfit` repository
- Optional but recommended: the MacBook's own OpenClaw install, so agents on that machine can call the CLI through normal shell execution

## Option A: Install from a local checkout

On the MacBook:

```bash
git clone <repo-url-or-local-copy> contextfit
cd contextfit
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify:

```bash
contextfit --help
pytest tests/ -q
```

## Option B: Copy from the current machine

From the MacBook, copy the project directory from this host or another synced location:

```bash
rsync -av --exclude .venv --exclude .git \
  user@source-host:/home/example/.openclaw/workspace-main/contextfit/ \
  ~/contextfit/

cd ~/contextfit
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Build a knowledge base

```bash
mkdir -p ~/contextfit-data/docs
# Put .txt or .md files under ~/contextfit-data/docs

contextfit --kb ~/contextfit-data/kb ingest ~/contextfit-data/docs \
  --chunk-size 512 \
  --overlap 64 \
  --train-sid-generator
```

This creates:

```text
~/contextfit-data/kb/
  chunks/
    chunks.bin
    index.json
  inverted/
    meta.json
    postings.bin
  sid/
    semantic_ids.json
    learned_sid_generator.json
```

## Query

```bash
contextfit --kb ~/contextfit-data/kb query "What does this knowledge base know?" --method hybrid
contextfit --kb ~/contextfit-data/kb query "async retrieval" --method sid --top-k 3
contextfit --kb ~/contextfit-data/kb stats
```

For OpenClaw/agent consumption, use JSON:

```bash
contextfit --kb ~/contextfit-data/kb query "summarize deployment notes" --method hybrid --json
contextfit --kb ~/contextfit-data/kb stats --json
```

Query JSON includes:

- `query_tokens`
- `input_ids`
- retrieved chunks with scores, token arrays, previews, metadata, and semantic IDs
- generated SID predictions

## Using from OpenClaw on the MacBook

Once installed, the MacBook's OpenClaw agent can call the CLI with shell commands, for example:

```bash
cd ~/contextfit
source .venv/bin/activate
contextfit --kb ~/contextfit-data/kb query "summarize the deployment notes" --method hybrid --json
```

For persistent convenience, expose the CLI in the shell used by OpenClaw:

```bash
ln -sf ~/contextfit/.venv/bin/contextfit ~/.local/bin/contextfit
```

Make sure `~/.local/bin` is on PATH for the OpenClaw runtime shell.

## Current CLI limitations

The CLI is local-only right now:

- No network daemon/server mode yet
- No REST API yet
- No OpenClaw plugin wrapper yet
- No long-running watch mode yet

Recommended next additions for agent use:

1. `contextfit serve --host 127.0.0.1 --port 8765`
2. `contextfit ingest --watch <dir>`
3. OpenClaw skill/plugin wrapper that calls the CLI
