# ContextFit for Claude Desktop on macOS

This guide walks through setting up ContextFit as a **local memory tool for Claude Desktop** on a new MacBook.

Claude Desktop is the chat UI. ContextFit is the local search/memory engine. MCP is the bridge between them.

## What you will have at the end

Claude Desktop will get these ContextFit tools:

- `contextfit_search` — search a local ContextFit knowledge base and return source-backed chunks
- `contextfit_get_chunk` — fetch the full text for a returned `chunk_id`
- `contextfit_stats` — show local KB stats
- `contextfit_list_vaults` — list named local vaults, if configured
- `contextfit_search_vault` — search one named vault
- `contextfit_search_all_vaults` — search all registered vaults

Your source files stay on your Mac. Claude only sees the snippets ContextFit returns into the active chat.

---

## Quick MacBook setup

Assume:

- Claude Desktop is installed.
- Your documents are in `~/Documents/contextfit-demo`.
- Your ContextFit knowledge base will live at `~/contextfit_kb`.

### 1. Install command-line prerequisites

Open **Terminal** and install Apple command-line tools if needed:

```bash
xcode-select --install
```

If you do not already have Python 3 available, install it with Homebrew or from python.org.

Check:

```bash
python3 --version
git --version
```

### 2. Install ContextFit

Create a virtual environment and install ContextFit from PyPI:

```bash
mkdir -p ~/contextfit
cd ~/contextfit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install contextfit
```

Contributor install from source:

```bash
cd ~/Documents
git clone https://github.com/ContextFit/cf.git
cd cf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Confirm the CLI works:

```bash
contextfit --help
```

If `contextfit` is not found, use the full path to the venv executable later:

```bash
pwd
# Example output for PyPI install: /Users/YOU/contextfit
# ContextFit executable: /Users/YOU/contextfit/.venv/bin/contextfit
```

### 3. Create a small local knowledge base

Create a demo folder if you do not already have one:

```bash
mkdir -p ~/Documents/contextfit-demo
cat > ~/Documents/contextfit-demo/project-orion.md <<'EOF'
# Project Orion

The launch checklist has three open items: pricing review, docs polish, and support handoff.
The preferred launch voice is concise, technical, and practical.
EOF
```

Ingest it into ContextFit:

```bash
contextfit --kb ~/contextfit_kb ingest ~/Documents/contextfit-demo \
  --defer-index-build \
  --rebuild-index-after-ingest
```

Test search before opening Claude:

```bash
contextfit --kb ~/contextfit_kb query "What are the open launch items?" --json
```

You should see JSON with retrieved chunks from `project-orion.md`.

### 4. Find your ContextFit executable path

Claude Desktop often runs with a smaller PATH than your Terminal. The safest setup is to use the absolute path:

```bash
which contextfit
```

If you installed with the venv above, this will usually be:

```text
/Users/YOU/contextfit/.venv/bin/contextfit
```

Use that path in the Claude config below.

### 5. Open Claude Desktop's config file

Run this in Terminal:

```bash
mkdir -p "$HOME/Library/Application Support/Claude"
open -e "$HOME/Library/Application Support/Claude/claude_desktop_config.json"
```

If the file is blank or does not exist yet, paste this JSON.

Replace:

- `/Users/YOU/contextfit/.venv/bin/contextfit` with your `which contextfit` result
- `/Users/YOU/contextfit_kb` with your actual KB path

```json
{
  "mcpServers": {
    "contextfit": {
      "command": "/Users/YOU/contextfit/.venv/bin/contextfit",
      "args": [
        "--kb",
        "/Users/YOU/contextfit_kb",
        "mcp",
        "--top-k",
        "8",
        "--method",
        "hybrid"
      ]
    }
  }
}
```

Save the file.

If you already have other MCP servers configured, add only the `"contextfit"` block inside the existing `"mcpServers"` object.

### 6. Restart Claude Desktop completely

Quit Claude Desktop fully and reopen it.

On macOS, use **Claude → Quit Claude** or press `Cmd+Q`. Closing the window is not always enough.

### 7. Try it in Claude

Ask Claude:

```text
Use ContextFit to search my local memory for Project Orion. What are the open launch items? Cite the source chunks.
```

Or:

```text
List the ContextFit tools you can use.
```

---

## Raw MCP smoke test

If Claude does not show the tools, test the MCP server directly in Terminal:

```bash
printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n' \
  | /Users/YOU/contextfit/.venv/bin/contextfit --kb /Users/YOU/contextfit_kb mcp
```

You should see JSON-RPC responses that include tools such as `contextfit_search`.

If this fails, fix the command path or KB path before trying Claude again.

---

## Multiple knowledge bases / vaults

For multiple local collections, use a vault registry instead of one MCP server per folder.

Create `~/.contextfit/vaults.json`:

```json
{
  "vaults": {
    "work": {
      "kb_path": "/Users/YOU/contextfit_work_kb",
      "description": "Work docs, notes, and project memory"
    },
    "research": {
      "kb_path": "/Users/YOU/contextfit_research_kb",
      "description": "Papers, references, and experiments"
    },
    "personal": "/Users/YOU/contextfit_personal_kb"
  }
}
```

Then add `--vault-registry` to Claude's ContextFit args:

```json
{
  "mcpServers": {
    "contextfit": {
      "command": "/Users/YOU/contextfit/.venv/bin/contextfit",
      "args": [
        "--kb",
        "/Users/YOU/contextfit_kb",
        "mcp",
        "--vault-registry",
        "/Users/YOU/.contextfit/vaults.json"
      ]
    }
  }
}
```

Then in Claude:

```text
List my ContextFit vaults.
Search the work vault for the Acme renewal audit.
Search all vaults for website tokenization.
```

If the registry file does not exist, ContextFit still exposes a single `default` vault from `--kb`.

---

## CLI integration for local agents

MCP is best for interactive desktop clients like Claude Desktop. Local agent runtimes, shell scripts, cron jobs, OpenClaw, and Hermes can use JSON CLI output directly:

```bash
contextfit vaults list --json
contextfit search "Acme renewal audit" --vault work --json --extractive auto --compact
contextfit search-all "website tokenization" --json --extractive auto --compact
contextfit chunk 42 --vault work --json
```

The CLI and MCP server use the same local vault registry and retrieval engine. Use `--extractive auto --compact` when an agent needs prompt-ready source-backed evidence instead of full chunk previews or JSON-heavy metadata; ContextFit will project matching `.tmd` rows, bullets, or spans without an LLM.

---

## Troubleshooting checklist

### Claude does not show ContextFit tools

1. Quit and reopen Claude Desktop with `Cmd+Q`.
2. Confirm `claude_desktop_config.json` is valid JSON.
3. Use the absolute path from `which contextfit` for `command`.
4. Use absolute paths in `args`; avoid `~` in Claude config.
5. Run the raw MCP smoke test above.

### `contextfit` works in Terminal but not in Claude

Use the full venv executable path:

```json
"command": "/Users/YOU/contextfit/.venv/bin/contextfit"
```

Claude may not load your shell profile, so relying on PATH can fail.

### Search returns nothing

Run:

```bash
contextfit --kb ~/contextfit_kb stats --json
contextfit --kb ~/contextfit_kb query "test" --json
```

If the KB has zero chunks, ingest your source folder again.

### Privacy reminder

- ContextFit reads only the KB path or vaults you configure.
- Your original files stay local.
- Retrieved snippets are sent into the active Claude chat as tool results.
- Only index sensitive folders if you are comfortable with relevant snippets appearing in Claude conversations.
