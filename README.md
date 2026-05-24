# ContextFit

**A token-native knowledge base designed for LLM scale.**

ContextFit keeps everything—storage, indexing, search, relationships, traversal, and commonality detection—inside discrete token-ID space until the very last step, when you decode only the final retrieved token chunks for the LLM's output.

Latest LongMemEval-S retrieval artifact: pure token-native ContextFit reaches **95.1% Any@5** with conversation-aware parent/child chunks. With optional OpenAI fusion, ContextFit reaches **96.6% Any@5** and **98.7% Any@10** evidence retrieval with no vector database required; adding auditable evidence-certificate reranking reaches **98.3% Any@5**, **99.2% Any@10**, and **86.4% All@5** in the same local retrieval harness with zero paired top-5 losses versus the 96.6% fusion baseline. Current end-to-end QA progress reports **85.2%** overall with a GPT-4o-only selective-fusion run, and **87.2%** overall / **87.6%** task-averaged with a GPT-5-mini answerer/extractor plus GPT-4o judging. These are local LongMemEval-style evaluations, not official leaderboard submissions. See [`benchmarks/longmemeval_contextfit_report.md`](benchmarks/longmemeval_contextfit_report.md).

## Why Token-Native?

- **~2× smaller storage** than raw text (no repeated tokenization)
- **Blazing-fast integer-only operations** (no float embeddings)
- **Hierarchical "geo-map-style" traversal** for multi-hop reasoning
- **Neural-network-like chunk relationships** via token overlap graphs
- **Automatic commonality discovery** without vector spaces
- **Direct LLM injection** — feed `input_ids` directly, no conversion
- **Structure-aware ingestion** — Markdown sections, text paragraphs, and TMD ledger rows become retrievable units

## Fastest Path

Use this when you just want to confirm ContextFit works locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install contextfit

mkdir -p /tmp/contextfit-demo
cat > /tmp/contextfit-demo/project-orion.md <<'EOF'
# Project Orion

The launch checklist has three open items: pricing review, docs polish, and support handoff.
The preferred launch voice is concise, technical, and practical.
EOF

contextfit --kb /tmp/contextfit-kb ingest /tmp/contextfit-demo \
  --defer-index-build \
  --rebuild-index-after-ingest

contextfit --kb /tmp/contextfit-kb query "What are the open launch items?" --json
```

You should see JSON results with chunks from `project-orion.md`. From there:

- For Claude Desktop, follow [`docs/CLAUDE_DESKTOP_MCP.md`](docs/CLAUDE_DESKTOP_MCP.md).
- For local agents and automation, use `contextfit search ... --json --extractive auto --compact`.
- For contributor development, clone this repo and run `python -m pip install -e .`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ContextFit                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Storage   │  │   Index     │  │        Graph            │  │
│  │             │  │             │  │                         │  │
│  │ Token Arrays│  │ Inverted    │  │ Chunk Relationships     │  │
│  │ Chunk Store │  │ Suffix/FM   │  │ Community Detection     │  │
│  │ Compression │  │ BM25 Tokens │  │ Commonality Mining      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                  │
│  ┌─────────────────────────────┐  ┌─────────────────────────┐   │
│  │        Hierarchy            │  │       Retrieval         │   │
│  │                             │  │                         │   │
│  │ Level 0: Raw Chunks         │  │ Query Tokenization      │   │
│  │ Level 1+: Summary Clusters  │  │ Graph Traversal         │   │
│  │ Geo-Map Navigation          │  │ Direct input_ids Output │   │
│  └─────────────────────────────┘  └─────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Semantic IDs (SIDs)                      ││
│  │                                                             ││
│  │  Hierarchical token sequences → generative retrieval        ││
│  │  Similar chunks share prefixes → trie-like navigation       ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Storage Layer
- Token arrays (uint16/uint32 IDs)
- Memory-mapped files for large corpora
- Delta encoding + Zstd compression
- Chunk metadata headers

### 2. Index Layer
- **Inverted Index**: tokenID → [(chunkID, positions)] using Roaring bitmaps
- **Suffix Array / FM-Index**: Instant exact n-gram search
- **BM25 on Tokens**: TF-IDF scoring with token IDs as terms
- **Binary postings pack**: one compact `postings.bin` instead of JSON-per-token files

### 3. Graph Layer
- Nodes = chunks (or Semantic IDs)
- Edges = token n-gram overlap, Jaccard similarity, co-occurrence
- MinHash + LSH for fast similarity without floats
- Community detection for commonality discovery

### 4. Hierarchy Layer
- Level 0: Raw token chunks (256–1024 tokens each)
- Level 1+: Clustered summaries as token sequences
- GraphRAG-style community summaries
- Integer pointers for zoom navigation

### 5. Retrieval Layer
- Tokenize query → search indexes → traverse graph → collect token IDs
- Feed directly as `input_ids` to any LLM
- No detokenization until final generation

### 5a. Structure-Aware Chunking
- TMD ledger (`.tmd`) files chunk by rows while preserving schema/front matter context; TMD ledger is a new ContextFit-proposed Tabular Markdown file format for row-addressable, human-readable ledgers
- `.md` files chunk by heading/section boundaries and attach `heading_path` metadata
- `.txt` files chunk by paragraphs/separators with paragraph-level overlap
- `.json` / `.jsonl` files chunk by object/event records while preserving path, line, and index metadata
- `.csv` / `.tsv` files chunk by source rows while preserving headers as fields
- `.eml` files chunk email messages with sender, recipient, subject, and date context preserved
- `.ics` files chunk calendar events with summary, time, location, recurrence, and attendee metadata
- Code files (`.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.sh`, `.sql`, `.css`, `.html`, and more) chunk by generic symbol/import boundaries with language, symbol, and line-range metadata.
- Unknown formats fall back to conservative token windows

### 6. Semantic IDs
- Assign each chunk a short hierarchical SID token sequence
- Similar chunks share prefixes via MinHash-band residual buckets
- Resolve generated/predicted SID prefixes through a trie with prefix backoff
- Retrieval mode: `--method sid` or hybrid SID + BM25

### 7. SID Generator
- Predicts SID prefixes from query tokens without detokenizing
- Combines BM25 candidate chunks, MinHash similarity, and LSH neighbors
- Candidate chunks vote for hierarchical SID prefixes
- Returns generated SID predictions plus resolved chunk IDs

### 8. Learned SID Generator
- Trains a sparse token→SID associative model from stored chunks
- Uses beam search over valid SID prefixes
- No neural dependency yet; still token-native and deterministic
- CLI: `contextfit ingest ./docs --train-sid-generator`

## Getting Started

```bash
# Install ContextFit
pip install contextfit

# Ingest a knowledge base into an explicit local KB path
contextfit --kb ~/contextfit_kb ingest ./documents --tokenizer tiktoken

# Query
contextfit --kb ~/contextfit_kb query "What is ContextFit?"

# Query through Semantic IDs
contextfit --kb ~/contextfit_kb query "async retrieval" --method sid

# Agent-friendly machine-readable output
contextfit --kb ~/contextfit_kb query "What is ContextFit?" --method hybrid --json
contextfit --kb ~/contextfit_kb search "Acme renewal audit" --json --extractive auto --compact
contextfit --kb ~/contextfit_kb stats --json

# Run a deterministic sample benchmark
python examples/benchmark_sample_corpus.py --docs-per-topic 100 --json

# Run needle-in-a-haystack benchmark
python examples/benchmark_needle_haystack.py --needles 20 --distractors 200 --top-k 5 --json

# Ingest and train the learned SID generator
contextfit --kb ~/contextfit_kb ingest ./documents --train-sid-generator
```

For contributor installs from source, clone the repo and run `pip install -e .` from the project root.

For installing on a MacBook/OpenClaw node, see [`docs/MACBOOK_CLI_DEPLOY.md`](docs/MACBOOK_CLI_DEPLOY.md).

For OpenClaw integration, including the `contextfit_search` tool and `contextfit` context engine plugin, see [`docs/OPENCLAW_INTEGRATION.md`](docs/OPENCLAW_INTEGRATION.md).

For Claude Desktop, ContextFit includes a local MCP stdio server. See [`docs/CLAUDE_DESKTOP_MCP.md`](docs/CLAUDE_DESKTOP_MCP.md) or run:

```bash
contextfit --kb ~/contextfit_kb mcp
```

`--json` is intended for OpenClaw/agent use. Query JSON includes `input_ids`, retrieved chunk metadata, SID predictions, semantic IDs, and decoded previews. `contextfit search --json --extractive auto` also returns deterministic query-focused evidence (`tmd_row`, `bullet`, or `span`) with source line and row IDs when available. Add `--compact` to emit a TMD-like `compact_context` string that preserves citations while avoiding JSON-heavy metadata in an LLM prompt.

## Current Storage Layout

```text
contextfit_kb/
  chunks/
    chunks.bin        # zstd-compressed token-array records
    index.json        # chunk_id → byte offset/length
  inverted/
    meta.json         # corpus/index metadata
    postings.bin      # compact binary token → roaring bitmap + positions pack
  sid/
    semantic_ids.json
    learned_sid_generator.json
```

The inverted index now saves as a single binary postings pack by default. Legacy JSON-per-token indexes still load for compatibility.

## Project Status

🚧 **Early Development** — Architecture phase

## References

- TERAG: Token-Efficient GraphRAG (3–11% token reduction)
- Semantic IDs / Generative Retrieval
- GraphRAG community detection
- Letta's token-space learning

## License

MIT
