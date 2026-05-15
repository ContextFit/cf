# ContextFit

**A token-native knowledge base designed for LLM scale.**

ContextFit keeps everything—storage, indexing, search, relationships, traversal, and commonality detection—inside discrete token-ID space until the very last step, when you decode only the final retrieved token chunks for the LLM's output.

## Why Token-Native?

- **~2× smaller storage** than raw text (no repeated tokenization)
- **Blazing-fast integer-only operations** (no float embeddings)
- **Hierarchical "geo-map-style" traversal** for multi-hop reasoning
- **Neural-network-like chunk relationships** via token overlap graphs
- **Automatic commonality discovery** without vector spaces
- **Direct LLM injection** — feed `input_ids` directly, no conversion
- **Structure-aware ingestion** — Markdown sections, text paragraphs, and TMD ledger rows become retrievable units

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

# Ingest a knowledge base
contextfit ingest ./documents --tokenizer tiktoken

# Query
contextfit query "What is ContextFit?"

# Query through Semantic IDs
contextfit query "async retrieval" --method sid

# Agent-friendly machine-readable output
contextfit query "What is ContextFit?" --method hybrid --json
contextfit search "Acme renewal audit" --json --extractive auto --compact
contextfit stats --json

# Run a deterministic sample benchmark
python examples/benchmark_sample_corpus.py --docs-per-topic 100 --json

# Run needle-in-a-haystack benchmark
python examples/benchmark_needle_haystack.py --needles 20 --distractors 200 --top-k 5 --json

# Ingest and train the learned SID generator
contextfit ingest ./documents --train-sid-generator
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
