# ContextFit — Token-Native Agent Memory

ContextFit is an open-source memory retrieval engine for AI agents. It asks a simple question: what if conversational memory did not need to become vectors before it could be useful?

Interactive demo: https://context.fit/demo.html shows four memory scenarios where ContextFit retrieves preferences, open loops, temporal updates, and multi-session evidence with citations.

Instead of sending every session through an embedding model and hiding meaning inside cosine distance, ContextFit keeps memory close to the text. It indexes tokenized conversations, extracts deterministic memory atoms, scores episodes by the kind of memory they contain, routes queries to the right retrieval mode, and reranks with transparent token-native signals.

The emotional point is trust. Agent memory is intimate: preferences, decisions, constraints, goals, open loops, and the history of what someone told you. ContextFit is designed so that memory can be inspected, explained, moved, backed up, and run locally.

## Why people care

- **No vector database required.** The index is files on disk.
- **No required embedding API.** The core retrieval path is token-native and local.
- **No GPU required.** Runs on CPU.
- **Interpretable retrieval.** Results can be explained by atoms, routes, scores, and source evidence.
- **Compact citation handles.** Prompt context can use tiny `@r1` handles while exact source/chunk/line/row provenance and scores stay in an expiring sidecar reference map.
- **Agent-specific memory behavior.** Built for preferences, goals, constraints, decisions, temporal updates, open loops, and multi-session synthesis.

## Core primitives

1. **Memory atoms** — deterministic, domain-agnostic fact extraction from conversational text.
2. **Episode relevance scoring** — structural numeric ranking for vague advice and episodic inference.
3. **Query router** — near-zero-cost dispatch to the retrieval mode that matches the question.
4. **Structural session reranker** — token-native post-retrieval reranking with question-type slot matching.
5. **Preference reranker** — user-authored taste evidence for personalized recommendations.
6. **Evidence-coverage reranker** — complementary-evidence ranking for multi-session synthesis.

## Benchmarks

On the 499-case domain-agnostic agent-memory benchmark (Mem0 measured on the original 79-case subset):

| System | R@1 | R@3 | MRR | Cost |
|---|---:|---:|---:|---:|
| Mem0 (GPT-4o-mini + embed, 79-case) | 54.4% | 91.1% | 0.716 | LLM + embed API |
| Cohere embed-english-v3 | 58.7% | 91.4% | 0.751 | embed API |
| ContextFit + routed rerankers | 62.3% | 93.0% | 0.777 | $0 core path |
| OpenAI text-embedding-3-small | 63.1% | 96.6% | 0.792 | embed API |

Behavior highlights:

- Preference recommendation: ContextFit 85.5% R@1 vs OpenAI 77.4%.
- Multi-session synthesis: ContextFit 82.1% R@1 vs OpenAI 87.5%.
- LongMemEval-S: pure token-native ContextFit with conversation-aware parent/child chunks reaches 95.1% Any@5. It matches OpenAI fusion on preference Any@5 (83.3%) and narrows multi-session Any@5 to within ~0.8 pts. The companion-evidence coverage reranker preserves 95.1% Any@5 while improving overall All@5 from 77.9% to 80.4% and multi-session All@5 from 55.4% to 65.3%.

## Install

```bash
pip install contextfit
```

## Minimal API

```python
from contextfit import RetrievalEngine

engine = RetrievalEngine()
engine.ingest_sessions(sessions)
engine.save("./memory_index")

result = engine.query_auto(
    "what should I cook for dinner tonight?",
    top_k=5,
)

print(result["route"])
print(result["session_ids"])
```

## More

- Full whitepaper markdown: https://context.fit/token-native-agent-memory.md
- Human website: https://context.fit/
- GitHub: https://github.com/ContextFit/cf
- Creator: Christophe Ponsart — https://x.com/cponsart


## Structure-aware ingestion

ContextFit now chooses semantic file boundaries before final token encoding:

- Markdown (`.md`) chunks by headings and semantic blocks, with `heading_path`, `section_level`, and `chunk_ordinal` metadata.
- Plain text (`.txt`) chunks by paragraphs/separators with paragraph-level overlap.
- TMD ledger (`.tmd`) chunks by source rows while preserving schema/front-matter context. TMD ledger is a new ContextFit-proposed Tabular Markdown file format for row-addressable, human-readable ledgers.
- JSON / JSONL (`.json`, `.jsonl`) chunks by object/event records with path, line, and index metadata.
- CSV / TSV (`.csv`, `.tsv`) chunks by source rows while preserving headers as fields.
- Email (`.eml`) chunks messages with sender, recipient, subject, and date context preserved.
- Calendar (`.ics`) chunks events with summary, time, location, recurrence, and attendee metadata.
- Code files (`.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.sh`, `.sql`, `.css`, `.html`, and more) chunk by generic symbol/import boundaries with language, symbol, and line-range metadata.
- Unknown formats fall back to conservative token windows.

The retrieval engine and CLI both use this routing for file ingestion. The tokenizer remains the source of truth for stored token IDs.
