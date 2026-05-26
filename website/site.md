# ContextFit — Token-Native Agent Memory

ContextFit is an open-source memory retrieval engine for AI agents. It asks a simple question: what if conversational memory did not need to become vectors before it could be useful?

Interactive demo: https://context.fit/demo.html shows four memory scenarios where ContextFit retrieves preferences, open loops, temporal updates, and multi-session evidence with citations.

Instead of sending every session through an embedding model and hiding meaning inside cosine distance, ContextFit keeps memory close to the text. It indexes tokenized conversations, extracts deterministic memory atoms, scores episodes by the kind of memory they contain, routes queries to the right retrieval mode, and reranks with transparent token-native signals, including auditable evidence certificates.

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
7. **Evidence-certificate reranker** — auditable promotion rules that move strong answer evidence up only when generic reason codes fire.

## Benchmarks

LongMemEval-S session retrieval asks a clean retrieval question: did the system put a ground-truth answer session in the top 5, with no answer generation and no LLM judge.

| System | R@1 | R@3 | R@5 | R@10 | Retrieval path |
|---|---:|---:|---:|---:|---|
| ContextFit token-native | 81.80% | 90.40% | 96.20% | 97.80% | no embeddings, no vector DB, no LLM in retrieval |
| ContextFit + OpenAI fusion | 84.60% | 95.20% | 99.00% | 99.60% | route-gated chunk-vector fusion, no vector DB |
| gbrain-hybrid published reference | - | - | 97.60% | - | published reference |
| MemPalace raw published reference | - | - | 96.60% | - | published reference |

Agent-memory behavior highlights from the internal engineering eval:

- Preference recommendation: ContextFit 85.5% R@1 vs OpenAI 77.4%.
- Open-loop retrieval: ContextFit 80.3% R@1 vs OpenAI 63.9%.
- Temporal supersession: ContextFit 49.2% R@1 vs OpenAI 47.6%.
- Multi-session synthesis: ContextFit 82.1% R@1 vs OpenAI 87.5%.

The 499-case agent-memory eval is useful for engineering, but it is not the homepage leaderboard claim: Mem0 was measured on a 79-case subset, and aggregate embedding recall does not capture ContextFit's product advantage. The product claim is that ContextFit adds routing, source aggregation, temporal handling, provenance handles, abstention, and local deployment controls around retrieval.

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
- LongMemEval-S retrieval artifact: https://context.fit/longmemeval-fusion-20260519.html
- LongMemEval-S end-to-end QA artifact: https://context.fit/longmemeval-fusion-qa-20260519.html
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
