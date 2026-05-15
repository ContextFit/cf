# ContextFit Performance Testing

ContextFit includes deterministic sample-corpus tests and a local benchmark script.

## Test coverage

`tests/test_sample_corpus_performance.py` validates:

- Retrieval quality across BM25, SID, and hybrid methods
- Learned SID generator correctness on a multi-topic corpus
- Binary postings pack save/load behavior
- Rough ingest/load/query latency thresholds
- Binary pack compatibility with legacy JSON postings

Run:

```bash
pytest tests/test_sample_corpus_performance.py -q
```

The thresholds are intentionally generous so they are useful in CI and on laptops:

- sample corpus build + learned SID training + save: `< 10s`
- load: `< 3s`
- four hybrid queries: `< 1s total`
- retrieval accuracy@1 on deterministic topic queries: `100%`

## Local benchmark

Run:

```bash
python examples/benchmark_sample_corpus.py --docs-per-topic 100 --json
```

Example output shape:

```json
{
  "docs_per_topic": 100,
  "topics": 4,
  "chunks": 1200,
  "accuracy_at_1": 1.0,
  "ingest_seconds": 1.4,
  "load_seconds": 1.0,
  "total_query_seconds": 0.05,
  "avg_query_ms": 12.5,
  "postings_bytes": 1024000,
  "storage_bytes": 240000,
  "vocab_size": 75,
  "learned_sid_trained_chunks": 1200
}
```

Use this script when testing storage/index changes on a MacBook or another OpenClaw node.

## Needle-in-a-Haystack benchmark

ContextFit includes a synthetic NIAH/RULER-style retrieval benchmark:

```bash
python examples/benchmark_needle_haystack.py \
  --needles 20 \
  --distractors 200 \
  --top-k 5 \
  --grep-baseline \
  --json
```

It hides exact passcode facts like:

```text
IMPORTANT NEEDLE FACT: The passcode for NEEDLE-0007 is CODE-123456.
```

inside many distractor documents. It reports:

- recall@k by method (`bm25`, `sid`, `hybrid`)
- answer recall@k
- query latency
- build/load time
- storage/index size
- optional linear text-scan baseline

This is the closest local harness to common long-context "needle in a haystack" checks. It is retrieval-only; an end-to-end LLM benchmark would add a generation step and grade whether the LLM outputs the exact passcode from retrieved `input_ids`.

## Notes

The binary postings pack currently preserves exact positions. On small corpora it may be similar in size to JSON postings, but it avoids per-token file explosion and gives predictable one-file locality. Future work can delta-code/compress positions inside `postings.bin` for a better size profile.
