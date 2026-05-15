# ContextFit Architecture

## Design Philosophy

**Stay in token space until the very last moment.**

Traditional RAG pipelines:
```
Text → Tokenize → Embed (floats) → Vector Search → Detokenize → Re-tokenize → LLM
```

ContextFit:
```
Text → Tokenize → Store (integers) → Integer Search/Traverse → input_ids → LLM
```

---

## 1. Core Storage: Pure Token Arrays

### Chunk Structure

```
┌──────────────────────────────────────────────────────────┐
│ Chunk Header (fixed size: 26 bytes)                      │
├──────────────────────────────────────────────────────────┤
│ chunk_id:     u64                                        │
│ level:        u8           (hierarchy level, 0 = finest) │
│ parent_id:    u64          (pointer to parent cluster)   │
│ token_count:  u32                                        │
│ metadata_len: u32          (byte length of JSON payload) │
├──────────────────────────────────────────────────────────┤
│ Token Array                                              │
├──────────────────────────────────────────────────────────┤
│ tokens:    [u32; token_count]                            │
├──────────────────────────────────────────────────────────┤
│ Metadata (variable length)                               │
├──────────────────────────────────────────────────────────┤
│ metadata:  UTF-8 JSON, metadata_len bytes                │
│   e.g. {"source":"...","from":"...","subject":"..."}      │
└──────────────────────────────────────────────────────────┘
```

> **Note:** The format was updated from a fixed 32-byte metadata slot to
> variable-length JSON. This allows rich per-chunk metadata (email headers,
> document paths, dates) without truncation.

### Storage Format

**Option A: Memory-Mapped Flat Files**
- One file per level
- Fixed-size headers enable random access
- Variable-length token arrays with length prefix

**Option B: Chunk Store (recommended)**
- SQLite or RocksDB for metadata
- Separate blob storage for token arrays
- Enables efficient incremental updates

### Compression

1. **Delta encoding** for token IDs (often sequential)
2. **Zstd** compression (excellent for integer sequences)
3. **Expected savings**: ~50% vs raw token arrays

### Chunk Sizing

| Use Case | Chunk Size | Rationale |
|----------|------------|-----------|
| Dense search | 256 tokens | Higher precision |
| General RAG | 512 tokens | Balanced |
| Long-context | 1024 tokens | Fewer chunks |

### Structure-Aware File Ingestion

Before final token encoding, ContextFit chooses better boundaries for common structured text formats:

- **TMD (`.tmd`)**: row-aware chunks with schema/front-matter context. Ledger rows remain atomic and source-verifiable.
- **Markdown (`.md`)**: heading/block-aware chunks. Each chunk carries `chunk_type=markdown_section`, `heading_path`, `section_level`, and `chunk_ordinal`; paragraphs, lists, tables, blockquotes, and code fences stay intact where possible.
- **Plain text (`.txt`)**: paragraph/separator-aware chunks with whole-paragraph overlap.
- **Fallback**: unknown formats use the conservative sliding token window.

The stored representation is unchanged: every chunk is still encoded as token IDs, and retrieval remains token-native.

### Email Ingestion

When ingesting `.eml`, `.md` (email exports), or similar files, ContextFit
extracts structured metadata (From, To, Subject, Date) and prepends it as
a text preamble to every chunk from that email. This ensures contact and
subject information appears in every chunk's token array, making BM25
search sender- and subject-aware out of the box.

Supported email formats:
- Raw MIME (`.eml`)
- Markdown email exports (`# Email: ...` / `**From:** ...` headers)
- Base64-encoded bodies (auto-decoded)
- Multipart HTML/plain (HTML stripped to clean text)

---

## 2. Index Layer: Integer-Only Search

### 2.1 Inverted Index

```
tokenID → RoaringBitmap<chunkID>
```

Or with positions:
```
tokenID → [(chunkID, [positions])]
```

**Implementation:**
- Hash map: `Dict[u32, RoaringBitmap]`
- Use `pyroaring` (Python) or `roaring-rs` (Rust)
- O(1) lookup, efficient intersection for multi-token queries
- Persisted as a compact binary postings pack: `inverted/postings.bin`

**Query flow:**
```python
def search_phrase(tokens: List[int]) -> Set[int]:
    candidates = index[tokens[0]]
    for token in tokens[1:]:
        candidates &= index[token]  # Bitmap intersection
    return candidates
```

**Binary pack layout (`postings.bin`):**

```text
magic:          8 bytes  (CFIDX1)
posting_count:  u64

repeated postings:
  token_id:      u64
  bitmap_len:    u32
  bitmap_bytes:  serialized Roaring bitmap
  pos_chunks:    u32

  repeated position blocks:
    chunk_id:    u64
    pos_count:   u32
    positions:   u32[pos_count]
```

`meta.json` stores corpus-wide fields like `total_chunks`, `chunk_lengths`, and `store_positions`. Older JSON-per-token postings remain load-compatible, but new saves default to binary.

**Lazy loading:** After segment-based ingest, the inverted index is loaded
as a `LazyInvertedIndex` — postings are memory-mapped from `postings.bin`
and loaded on demand. `save()` on a lazy index only refreshes `meta.json`;
it never rewrites `postings.bin` (which would destroy un-loaded postings).

### 2.2 Suffix Array / FM-Index

Treat entire corpus as one token sequence. Build suffix array on token IDs.

**Capabilities:**
- Exact n-gram search in O(log n)
- Substring matching
- Pattern counting

**Libraries:**
- `infini-gram` (designed for token n-grams)
- `divsufsort` + custom wrapper
- Rust: `suffix` crate

### 2.3 BM25 on Tokens

Compute TF-IDF directly on token IDs:

```python
def bm25_score(query_tokens, chunk_tokens, corpus_stats):
    score = 0
    for token in query_tokens:
        tf = chunk_tokens.count(token)
        idf = log((N - df[token] + 0.5) / (df[token] + 0.5))
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len/avglen))
    return score
```

**Key insight:** No text needed. Token IDs are the terms.

**Tuned parameters:** `k1=1.5`, `b=0.90`. Higher `b` (vs textbook 0.75) applies
stronger length normalization — important for email corpora where chunk lengths
vary widely (50-token notifications vs 2,000-token threads).

### 2.4 Query Expansion

Before BM25 search, the query token set is expanded via two layers:

**Layer 1 — Text expansion (`QueryExpander`):**
- Porter stemmer variants (e.g. `payment` → `pay`, `paying`)
- Domain synonym dictionary (e.g. `acme` → `acme corp`, `acme.example`)
- Zero latency; pure lookup table

**Layer 2 — Embedding expansion (`SemanticExpander`):**
- At **index time**: each corpus token string is embedded via
  `text-embedding-3-small` (one API call batch; cached to
  `<kb>/expanders/embedding_expander.npz`)
- At **query time**: cosine similarity lookup against cached vectors;
  zero API calls, ~1ms
- Only **word-initial tokens** (leading-space or standalone ≥4-char alpha)
  are used as expansion pivots — BPE suffix fragments (`rent`, `al`) are
  excluded to avoid noisy neighbors
- Weights use a power curve (`steep=3.0`) so only tight semantic neighbors
  (cosine ≥ ~0.85) contribute meaningfully
- Falls back gracefully if the cache is absent

Building the embedding cache:
```bash
python scripts/build_embedding_expander.py ./my-kb
python scripts/build_embedding_expander.py ./my-kb --dry-run  # cost estimate
```

The embedded token approach is token-native: expansion happens in token ID
space and requires no text decoding at query time.

---

## 3. Graph Layer: Neural-Network-Style Relationships

### 3.1 Graph Structure

```
Nodes:  chunks (or Semantic IDs)
Edges:  weighted by token overlap

Chunk A ──[0.7]── Chunk B
   │                │
   └──[0.3]── Chunk C
```

### 3.2 Similarity Without Embeddings

**Method 1: Jaccard on Token Sets**
```python
similarity = len(A & B) / len(A | B)
```

**Method 2: N-gram Overlap**
```python
def ngram_overlap(a, b, n=3):
    ngrams_a = set(zip(*[a[i:] for i in range(n)]))
    ngrams_b = set(zip(*[b[i:] for i in range(n)]))
    return len(ngrams_a & ngrams_b) / min(len(ngrams_a), len(ngrams_b))
```

**Method 3: MinHash + LSH (scalable)**
```python
# Generate integer signatures
def minhash(tokens, num_hashes=128):
    signatures = []
    for seed in range(num_hashes):
        min_hash = min(hash((seed, t)) for t in tokens)
        signatures.append(min_hash)
    return signatures

# LSH for approximate nearest neighbors
# Band chunks by signature bands → same band = candidate pair
```

### 3.3 Community Detection (GraphRAG-style)

**Algorithm: Leiden / Louvain on Token-Overlap Graph**

1. Build sparse adjacency matrix from chunk similarities
2. Run community detection (integer edge weights)
3. Each community = related chunks
4. Generate community summary (stored as tokens)

**Commonality Mining:**
- Frequent pattern mining on token n-grams across chunks
- High-frequency patterns become "hub tokens"
- Connect semantically related but lexically different chunks

### 3.4 Graph Traversal

```python
def traverse(query_tokens, start_chunks, depth=2):
    visited = set()
    frontier = start_chunks
    
    for _ in range(depth):
        next_frontier = []
        for chunk in frontier:
            if chunk in visited:
                continue
            visited.add(chunk)
            # Follow edges weighted by relevance to query
            neighbors = graph.neighbors(chunk, query_tokens)
            next_frontier.extend(neighbors)
        frontier = next_frontier
    
    return visited
```

---

## 4. Hierarchy Layer: Geo-Map Navigation

### 4.1 Level Structure

```
Level 3 (coarsest):  [Global Summary]
                          │
Level 2:             [Domain Summaries]
                     /       |       \
Level 1:        [Topic Clusters]
                /    |    \
Level 0:    [Raw Chunks]  ← 256-1024 tokens each
```

### 4.2 Building the Hierarchy

**Bottom-Up Construction:**

1. **Level 0**: Chunk raw documents into 512-token pieces
2. **Cluster**: Group Level 0 chunks by token overlap (k-means on MinHash signatures)
3. **Summarize**: For each cluster, generate summary (LLM or extractive)
4. **Store**: Summary as token sequence → Level 1 chunk
5. **Repeat**: Cluster Level 1 → Level 2, etc.

**Summary Options:**
- **Extractive**: Most frequent n-grams, key sentences
- **Generative**: LLM summarization (one-time cost, stored as tokens)

### 4.3 Traversal Algorithm

```python
def hierarchical_search(query_tokens, top_level=3):
    candidates = []
    
    # Start at top
    current_level = top_level
    current_nodes = get_level(top_level)
    
    while current_level >= 0:
        # Score nodes at this level
        scored = [(node, bm25(query_tokens, node.tokens)) for node in current_nodes]
        scored.sort(key=lambda x: -x[1])
        
        # Take top-k
        best = scored[:k]
        
        if current_level == 0:
            candidates = [node for node, _ in best]
            break
        
        # Zoom in: get children
        current_nodes = flatten([node.children for node, _ in best])
        current_level -= 1
    
    return candidates
```

---

## 5. Semantic IDs (SIDs)

### 5.1 Concept

Assign each chunk a short hierarchical token sequence (4-8 special tokens).

```
Chunk about "Python async programming"
→ SID: [SID_TECH, SID_PROG, SID_PYTHON, SID_ASYNC]

Similar chunks share prefixes:
  [SID_TECH, SID_PROG, SID_PYTHON, SID_ASYNC]
  [SID_TECH, SID_PROG, SID_PYTHON, SID_GIL]
  [SID_TECH, SID_PROG, SID_RUST, SID_ASYNC]
```

### 5.2 Construction

**Residual Quantization:**
1. Cluster chunks → assign first SID token
2. Within each cluster, sub-cluster → second SID token
3. Repeat for desired depth

**Training (optional):**
- Train small encoder to predict SID from chunk tokens
- Or use hierarchical k-means on MinHash signatures (no neural net needed)

### 5.3 Generative Retrieval

The LLM can *generate* relevant SIDs:

```
Query: "How does Python handle async?"
→ LLM generates: [SID_TECH, SID_PROG, SID_PYTHON, ...]
→ Prefix match against SID trie
→ Retrieve matching chunks
```

### 5.4 Current Implementation

ContextFit now includes a first-pass `SemanticIDIndex`:

- **SID format:** tuple of reserved integer token IDs
- **Default depth:** 4 SID tokens per chunk
- **Default branching:** 256 buckets per SID level
- **Token range:** starts at `2_000_000_000` so SID tokens never collide with normal tokenizer vocabulary IDs
- **Assignment:** MinHash signature bands are hashed into per-level buckets. This is a discrete, embedding-free approximation of residual quantization.
- **Lookup:** SID prefixes are indexed in a trie-like map: `prefix → chunk IDs`
- **Backoff:** Query full SID first, then progressively shorter prefixes until candidates are found

Retrieval modes:

```bash
contextfit query "how does async retrieval work" --method sid
contextfit query "how does async retrieval work" --method hybrid
```

Hybrid retrieval blends BM25 scores with SID prefix strength, giving us lexical precision plus token-native neighborhood traversal.

### 5.5 SID Generator

ContextFit includes a first-pass `SIDGenerator` for natural-language query → SID-prefix prediction.

It is still fully token-native:

1. Tokenize query
2. Run BM25 over token IDs to get lexical candidate chunks
3. Build a MinHash query signature and query LSH buckets
4. Candidate chunks vote for their own SID prefixes
5. The highest-scoring generated prefixes are resolved through the SID trie

The generator returns `SIDPrediction` records:

```python
SIDPrediction(
    prefix=(2000000123, 2000000341, ...),
    score=2.73,
    depth=3,
    support=4,
    candidate_chunks=(12, 18, 44, 51),
)
```

This is a bridge toward true generative retrieval. The current implementation is a deterministic retriever-generator; later versions can train a small model or fine-tune an LLM to emit these same SID tokens directly.

### 5.6 Learned SID Generator

ContextFit also includes `LearnedSIDGenerator`, a sparse learned query→SID model.

Training:

1. For each chunk, read its token IDs and assigned SID path
2. Each unique chunk token votes for every SID token in that chunk's path
3. Store per-level token→SID-token counts and valid prefix→child transitions

Inference:

1. Tokenize the query
2. Score SID tokens at each level from query-token votes plus priors
3. Beam search through observed valid SID prefixes
4. Resolve generated prefixes through the SID trie

This is not a neural network yet, but it is a genuine learned generator: it learns a discrete associative model from the corpus and emits SID prefixes directly without using BM25 candidates as the primary generator.

CLI:

```bash
contextfit ingest ./documents --train-sid-generator
contextfit query "how does async retrieval work" --method sid
```

---

## 6. End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ User Query: "How does async work in Python?"                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. TOKENIZE                                                     │
│    "How does async work in Python?"                             │
│    → [2437, 1587, 14461, 990, 287, 11361, 30]                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. QUERY EXPANSION                                              │
│    - Synonym/stem expansion (QueryExpander, zero latency)       │
│    - Embedding expansion (SemanticExpander, cosine lookup only) │
│    - → Expanded token set: [2437, 1587, 14461, 990, ...]        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. INDEX SEARCH (all integer operations)                        │
│    - Inverted index lookup on expanded token set                │
│    - BM25 scoring on candidate chunks (k1=1.5, b=0.90)         │
│    - → Candidate chunk IDs: [42, 187, 203, ...]                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. GRAPH TRAVERSAL (integer edge weights)                       │
│    - Start from candidates                                      │
│    - Follow high-weight edges to related chunks                 │
│    - Community-aware expansion                                  │
│    - → Expanded set: [42, 187, 203, 89, 156, ...]              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. HIERARCHY NAVIGATION (integer pointers)                      │
│    - Check parent summaries for broader context                 │
│    - Zoom into siblings for related details                     │
│    - → Final chunk set with context                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. COLLECT TOKEN ARRAYS (no detokenization!)                    │
│    - Gather token arrays for selected chunks                    │
│    - Order by relevance / hierarchy                             │
│    - → [token_array_1, token_array_2, ...]                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. LLM INPUT                                                    │
│    input_ids = system_tokens + context_tokens + query_tokens    │
│    (All integers, no text conversion until generation)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. GENERATE (only now decode to text)                           │
│    model.generate(input_ids) → output tokens → decode → text    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Priorities

### Phase 1: Core Storage + Basic Index
- [ ] Chunk store with token arrays
- [ ] Inverted index with Roaring bitmaps
- [ ] BM25 scoring on tokens
- [ ] Basic retrieval → input_ids output

### Phase 2: Graph Layer
- [ ] MinHash signatures for all chunks
- [ ] LSH bucketing for similarity
- [ ] Sparse adjacency graph
- [ ] Graph traversal

### Phase 3: Hierarchy
- [ ] Clustering (k-means on MinHash)
- [ ] Summary generation (extractive first)
- [ ] Multi-level navigation

### Phase 4: Semantic IDs
- [ ] Residual quantization
- [ ] SID trie
- [ ] Generative retrieval integration

### Phase 5: Optimization
- [ ] Memory mapping
- [ ] Zstd compression
- [ ] Incremental updates
- [ ] Distributed storage

---

## 8. Technology Stack

| Component | Python | Rust |
|-----------|--------|------|
| Tokenizer | `tiktoken`, `sentencepiece` | `tiktoken-rs` |
| Storage | `sqlite3`, `rocksdb` | `rocksdb`, `sled` |
| Bitmaps | `pyroaring` | `roaring` |
| Compression | `zstandard` | `zstd` |
| Suffix Array | `pysuffixarray` | `suffix` |
| Graph | `networkx`, `igraph` | `petgraph` |
| LSH | `datasketch` | custom |

**Recommended:** Start with Python for rapid prototyping, Rust for performance-critical paths (index, search).

---

## 9. Metrics & Goals

| Metric | Traditional RAG | ContextFit Target |
|--------|-----------------|-------------------|
| Storage size | 1× (text + embeddings) | 0.3-0.5× |
| Indexing speed | Minutes | Seconds |
| Query latency | 50-200ms | <10ms |
| Token efficiency | 100% (full chunks) | 3-11% (TERAG-style) |
| Multi-hop | Expensive | Native (graph) |
