"""Tests for Semantic IDs."""

import tempfile
from pathlib import Path

import numpy as np

from contextfit import RetrievalEngine
from contextfit.graph.similarity import MinHasher
from contextfit.sid.generator import SIDGenerator
from contextfit.sid.semantic import SID_TOKEN_BASE, SemanticIDIndex


def test_sid_assignment_is_hierarchical_and_indexed():
    hasher = MinHasher(num_perm=32, ngram_size=2)
    index = SemanticIDIndex(depth=4, branching_factor=16)

    tokens = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint32)
    sid = index.assign_tokens(42, tokens, hasher)

    assert sid.chunk_id == 42
    assert sid.depth == 4
    assert all(token >= SID_TOKEN_BASE for token in sid.tokens)
    assert index.get(42) == sid

    # Every prefix should resolve back to the chunk.
    for depth in range(1, sid.depth + 1):
        assert index.chunks_for_prefix(sid.prefix(depth)) == [42]


def test_sid_prefix_backoff_finds_related_chunks():
    index = SemanticIDIndex(depth=4, branching_factor=16)

    # Manually add two chunks sharing a prefix and one unrelated chunk.
    sid_a = (SID_TOKEN_BASE + 1, SID_TOKEN_BASE + 20, SID_TOKEN_BASE + 40, SID_TOKEN_BASE + 60)
    sid_b = (SID_TOKEN_BASE + 1, SID_TOKEN_BASE + 20, SID_TOKEN_BASE + 41, SID_TOKEN_BASE + 61)
    sid_c = (SID_TOKEN_BASE + 2, SID_TOKEN_BASE + 21, SID_TOKEN_BASE + 42, SID_TOKEN_BASE + 62)

    from contextfit.sid.semantic import SemanticID

    index.add(SemanticID(1, sid_a))
    index.add(SemanticID(2, sid_b))
    index.add(SemanticID(3, sid_c))

    results = index.query_sid(sid_a, min_prefix_depth=2)
    assert results == [(1, 1.0)]

    # Unknown full SID backs off to depth 2 and finds both prefix siblings.
    query_prefix = (sid_a[0], sid_a[1], SID_TOKEN_BASE + 99, SID_TOKEN_BASE + 100)
    backed_off = index.query_sid(query_prefix, min_prefix_depth=2)
    assert backed_off == [(1, 0.5), (2, 0.5)]


def test_sid_save_load_roundtrip():
    hasher = MinHasher(num_perm=32, ngram_size=2)
    index = SemanticIDIndex(depth=3, branching_factor=8)
    index.assign_tokens(7, np.array([10, 20, 30, 40], dtype=np.uint32), hasher)

    with tempfile.TemporaryDirectory() as tmpdir:
        index.save(tmpdir)
        loaded = SemanticIDIndex.load(tmpdir)

    assert loaded.stats() == index.stats()
    assert loaded.get(7) == index.get(7)


def test_retrieval_engine_indexes_and_returns_sids():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = RetrievalEngine.create(Path(tmpdir) / "kb")
        engine.ingest_text("Python async await event loop coroutine future task.", chunk_size=64)
        engine.ingest_text("Rust async future executor tokio task poll wake.", chunk_size=64)

        assert engine.sid_index is not None
        assert len(engine.sid_index) == 2

        result = engine.query("async future task", method="hybrid", top_k=2)
        assert result.chunks
        assert result.semantic_ids is not None
        assert len(result.semantic_ids) == len(result.chunks)

        engine.save(Path(tmpdir) / "kb")
        loaded = RetrievalEngine.load(Path(tmpdir) / "kb")
        assert loaded.sid_index is not None
        assert len(loaded.sid_index) == 2


def test_sid_generator_predicts_prefixes_and_retrieves_chunks():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = RetrievalEngine.create(Path(tmpdir) / "kb")
        python_chunks = engine.ingest_text(
            "Python async await event loop coroutine future task.",
            chunk_size=64,
        )
        engine.ingest_text(
            "Rust ownership borrowing lifetimes compiler memory safety.",
            chunk_size=64,
        )

        assert engine.sid_index is not None
        generator = SIDGenerator(engine.sid_index, engine.bm25, engine.hasher, engine.lsh)
        query_tokens = engine.tokenizer.encode("python async coroutine task")

        predictions = generator.predict(query_tokens, top_k=3)
        assert predictions
        assert predictions[0].depth >= 1
        assert predictions[0].score > 0

        retrieved = generator.retrieve(query_tokens, top_k=2)
        assert retrieved
        assert retrieved[0][0] == python_chunks[0].chunk_id


def test_engine_sid_method_uses_generator_predictions():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = RetrievalEngine.create(Path(tmpdir) / "kb")
        engine.ingest_text("Python async await event loop coroutine future task.", chunk_size=64)
        engine.ingest_text(
            "Rust ownership borrowing lifetimes compiler memory safety.",
            chunk_size=64,
        )

        result = engine.query("python async coroutine task", method="sid", top_k=1)

        assert result.chunks
        assert result.semantic_ids is not None
        assert result.sid_predictions is not None
        assert result.sid_predictions[0].score > 0


def test_learned_sid_generator_trains_predicts_and_persists():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = RetrievalEngine.create(Path(tmpdir) / "kb")
        python_chunks = engine.ingest_text(
            "Python async await event loop coroutine future task.",
            chunk_size=64,
        )
        engine.ingest_text(
            "Rust ownership borrowing lifetimes compiler memory safety.",
            chunk_size=64,
        )

        learned = engine.train_learned_sid_generator()
        assert learned.trained_chunks == 2

        query_tokens = engine.tokenizer.encode("python async coroutine task")
        predictions = learned.predict(query_tokens, top_k=3)
        assert predictions
        assert predictions[0].candidate_chunks

        retrieved = learned.retrieve(query_tokens, top_k=2)
        assert retrieved
        assert retrieved[0][0] == python_chunks[0].chunk_id

        engine.save(Path(tmpdir) / "kb")
        loaded = RetrievalEngine.load(Path(tmpdir) / "kb")
        assert loaded.learned_sid_generator is not None
        assert loaded.learned_sid_generator.trained_chunks == 2


def test_engine_sid_method_prefers_learned_generator_when_trained():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = RetrievalEngine.create(Path(tmpdir) / "kb")
        python_chunks = engine.ingest_text(
            "Python async await event loop coroutine future task.",
            chunk_size=64,
        )
        engine.ingest_text(
            "Rust ownership borrowing lifetimes compiler memory safety.",
            chunk_size=64,
        )
        engine.train_learned_sid_generator()

        result = engine.query("python async coroutine task", method="sid", top_k=1)
        assert result.chunks
        assert result.chunks[0].chunk_id == python_chunks[0].chunk_id
        assert result.sid_predictions is not None
