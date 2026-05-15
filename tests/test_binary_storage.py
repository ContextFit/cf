"""Tests for compact binary storage formats."""

import tempfile
from pathlib import Path

import numpy as np

from contextfit import InvertedIndex, RetrievalEngine


def test_inverted_index_binary_roundtrip_preserves_search_and_positions():
    index = InvertedIndex(store_positions=True)
    index.add_chunk(0, np.array([10, 20, 30, 20], dtype=np.uint32))
    index.add_chunk(1, np.array([20, 30, 40], dtype=np.uint32))

    with tempfile.TemporaryDirectory() as tmpdir:
        index.save_binary(tmpdir)
        assert (Path(tmpdir) / "postings.bin").exists()
        assert not (Path(tmpdir) / "postings").exists()

        loaded = InvertedIndex.load(tmpdir)

    assert loaded.search([20]) == {0, 1}
    assert loaded.search([10, 30], mode="and") == {0}
    assert loaded.search_phrase([20, 30]) == {0, 1}
    assert loaded.get_tf(20, 0) == 2
    assert loaded.chunk_length(1) == 3
    assert loaded.stats()["vocab_size"] == 4


def test_inverted_index_legacy_json_still_loads():
    index = InvertedIndex(store_positions=True)
    index.add_chunk(0, [1, 2, 3])

    with tempfile.TemporaryDirectory() as tmpdir:
        index.save_json(tmpdir)
        assert (Path(tmpdir) / "postings" / "1.json").exists()
        loaded = InvertedIndex.load(tmpdir)

    assert loaded.search_phrase([1, 2]) == {0}


def test_retrieval_engine_saves_binary_inverted_pack_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = Path(tmpdir) / "kb"
        engine = RetrievalEngine.create(kb)
        engine.ingest_text("Python async await event loop coroutine future task.", chunk_size=64)
        engine.train_learned_sid_generator()
        engine.save(kb)

        assert (kb / "inverted" / "postings.bin").exists()
        assert not (kb / "inverted" / "postings").exists()

        loaded = RetrievalEngine.load(kb)
        result = loaded.query("python async task", method="sid", top_k=1)

    assert result.chunks
    assert result.semantic_ids is not None
    assert result.sid_predictions is not None
