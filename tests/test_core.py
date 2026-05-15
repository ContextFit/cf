"""Tests for core components."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from contextfit.core.chunk import Chunk, ChunkStore
from contextfit.core.tokenizer import Tokenizer


class TestTokenizer:
    """Test tokenizer functionality."""
    
    def test_load_tiktoken(self):
        tok = Tokenizer.load("cl100k_base")
        assert tok.vocab_size > 0
        assert "tiktoken" in tok.name
    
    def test_encode_decode(self):
        tok = Tokenizer.load("cl100k_base")
        text = "Hello, world!"
        
        tokens = tok.encode(text)
        assert isinstance(tokens, np.ndarray)
        assert tokens.dtype == np.uint32
        assert len(tokens) > 0
        
        decoded = tok.decode(tokens)
        assert decoded == text
    
    def test_chunk_text(self):
        tok = Tokenizer.load("cl100k_base")
        text = "This is a test. " * 100
        
        chunks = tok.chunk_text(text, chunk_size=50, overlap=10)
        assert len(chunks) > 1
        
        for chunk in chunks:
            assert len(chunk) <= 50


class TestChunk:
    """Test chunk serialization."""
    
    def test_roundtrip(self):
        tokens = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        chunk = Chunk(
            chunk_id=42,
            tokens=tokens,
            level=1,
            parent_id=10,
            metadata={"foo": "bar"},
        )
        
        data = chunk.to_bytes()
        restored = Chunk.from_bytes(data)
        
        assert restored.chunk_id == chunk.chunk_id
        assert restored.level == chunk.level
        assert restored.parent_id == chunk.parent_id
        assert np.array_equal(restored.tokens, chunk.tokens)


class TestChunkStore:
    """Test chunk storage."""
    
    def test_add_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChunkStore(Path(tmpdir))
            
            tokens = np.array([10, 20, 30], dtype=np.uint32)
            chunk = store.add(tokens, level=0)
            
            assert chunk.chunk_id == 0
            assert len(store) == 1
            
            retrieved = store.get(0)
            assert retrieved is not None
            assert np.array_equal(retrieved.tokens, tokens)
    
    def test_compression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ChunkStore(Path(tmpdir), compress=True)
            
            # Add a chunk with repeated patterns (compresses well)
            tokens = np.array([1, 2, 3] * 100, dtype=np.uint32)
            store.add(tokens)
            store.flush(force=True)
            
            stats = store.stats()
            assert stats["compressed"] is True
            assert stats["chunk_count"] == 1

    def test_flush_persists_index_atomically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            store = ChunkStore(path, autosave_every=1000)

            store.add(np.array([10, 20, 30], dtype=np.uint32))
            assert not (path / "index.json").exists()

            store.flush(force=True)

            index_path = path / "index.json"
            assert index_path.exists()
            assert not (path / "index.json.tmp").exists()

            data = json.loads(index_path.read_text())
            assert data["next_id"] == 1
            assert data["index"]["0"][0] == 0

    def test_autosave_writes_periodically_without_per_chunk_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            store = ChunkStore(path, autosave_every=2)

            store.add(np.array([1, 2, 3], dtype=np.uint32))
            assert not (path / "index.json").exists()

            store.add(np.array([4, 5, 6], dtype=np.uint32))
            assert (path / "index.json").exists()

            reloaded = ChunkStore(path, autosave_every=2)
            assert len(reloaded) == 2
