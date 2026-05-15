"""
Chunk storage: Pure token arrays with compact metadata.

A Chunk is the fundamental unit - a sequence of token IDs with metadata.
ChunkStore manages persistence with optional compression.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import zstandard as zstd

# Header format: chunk_id (u64), level (u8), parent_id (u64), token_count (u32), metadata_len (u32)
# Variable-length metadata follows after the fixed token payload.
HEADER_FORMAT = "<QBQII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


@dataclass
class Chunk:
    """A token sequence with metadata."""
    
    chunk_id: int
    tokens: np.ndarray  # dtype=uint32
    level: int = 0
    parent_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    
    @property
    def token_count(self) -> int:
        return len(self.tokens)
    
    def to_bytes(self) -> bytes:
        """Serialize chunk to bytes."""
        token_bytes = self.tokens.astype(np.uint32).tobytes()
        metadata_bytes = json.dumps(self.metadata, separators=(",", ":")).encode("utf-8")
        header = struct.pack(
            HEADER_FORMAT,
            self.chunk_id,
            self.level,
            self.parent_id or 0,
            self.token_count,
            len(metadata_bytes),
        )
        return header + token_bytes + metadata_bytes
    
    @classmethod
    def from_bytes(cls, data: bytes) -> Chunk:
        """Deserialize chunk from bytes."""
        chunk_id, level, parent_id, token_count, metadata_len = struct.unpack(
            HEADER_FORMAT, data[:HEADER_SIZE]
        )
        # Parse tokens
        token_start = HEADER_SIZE
        token_end = token_start + token_count * 4
        tokens = np.frombuffer(data[token_start:token_end], dtype=np.uint32)
        # Parse variable-length metadata
        metadata_bytes = data[token_end:token_end + metadata_len]
        try:
            metadata = json.loads(metadata_bytes.decode("utf-8")) if metadata_bytes else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            metadata = {}

        return cls(
            chunk_id=chunk_id,
            tokens=tokens,
            level=level,
            parent_id=parent_id if parent_id != 0 else None,
            metadata=metadata,
        )
    
    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id}, level={self.level}, tokens={self.token_count})"


class ChunkStore:
    """
    Persistent storage for chunks.
    
    Uses memory-mapped files with optional Zstd compression.
    """
    
    def __init__(
        self,
        path: Path | str,
        compress: bool = True,
        compression_level: int = 3,
        autosave_every: int = 100,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.compress = compress
        self.compressor = zstd.ZstdCompressor(level=compression_level) if compress else None
        self.decompressor = zstd.ZstdDecompressor() if compress else None
        self.autosave_every = max(1, autosave_every)
        
        # In-memory index: chunk_id -> file offset (for mmap)
        self._index: dict[int, tuple[int, int]] = {}  # id -> (offset, length)
        self._next_id = 0
        self._dirty = False
        self._dirty_chunks = 0
        
        # Data file
        self._data_file = self.path / "chunks.bin"
        self._index_file = self.path / "index.json"
        self._index_tmp_file = self.path / "index.json.tmp"
        
        self._load_index()
    
    def _load_index(self) -> None:
        """Load index from disk if exists."""
        if self._index_file.exists():
            with open(self._index_file) as f:
                data = json.load(f)
                self._index = {int(k): tuple(v) for k, v in data["index"].items()}
                self._next_id = data.get("next_id", 0)
    
    def _save_index(self) -> None:
        """Persist index to disk atomically."""
        payload = {"index": self._index, "next_id": self._next_id}
        with open(self._index_tmp_file, "w") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(self._index_tmp_file, self._index_file)
        self._dirty = False
        self._dirty_chunks = 0

    def flush(self, force: bool = False) -> None:
        """Persist pending index changes when needed."""
        if force:
            if self._dirty or not self._index_file.exists():
                self._save_index()
            return
        if self._dirty and self._dirty_chunks >= self.autosave_every:
            self._save_index()
    
    def add(self, tokens: np.ndarray, level: int = 0, parent_id: int | None = None, metadata: dict | None = None) -> Chunk:
        """Add a new chunk and return it."""
        chunk = Chunk(
            chunk_id=self._next_id,
            tokens=np.asarray(tokens, dtype=np.uint32),
            level=level,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self._next_id += 1
        
        # Serialize
        raw_bytes = chunk.to_bytes()
        if self.compress:
            data = self.compressor.compress(raw_bytes)
        else:
            data = raw_bytes
        
        # Append to file
        with open(self._data_file, "ab") as f:
            offset = f.tell()
            length = len(data)
            # Write length prefix for variable-size records
            f.write(struct.pack("<I", length))
            f.write(data)
        
        self._index[chunk.chunk_id] = (offset, length + 4)
        self._dirty = True
        self._dirty_chunks += 1
        self.flush()
        
        return chunk

    def close(self) -> None:
        """Flush pending state to disk."""
        self.flush(force=True)
        
    
    def get(self, chunk_id: int) -> Chunk | None:
        """Retrieve a chunk by ID."""
        if chunk_id not in self._index:
            return None
        
        offset, total_length = self._index[chunk_id]
        
        with open(self._data_file, "rb") as f:
            f.seek(offset)
            length = struct.unpack("<I", f.read(4))[0]
            data = f.read(length)
        
        if self.compress:
            raw_bytes = self.decompressor.decompress(data)
        else:
            raw_bytes = data
        
        return Chunk.from_bytes(raw_bytes)
    
    def iter_chunks(self, level: int | None = None) -> Iterator[Chunk]:
        """Iterate all chunks, optionally filtering by level."""
        for chunk_id in sorted(self._index.keys()):
            chunk = self.get(chunk_id)
            if chunk is not None:
                if level is None or chunk.level == level:
                    yield chunk
    
    def __len__(self) -> int:
        return len(self._index)
    
    def __contains__(self, chunk_id: int) -> bool:
        return chunk_id in self._index
    
    def stats(self) -> dict:
        """Return storage statistics."""
        data_size = self._data_file.stat().st_size if self._data_file.exists() else 0
        return {
            "chunk_count": len(self._index),
            "data_size_bytes": data_size,
            "compressed": self.compress,
            "autosave_every": self.autosave_every,
            "dirty_chunks": self._dirty_chunks,
        }

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
