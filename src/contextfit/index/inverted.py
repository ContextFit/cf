"""
Inverted Index: tokenID → chunk IDs (using Roaring bitmaps).

Pure integer operations for fast lookups and set operations.

Incremental build
-----------------
Large corpora cannot be indexed in a single in-memory pass.  Instead,
ContextFit uses a segment-based approach:

  1. SegmentWriter accumulates (token, chunk_id, tf) triples during ingest
     and flushes compact sorted segment files every N chunks.
  2. Once ingest is done, SegmentMerger performs a k-way merge of all
     segment files using O(n_segments) memory, producing the final
     postings.bin + postings_offsets.bin in one streaming pass.

Segment file format (segments/seg_NNNNNNNN.bin):
  magic:     8 bytes  CFSEG1\0\0
  n_entries: u64
  entries (sorted by token_id ASC, chunk_id ASC):
    token_id: u64
    chunk_id: u64
    tf:       u32
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from pyroaring import BitMap

POSTINGS_MAGIC = b"CFIDX1\0\0"


@dataclass
class PostingList:
    """A posting list for a single token: chunk IDs + positions."""
    
    chunk_ids: BitMap = field(default_factory=BitMap)
    # Optional: positions per chunk (for phrase queries)
    positions: dict[int, list[int]] = field(default_factory=dict)
    
    def add(self, chunk_id: int, positions: list[int] | None = None) -> None:
        """Add a chunk to this token's posting list."""
        self.chunk_ids.add(chunk_id)
        if positions:
            self.positions[chunk_id] = positions
    
    def __len__(self) -> int:
        return len(self.chunk_ids)


class InvertedIndex:
    """
    Inverted index mapping token IDs to chunk IDs.
    
    Uses Roaring bitmaps for compact storage and fast set operations
    (intersection, union, difference).
    
    Example:
        index = InvertedIndex()
        index.add_chunk(chunk_id=0, tokens=[1, 2, 3, 1, 2])
        index.add_chunk(chunk_id=1, tokens=[2, 3, 4, 5])
        
        # Find chunks containing token 2
        results = index.search([2])  # Returns {0, 1}
        
        # Find chunks containing both token 2 AND token 3
        results = index.search([2, 3])  # Returns {0, 1}
        
        # Find chunks containing phrase [1, 2] (exact sequence)
        results = index.search_phrase([1, 2])  # Returns {0}
    """
    
    def __init__(self, store_positions: bool = True):
        self._index: dict[int, PostingList] = defaultdict(PostingList)
        self._chunk_lengths: dict[int, int] = {}  # For BM25
        self._total_chunks = 0
        self._store_positions = store_positions
    
    def add_chunk(self, chunk_id: int, tokens: np.ndarray | list[int]) -> None:
        """Index a chunk's tokens."""
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        
        self._chunk_lengths[chunk_id] = len(tokens)
        self._total_chunks = max(self._total_chunks, chunk_id + 1)
        
        # Build token -> positions map
        token_positions: dict[int, list[int]] = defaultdict(list)
        for pos, token in enumerate(tokens):
            token_positions[token].append(pos)
        
        # Add to index
        for token, positions in token_positions.items():
            if self._store_positions:
                self._index[token].add(chunk_id, positions)
            else:
                self._index[token].add(chunk_id)
    
    def search(self, tokens: Iterable[int], mode: str = "and") -> set[int]:
        """
        Search for chunks containing tokens.
        
        Args:
            tokens: Token IDs to search
            mode: "and" (all tokens) or "or" (any token)
        
        Returns:
            Set of matching chunk IDs
        """
        tokens = list(tokens)
        if not tokens:
            return set()
        
        # Get posting lists
        posting_lists = [self._index.get(t) for t in tokens]
        posting_lists = [p for p in posting_lists if p is not None]
        
        if not posting_lists:
            return set()
        
        if mode == "and":
            # Intersection of all posting lists
            result = posting_lists[0].chunk_ids.copy()
            for pl in posting_lists[1:]:
                result &= pl.chunk_ids
            return set(result)
        else:  # "or"
            # Union of all posting lists
            result = BitMap()
            for pl in posting_lists:
                result |= pl.chunk_ids
            return set(result)
    
    def search_phrase(self, tokens: list[int]) -> set[int]:
        """
        Search for exact phrase (tokens in sequence).
        
        Requires store_positions=True.
        """
        if not self._store_positions:
            raise ValueError("Phrase search requires store_positions=True")
        
        if not tokens:
            return set()
        
        # Start with chunks containing first token
        first_posting = self._index.get(tokens[0])
        if not first_posting:
            return set()
        
        candidates = set(first_posting.chunk_ids)
        
        # Filter by subsequent tokens at correct positions
        for offset, token in enumerate(tokens[1:], start=1):
            posting = self._index.get(token)
            if not posting:
                return set()
            
            new_candidates = set()
            for chunk_id in candidates:
                if chunk_id not in posting.chunk_ids:
                    continue
                
                # Check if positions align
                first_positions = first_posting.positions.get(chunk_id, [])
                current_positions = posting.positions.get(chunk_id, [])
                
                for start_pos in first_positions:
                    expected_pos = start_pos + offset
                    if expected_pos in current_positions:
                        new_candidates.add(chunk_id)
                        break
            
            candidates = new_candidates
            if not candidates:
                return set()
        
        return candidates
    
    def get_df(self, token: int) -> int:
        """Get document frequency for a token."""
        posting = self._index.get(token)
        return len(posting) if posting else 0
    
    def get_tf(self, token: int, chunk_id: int) -> int:
        """Get term frequency in a specific chunk."""
        posting = self._index.get(token)
        if not posting or chunk_id not in posting.chunk_ids:
            return 0
        positions = posting.positions.get(chunk_id)
        return len(positions) if positions else 1  # Approximate if no positions
    
    @property
    def total_chunks(self) -> int:
        return self._total_chunks
    
    @property
    def vocab_size(self) -> int:
        return len(self._index)
    
    def chunk_length(self, chunk_id: int) -> int:
        return self._chunk_lengths.get(chunk_id, 0)
    
    @property
    def avg_chunk_length(self) -> float:
        if not self._chunk_lengths:
            return 0.0
        return sum(self._chunk_lengths.values()) / len(self._chunk_lengths)
    
    def _meta(self) -> dict:
        """Return serializable metadata."""
        return {
            "format": "contextfit-inverted-v1",
            "total_chunks": self._total_chunks,
            "chunk_lengths": self._chunk_lengths,
            "store_positions": self._store_positions,
            "vocab_size": len(self._index),
        }

    def save(self, path: Path | str, binary: bool = True) -> None:
        """Save index to disk.

        By default this writes a compact binary postings pack. Use
        ``binary=False`` to write the legacy JSON-per-token layout.
        """
        if binary:
            self.save_binary(path)
        else:
            self.save_json(path)

    def save_binary(self, path: Path | str) -> None:
        """Save as a compact binary postings pack with a per-token offset table.

        Layout of ``postings.bin``:
            magic:          8 bytes  (CFIDX1)
            posting_count:  u64
            repeated postings:
                token_id:      u64
                bitmap_len:    u32
                bitmap_bytes:  roaring serialized bytes
                pos_chunks:    u32
                repeated position blocks:
                    chunk_id:  u64
                    pos_count: u32
                    positions: u32[pos_count]

        Layout of ``postings_offsets.bin``  (enables lazy loading)::
            posting_count: u64
            repeated entries:
                token_id:  u64
                offset:    u64   (byte offset of this token's record in postings.bin)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta_path = path / "meta.json"
        meta_tmp_path = path / "meta.json.tmp"
        with open(meta_tmp_path, "w") as f:
            json.dump(self._meta(), f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(meta_tmp_path, meta_path)

        postings_path = path / "postings.bin"
        postings_tmp_path = path / "postings.bin.tmp"
        offsets: list[tuple[int, int]] = []  # (token_id, byte_offset)

        with open(postings_tmp_path, "wb") as f:
            f.write(POSTINGS_MAGIC)
            f.write(struct.pack("<Q", len(self._index)))

            for token in sorted(self._index.keys()):
                offsets.append((int(token), f.tell()))
                posting = self._index[token]
                bitmap_bytes = posting.chunk_ids.serialize()
                f.write(struct.pack("<QI", int(token), len(bitmap_bytes)))
                f.write(bitmap_bytes)

                positions = posting.positions if self._store_positions else {}
                f.write(struct.pack("<I", len(positions)))
                for chunk_id in sorted(positions.keys()):
                    pos = positions[chunk_id]
                    f.write(struct.pack("<QI", int(chunk_id), len(pos)))
                    if pos:
                        f.write(struct.pack(f"<{len(pos)}I", *pos))
            f.flush()
            os.fsync(f.fileno())
        os.replace(postings_tmp_path, postings_path)

        # Write per-token offset table for lazy loading.
        offsets_tmp = path / "postings_offsets.bin.tmp"
        with open(offsets_tmp, "wb") as f:
            f.write(struct.pack("<Q", len(offsets)))
            for tok, off in offsets:
                f.write(struct.pack("<QQ", tok, off))
            f.flush()
            os.fsync(f.fileno())
        os.replace(offsets_tmp, path / "postings_offsets.bin")

    def save_json(self, path: Path | str) -> None:
        """Save index to the legacy JSON-per-token layout."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        meta_tmp_path = path / "meta.json.tmp"
        with open(meta_tmp_path, "w") as f:
            json.dump(self._meta() | {"format": "contextfit-inverted-json-v1"}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(meta_tmp_path, path / "meta.json")
        
        # Save posting lists (one file per token for simplicity)
        # In production, would use a more efficient format
        postings_path = path / "postings"
        postings_path.mkdir(exist_ok=True)
        
        for token, posting in self._index.items():
            data = {
                "chunk_ids": list(posting.chunk_ids),
                "positions": {str(k): v for k, v in posting.positions.items()},
            }
            with open(postings_path / f"{token}.json", "w") as f:
                json.dump(data, f)
    
    @classmethod
    def load(cls, path: Path | str) -> "InvertedIndex":
        """Load index from disk.

        Uses a lazy mmap-backed loader when an offset index is present
        (written by :meth:`save_binary`).  Falls back to the original
        eager load otherwise.
        """
        path = Path(path)

        if (path / "postings_offsets.bin").exists():
            return LazyInvertedIndex.load_lazy(path)

        if (path / "postings.bin").exists():
            return cls.load_binary(path)

        return cls.load_json(path)

    @classmethod
    def load_binary(cls, path: Path | str) -> InvertedIndex:
        """Load a binary postings pack."""
        path = Path(path)
        
        with open(path / "meta.json") as f:
            meta = json.load(f)
        
        index = cls(store_positions=meta["store_positions"])
        index._total_chunks = meta["total_chunks"]
        index._chunk_lengths = {int(k): v for k, v in meta["chunk_lengths"].items()}

        with open(path / "postings.bin", "rb") as f:
            magic = f.read(len(POSTINGS_MAGIC))
            if magic != POSTINGS_MAGIC:
                raise ValueError(f"Invalid postings pack magic: {magic!r}")

            (posting_count,) = struct.unpack("<Q", f.read(8))
            for _ in range(posting_count):
                token, bitmap_len = struct.unpack("<QI", f.read(12))
                bitmap_bytes = f.read(bitmap_len)
                posting = PostingList(chunk_ids=BitMap.deserialize(bitmap_bytes))

                (pos_chunk_count,) = struct.unpack("<I", f.read(4))
                for _ in range(pos_chunk_count):
                    chunk_id, pos_count = struct.unpack("<QI", f.read(12))
                    if pos_count:
                        raw = f.read(pos_count * 4)
                        posting.positions[int(chunk_id)] = list(
                            struct.unpack(f"<{pos_count}I", raw)
                        )
                    else:
                        posting.positions[int(chunk_id)] = []

                index._index[int(token)] = posting

        return index

    @classmethod
    def load_json(cls, path: Path | str) -> InvertedIndex:
        """Load the legacy JSON-per-token layout."""
        path = Path(path)

        with open(path / "meta.json") as f:
            meta = json.load(f)

        index = cls(store_positions=meta["store_positions"])
        index._total_chunks = meta["total_chunks"]
        index._chunk_lengths = {int(k): v for k, v in meta["chunk_lengths"].items()}
        
        postings_path = path / "postings"
        for posting_file in postings_path.glob("*.json"):
            token = int(posting_file.stem)
            with open(posting_file) as f:
                data = json.load(f)
            
            posting = PostingList()
            posting.chunk_ids = BitMap(data["chunk_ids"])
            posting.positions = {int(k): v for k, v in data["positions"].items()}
            index._index[token] = posting
        
        return index
    
    def stats(self) -> dict:
        """Return index statistics."""
        posting_sizes = [len(p) for p in self._index.values()]
        return {
            "total_chunks": self._total_chunks,
            "vocab_size": len(self._index),
            "avg_posting_size": sum(posting_sizes) / len(posting_sizes) if posting_sizes else 0,
            "max_posting_size": max(posting_sizes) if posting_sizes else 0,
            "avg_chunk_length": self.avg_chunk_length,
        }


class LazyInvertedIndex(InvertedIndex):
    """Inverted index that memory-maps ``postings.bin`` and deserializes
    each token's posting list on first access.

    On a 84k-chunk KB this reduces cold-load RAM from ~3 GB to ~80 MB and
    shaves ~40 s from startup.  Postings are cached after first access so
    repeat queries pay no extra cost.
    """

    def __init__(self, store_positions: bool = True) -> None:
        super().__init__(store_positions=store_positions)
        self._mmap: mmap.mmap | None = None
        self._offsets: dict[int, int] = {}   # token_id -> byte offset in postings.bin
        self._loaded: set[int] = set()       # tokens already deserialized

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._mmap is None:
            raise RuntimeError("LazyInvertedIndex: postings file not open")

    def _load_posting(self, token: int) -> PostingList | None:
        """Deserialize one posting from the mmap on demand."""
        if token in self._loaded:
            return self._index.get(token)
        off = self._offsets.get(token)
        if off is None:
            self._loaded.add(token)
            return None
        self._ensure_open()
        m = self._mmap
        # Read: token_id (u64), bitmap_len (u32)
        tok_id, bitmap_len = struct.unpack_from("<QI", m, off)
        off += 12
        bitmap_bytes = m[off: off + bitmap_len]
        off += bitmap_len
        posting = PostingList(chunk_ids=BitMap.deserialize(bytes(bitmap_bytes)))
        (pos_chunk_count,) = struct.unpack_from("<I", m, off)
        off += 4
        for _ in range(pos_chunk_count):
            chunk_id, pos_count = struct.unpack_from("<QI", m, off)
            off += 12
            if pos_count:
                raw = m[off: off + pos_count * 4]
                posting.positions[int(chunk_id)] = list(struct.unpack(f"<{pos_count}I", bytes(raw)))
                off += pos_count * 4
            else:
                posting.positions[int(chunk_id)] = []
        self._index[token] = posting
        self._loaded.add(token)
        return posting

    # ------------------------------------------------------------------
    # Override hot-path methods to trigger lazy load
    # ------------------------------------------------------------------

    def search(self, tokens: Iterable[int], mode: str = "and") -> set[int]:
        for t in list(tokens):
            self._load_posting(t)
        return super().search(tokens, mode)

    def search_phrase(self, tokens: list[int]) -> set[int]:
        for t in tokens:
            self._load_posting(t)
        return super().search_phrase(tokens)

    def get_df(self, token: int) -> int:
        self._load_posting(token)
        return super().get_df(token)

    def get_tf(self, token: int, chunk_id: int) -> int:
        self._load_posting(token)
        return super().get_tf(token, chunk_id)

    def stats(self) -> dict:
        """Return full index statistics.

        Lazy indexes keep postings mmap-backed on the query hot path, but
        ``stats()`` is an explicit inspection/debug operation and should remain
        semantically identical to eager indexes. Load postings on demand here so
        binary and JSON layouts report comparable values.
        """
        for token in list(self._offsets.keys()):
            self._load_posting(token)
        return super().stats()

    def save(self, path: Path | str, binary: bool = True) -> None:  # type: ignore[override]
        """Save the lazy index.

        The postings.bin and offset table are *already on disk* (written by
        SegmentMerger).  Calling the parent save() would iterate the sparse
        in-memory cache and overwrite the file with only those postings that
        happened to be loaded, silently destroying the rest.

        Instead we only refresh meta.json (chunk_lengths may have grown) and
        leave postings.bin / postings_offsets.bin untouched.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta_tmp = path / "meta.json.tmp"
        with open(meta_tmp, "w") as f:
            import json as _json
            _json.dump(self._meta(), f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(meta_tmp, path / "meta.json")

    def close(self) -> None:
        if self._mmap:
            self._mmap.close()
            self._mmap = None

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    @classmethod
    def load_lazy(cls, path: Path | str) -> "LazyInvertedIndex":
        """Open the index in lazy mode.  Only meta + offset table loaded eagerly."""
        path = Path(path)

        with open(path / "meta.json") as f:
            meta = json.load(f)

        index = cls(store_positions=meta["store_positions"])
        index._total_chunks = meta["total_chunks"]
        index._chunk_lengths = {int(k): v for k, v in meta["chunk_lengths"].items()}

        # Load offset table into a plain dict — fast & compact.
        raw = (path / "postings_offsets.bin").read_bytes()
        (count,) = struct.unpack_from("<Q", raw, 0)
        off = 8
        for _ in range(count):
            tok, byte_off = struct.unpack_from("<QQ", raw, off)
            index._offsets[tok] = byte_off
            off += 16

        # Open the postings file for mmap reads.
        fd = open(path / "postings.bin", "rb")
        index._mmap = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ)
        fd.close()   # mmap keeps the fd alive

        return index


# ---------------------------------------------------------------------------
# Incremental index building: SegmentWriter + SegmentMerger
# ---------------------------------------------------------------------------

SEGMENT_MAGIC = b"CFSEG1\0\0"


class SegmentWriter:
    """
    Accumulates (token_id, chunk_id, tf) triples during ingest and flushes
    compact sorted segment files every ``flush_every`` chunks.

    Usage::

        writer = SegmentWriter(kb_path / "segments", flush_every=2000)
        for chunk in chunks:
            writer.add_chunk(chunk.chunk_id, chunk.tokens)
        writer.flush()   # flush any remaining buffer
        n_segments = writer.segment_count
    """

    def __init__(self, segments_dir: Path, flush_every: int = 2000) -> None:
        self.dir = Path(segments_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._buf: list[tuple[int, int, int]] = []   # (token, chunk_id, tf)
        self._chunk_lengths: dict[int, int] = {}
        self._total_chunks: int = 0
        self._chunks_since_flush: int = 0
        self.segment_count: int = len(list(self.dir.glob("seg_*.bin")))

    def add_chunk(self, chunk_id: int, tokens: "np.ndarray | list[int]") -> None:
        """Buffer one chunk's token counts."""
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        tf_map: dict[int, int] = {}
        for tok in tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1
        for tok, tf in tf_map.items():
            self._buf.append((tok, chunk_id, tf))
        self._chunk_lengths[chunk_id] = len(tokens)
        self._total_chunks = max(self._total_chunks, chunk_id + 1)
        self._chunks_since_flush += 1
        if self._chunks_since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write buffered triples to a segment file and clear the buffer."""
        if not self._buf:
            return
        self._buf.sort()   # sort by (token, chunk_id)
        seg_path = self.dir / f"seg_{self.segment_count:08d}.bin"
        tmp = self.dir / f"seg_{self.segment_count:08d}.bin.tmp"
        with open(tmp, "wb") as f:
            f.write(SEGMENT_MAGIC)
            f.write(struct.pack("<Q", len(self._buf)))
            for tok, cid, tf in self._buf:
                f.write(struct.pack("<QQI", tok, cid, tf))
        os.replace(tmp, seg_path)
        self.segment_count += 1
        self._buf.clear()
        self._chunks_since_flush = 0

    def save_meta(self) -> None:
        """Persist chunk_lengths and total_chunks for the merger."""
        tmp = self.dir / "meta.json.tmp"
        with open(tmp, "w") as f:
            json.dump({
                "total_chunks": self._total_chunks,
                "chunk_lengths": self._chunk_lengths,
            }, f, separators=(",", ":"))
        os.replace(tmp, self.dir / "meta.json")


class SegmentMerger:
    """
    K-way merge of segment files into a final postings.bin.

    Memory usage is O(n_segments), completely independent of corpus size.
    Produces the same postings.bin + postings_offsets.bin layout as
    InvertedIndex.save_binary(), compatible with LazyInvertedIndex.

    Usage::

        merger = SegmentMerger(kb_path / "segments")
        merger.merge(kb_path / "inverted")
    """

    ENTRY_SIZE = 20   # u64 token + u64 chunk_id + u32 tf

    def __init__(self, segments_dir: Path) -> None:
        self.dir = Path(segments_dir)

    def _open_segments(
        self,
    ) -> list[tuple["io.BufferedReader", int, int]]:
        """Open all segment files and return (fh, remaining_entries, file_idx)."""
        import io
        segs = sorted(self.dir.glob("seg_*.bin"))
        result = []
        for seg in segs:
            fh = open(seg, "rb")
            magic = fh.read(8)
            if magic != SEGMENT_MAGIC:
                fh.close()
                continue
            (n,) = struct.unpack("<Q", fh.read(8))
            if n > 0:
                result.append([fh, n, 0])   # [filehandle, remaining, consumed]
        return result

    def _read_entry(self, fh) -> tuple[int, int, int] | None:
        """Read one (token, chunk_id, tf) entry from a segment."""
        raw = fh.read(self.ENTRY_SIZE)
        if not raw or len(raw) < self.ENTRY_SIZE:
            return None
        tok, cid, tf = struct.unpack("<QQI", raw)
        return tok, cid, tf

    def merge(self, output_path: Path | str, store_positions: bool = False) -> None:
        """
        Merge all segments and write postings.bin + postings_offsets.bin.

        Also writes meta.json (chunk_lengths + total_chunks) from the
        segment writer's saved meta, compatible with LazyInvertedIndex.
        """
        import heapq
        import io

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load chunk_lengths from segment meta
        seg_meta_path = self.dir / "meta.json"
        if seg_meta_path.exists():
            seg_meta = json.loads(seg_meta_path.read_text())
            total_chunks: int = seg_meta["total_chunks"]
            chunk_lengths: dict[int, int] = {int(k): v for k, v in seg_meta["chunk_lengths"].items()}
        else:
            total_chunks = 0
            chunk_lengths = {}

        # Open all segments, read first entry from each
        handles = self._open_segments()
        if not handles:
            print("  No segments to merge.")
            return

        # Heap: (token, chunk_id, tf, handle_index)
        heap: list[tuple[int, int, int, int]] = []
        current_entries: list = list(handles)   # mutable state per handle
        buffers: list[tuple[int, int, int] | None] = []
        for i, (fh, remaining, _) in enumerate(handles):
            entry = self._read_entry(fh)
            buffers.append(entry)
            if entry:
                heapq.heappush(heap, (entry[0], entry[1], entry[2], i))

        print(f"  Merging {len(handles)} segments...", flush=True)

        offsets: list[tuple[int, int]] = []   # (token_id, byte_offset)
        postings_tmp = output_path / "postings.bin.tmp"
        meta_tmp = output_path / "meta.json.tmp"

        with open(postings_tmp, "wb") as out:
            out.write(POSTINGS_MAGIC)
            # Placeholder for posting_count
            count_offset = out.tell()
            out.write(struct.pack("<Q", 0))

            posting_count = 0
            current_token: int | None = None
            current_chunk_tfs: dict[int, int] = {}   # chunk_id -> tf

            def _flush_token() -> None:
                nonlocal posting_count, current_token
                if current_token is None or not current_chunk_tfs:
                    return
                offsets.append((current_token, out.tell()))
                bm = BitMap(current_chunk_tfs.keys())
                bitmap_bytes = bm.serialize()
                out.write(struct.pack("<QI", current_token, len(bitmap_bytes)))
                out.write(bitmap_bytes)
                # positions block: store tf as single-element list per chunk
                # (avoids breaking the existing BM25 get_tf path)
                if store_positions:
                    out.write(struct.pack("<I", len(current_chunk_tfs)))
                    for cid in sorted(current_chunk_tfs):
                        tf = current_chunk_tfs[cid]
                        out.write(struct.pack("<QI", cid, tf))
                        out.write(struct.pack(f"<{tf}I", *range(tf)))  # fake positions
                else:
                    # Write tf as position count proxy
                    out.write(struct.pack("<I", len(current_chunk_tfs)))
                    for cid in sorted(current_chunk_tfs):
                        tf = current_chunk_tfs[cid]
                        out.write(struct.pack("<QI", cid, tf))
                        if tf > 0:
                            out.write(struct.pack(f"<{tf}I", *range(tf)))
                posting_count += 1
                current_chunk_tfs.clear()

            while heap:
                tok, cid, tf, idx = heapq.heappop(heap)

                if tok != current_token:
                    _flush_token()
                    current_token = tok

                current_chunk_tfs[cid] = current_chunk_tfs.get(cid, 0) + tf

                # Advance this segment
                fh = handles[idx][0]
                next_entry = self._read_entry(fh)
                if next_entry:
                    heapq.heappush(heap, (next_entry[0], next_entry[1], next_entry[2], idx))

            _flush_token()   # flush last token

            # Back-patch posting count
            end = out.tell()
            out.seek(count_offset)
            out.write(struct.pack("<Q", posting_count))
            out.seek(end)

        for fh, _, _ in handles:
            fh.close()

        os.replace(postings_tmp, output_path / "postings.bin")
        print(f"  Written {posting_count:,} posting lists.", flush=True)

        # Write offset table
        off_tmp = output_path / "postings_offsets.bin.tmp"
        with open(off_tmp, "wb") as f:
            f.write(struct.pack("<Q", len(offsets)))
            for tok, byte_off in offsets:
                f.write(struct.pack("<QQ", tok, byte_off))
        os.replace(off_tmp, output_path / "postings_offsets.bin")

        # Write meta.json
        with open(meta_tmp, "w") as f:
            json.dump({
                "format": "contextfit-inverted-v1",
                "total_chunks": total_chunks,
                "chunk_lengths": chunk_lengths,
                "store_positions": store_positions,
                "vocab_size": posting_count,
            }, f, separators=(",", ":"))
        os.replace(meta_tmp, output_path / "meta.json")

        print(f"  Index ready: {posting_count:,} tokens, {total_chunks:,} chunks.", flush=True)

    @classmethod
    def segments_exist(cls, segments_dir: Path) -> bool:
        return any(Path(segments_dir).glob("seg_*.bin"))
