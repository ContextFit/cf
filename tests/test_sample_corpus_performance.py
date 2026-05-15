"""Sample-corpus correctness and performance tests.

These tests are intentionally benchmark-style without requiring pytest-benchmark.
Thresholds are generous enough for CI/dev laptops but still catch obvious
regressions like JSON-per-token persistence returning, broken SID retrieval, or
multi-second query paths on small corpora.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from contextfit import InvertedIndex, RetrievalEngine

TOPICS = {
    "python_async": [
        "python",
        "async",
        "await",
        "coroutine",
        "asyncio",
        "event loop",
        "future",
        "task",
    ],
    "rust_memory": [
        "rust",
        "ownership",
        "borrowing",
        "lifetimes",
        "compiler",
        "memory safety",
        "trait",
        "cargo",
    ],
    "drone_mapping": [
        "drone",
        "orthomosaic",
        "open drone map",
        "tileset",
        "gps",
        "photogrammetry",
        "mesh",
        "maplibre",
    ],
    "agent_memory": [
        "agent",
        "memory",
        "semantic id",
        "retrieval",
        "knowledge graph",
        "context window",
        "token store",
        "commonality",
    ],
}


def make_document(topic: str, i: int, repeats: int = 6) -> str:
    """Create deterministic topic-heavy sample text."""
    terms = TOPICS[topic]
    sentences = []
    for j in range(repeats):
        rotated = terms[j % len(terms) :] + terms[: j % len(terms)]
        sentences.append(
            f"Document {i} section {j} discusses {topic}. "
            f"The key signals are {', '.join(rotated[:5])}. "
            f"This sample connects {rotated[0]} with {rotated[1]} and {rotated[2]}."
        )
    return "\n".join(sentences)


def build_sample_engine(kb: Path, docs_per_topic: int = 50) -> tuple[RetrievalEngine, float]:
    """Build and train a deterministic sample KB."""
    engine = RetrievalEngine.create(kb)
    start = time.perf_counter()
    for topic in TOPICS:
        for i in range(docs_per_topic):
            engine.ingest_text(
                make_document(topic, i),
                chunk_size=96,
                overlap=16,
                metadata={"topic": topic},
            )
    engine.train_learned_sid_generator()
    engine.save(kb)
    elapsed = time.perf_counter() - start
    return engine, elapsed


def assert_top_topic(result, expected_topic: str) -> None:
    assert result.chunks, "expected at least one retrieved chunk"
    assert result.chunks[0].metadata.get("topic") == expected_topic
    assert result.semantic_ids is not None
    assert result.sid_predictions is not None


def test_sample_corpus_retrieval_quality_across_methods():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine, _ = build_sample_engine(Path(tmpdir) / "kb", docs_per_topic=10)

        checks = [
            ("how do python coroutines use await and asyncio tasks", "python_async"),
            ("rust compiler ownership borrowing lifetimes memory safety", "rust_memory"),
            ("drone photogrammetry orthomosaic gps tileset mesh", "drone_mapping"),
            ("agent memory semantic id token store retrieval graph", "agent_memory"),
        ]

        for query, expected_topic in checks:
            hybrid = engine.query(query, method="hybrid", top_k=3)
            sid = engine.query(query, method="sid", top_k=3)
            bm25 = engine.query(query, method="bm25", top_k=3)

            assert_top_topic(hybrid, expected_topic)
            assert_top_topic(sid, expected_topic)
            assert bm25.chunks[0].metadata.get("topic") == expected_topic


def test_sample_corpus_persistence_and_binary_pack_are_fast_enough():
    with tempfile.TemporaryDirectory() as tmpdir:
        kb = Path(tmpdir) / "kb"
        _engine, build_seconds = build_sample_engine(kb, docs_per_topic=25)

        # Generous threshold: this should be comfortably sub-second on the DGX,
        # and still reasonable on a MacBook dev laptop.
        assert build_seconds < 10.0
        assert (kb / "inverted" / "postings.bin").exists()
        assert not (kb / "inverted" / "postings").exists()

        load_start = time.perf_counter()
        loaded = RetrievalEngine.load(kb)
        load_seconds = time.perf_counter() - load_start
        assert load_seconds < 3.0

        queries = [
            "python asyncio coroutine task",
            "rust ownership lifetimes compiler",
            "drone orthomosaic photogrammetry tileset",
            "agent memory semantic id retrieval",
        ]
        query_start = time.perf_counter()
        results = [loaded.query(q, method="hybrid", top_k=5) for q in queries]
        total_query_seconds = time.perf_counter() - query_start

        assert all(result.chunks for result in results)
        assert total_query_seconds < 1.0


def test_binary_postings_pack_avoids_many_json_files_for_sample_corpus():
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = RetrievalEngine.create(Path(tmpdir) / "kb")
        for topic in TOPICS:
            for i in range(15):
                engine.ingest_text(make_document(topic, i), chunk_size=96, overlap=16)

        binary_path = Path(tmpdir) / "binary"
        json_path = Path(tmpdir) / "json"
        engine.inverted.save_binary(binary_path)
        engine.inverted.save_json(json_path)

        binary_bytes = (binary_path / "postings.bin").stat().st_size
        json_postings = list((json_path / "postings").glob("*.json"))
        json_bytes = sum(path.stat().st_size for path in json_postings)

        # The current binary pack prioritizes one-file locality and fast loading while
        # preserving positions. On tiny corpora raw position blocks can be near/slightly
        # above JSON size, but it should stay in the same order of magnitude and avoid
        # per-token file explosion.
        assert len(json_postings) > 10
        assert binary_bytes <= json_bytes * 1.25
        assert InvertedIndex.load(binary_path).stats() == InvertedIndex.load(json_path).stats()
