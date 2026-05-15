"""Tests for exact/token search behavior."""

from pathlib import Path

from contextfit.retrieval.engine import RetrievalEngine


def test_exact_method_prioritizes_literal_phrase(tmp_path: Path):
    kb = tmp_path / "kb"
    engine = RetrievalEngine.create(kb)
    engine.ingest_text("The purchased shoe was Brooks Glycerin 22 in 10.5 Medium.", metadata={"source": "purchase.md"})
    engine.ingest_text("Running shoes and foot scans are related concepts.", metadata={"source": "general.md"})

    result = engine.query("Brooks Glycerin 22", method="exact", top_k=5)

    assert len(result.chunks) == 1
    assert result.chunks[0].metadata["source"] == "purchase.md"
    assert result.scores[0] >= 1_000_000.0


def test_hybrid_includes_literal_match_above_related_text(tmp_path: Path):
    kb = tmp_path / "kb"
    engine = RetrievalEngine.create(kb)
    engine.ingest_text("Running shoes and foot scans are related concepts.", metadata={"source": "general.md"})
    engine.ingest_text("The purchased shoe was Brooks Glycerin 22 in 10.5 Medium.", metadata={"source": "purchase.md"})

    result = engine.query("Brooks Glycerin 22", method="hybrid", top_k=2)

    assert result.chunks[0].metadata["source"] == "purchase.md"
    assert result.scores[0] >= 1_000_000.0
