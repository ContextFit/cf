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


def test_hybrid_rrf_does_not_let_sid_swamp_bm25_winner(tmp_path: Path):
    kb = tmp_path / "kb"
    engine = RetrievalEngine.create(kb)

    engine.ingest_text(
        "My favorite hobby is collecting vintage typewriters from the 1920s.",
        metadata={"source": "target.md"},
    )
    distractor = " ".join(
        [
            "I like various activities including hiking, reading, cooking, "
            "traveling, gardening, painting, music, films, theater, sports, "
            "games, photography, cycling, swimming, running, yoga, meditation, "
            "writing, learning languages, and documentaries."
        ]
        * 8
    )
    for i in range(5):
        engine.ingest_text(distractor, metadata={"source": f"distractor_{i}.md"})

    result = engine.query("vintage typewriters", method="hybrid_rrf", top_k=3)
    sources = [chunk.metadata.get("source") for chunk in result.chunks]

    assert sources[0] == "target.md"
