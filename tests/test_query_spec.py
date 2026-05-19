from __future__ import annotations

from pathlib import Path

from contextfit.retrieval.engine import RetrievalEngine


def test_structured_metadata_filters_prefilter_token_native_results(tmp_path: Path) -> None:
    engine = RetrievalEngine.create(tmp_path / "kb")
    engine.ingest_text(
        "ContextFit metadata prefilter decision from April.",
        metadata={"source": "april.md", "kind": "decision", "date": "2026-04-01"},
    )
    engine.ingest_text(
        "ContextFit metadata prefilter decision from May.",
        metadata={"source": "may.md", "kind": "decision", "date": "2026-05-17"},
    )
    engine.ingest_text(
        "ContextFit metadata prefilter note from May.",
        metadata={"source": "note.md", "kind": "note", "date": "2026-05-17"},
    )

    result = engine.query(
        "ContextFit metadata prefilter",
        top_k=5,
        method="hybrid",
        filters=[
            {"field": "kind", "op": "exact", "value": "decision"},
            {"field": "date", "op": "on_or_after", "value": "2026-05-01"},
        ],
    )

    assert [chunk.metadata["source"] for chunk in result.chunks] == ["may.md"]
    assert result.filter_trace is not None
    assert result.filter_trace["matched_chunks"] == 1
    assert result.filter_trace["broadened"] is False


def test_query_spec_can_broaden_when_agent_filter_is_too_narrow(tmp_path: Path) -> None:
    engine = RetrievalEngine.create(tmp_path / "kb")
    engine.ingest_text(
        "ContextFit agent retrieval uses token-native search.",
        metadata={"source": "agent.md", "kind": "decision"},
    )

    result = engine.query(
        "",
        top_k=1,
        method="hybrid",
        query_spec={
            "query": "ContextFit agent retrieval",
            "filters": [{"field": "kind", "op": "exact", "value": "missing"}],
            "min_filter_matches": 1,
        },
    )

    assert [chunk.metadata["source"] for chunk in result.chunks] == ["agent.md"]
    assert result.filter_trace is not None
    assert result.filter_trace["matched_chunks"] == 0
    assert result.filter_trace["broadened"] is True


def test_broad_filters_post_filter_instead_of_pushdown(tmp_path: Path) -> None:
    engine = RetrievalEngine.create(tmp_path / "kb")
    engine.ingest_text(
        "ContextFit metadata prefilter shared topic decision.",
        metadata={"source": "decision.md", "kind": "decision"},
    )
    engine.ingest_text(
        "ContextFit metadata prefilter shared topic note one.",
        metadata={"source": "note-1.md", "kind": "note"},
    )
    engine.ingest_text(
        "ContextFit metadata prefilter shared topic note two.",
        metadata={"source": "note-2.md", "kind": "note"},
    )

    result = engine.query(
        "ContextFit metadata prefilter shared topic",
        top_k=5,
        method="hybrid",
        filters=[{"field": "kind", "op": "exact", "value": "note"}],
        filter_pushdown_threshold=0.50,
    )

    assert {chunk.metadata["source"] for chunk in result.chunks} == {
        "note-1.md",
        "note-2.md",
    }
    assert result.filter_trace is not None
    assert result.filter_trace["pushdown"] is False


def test_range_filters_survive_save_load(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb"
    engine = RetrievalEngine.create(kb_path)
    engine.ingest_text(
        "ContextFit metadata prefilter old decision.",
        metadata={"source": "old.md", "date": "2026-04-01"},
    )
    engine.ingest_text(
        "ContextFit metadata prefilter new decision.",
        metadata={"source": "new.md", "date": "2026-05-17"},
    )
    engine.save(kb_path)

    loaded = RetrievalEngine.load(kb_path)
    result = loaded.query(
        "ContextFit metadata prefilter decision",
        top_k=5,
        method="hybrid",
        filters=[{"field": "date", "op": "on_or_after", "value": "2026-05-01"}],
    )

    assert [chunk.metadata["source"] for chunk in result.chunks] == ["new.md"]
