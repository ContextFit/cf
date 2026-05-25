from pathlib import Path

from contextfit.extractors import document, tmd
from contextfit.cli import _preprocess_file
from contextfit.retrieval.engine import RetrievalEngine


def test_markdown_chunking_preserves_heading_path_and_blocks(tmp_path):
    path = tmp_path / "guide.md"
    text = """# Project Guide

Intro paragraph.

## Install

Run this first.

```python
print('hello')
```

## Deploy

- build
- ship
"""
    chunks = document.chunk_markdown(path, text, chunk_size=40, overlap=0)

    assert len(chunks) >= 2
    paths = [c["metadata"].get("heading_path", "") for c in chunks]
    assert any("Project Guide > Install" in p for p in paths)
    assert any("Project Guide > Deploy" in p for p in paths)
    assert any("print('hello')" in c["text"] for c in chunks)
    assert all(c["metadata"]["chunk_type"] == "markdown_section" for c in chunks)
    assert all(isinstance(c["metadata"].get("line_start"), int) for c in chunks)
    assert all(isinstance(c["metadata"].get("line_end"), int) for c in chunks)

    install = next(c for c in chunks if "print('hello')" in c["text"])
    assert install["metadata"]["line_start"] == 5
    assert install["metadata"]["line_end"] == 11

    deploy = next(c for c in chunks if "- build" in c["text"])
    assert deploy["metadata"]["line_start"] == 13
    assert deploy["metadata"]["line_end"] == 16


def test_plain_text_chunking_uses_paragraph_groups(tmp_path):
    path = tmp_path / "notes.txt"
    text = "\n\n".join([f"Paragraph {i} " + "word " * 35 for i in range(4)])
    chunks = document.chunk_text(path, text, chunk_size=80, overlap=0)

    assert len(chunks) >= 2
    assert all(c["metadata"]["chunk_type"] == "paragraph_group" for c in chunks)
    assert "Paragraph 0" in chunks[0]["text"]


def test_tmd_chunking_preserves_rows_and_header(tmp_path):
    path = tmp_path / "ledger.tmd"
    text = """---
schema:
  amount: number
---
# Ledger

row1[]: amount=10, note=coffee
row2[]: amount=20, note=lunch
"""
    chunks = tmd.chunk_tmd(path, text, chunk_size=20, overlap=0)

    assert chunks
    assert chunks[0]["metadata"]["chunk_type"] == "rows"
    assert "# Ledger" in chunks[0]["text"]
    assert "row1[]" in chunks[0]["text"]
    assert chunks[0]["metadata"]["row_ids"] == ["row1"]
    assert chunks[0]["metadata"]["chunk_ordinal"] == 0
    assert chunks[0]["metadata"]["line_start"] == 7
    assert chunks[0]["metadata"]["line_end"] == 7


def test_tmd_chunking_preserves_explicit_row_ids_and_line_ranges(tmp_path):
    path = tmp_path / "campaign.tmd"
    text = """---
schema:
  status: string
---
# Campaign

campaign[video-01]: status=idea
campaign[video-04]: status=blocked
campaign[video-08]: status=ready
"""
    chunks = tmd.chunk_tmd(path, text, chunk_size=200, overlap=0)

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["row_ids"] == ["video-01", "video-04", "video-08"]
    assert chunks[0]["metadata"]["line_start"] == 7
    assert chunks[0]["metadata"]["line_end"] == 9


def test_cli_preprocess_uses_structure_aware_chunks(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\n## Alpha\n\nAlpha text.\n\n## Beta\n\nBeta text.")

    result = _preprocess_file(
        path,
        tokenizer_name="cl100k_base",
        chunk_size=20,
        overlap=0,
        max_file_bytes=0,
        max_file_tokens=0,
    )

    assert result["status"] == "ok"
    assert "token_items" in result
    metas = [item["metadata"] for item in result["token_items"]]
    assert any(meta.get("chunk_type") == "markdown_section" for meta in metas)
    assert any("Guide > Alpha" in meta.get("heading_path", "") for meta in metas)
    assert all("line_start" in meta and "line_end" in meta for meta in metas)


def test_engine_ingest_file_uses_markdown_structure(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\n## Alpha\n\nAlpha text.\n\n## Beta\n\nBeta text.")
    engine = RetrievalEngine.create(tmp_path / "kb")

    chunks = engine.ingest_file(path, chunk_size=20, overlap=0)

    assert chunks
    metas = [chunk.metadata for chunk in chunks]
    assert any(meta.get("chunk_type") == "markdown_section" for meta in metas)
    assert any("Guide > Alpha" in meta.get("heading_path", "") for meta in metas)
    assert all("line_start" in meta and "line_end" in meta for meta in metas)
