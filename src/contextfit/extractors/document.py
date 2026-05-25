"""
Document metadata extractor.

Works with Markdown, plain text, RST, and similar documents.
Extracts: title, author, date, tags from YAML front-matter or
first-heading / comment conventions.

Fields produced:
  title     — document title (H1 or front-matter)
  author    — author name (front-matter or "Author:" line)
  date      — date string (front-matter or "Date:" line)
  tags      — comma-joined tags from front-matter
  section   — parent directory name (useful for KB organisation)
  source    — file path
  filename  — base filename
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _frontmatter(text: str) -> dict[str, str]:
    """Parse YAML-ish front-matter between --- delimiters."""
    m = re.match(r"^---\n(.+?)\n---", text, re.S)
    if not m:
        return {}
    fields: dict[str, str] = {}
    for line in m.group(1).splitlines():
        kv = re.match(r"(\w+)\s*:\s*(.+)", line)
        if kv:
            fields[kv.group(1).lower()] = kv.group(2).strip().strip('"\'')
    return fields


def extract(path: Path, text: str) -> dict[str, str]:
    """Document metadata extractor."""
    header = text[:4096]
    fields = _frontmatter(header)

    # Title: front-matter or first H1/H2
    if "title" not in fields:
        h = re.search(r"^#{1,2}\s+(.+)", header, re.M)
        if h:
            fields["title"] = h.group(1).strip()
        else:
            fields["title"] = path.stem.replace("-", " ").replace("_", " ")

    # Author
    if "author" not in fields:
        a = re.search(r"(?i)^author[:\s]+(.+)", header, re.M)
        if a:
            fields["author"] = a.group(1).strip()

    # Date
    if "date" not in fields:
        d = re.search(r"(?i)^date[:\s]+(.+)", header, re.M)
        if d:
            fields["date"] = d.group(1).strip()

    # Tags
    if "tags" in fields and isinstance(fields["tags"], str):
        # Normalise list-style tags
        fields["tags"] = re.sub(r"[\[\]'\"]", "", fields["tags"])

    # Section from parent dir
    fields["section"] = path.parent.name

    fields["source"] = str(path)
    fields["filename"] = path.name
    return fields


def _token_estimate(text: str) -> float:
    """Cheap token estimate used before tokenizer-specific encoding."""
    return len(text.split()) * 1.3


def _frontmatter_end(lines: list[str]) -> int:
    """Return line index after YAML front-matter, or 0 when absent."""
    if not lines or lines[0].strip() != "---":
        return 0
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            return i + 1
    return 0


def _emit_markdown_chunk(
    chunks: list[dict],
    *,
    base_meta: dict[str, Any],
    header_text: str,
    heading_path: list[str],
    blocks: list[tuple[str, int, int]],
    ordinal: int,
) -> int:
    body = "\n\n".join(block.strip() for block, _start, _end in blocks if block.strip()).strip()
    if not body:
        return ordinal

    line_start = min(start for block, start, _end in blocks if block.strip())
    line_end = max(end for block, _start, end in blocks if block.strip())
    breadcrumb = " > ".join(heading_path)
    context_lines = []
    if header_text:
        context_lines.append(header_text)
    if breadcrumb:
        context_lines.append(f"Heading path: {breadcrumb}")
    chunk_text = "\n\n".join(context_lines + [body]).strip()
    context_text = "\n\n".join(context_lines).strip()
    context_line_count = context_text.count("\n") + 2 if context_text else 0

    chunks.append({
        "text": chunk_text,
        "metadata": {
            **base_meta,
            "chunk_type": "markdown_section",
            "heading_path": breadcrumb,
            "section_level": str(len(heading_path)),
            "chunk_ordinal": str(ordinal),
            "line_start": line_start,
            "line_end": line_end,
            "chunk_context_line_count": context_line_count,
        },
    })
    return ordinal + 1


def chunk_markdown(path: Path, text: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """
    Structure-aware chunking for Markdown documents.

    Strategy:
    - Preserve YAML front-matter and document title as lightweight context.
    - Use headings as primary boundaries.
    - Keep semantic blocks intact: paragraphs, lists, tables, blockquotes, code fences.
    - Split oversized sections by block, not arbitrary token windows.
    - Add heading-path metadata for explainable retrieval.
    """
    base_meta = extract(path, text)
    lines = text.splitlines()
    start = _frontmatter_end(lines)
    header_lines = lines[:start]

    # Keep the first H1 as document-level context and not as body-only noise.
    if start < len(lines) and re.match(r"^#\s+", lines[start].strip()):
        header_lines.append(lines[start])
        start += 1

    header_text = "\n".join(header_lines).strip()
    chunks: list[dict] = []
    heading_stack: list[str] = [base_meta.get("title", path.stem)] if base_meta.get("title") else []
    current_blocks: list[tuple[str, int, int]] = []
    current_block: list[tuple[str, int]] = []
    in_code = False
    ordinal = 0

    def flush_block() -> None:
        nonlocal current_block
        block = "\n".join(line for line, _line_no in current_block).strip("\n")
        if block.strip():
            line_numbers = [line_no for line, _line_no in current_block if line.strip()]
            current_blocks.append((block, min(line_numbers), max(line_numbers)))
        current_block = []

    def flush_section() -> None:
        nonlocal current_blocks, ordinal
        flush_block()
        if not current_blocks:
            return

        budget = max(64, chunk_size * 0.8)
        acc: list[tuple[str, int, int]] = []
        acc_est = _token_estimate(header_text) + _token_estimate(" > ".join(heading_stack))
        previous_tail: list[tuple[str, int, int]] = []

        for block in current_blocks:
            block_est = _token_estimate(block[0])
            if acc and acc_est + block_est > budget:
                ordinal = _emit_markdown_chunk(
                    chunks,
                    base_meta=base_meta,
                    header_text=header_text,
                    heading_path=heading_stack,
                    blocks=acc,
                    ordinal=ordinal,
                )
                previous_tail = acc[-1:] if overlap > 0 else []
                acc = previous_tail.copy()
                acc_est = _token_estimate(header_text) + _token_estimate(" > ".join(heading_stack)) + sum(_token_estimate(b[0]) for b in acc)
            acc.append(block)
            acc_est += block_est

        if acc and acc != previous_tail:
            ordinal = _emit_markdown_chunk(
                chunks,
                base_meta=base_meta,
                header_text=header_text,
                heading_path=heading_stack,
                blocks=acc,
                ordinal=ordinal,
            )
        current_blocks = []

    for line_no, line in enumerate(lines[start:], start + 1):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            current_block.append((line, line_no))
            in_code = not in_code
            if not in_code:
                flush_block()
            continue

        if not in_code:
            heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
            if heading:
                flush_section()
                level = len(heading.group(1))
                title = heading.group(2).strip()
                if level == 1:
                    heading_stack = [title]
                else:
                    heading_stack = heading_stack[: max(1, level - 1)]
                    heading_stack.append(title)
                current_block = [(line, line_no)]
                flush_block()
                continue

            if stripped == "":
                flush_block()
                continue

            starts_structured = bool(re.match(r"^(- |\* |\d+\. |>|\|)", stripped))
            if current_block and starts_structured:
                prev = current_block[-1][0].strip()
                prev_structured = bool(re.match(r"^(- |\* |\d+\. |>|\|)", prev))
                if not prev_structured:
                    flush_block()

        current_block.append((line, line_no))

    flush_section()
    return chunks if chunks else [{"text": text, "metadata": {**base_meta, "chunk_type": "document", "line_start": 1, "line_end": len(lines) or 1}}]


def chunk_text(path: Path, text: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """
    Paragraph-aware chunking for plain text.

    Uses blank lines and common separators as soft boundaries, carries file-level
    metadata, and overlaps by whole paragraphs instead of raw token tails.
    """
    base_meta = extract(path, text)
    parts = [p.strip() for p in re.split(r"\n\s*\n|\n-{3,}\n", text) if p.strip()]
    if not parts:
        return [{"text": text, "metadata": {**base_meta, "chunk_type": "text"}}]

    chunks: list[dict] = []
    acc: list[str] = []
    acc_est = 0.0
    budget = max(64, chunk_size * 0.8)
    ordinal = 0

    def emit(blocks: list[str], ordinal_value: int) -> int:
        chunks.append({
            "text": "\n\n".join(blocks).strip(),
            "metadata": {
                **base_meta,
                "chunk_type": "paragraph_group",
                "chunk_ordinal": str(ordinal_value),
            },
        })
        return ordinal_value + 1

    for part in parts:
        part_est = _token_estimate(part)
        if acc and acc_est + part_est > budget:
            ordinal = emit(acc, ordinal)
            tail = acc[-1:] if overlap > 0 else []
            acc = tail.copy()
            acc_est = sum(_token_estimate(p) for p in acc)
        acc.append(part)
        acc_est += part_est

    if acc:
        emit(acc, ordinal)
    return chunks
