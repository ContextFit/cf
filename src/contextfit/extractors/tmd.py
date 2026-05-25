"""
TMD Extractor: Extract metadata from Tabular Markdown files.

Provides row-level metadata extraction for better semantic chunking.
"""

from __future__ import annotations

import re
import yaml
from pathlib import Path


def extract(path: Path, text: str) -> dict[str, str]:
    """
    Extract metadata from a TMD file.
    
    Returns metadata dict with:
    - source: file path
    - domain: 'tmd'
    - title: document title
    - table: table name
    - schema_fields: comma-separated field names
    - computed_fields: comma-separated computed field names
    - row_count: number of data rows
    """
    metadata: dict[str, str] = {
        "source": str(path),
        "domain": "tmd",
    }
    
    lines = text.split('\n')
    
    # Parse front matter
    if lines and lines[0].strip() == '---':
        end_idx = None
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                end_idx = i
                break
        
        if end_idx:
            yaml_content = '\n'.join(lines[1:end_idx])
            try:
                front = yaml.safe_load(yaml_content) or {}
                
                if "schema" in front:
                    metadata["schema_fields"] = ", ".join(front["schema"].keys())
                
                if "computed" in front:
                    metadata["computed_fields"] = ", ".join(front["computed"].keys())
                    
            except yaml.YAMLError:
                pass
    
    # Find title (H1)
    for line in lines:
        if line.startswith('# '):
            metadata["title"] = line[2:].strip()
            metadata["table"] = metadata["title"].lower().replace(' ', '_')
            break
    
    # Count rows
    row_pattern = re.compile(r'^\w+\[[^\]]*\]:\s*.+$')
    row_count = sum(1 for line in lines if row_pattern.match(line.strip()))
    metadata["row_count"] = str(row_count)
    
    return metadata


def chunk_tmd(path: Path, text: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """
    Smart chunking for TMD files.
    
    Strategy:
    - Keep schema/header with each chunk
    - Chunk by row boundaries when possible
    - Include table context in metadata
    
    Returns list of dicts with 'text' and 'metadata' keys.
    """
    chunks = []
    lines = text.split('\n')
    
    # Extract header (front matter + title)
    header_lines = []
    data_start = 0
    
    # Front matter
    if lines and lines[0].strip() == '---':
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                header_lines = lines[:i+1]
                data_start = i + 1
                break
    
    # Title
    for i, line in enumerate(lines[data_start:], data_start):
        stripped = line.strip()
        if stripped.startswith('# '):
            header_lines.append(line)
            data_start = i + 1
            break
        elif stripped and not stripped.startswith('#'):
            break
    
    header_text = '\n'.join(header_lines).strip()
    
    # Find all data rows
    row_pattern = re.compile(r'^(\w+)\[([^\]]*)\]:\s*(.+)$')
    rows: list[tuple[str, int, str]] = []
    prose_lines: list[tuple[str, int]] = []
    
    for line_no, line in enumerate(lines[data_start:], data_start + 1):
        stripped = line.strip()
        row_match = row_pattern.match(stripped)
        if row_match:
            key = row_match.group(1)
            row_id = row_match.group(2).strip() or key
            rows.append((line, line_no, row_id))
        elif stripped and not stripped.startswith('##'):
            prose_lines.append((line, line_no))
    
    # Base metadata
    base_meta = extract(path, text)
    
    # Chunk rows, prepending header to each chunk
    current_chunk_rows: list[tuple[str, int, str]] = []
    current_tokens_estimate = len(header_text.split()) * 1.3  # Rough token estimate
    ordinal = 0

    def emit_rows(chunk_rows: list[tuple[str, int, str]], ordinal_value: int) -> int:
        row_texts = [row for row, _line_no, _row_id in chunk_rows]
        row_ids = [row_id for _row, _line_no, row_id in chunk_rows]
        line_numbers = [line_no for _row, line_no, _row_id in chunk_rows]
        chunk_text = header_text + '\n\n' + '\n'.join(row_texts) if header_text else '\n'.join(row_texts)
        context_line_count = header_text.count('\n') + 2 if header_text else 0
        chunks.append({
            'text': chunk_text,
            'metadata': {
                **base_meta,
                'chunk_type': 'rows',
                'row_count_in_chunk': str(len(chunk_rows)),
                'row_ids': row_ids,
                'chunk_ordinal': ordinal_value,
                'line_start': min(line_numbers),
                'line_end': max(line_numbers),
                'chunk_context_line_count': context_line_count,
            }
        })
        return ordinal_value + 1
    
    for row in rows:
        row_text, _line_no, _row_id = row
        row_tokens = len(row_text.split()) * 1.3
        
        if current_tokens_estimate + row_tokens > chunk_size * 0.8 and current_chunk_rows:
            ordinal = emit_rows(current_chunk_rows, ordinal)
            current_chunk_rows = []
            current_tokens_estimate = len(header_text.split()) * 1.3
        
        current_chunk_rows.append(row)
        current_tokens_estimate += row_tokens
    
    # Emit final chunk
    if current_chunk_rows:
        ordinal = emit_rows(current_chunk_rows, ordinal)
    
    # Add prose as separate chunk if substantial
    prose_text = '\n'.join(line for line, _line_no in prose_lines).strip()
    if prose_text and len(prose_text) > 100:
        prose_line_numbers = [line_no for line, line_no in prose_lines if line.strip()]
        chunks.append({
            'text': header_text + '\n\n' + prose_text,
            'metadata': {
                **base_meta,
                'chunk_type': 'prose',
                'chunk_ordinal': ordinal,
                'line_start': min(prose_line_numbers),
                'line_end': max(prose_line_numbers),
                'chunk_context_line_count': header_text.count('\n') + 2 if header_text else 0,
            }
        })
    
    line_count = len(text.splitlines()) or 1
    return chunks if chunks else [{'text': text, 'metadata': {**base_meta, 'line_start': 1, 'line_end': line_count}}]
