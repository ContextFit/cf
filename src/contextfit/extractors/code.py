"""Generic, dependency-free code structure extraction.

This module intentionally avoids language-specific parser dependencies. It uses
common function/class/import/comment patterns to keep code symbols and nearby
context together across many languages. Tree-sitter can be layered on later, but
this gives useful source-verifiable chunks with line ranges and symbol metadata
without changing ContextFit's local/token-native footprint.
"""

from __future__ import annotations

import re
from pathlib import Path

from contextfit.extractors import document


CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp",
    ".cs", ".rb", ".php", ".swift", ".kt", ".kts", ".scala",
    ".sh", ".bash", ".zsh", ".fish", ".sql", ".css", ".scss", ".html", ".vue",
}

_LANGUAGE_BY_EXT = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java",
    ".c": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".h": "c/cpp", ".hpp": "cpp",
    ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".fish": "shell",
    ".sql": "sql", ".css": "css", ".scss": "scss", ".html": "html", ".vue": "vue",
}

_SYMBOL_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("class", "python", re.compile(r"^\s*class\s+([A-Za-z_][\w]*)\b")),
    ("function", "python", re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][\w]*)\s*\(")),
    ("class", "generic", re.compile(r"^\s*(?:export\s+)?(?:abstract\s+|final\s+|public\s+|private\s+|protected\s+|static\s+)*class\s+([A-Za-z_$][\w$]*)\b")),
    ("function", "generic", re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(")),
    ("function", "generic", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>")),
    ("function", "generic", re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][\w]*)\s*\(")),
    ("function", "generic", re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][\w]*)\s*\(")),
    ("function", "generic", re.compile(r"^\s*(?:public|private|protected|static|final|override|internal|open|suspend|async|virtual|extern|inline|constexpr|def)\s+.*?\b([A-Za-z_][\w]*)\s*\([^;]*\)\s*(?:\{|=>)?\s*$")),
    ("function", "ruby", re.compile(r"^\s*def\s+([A-Za-z_][\w!?=]*)\b")),
    ("section", "sql", re.compile(r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE|VIEW|TABLE|INDEX|TRIGGER)\s+([\w.\"]+)", re.I)),
    ("selector", "css", re.compile(r"^\s*([^@{}][^{;]{1,100})\s*\{\s*$")),
]

_IMPORT_RE = re.compile(r"^\s*(?:import\s+|from\s+\S+\s+import\s+|export\s+.*from\s+|require\(|#include\s+|use\s+|package\s+|using\s+|namespace\s+)")
_COMMENT_RE = re.compile(r"^\s*(?:#|//|/\*|\*|--|<!--)")


def is_code_path(path: Path | str) -> bool:
    return Path(path).suffix.lower() in CODE_EXTENSIONS


def language_for_path(path: Path | str) -> str:
    return _LANGUAGE_BY_EXT.get(Path(path).suffix.lower(), Path(path).suffix.lower().lstrip(".") or "code")


def _symbol_at(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "//", "/*", "*", "--")):
        return None
    for kind, _family, pattern in _SYMBOL_PATTERNS:
        m = pattern.match(line)
        if m:
            name = " ".join(m.group(1).strip().split())
            if name and not name.startswith(("if", "for", "while", "switch", "catch")):
                return kind, name
    return None


def _line_no_ranges(lines: list[str]) -> list[tuple[int, int, str, str]]:
    starts: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        sym = _symbol_at(line)
        if sym:
            kind, name = sym
            starts.append((i, kind, name))
    ranges: list[tuple[int, int, str, str]] = []
    for idx, (start, kind, name) in enumerate(starts):
        end = (starts[idx + 1][0] - 1) if idx + 1 < len(starts) else len(lines) - 1
        ranges.append((start, max(start, end), kind, name))
    return ranges


def _preamble_end(lines: list[str], first_symbol_start: int | None) -> int:
    limit = first_symbol_start if first_symbol_start is not None else len(lines)
    last = -1
    for i in range(limit):
        line = lines[i]
        if _IMPORT_RE.match(line) or _COMMENT_RE.match(line) or not line.strip():
            last = i
        elif last >= 0:
            # Stop after a contiguous header/import block.
            break
    return last


def _emit_chunk(chunks: list[dict], *, path: Path, text: str, meta: dict[str, str], chunk_size: int, overlap: int) -> None:
    if document._token_estimate(text) <= max(64, chunk_size * 0.95):
        chunks.append({"text": text, "metadata": meta})
        return
    for ordinal, sub in enumerate(document.chunk_text(path, text, chunk_size=chunk_size, overlap=overlap)):
        chunks.append({
            "text": sub["text"],
            "metadata": {**meta, "subchunk_ordinal": str(ordinal)},
        })


def chunk_code(path: Path, text: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """Chunk source code around generic symbol/import boundaries."""
    base_meta = document.extract(path, text)
    language = language_for_path(path)
    lines = text.splitlines()
    if not lines:
        return [{"text": "", "metadata": {**base_meta, "chunk_type": "code_file", "language": language, "record_count": "0"}}]

    ranges = _line_no_ranges(lines)
    chunks: list[dict] = []
    ordinal = 0

    first_start = ranges[0][0] if ranges else None
    pre_end = _preamble_end(lines, first_start)
    if pre_end >= 0:
        pre_text = "\n".join(lines[:pre_end + 1]).strip("\n")
        if pre_text.strip():
            _emit_chunk(
                chunks,
                path=path,
                text=pre_text,
                meta={
                    **base_meta,
                    "chunk_type": "code_preamble",
                    "chunk_ordinal": str(ordinal),
                    "language": language,
                    "line_start": "1",
                    "line_end": str(pre_end + 1),
                },
                chunk_size=chunk_size,
                overlap=overlap,
            )
            ordinal += 1

    covered_until = pre_end
    for start, end, kind, name in ranges:
        if start <= covered_until:
            continue
        # Keep immediately preceding doc/comment block with the symbol.
        doc_start = start
        j = start - 1
        while j > covered_until and (_COMMENT_RE.match(lines[j]) or not lines[j].strip()):
            doc_start = j
            j -= 1
        block = "\n".join(lines[doc_start:end + 1]).strip("\n")
        _emit_chunk(
            chunks,
            path=path,
            text=block,
            meta={
                **base_meta,
                "chunk_type": "code_symbol",
                "chunk_ordinal": str(ordinal),
                "language": language,
                "symbol_kind": kind,
                "symbol_name": name,
                "line_start": str(doc_start + 1),
                "line_end": str(end + 1),
            },
            chunk_size=chunk_size,
            overlap=overlap,
        )
        ordinal += 1
        covered_until = end

    if not chunks:
        meta = {**base_meta, "chunk_type": "code_file", "chunk_ordinal": "0", "language": language, "line_start": "1", "line_end": str(len(lines))}
        _emit_chunk(chunks, path=path, text=text, meta=meta, chunk_size=chunk_size, overlap=overlap)
    return chunks
