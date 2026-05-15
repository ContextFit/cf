"""
Auto extractor — sniffs file type and delegates to the right extractor.

Heuristics (in order):
  1. .eml extension → email extractor
  2. Markdown with **From:** / **Date:** headers → email extractor
  3. YAML front-matter with title/author → document extractor
  4. Plain markdown/text → document extractor
  5. Fallback → null extractor (source + filename only)
"""

from __future__ import annotations

from pathlib import Path

from contextfit.extractors import email as _email
from contextfit.extractors import document as _document
from contextfit.extractors.base import null_extractor


def extract(path: Path, text: str) -> dict[str, str]:
    """Auto-detecting metadata extractor."""
    header = text[:2048].lower()

    # TMD (Tabular Markdown) signals
    if (
        path.suffix == ".tmd"
        or (path.suffix == ".md" and re.search(r"^\w+\[[^\]]*\]:\s*.+", text, re.MULTILINE))
    ):
        from contextfit.extractors import tmd as _tmd
        return _tmd.extract(path, text)

    # Email signals
    if (
        path.suffix in (".eml",)
        or "**from:**" in header
        or re.search(r"(?m)^from:\s+\S+@\S+", header)
        or re.search(r"\*\*date:\*\*", header)
    ):
        return _email.extract(path, text)

    # Document signals
    if (
        text.startswith("---\n")          # YAML front-matter
        or re.search(r"(?m)^#{1,3}\s", text[:512])  # any heading
        or path.suffix in (".md", ".rst", ".txt")
    ):
        return _document.extract(path, text)

    return null_extractor(path, text)


import re  # noqa: E402  (needed for the function body above)
