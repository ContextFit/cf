"""
MetadataExtractor protocol — pluggable metadata extraction for any data type.

An extractor takes a file path + raw text and returns a flat dict of
string field→value pairs.  ContextFit stores these verbatim in the
MetadataIndex; the engine treats them as opaque structured annotations.

Built-in extractors:
  email     — parses From/To/Subject/Date from .md email exports
  document  — extracts title/author/date from markdown/text documents
  auto      — sniffs file type and delegates to the right extractor

Custom extractors:
  Any callable matching  (path: Path, text: str) -> dict[str, str]
  can be passed to MetadataIndex.add_file_chunks() or engine.ingest_file().

Example:
    def my_extractor(path, text):
        return {"author": re.search(r'author: (.+)', text).group(1),
                "project": path.parent.name}

    engine.ingest_file(path, metadata_extractor=my_extractor)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class MetadataExtractor(Protocol):
    """Protocol for metadata extractors."""

    def __call__(self, path: Path, text: str) -> dict[str, str]:
        """Extract metadata from a file.

        Args:
            path: File path (use for filename, extension, parent dir hints).
            text: Raw file text (first N bytes is fine; extractors should
                  only look at headers / frontmatter).

        Returns:
            Flat dict of string field→value.  Any fields are allowed;
            ContextFit indexes all of them.  Common conventions:
              title    — human-readable title
              author   — primary author / sender
              date     — ISO-ish date string
              source   — original file path (auto-added if missing)
              tags     — comma-separated tags
        """
        ...


# Re-export convenience
def null_extractor(path: Path, text: str) -> dict[str, str]:
    """Extractor that only records the source path (no parsing)."""
    return {"source": str(path), "filename": path.name}
