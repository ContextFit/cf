"""
Email metadata extractor.

Parses exported email .md / .txt / .eml files and extracts structured
header fields.  This is email-specific logic that lives here, NOT in
MetadataIndex (which is completely generic).

Fields produced:
  from          — raw From header (decoded)
  from_email    — extracted email address, lowercase
  from_domain   — domain part of from_email
  to            — raw To header
  subject       — decoded Subject (falls back to markdown H1)
  date          — raw Date header
  category      — Category header (optional)
  source        — file path
  filename      — base filename
"""

from __future__ import annotations

import base64
import re
from pathlib import Path


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)
_RFC2047_RE = re.compile(r"=\?utf-8\?[BQ]\?([A-Za-z0-9+/=_]+)\?=", re.I)


def _decode_rfc2047(s: str) -> str:
    parts = _RFC2047_RE.findall(s)
    if not parts:
        return s
    try:
        return " ".join(
            base64.b64decode(p.replace("_", " ")).decode("utf-8", errors="replace")
            for p in parts
        )
    except Exception:
        return s


def extract(path: Path, text: str) -> dict[str, str]:
    """Email metadata extractor — call signature matches MetadataExtractor protocol."""
    fields: dict[str, str] = {}
    header_text = text[:4096]  # only read headers

    for pattern, key in [
        (r"\*\*From:\*\*\s*(.+)",    "from"),
        (r"\*\*To:\*\*\s*(.+)",      "to"),
        (r"\*\*Subject:\*\*\s*(.+)", "subject"),
        (r"\*\*Date:\*\*\s*(.+)",    "date"),
        (r"\*\*Category:\*\*\s*(.+)","category"),
        (r"(?m)^From:\s*(.+)",       "from"),
        (r"(?m)^To:\s*(.+)",         "to"),
        (r"(?m)^Subject:\s*(.+)",    "subject"),
        (r"(?m)^Date:\s*(.+)",       "date"),
    ]:
        m = re.search(pattern, header_text)
        if m and key not in fields:
            fields[key] = _decode_rfc2047(m.group(1).strip())

    # Subject fallback: markdown H1 "# Email: ..."
    if not fields.get("subject"):
        h1 = re.search(r"^#\s+Email:\s+(.+)", header_text, re.M)
        if h1:
            fields["subject"] = _decode_rfc2047(h1.group(1).strip())

    # Normalise sender
    if "from" in fields:
        em = _EMAIL_RE.search(fields["from"])
        fields["from_email"] = em.group(0).lower() if em else ""
        fields["from_domain"] = fields["from_email"].split("@")[-1] if fields["from_email"] else ""

    fields["source"] = str(path)
    fields["filename"] = path.name
    return fields


# Re-export the CLI email normalizer for a structure-aware engine path without
# duplicating MIME/markdown cleanup logic here.
def clean_email_text(path: Path, raw: str) -> tuple[str, dict[str, str]]:
    from contextfit.cli import _extract_email_text
    text, meta = _extract_email_text(raw, path)
    return text, {str(k): str(v) for k, v in meta.items()}


def chunk_email(path: Path, raw: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """Chunk an email message with header context preserved in each chunk."""
    from contextfit.extractors import document

    text, meta = clean_email_text(path, raw)
    chunks = document.chunk_text(path, text, chunk_size=chunk_size, overlap=overlap)
    for ordinal, chunk in enumerate(chunks):
        chunk["metadata"] = {
            **chunk.get("metadata", {}),
            **meta,
            "chunk_type": "email_message",
            "chunk_ordinal": str(ordinal),
        }
    return chunks
