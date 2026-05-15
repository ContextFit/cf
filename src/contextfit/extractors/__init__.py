"""
ContextFit metadata extractors.

Usage:
    from contextfit.extractors import auto, email, document
    from contextfit.extractors.base import null_extractor

    # Pass to ingest:
    engine.ingest_file(path, metadata_extractor=auto.extract)
    engine.ingest_file(path, metadata_extractor=email.extract)

    # Or any callable (path, text) -> dict[str, str]:
    engine.ingest_file(path, metadata_extractor=my_custom_fn)
"""

from contextfit.extractors.base import MetadataExtractor, null_extractor
from contextfit.extractors import auto, document, email

__all__ = [
    "MetadataExtractor",
    "null_extractor",
    "auto",
    "document",
    "email",
]
