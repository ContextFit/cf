"""
SMD Extractor: Extract metadata from Story Markdown files.

SMD files are presentation/story artifacts. ContextFit chunks them by scene so
agents can retrieve argument-level units such as claims, beats, asks, and
takeaways without treating the whole deck as plain Markdown.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


SCENE_RE = re.compile(r"^::scene(?:\s+(.*?))?\s*$")
DIRECTIVE_RE = re.compile(r"^::+([A-Za-z][A-Za-z0-9_-]*)(?:\s+(.*?))?\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+\{#[A-Za-z0-9_.:-]+\})?\s*$")


def extract(path: Path, text: str) -> dict[str, str]:
    """Extract deck/story-level metadata from an SMD file."""
    front, body = _split_frontmatter(text)
    lines = body.splitlines()
    metadata: dict[str, str] = {
        "source": str(path),
        "domain": "smd",
    }

    title = front.get("title") or _first_heading(lines)
    if title:
        metadata["title"] = str(title)
    if "type" in front:
        metadata["story_type"] = str(front["type"])
    if "audience" in front:
        metadata["audience"] = str(front["audience"])
    if "audiences" in front:
        audiences = front["audiences"]
        if isinstance(audiences, dict):
            metadata["audiences"] = ", ".join(str(key) for key in audiences.keys())
        else:
            metadata["audiences"] = str(audiences)
    if "render" in front and isinstance(front["render"], dict):
        render = front["render"]
        if "template" in render:
            metadata["render_template"] = str(render["template"])
        if "aspect" in render:
            metadata["render_aspect"] = str(render["aspect"])

    metadata["scene_count"] = str(len(list(_iter_scene_ranges(lines))))
    for name in ("claim", "beat", "evidence", "ask", "takeaway", "agent-notes"):
        metadata[f"{name.replace('-', '_')}_count"] = str(_count_directive(lines, name))

    return metadata


def chunk_smd(path: Path, text: str, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """
    Chunk SMD by scenes.

    Each chunk includes document title/frontmatter context plus one scene. Scene
    metadata preserves stable IDs, roles, reveal IDs, source refs, and line
    ranges for provenance.
    """
    front, body = _split_frontmatter(text)
    lines = body.splitlines()
    base_meta = extract(path, text)
    header = _header_context(front, lines)
    scene_ranges = list(_iter_scene_ranges(lines))
    chunks: list[dict] = []

    for ordinal, scene in enumerate(scene_ranges):
        start, end = scene["start"], scene["end"]
        scene_lines = lines[start:end]
        attrs = _parse_attrs(scene.get("attrs", ""))
        heading = _nearest_heading_before(lines, start)
        scene_text = "\n".join(scene_lines).strip()
        chunk_text = f"{header}\n\n{scene_text}".strip() if header else scene_text
        child_directives = _scene_directives(scene_lines)

        metadata = {
            **base_meta,
            "chunk_type": "smd_scene",
            "chunk_ordinal": ordinal,
            "scene_id": str(attrs.get("id", "")),
            "scene_role": str(attrs.get("role", "")),
            "scene_title": heading.get("title", ""),
            "heading_path": heading.get("title", ""),
            "line_start": start + 1,
            "line_end": end,
            "chunk_context_line_count": header.count("\n") + 2 if header else 0,
        }

        ids = [str(item["attrs"]["id"]) for item in child_directives if "id" in item["attrs"]]
        roles = [str(item["attrs"]["role"]) for item in child_directives if "role" in item["attrs"]]
        sources = [
            str(item["attrs"][key])
            for item in child_directives
            for key in ("source", "sources")
            if key in item["attrs"]
        ]
        reveals = [
            str(item["attrs"]["reveal"])
            for item in child_directives
            if "reveal" in item["attrs"]
        ]
        directive_names = sorted({str(item["name"]) for item in child_directives})

        if ids:
            metadata["story_atom_ids"] = ids
        if roles:
            metadata["story_roles"] = sorted(set(roles))
        if sources:
            metadata["story_sources"] = sources
        if reveals:
            metadata["story_reveals"] = reveals
        if directive_names:
            metadata["story_directives"] = directive_names

        chunks.append({"text": chunk_text, "metadata": metadata})

    line_count = len(text.splitlines()) or 1
    return chunks if chunks else [{"text": text, "metadata": {**base_meta, "line_start": 1, "line_end": line_count}}]


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    raw = text[4:end].strip()
    body = text[end + 4 :].lstrip("\n")
    try:
        data = yaml.safe_load(raw) if raw else {}
    except yaml.YAMLError:
        return {}, text
    return data if isinstance(data, dict) else {}, body


def _header_context(front: dict[str, Any], lines: list[str]) -> str:
    parts: list[str] = []
    if front:
        title = front.get("title")
        story_type = front.get("type")
        if title:
            parts.append(f"Story: {title}")
        if story_type:
            parts.append(f"Type: {story_type}")
    first_heading = _first_heading(lines)
    if first_heading and first_heading not in "\n".join(parts):
        parts.append(f"# {first_heading}")
    return "\n".join(parts)


def _first_heading(lines: list[str]) -> str | None:
    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            return match.group(2).strip()
    return None


def _iter_scene_ranges(lines: list[str]) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for idx, line in enumerate(lines):
        match = SCENE_RE.match(line)
        if not match:
            continue
        if current is not None:
            current["end"] = _trim_trailing_heading(lines, current["start"], idx)
            scenes.append(current)
        current = {"start": idx, "attrs": match.group(1) or ""}
    if current is not None:
        current["end"] = len(lines)
        scenes.append(current)
    return scenes


def _trim_trailing_heading(lines: list[str], start: int, end: int) -> int:
    trimmed = end
    while trimmed > start and lines[trimmed - 1].strip() == "":
        trimmed -= 1
    if trimmed > start and HEADING_RE.match(lines[trimmed - 1]):
        trimmed -= 1
    return trimmed


def _nearest_heading_before(lines: list[str], idx: int) -> dict[str, str]:
    for cursor in range(idx - 1, -1, -1):
        match = HEADING_RE.match(lines[cursor])
        if match:
            return {"title": match.group(2).strip(), "line": str(cursor + 1)}
    return {}


def _scene_directives(lines: list[str]) -> list[dict[str, Any]]:
    directives = []
    for line in lines:
        match = DIRECTIVE_RE.match(line)
        if not match:
            continue
        name, raw_attrs = match.groups()
        if name == "scene":
            continue
        directives.append({"name": name, "attrs": _parse_attrs(raw_attrs or "")})
    return directives


def _count_directive(lines: list[str], name: str) -> int:
    return sum(1 for line in lines if re.match(rf"^::+{re.escape(name)}(?:\s|$)", line))


def _parse_attrs(raw: str) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for token in raw.split():
        if "=" not in token:
            attrs[token] = True
            continue
        key, value = token.split("=", 1)
        attrs[key] = value.strip("\"'")
    return attrs
