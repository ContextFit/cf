"""Conversation/session-aware chunking utilities.

These chunkers choose boundaries that preserve dialogue structure before the
retrieval engine encodes final token IDs.  They are deliberately deterministic
and metadata-rich: no LLM calls, no embeddings, no label-dependent fields.
"""

from __future__ import annotations

from typing import Any


def _token_estimate(text: str) -> float:
    """Cheap token estimate used before tokenizer-specific encoding."""
    return len(text.split()) * 1.3


def _turn_text(i: int, turn: dict[str, Any], include_answer_marker: bool = False) -> str:
    role = str(turn.get("role", "unknown"))
    content = str(turn.get("content", ""))
    has_answer = " [HAS_ANSWER]" if include_answer_marker and turn.get("has_answer") else ""
    return f"Turn {i} ({role}){has_answer}: {content}"


def conversation_to_text(
    session_id: str,
    date: str,
    turns: list[dict[str, Any]],
    *,
    include_answer_marker: bool = False,
) -> str:
    """Render a full conversation as a parent/session text node."""
    header = f"Session ID: {session_id}\nDate: {date}"
    lines = [header, ""]
    for i, turn in enumerate(turns, 1):
        lines.append(_turn_text(i, turn, include_answer_marker=include_answer_marker))
    return "\n".join(lines)


def chunk_conversation(
    session_id: str,
    date: str,
    turns: list[dict[str, Any]],
    *,
    chunk_size: int = 512,
    overlap: int = 64,
    base_metadata: dict[str, Any] | None = None,
    include_answer_marker: bool = False,
) -> list[dict[str, Any]]:
    """Chunk a conversation by turn/exchange boundaries.

    Strategy:
    - keep a compact session header in every chunk;
    - keep adjacent user/assistant turns together when they fit;
    - never split a turn unless a single turn exceeds the chunk budget;
    - overlap by whole turns, not raw token tails;
    - attach turn range and role metadata to each chunk.

    `include_answer_marker` defaults to False so benchmark labels such as
    LongMemEval's `has_answer` do not leak into retrieval unless explicitly
    requested for backwards-compatible experiments.
    """
    meta_base = dict(base_metadata or {})
    meta_base.setdefault("session_id", session_id)
    meta_base.setdefault("date", date)
    meta_base.setdefault("kind", "session")

    header = f"Session ID: {session_id}\nDate: {date}"
    budget = max(64.0, chunk_size * 0.85)
    header_est = _token_estimate(header)
    overlap_turns = 1 if overlap > 0 else 0

    units: list[tuple[int, str, str]] = []
    for i, turn in enumerate(turns, 1):
        role = str(turn.get("role", "unknown"))
        text = _turn_text(i, turn, include_answer_marker=include_answer_marker)
        if text.strip():
            units.append((i, role, text))

    if not units:
        return [
            {
                "text": header,
                "metadata": {
                    **meta_base,
                    "chunk_type": "conversation",
                    "turn_start": "",
                    "turn_end": "",
                    "chunk_ordinal": "0",
                },
            }
        ]

    chunks: list[dict[str, Any]] = []
    acc: list[tuple[int, str, str]] = []
    acc_est = header_est
    ordinal = 0

    def emit(blocks: list[tuple[int, str, str]], ordinal_value: int) -> int:
        if not blocks:
            return ordinal_value
        text = header + "\n\n" + "\n".join(block[2] for block in blocks)
        roles = sorted({block[1] for block in blocks})
        chunks.append({
            "text": text,
            "metadata": {
                **meta_base,
                "chunk_type": "conversation_turns",
                "turn_start": str(blocks[0][0]),
                "turn_end": str(blocks[-1][0]),
                "turn_roles": ",".join(roles),
                "chunk_ordinal": str(ordinal_value),
            },
        })
        return ordinal_value + 1

    for unit in units:
        unit_est = _token_estimate(unit[2])
        if acc and acc_est + unit_est > budget:
            ordinal = emit(acc, ordinal)
            acc = acc[-overlap_turns:] if overlap_turns else []
            acc_est = header_est + sum(_token_estimate(u[2]) for u in acc)
        acc.append(unit)
        acc_est += unit_est

    if acc:
        emit(acc, ordinal)

    return chunks
