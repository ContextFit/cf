"""Structured query specifications for agent-facing retrieval.

These types intentionally do not plan queries with an LLM. They define the
deterministic contract an agent, MCP host, CLI, or local planner can pass into
ContextFit before token-native retrieval runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any


@dataclass(frozen=True)
class MetadataPredicate:
    """One structured metadata predicate.

    Operators:
      - contains: case-insensitive substring/token match
      - exact: full field equality
      - in: exact match against one of several values
      - gt/gte/lt/lte: numeric, ISO-date, or lexical comparison
      - after/before/on_or_after/on_or_before: aliases for range comparisons
      - exists: field is present and non-empty
    """

    field: str
    op: str = "contains"
    value: Any | None = None
    values: list[Any] = dc_field(default_factory=list)

    @classmethod
    def from_obj(cls, obj: MetadataPredicate | dict[str, Any]) -> MetadataPredicate:
        if isinstance(obj, MetadataPredicate):
            return obj
        if not isinstance(obj, dict):
            raise TypeError("metadata filter predicates must be dictionaries")

        field_name = str(obj.get("field") or "").strip()
        if not field_name:
            raise ValueError("metadata filter predicate is missing field")

        raw_values = obj.get("values")
        values = list(raw_values) if isinstance(raw_values, list) else []
        return cls(
            field=field_name,
            op=str(obj.get("op") or obj.get("mode") or "contains").strip().lower(),
            value=obj.get("value"),
            values=values,
        )


@dataclass(frozen=True)
class QuerySpec:
    """Agent-facing retrieval request with optional structured filters."""

    query: str
    filters: list[MetadataPredicate] = dc_field(default_factory=list)
    filter_mode: str = "and"
    min_filter_matches: int = 0
    filter_pushdown_threshold: float = 0.50

    @classmethod
    def from_obj(cls, obj: QuerySpec | dict[str, Any]) -> QuerySpec:
        if isinstance(obj, QuerySpec):
            return obj
        if not isinstance(obj, dict):
            raise TypeError("query_spec must be a dictionary")

        query = str(
            obj.get("query")
            or obj.get("semantic_query")
            or obj.get("natural_language_query")
            or ""
        ).strip()
        if not query:
            raise ValueError("query_spec is missing query")

        raw_filters = obj.get("filters") or obj.get("metadata_filters") or []
        if isinstance(raw_filters, dict):
            raw_filters = _filters_from_mapping(raw_filters)
        if not isinstance(raw_filters, list):
            raise ValueError("query_spec filters must be a list or object")

        return cls(
            query=query,
            filters=[MetadataPredicate.from_obj(f) for f in raw_filters],
            filter_mode=str(obj.get("filter_mode") or obj.get("combine") or "and").lower(),
            min_filter_matches=int(obj.get("min_filter_matches") or 0),
            filter_pushdown_threshold=float(obj.get("filter_pushdown_threshold") or 0.50),
        )


def _filters_from_mapping(filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert concise MCP-style filter maps into predicate objects."""
    out: list[dict[str, Any]] = []
    for field_name, raw in filters.items():
        if isinstance(raw, dict):
            if "field" in raw:
                out.append(dict(raw))
            else:
                for op, value in raw.items():
                    out.append({"field": field_name, "op": op, "value": value})
        elif isinstance(raw, list):
            out.append({"field": field_name, "op": "in", "values": raw})
        else:
            out.append({"field": field_name, "op": "exact", "value": raw})
    return out


__all__ = ["MetadataPredicate", "QuerySpec"]
