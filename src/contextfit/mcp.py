"""Minimal MCP stdio server for Claude Desktop.

This module intentionally avoids a runtime dependency on the Python `mcp` package.
Claude Desktop speaks JSON-RPC over stdio; the small subset implemented here is
sufficient for tool discovery and tool calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from contextfit.retrieval.engine import RetrievalEngine
from contextfit.retrieval.extractive import EvidenceItem, extract_evidence, format_evidence_compact

MCP_PROTOCOL_VERSION = "2024-11-05"
DEFAULT_VAULT_REGISTRY = Path("~/.contextfit/vaults.json")
METHODS = {"exact", "bm25", "sid", "graph", "hierarchy", "hybrid"}


@dataclass(frozen=True)
class VaultConfig:
    """One named local ContextFit knowledge base."""

    name: str
    kb_path: Path
    description: str = ""


@dataclass
class MCPServerConfig:
    kb_path: Path
    tokenizer: str = "cl100k_base"
    default_top_k: int = 5
    default_method: str = "hybrid"
    max_preview_chars: int = 1200
    extractive: str = "auto"
    max_evidence_chars: int = 1200
    vault_registry: Path | None = DEFAULT_VAULT_REGISTRY


class ContextFitMCPServer:
    """Serve ContextFit retrieval as Claude Desktop MCP tools."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._engine: RetrievalEngine | None = None
        self._engines: dict[str, RetrievalEngine] = {}
        self._vaults: dict[str, VaultConfig] | None = None

    @property
    def engine(self) -> RetrievalEngine:
        if self._engine is None:
            self._engine = self._load_engine(self.config.kb_path)
        return self._engine

    @property
    def vaults(self) -> dict[str, VaultConfig]:
        if self._vaults is None:
            self._vaults = self._load_vaults()
        return self._vaults

    def _load_engine(self, kb_path: Path) -> RetrievalEngine:
        return RetrievalEngine.load(kb_path, tokenizer_name=self.config.tokenizer)

    def _engine_for_vault(self, vault_name: str) -> RetrievalEngine:
        vault = self._resolve_vault(vault_name)
        if vault.name not in self._engines:
            self._engines[vault.name] = self._load_engine(vault.kb_path)
        return self._engines[vault.name]

    def _resolve_vault(self, vault_name: str) -> VaultConfig:
        name = str(vault_name or "").strip()
        if not name:
            raise ValueError("vault_name must be provided")
        vault = self.vaults.get(name)
        if vault is None:
            available = ", ".join(sorted(self.vaults)) or "none"
            raise ValueError(f"unknown vault '{name}'. Available vaults: {available}")
        return vault

    def _load_vaults(self) -> dict[str, VaultConfig]:
        registry_path = self.config.vault_registry
        if registry_path is not None:
            registry_path = registry_path.expanduser()
        if registry_path is not None and registry_path.exists():
            return _load_vault_registry(registry_path)
        return {
            "default": VaultConfig(
                name="default",
                kb_path=self.config.kb_path.expanduser().resolve(),
                description="Default ContextFit knowledge base from --kb.",
            )
        }

    def serve(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self.handle_request(request)
            except Exception as exc:  # pragma: no cover - defensive stdio server boundary
                response = self._error_response(None, -32603, f"Internal error: {exc}")
                print(traceback.format_exc(), file=sys.stderr, flush=True)

            if response is not None:
                stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
                stdout.flush()

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        # MCP notifications do not have an id and should not receive a response.
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}

        if method == "initialize":
            return self._result_response(
                request_id,
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "contextfit", "version": "0.1.1"},
                },
            )
        if method == "notifications/initialized":
            return None
        if method == "ping":
            return self._result_response(request_id, {})
        if method == "tools/list":
            return self._result_response(request_id, {"tools": self._tools()})
        if method == "tools/call":
            return self._handle_tool_call(request_id, params)

        if request_id is None:
            return None
        return self._error_response(request_id, -32601, f"Method not found: {method}")

    def _handle_tool_call(self, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        try:
            if name == "contextfit_search":
                result_text = self.search(**arguments)
            elif name == "contextfit_get_chunk":
                result_text = self.get_chunk(**arguments)
            elif name == "contextfit_stats":
                result_text = self.stats()
            elif name == "contextfit_list_vaults":
                result_text = self.list_vaults()
            elif name == "contextfit_search_vault":
                result_text = self.search_vault(**arguments)
            elif name == "contextfit_search_all_vaults":
                result_text = self.search_all_vaults(**arguments)
            else:
                return self._error_response(request_id, -32602, f"Unknown tool: {name}")
            return self._result_response(
                request_id,
                {"content": [{"type": "text", "text": result_text}], "isError": False},
            )
        except Exception as exc:
            return self._result_response(
                request_id,
                {
                    "content": [
                        {"type": "text", "text": f"ContextFit tool error: {exc}"}
                    ],
                    "isError": True,
                },
            )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        method: str | None = None,
        max_tokens: int = 4096,
        extractive: str | None = None,
        filters: list[dict[str, Any]] | dict[str, Any] | None = None,
        filter_mode: str = "and",
        min_filter_matches: int = 0,
        filter_pushdown_threshold: float = 0.50,
        query_spec: dict[str, Any] | None = None,
    ) -> str:
        return self._search_engine(
            engine=self.engine,
            kb_path=self.config.kb_path,
            query=query,
            top_k=top_k,
            method=method,
            max_tokens=max_tokens,
            vault_name=None,
            extractive=extractive,
            filters=filters,
            filter_mode=filter_mode,
            min_filter_matches=min_filter_matches,
            filter_pushdown_threshold=filter_pushdown_threshold,
            query_spec=query_spec,
        )

    def search_vault(
        self,
        vault_name: str,
        query: str,
        top_k: int | None = None,
        method: str | None = None,
        max_tokens: int = 4096,
        extractive: str | None = None,
        filters: list[dict[str, Any]] | dict[str, Any] | None = None,
        filter_mode: str = "and",
        min_filter_matches: int = 0,
        filter_pushdown_threshold: float = 0.50,
        query_spec: dict[str, Any] | None = None,
    ) -> str:
        vault = self._resolve_vault(vault_name)
        engine = self._engine_for_vault(vault.name)
        return self._search_engine(
            engine=engine,
            kb_path=vault.kb_path,
            query=query,
            top_k=top_k,
            method=method,
            max_tokens=max_tokens,
            vault_name=vault.name,
            extractive=extractive,
            filters=filters,
            filter_mode=filter_mode,
            min_filter_matches=min_filter_matches,
            filter_pushdown_threshold=filter_pushdown_threshold,
            query_spec=query_spec,
        )

    def search_all_vaults(
        self,
        query: str,
        top_k: int | None = None,
        method: str | None = None,
        max_tokens: int = 4096,
        extractive: str | None = None,
        filters: list[dict[str, Any]] | dict[str, Any] | None = None,
        filter_mode: str = "and",
        min_filter_matches: int = 0,
        filter_pushdown_threshold: float = 0.50,
        query_spec: dict[str, Any] | None = None,
    ) -> str:
        if not self.vaults:
            return "No ContextFit vaults are registered."
        lines = [
            f"ContextFit search across {len(self.vaults)} vault(s) for: {query}",
            "",
        ]
        for vault in self.vaults.values():
            lines.append(
                self.search_vault(
                    vault.name,
                    query,
                    top_k=top_k,
                    method=method,
                    max_tokens=max_tokens,
                    extractive=extractive,
                    filters=filters,
                    filter_mode=filter_mode,
                    min_filter_matches=min_filter_matches,
                    filter_pushdown_threshold=filter_pushdown_threshold,
                    query_spec=query_spec,
                )
            )
            lines.append("\n---\n")
        return "\n".join(lines).rstrip()

    def _search_engine(
        self,
        engine: RetrievalEngine,
        kb_path: Path,
        query: str,
        top_k: int | None,
        method: str | None,
        max_tokens: int,
        vault_name: str | None,
        extractive: str | None = None,
        filters: list[dict[str, Any]] | dict[str, Any] | None = None,
        filter_mode: str = "and",
        min_filter_matches: int = 0,
        filter_pushdown_threshold: float = 0.50,
        query_spec: dict[str, Any] | None = None,
    ) -> str:
        if query_spec and isinstance(query_spec, dict):
            query = str(
                query_spec.get("query")
                or query_spec.get("semantic_query")
                or query
                or ""
            )
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        top_k = int(top_k or self.config.default_top_k)
        method = method or self.config.default_method
        if method not in METHODS:
            raise ValueError(
                "method must be one of exact, bm25, sid, graph, hierarchy, hybrid"
            )

        result = engine.query(
            query,
            top_k=top_k,
            method=method,
            max_tokens=int(max_tokens),
            filters=filters,
            filter_mode=filter_mode,
            min_filter_matches=min_filter_matches,
            filter_pushdown_threshold=filter_pushdown_threshold,
            query_spec=query_spec,
        )
        title = f"ContextFit search results for: {query}"
        if vault_name:
            title = f"ContextFit search results in vault '{vault_name}' for: {query}"
        lines = [
            title,
            f"KB: {kb_path}",
            (
                f"Method: {result.method}; results: {len(result.chunks)}; "
                f"input tokens: {len(result.input_ids)}"
            ),
            "",
        ]
        if result.filter_trace:
            lines.extend(
                [
                    f"filters: {_compact_json(result.filter_trace)}",
                    "",
                ]
            )
        for rank, (chunk, score) in enumerate(
            zip(result.chunks, result.scores, strict=False),
            start=1,
        ):
            full_text = engine.tokenizer.decode(chunk.tokens.tolist())
            mode = extractive or self.config.extractive
            evidence: list[EvidenceItem] = []
            if mode != "none":
                evidence = extract_evidence(
                    full_text,
                    query,
                    mode=mode,  # type: ignore[arg-type]
                    max_chars=self.config.max_evidence_chars,
                    metadata=chunk.metadata,
                )
            preview = _trim(full_text, self.config.max_preview_chars)
            lines.extend(
                [
                    (
                        f"[{rank}] chunk_id={chunk.chunk_id} "
                        f"score={score:.4f} tokens={chunk.token_count}"
                    ),
                    f"source: {_source_label(chunk.metadata)}",
                    f"metadata: {_compact_json(chunk.metadata)}",
                    "evidence:" if evidence else "text:",
                    _compact_text_only(
                        format_evidence_compact(
                            evidence,
                            metadata=chunk.metadata,
                            chunk_id=chunk.chunk_id,
                            chunk_score=float(score),
                        )
                    ) if evidence else preview,
                    "",
                ]
            )
        if not result.chunks:
            lines.append("No matching chunks found.")
        return "\n".join(lines).rstrip()

    def get_chunk(self, chunk_id: int) -> str:
        chunk = self.engine.store.get(int(chunk_id))
        if chunk is None:
            raise ValueError(f"chunk not found: {chunk_id}")
        text = self.engine.tokenizer.decode(chunk.tokens.tolist())
        return "\n".join(
            [
                f"ContextFit chunk {chunk.chunk_id}",
                f"source: {_source_label(chunk.metadata)}",
                f"tokens: {chunk.token_count}",
                f"metadata: {_compact_json(chunk.metadata)}",
                "",
                text,
            ]
        )

    def stats(self) -> str:
        stats = self.engine.stats()
        return "ContextFit knowledge base stats:\n" + json.dumps(
            stats,
            indent=2,
            sort_keys=True,
        )

    def list_vaults(self) -> str:
        lines = ["ContextFit registered vaults:"]
        for vault in self.vaults.values():
            desc = f" — {vault.description}" if vault.description else ""
            lines.append(f"- {vault.name}: {vault.kb_path}{desc}")
        return "\n".join(lines)

    def _tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "contextfit_search",
                "description": (
                    "Search the default local ContextFit knowledge base and return "
                    "source-backed chunks for Claude to use as private context."
                ),
                "inputSchema": _search_schema(required=["query"]),
            },
            {
                "name": "contextfit_get_chunk",
                "description": (
                    "Fetch the full text and metadata for a specific ContextFit "
                    "chunk id from the default knowledge base."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "integer",
                            "description": "ContextFit chunk id.",
                        }
                    },
                    "required": ["chunk_id"],
                },
            },
            {
                "name": "contextfit_stats",
                "description": (
                    "Show basic stats for the default local ContextFit knowledge base."
                ),
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "contextfit_list_vaults",
                "description": "List registered local ContextFit vaults Claude can search.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "contextfit_search_vault",
                "description": "Search one named local ContextFit vault from the registry.",
                "inputSchema": _search_schema(
                    extra_properties={
                        "vault_name": {
                            "type": "string",
                            "description": "Registered vault name, e.g. work or personal.",
                        }
                    },
                    required=["vault_name", "query"],
                ),
            },
            {
                "name": "contextfit_search_all_vaults",
                "description": "Search every registered local ContextFit vault.",
                "inputSchema": _search_schema(required=["query"]),
            },
        ]

    @staticmethod
    def _result_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _load_vault_registry(path: Path) -> dict[str, VaultConfig]:
    raw = json.loads(path.read_text())
    vault_items = raw.get("vaults", raw)
    if not isinstance(vault_items, dict):
        raise ValueError("vault registry must contain an object named 'vaults'")

    vaults: dict[str, VaultConfig] = {}
    for name, entry in vault_items.items():
        if isinstance(entry, str):
            kb_path = entry
            description = ""
        elif isinstance(entry, dict):
            kb_path = entry.get("kb_path") or entry.get("kb") or entry.get("path")
            description = str(entry.get("description", ""))
        else:
            raise ValueError(f"invalid vault registry entry for '{name}'")
        if not kb_path:
            raise ValueError(f"vault '{name}' is missing kb_path")
        vaults[str(name)] = VaultConfig(
            name=str(name),
            kb_path=Path(str(kb_path)).expanduser().resolve(),
            description=description,
        )
    return vaults


def _search_schema(
    extra_properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "query": {
            "type": "string",
            "description": "Natural-language search query.",
        },
        "top_k": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
            "description": "Number of chunks to return. Defaults to 5.",
        },
        "method": {
            "type": "string",
            "enum": ["exact", "bm25", "sid", "graph", "hierarchy", "hybrid"],
            "description": "Retrieval method. Defaults to hybrid.",
        },
        "max_tokens": {
            "type": "integer",
            "minimum": 256,
            "maximum": 32768,
            "description": "Maximum retrieved token budget. Defaults to 4096.",
        },
        "extractive": {
            "type": "string",
            "enum": ["none", "spans", "rows", "bullets", "auto"],
            "description": (
                "Return deterministic query-focused evidence instead of full "
                "previews. Defaults to auto."
            ),
        },
        "filters": {
            "description": (
                "Optional structured metadata prefilters. Use either an array of "
                "{field, op, value|values} predicates or an object mapping fields "
                "to values/range operators. Retrieval remains token-native after "
                "these deterministic filters are applied."
            ),
            "oneOf": [
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "op": {
                                "type": "string",
                                "enum": [
                                    "contains",
                                    "exact",
                                    "in",
                                    "gt",
                                    "gte",
                                    "lt",
                                    "lte",
                                    "after",
                                    "before",
                                    "on_or_after",
                                    "on_or_before",
                                    "exists",
                                ],
                            },
                            "value": {},
                            "values": {"type": "array", "items": {}},
                        },
                        "required": ["field"],
                    },
                },
                {"type": "object"},
            ],
        },
        "filter_mode": {
            "type": "string",
            "enum": ["and", "or"],
            "description": "How to combine metadata filters. Defaults to and.",
        },
        "min_filter_matches": {
            "type": "integer",
            "minimum": 0,
            "description": (
                "Broaden by ignoring structured filters if they match fewer "
                "chunks than this threshold."
            ),
        },
        "filter_pushdown_threshold": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": (
                "Use post-filtering instead of filter pushdown when filters "
                "match more than this corpus fraction. Defaults to 0.5."
            ),
        },
        "query_spec": {
            "type": "object",
            "description": (
                "Optional agent-produced query spec containing query/semantic_query, "
                "filters, filter_mode, and min_filter_matches."
            ),
        },
    }
    if extra_properties:
        properties.update(extra_properties)
    return {"type": "object", "properties": properties, "required": required or []}


def _compact_text_only(value: str | tuple[str, dict[str, Any]]) -> str:
    if isinstance(value, tuple):
        return value[0]
    return value


def _format_evidence(items: list[EvidenceItem]) -> str:
    lines: list[str] = []
    for item in items:
        loc = ""
        if item.row_id:
            loc = f" row={item.row_id}"
        elif item.line_start is not None:
            loc = f" lines={item.line_start}"
            if item.line_end and item.line_end != item.line_start:
                loc += f"-{item.line_end}"
        lines.append(f"- [{item.kind}{loc}] {item.text}")
    return "\n".join(lines)


def _source_label(metadata: dict[str, Any]) -> str:
    for key in ("source", "path", "file", "file_path", "email_path", "uri"):
        value = metadata.get(key)
        if value:
            label = str(value)
            break
    else:
        label = "unknown"

    line_bits = []
    for key in ("line", "line_start", "start_line"):
        if metadata.get(key):
            line_bits.append(str(metadata[key]))
            break
    for key in ("line_end", "end_line"):
        if metadata.get(key):
            line_bits.append(str(metadata[key]))
            break
    if line_bits:
        label += ":" + "-".join(line_bits)
    return label


def _compact_json(value: Any, max_chars: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _trim(text, max_chars)


def _trim(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run ContextFit as a Claude Desktop MCP stdio server"
    )
    parser.add_argument("--kb", "-k", default="./contextfit_kb", help="Knowledge base path")
    parser.add_argument(
        "--vault-registry",
        default=str(DEFAULT_VAULT_REGISTRY),
        help="JSON registry of named vaults (default: ~/.contextfit/vaults.json)",
    )
    parser.add_argument("--tokenizer", "-t", default="cl100k_base", help="Tokenizer name")
    parser.add_argument("--top-k", type=int, default=5, help="Default search result count")
    parser.add_argument(
        "--method",
        default="hybrid",
        choices=["exact", "bm25", "sid", "graph", "hierarchy", "hybrid"],
        help="Default retrieval method",
    )
    parser.add_argument(
        "--max-preview-chars",
        type=int,
        default=1200,
        help="Maximum characters per search preview",
    )
    args = parser.parse_args(argv)

    server = ContextFitMCPServer(
        MCPServerConfig(
            kb_path=Path(args.kb).expanduser().resolve(),
            tokenizer=args.tokenizer,
            default_top_k=args.top_k,
            default_method=args.method,
            max_preview_chars=args.max_preview_chars,
            vault_registry=Path(args.vault_registry).expanduser(),
        )
    )
    server.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
