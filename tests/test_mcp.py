from __future__ import annotations

import json
from pathlib import Path

from contextfit.mcp import ContextFitMCPServer, MCPServerConfig
from contextfit.retrieval.engine import RetrievalEngine


def _build_kb(path: Path, text: str | None = None, source: str = "demo-note.md") -> None:
    engine = RetrievalEngine.create(path)
    engine.ingest_text(
        text or "ContextFit connects Claude Desktop to private local memory through MCP.",
        metadata={"source": source, "line_start": 1},
    )
    engine.save(path)


def _call_tool(server: ContextFitMCPServer, name: str, arguments: dict | None = None) -> str:
    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        }
    )
    assert response is not None
    assert response["result"]["isError"] is False
    return response["result"]["content"][0]["text"]


def test_mcp_initialize_and_tool_list(tmp_path: Path) -> None:
    server = ContextFitMCPServer(MCPServerConfig(kb_path=tmp_path / "kb"))

    init = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init is not None
    assert init["result"]["serverInfo"]["name"] == "contextfit"

    tools = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert tools is not None
    search_tool = next(tool for tool in tools["result"]["tools"] if tool["name"] == "contextfit_search")
    assert "extractive" in search_tool["inputSchema"]["properties"]
    tool_names = {tool["name"] for tool in tools["result"]["tools"]}
    assert {
        "contextfit_search",
        "contextfit_get_chunk",
        "contextfit_stats",
        "contextfit_list_vaults",
        "contextfit_search_vault",
        "contextfit_search_all_vaults",
    } <= tool_names


def test_mcp_search_returns_source_backed_chunks(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb"
    _build_kb(kb_path)
    server = ContextFitMCPServer(MCPServerConfig(kb_path=kb_path))

    text = _call_tool(
        server,
        "contextfit_search",
        {"query": "Claude Desktop MCP", "top_k": 1},
    )

    assert "ContextFit search results" in text
    assert "demo-note.md" in text
    assert "Claude Desktop" in text


def test_mcp_search_can_return_extractive_evidence(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb"
    _build_kb(
        kb_path,
        """project_launch[task-01]: status=blocked, target_date=2026-05-25, depends_on=release-01, notes=Waiting for package release
project_launch[task-02]: status=idea, target_date=2026-06-01, depends_on=task-01
""",
        "launch.tmd",
    )
    server = ContextFitMCPServer(MCPServerConfig(kb_path=kb_path))

    text = _call_tool(
        server,
        "contextfit_search",
        {"query": "task-01 blocked package release", "top_k": 1, "extractive": "auto"},
    )

    assert "evidence:" in text
    assert "c[" in text
    assert "e[1] k=row" in text
    assert "r=task-01" in text
    assert "Waiting for package release" in text


def test_mcp_search_accepts_structured_metadata_filters(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb"
    engine = RetrievalEngine.create(kb_path)
    engine.ingest_text(
        "ContextFit metadata prefilter decision from April.",
        metadata={"source": "april.md", "kind": "decision", "date": "2026-04-01"},
    )
    engine.ingest_text(
        "ContextFit metadata prefilter decision from May.",
        metadata={"source": "may.md", "kind": "decision", "date": "2026-05-17"},
    )
    engine.save(kb_path)
    server = ContextFitMCPServer(MCPServerConfig(kb_path=kb_path))

    text = _call_tool(
        server,
        "contextfit_search",
        {
            "query": "ContextFit metadata prefilter",
            "top_k": 3,
            "filters": [
                {"field": "date", "op": "on_or_after", "value": "2026-05-01"},
            ],
        },
    )

    assert "filters:" in text
    assert "may.md" in text
    assert "april.md" not in text


def test_mcp_get_chunk(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb"
    _build_kb(kb_path)
    server = ContextFitMCPServer(MCPServerConfig(kb_path=kb_path))

    text = _call_tool(server, "contextfit_get_chunk", {"chunk_id": 0})

    assert "ContextFit chunk 0" in text
    assert "private local memory" in text


def test_mcp_default_vault_when_registry_missing(tmp_path: Path) -> None:
    kb_path = tmp_path / "kb"
    _build_kb(kb_path)
    server = ContextFitMCPServer(
        MCPServerConfig(kb_path=kb_path, vault_registry=tmp_path / "missing.json")
    )

    vaults = _call_tool(server, "contextfit_list_vaults")
    assert "default" in vaults
    assert str(kb_path) in vaults

    text = _call_tool(
        server,
        "contextfit_search_vault",
        {"vault_name": "default", "query": "Claude Desktop"},
    )
    assert "vault 'default'" in text
    assert "Claude Desktop" in text


def test_mcp_vault_registry_searches_named_vaults(tmp_path: Path) -> None:
    work_kb = tmp_path / "work_kb"
    personal_kb = tmp_path / "personal_kb"
    _build_kb(work_kb, "Work vault contains Acme laptop renewal evidence.", "work.md")
    _build_kb(personal_kb, "Personal vault contains garden zucchini preferences.", "home.md")

    registry = tmp_path / "vaults.json"
    registry.write_text(
        json.dumps(
            {
                "vaults": {
                    "work": {
                        "kb_path": str(work_kb),
                        "description": "Work memory",
                    },
                    "personal": str(personal_kb),
                }
            }
        )
    )
    server = ContextFitMCPServer(MCPServerConfig(kb_path=work_kb, vault_registry=registry))

    vaults = _call_tool(server, "contextfit_list_vaults")
    assert "work" in vaults
    assert "personal" in vaults
    assert "Work memory" in vaults

    work_text = _call_tool(
        server,
        "contextfit_search_vault",
        {"vault_name": "work", "query": "Acme laptop", "top_k": 1},
    )
    assert "work.md" in work_text
    assert "Acme laptop" in work_text

    all_text = _call_tool(
        server,
        "contextfit_search_all_vaults",
        {"query": "vault", "top_k": 1},
    )
    assert "vault 'work'" in all_text
    assert "vault 'personal'" in all_text
