"""Tests for machine-readable CLI output."""

import json
import subprocess
import sys
from pathlib import Path


def run_contextfit(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "contextfit.cli", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def test_cli_query_json_and_stats_json(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("Python async await coroutine task.\n")
    kb = tmp_path / "kb"

    run_contextfit(
        ["--kb", str(kb), "ingest", str(docs), "--train-sid-generator"],
        cwd=repo,
    )

    query_proc = run_contextfit(
        ["--kb", str(kb), "query", "python async", "--method", "sid", "--json"],
        cwd=repo,
    )
    payload = json.loads(query_proc.stdout)

    assert payload["query"] == "python async"
    assert payload["method"] == "sid"
    assert payload["retrieved_chunks"] == 1
    assert payload["input_ids"]
    assert payload["sid_predictions"]
    assert payload["chunks"][0]["chunk_id"] == 0
    assert payload["chunks"][0]["semantic_id"]
    assert payload["chunks"][0]["tokens"] == payload["input_ids"]

    stats_proc = run_contextfit(["--kb", str(kb), "stats", "--json"], cwd=repo)
    stats = json.loads(stats_proc.stdout)
    assert stats["chunks"] == 1
    assert stats["semantic_ids"]["sid_count"] == 1
    assert stats["learned_sid_generator"]["trained_chunks"] == 1
    assert stats["manifest"]["phases"]["ingest_complete"] is True


def test_cli_deferred_ingest_build_index_and_train_sid(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("Rust ownership borrowing lifetimes memory safety.\n")
    kb = tmp_path / "kb"

    run_contextfit(
        ["--kb", str(kb), "ingest", str(docs), "--defer-index-build", "--checkpoint-every", "1"],
        cwd=repo,
    )
    run_contextfit(["--kb", str(kb), "build-index"], cwd=repo)
    run_contextfit(["--kb", str(kb), "train-sid"], cwd=repo)

    query_proc = run_contextfit(
        ["--kb", str(kb), "query", "rust borrowing", "--method", "sid", "--json"],
        cwd=repo,
    )
    payload = json.loads(query_proc.stdout)
    assert payload["retrieved_chunks"] == 1

    stats_proc = run_contextfit(["--kb", str(kb), "stats", "--json"], cwd=repo)
    stats = json.loads(stats_proc.stdout)
    assert stats["manifest"]["phases"]["index_built"] is True
    assert stats["manifest"]["phases"]["sid_trained"] is True


def test_cli_vault_search_all_and_chunk_json(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text("Acme renewal notes mention laptop inventory and approvals.\n")
    kb = tmp_path / "kb"
    registry = tmp_path / "vaults.json"

    run_contextfit(["--kb", str(kb), "ingest", str(docs)], cwd=repo)
    registry.write_text(
        json.dumps(
            {
                "vaults": {
                    "work-demo": {
                        "kb_path": str(kb),
                        "description": "Fictional work demo vault",
                    }
                }
            }
        )
    )

    vaults_proc = run_contextfit(
        ["vaults", "list", "--vault-registry", str(registry), "--json"],
        cwd=repo,
    )
    vaults = json.loads(vaults_proc.stdout)
    assert vaults["vaults"][0]["name"] == "work-demo"

    search_proc = run_contextfit(
        [
            "search",
            "Acme laptop",
            "--vault",
            "work-demo",
            "--vault-registry",
            str(registry),
            "--json",
        ],
        cwd=repo,
    )
    search_payload = json.loads(search_proc.stdout)
    assert search_payload["vault"] == "work-demo"
    assert search_payload["retrieved_chunks"] == 1

    evidence_proc = run_contextfit(
        [
            "search",
            "Acme laptop approvals",
            "--vault",
            "work-demo",
            "--vault-registry",
            str(registry),
            "--json",
            "--extractive",
            "auto",
            "--max-evidence-chars",
            "200",
        ],
        cwd=repo,
    )
    evidence_payload = json.loads(evidence_proc.stdout)
    assert evidence_payload["chunks"][0]["evidence"]
    assert "Acme renewal" in evidence_payload["chunks"][0]["evidence"][0]["text"]

    compact_proc = run_contextfit(
        [
            "search",
            "Acme laptop approvals",
            "--vault",
            "work-demo",
            "--vault-registry",
            str(registry),
            "--json",
            "--extractive",
            "auto",
            "--compact",
        ],
        cwd=repo,
    )
    compact_payload = json.loads(compact_proc.stdout)
    assert compact_payload["compact_context"].startswith("c[")
    assert "e[1] k=bul" in compact_payload["compact_context"]
    assert "metadata" not in compact_payload["chunks"][0]
    assert "tokens" not in compact_payload["chunks"][0]

    handles_proc = run_contextfit(
        [
            "search",
            "Acme laptop approvals",
            "--vault",
            "work-demo",
            "--vault-registry",
            str(registry),
            "--json",
            "--extractive",
            "auto",
            "--compact",
            "--citation-mode",
            "handles",
            "--reference-ttl-seconds",
            "60",
        ],
        cwd=repo,
    )
    handles_payload = json.loads(handles_proc.stdout)
    assert handles_payload["compact_context"].startswith("@r1 ")
    assert handles_payload["references"]["@r1"]["source"].endswith("note.md")
    assert "expires_at" in handles_payload["references"]["@r1"]

    all_proc = run_contextfit(
        ["search-all", "inventory", "--vault-registry", str(registry), "--json"],
        cwd=repo,
    )
    all_payload = json.loads(all_proc.stdout)
    assert all_payload["vaults"][0]["vault"] == "work-demo"

    chunk_proc = run_contextfit(
        [
            "chunk",
            "0",
            "--vault",
            "work-demo",
            "--vault-registry",
            str(registry),
            "--json",
        ],
        cwd=repo,
    )
    chunk_payload = json.loads(chunk_proc.stdout)
    assert chunk_payload["vault"] == "work-demo"
    assert "Acme renewal" in chunk_payload["text"]
