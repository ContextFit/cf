from contextfit.cli import _discover_files, _preprocess_file
from contextfit.extractors import code
from contextfit.retrieval.engine import RetrievalEngine


def test_python_code_chunks_by_symbols(tmp_path):
    path = tmp_path / "service.py"
    path.write_text(
        "import os\n\n"
        "class Greeter:\n"
        "    def hello(self, name):\n"
        "        return f'hello {name}'\n\n"
        "def build_message(user):\n"
        "    return Greeter().hello(user)\n"
    )

    chunks = code.chunk_code(path, path.read_text(), chunk_size=80, overlap=0)
    metas = [c["metadata"] for c in chunks]

    assert metas[0]["chunk_type"] == "code_preamble"
    assert metas[0]["language"] == "python"
    assert any(m.get("symbol_name") == "Greeter" and m.get("symbol_kind") == "class" for m in metas)
    assert any(m.get("symbol_name") == "hello" for m in metas)
    assert any(m.get("symbol_name") == "build_message" for m in metas)
    assert all("line_start" in m and "line_end" in m for m in metas)


def test_javascript_code_chunks_functions_and_arrows(tmp_path):
    path = tmp_path / "app.ts"
    path.write_text(
        "import { readFile } from 'node:fs/promises';\n\n"
        "export async function loadConfig(path: string) {\n"
        "  return JSON.parse(await readFile(path, 'utf8'));\n"
        "}\n\n"
        "const normalizeName = (name: string) => name.trim().toLowerCase();\n"
    )

    chunks = code.chunk_code(path, path.read_text(), chunk_size=80, overlap=0)
    symbols = {c["metadata"].get("symbol_name") for c in chunks}

    assert "loadConfig" in symbols
    assert "normalizeName" in symbols
    assert chunks[0]["metadata"]["chunk_type"] == "code_preamble"
    assert chunks[1]["metadata"]["language"] == "typescript"


def test_engine_ingest_file_uses_code_chunks(tmp_path):
    path = tmp_path / "worker.js"
    path.write_text("function runJob(job) {\n  return job.id;\n}\n")
    engine = RetrievalEngine.create(tmp_path / "kb")

    chunks = engine.ingest_file(path, chunk_size=80, overlap=0)

    assert chunks
    assert chunks[0].metadata["chunk_type"] == "code_symbol"
    assert chunks[0].metadata["symbol_name"] == "runJob"


def test_cli_discovers_and_preprocesses_code(tmp_path):
    path = tmp_path / "query.sql"
    path.write_text("CREATE VIEW active_users AS SELECT * FROM users WHERE active = true;\n")

    discovered = {p.name for p in _discover_files(tmp_path)}
    result = _preprocess_file(
        path,
        tokenizer_name="cl100k_base",
        chunk_size=80,
        overlap=0,
        max_file_bytes=0,
        max_file_tokens=0,
    )

    assert "query.sql" in discovered
    assert result["status"] == "ok"
    assert result["token_items"][0]["metadata"]["chunk_type"] == "code_symbol"
    assert result["token_items"][0]["metadata"]["language"] == "sql"
