import json

from contextfit.cli import _discover_files, _preprocess_file
from contextfit.extractors import structured
from contextfit.retrieval.engine import RetrievalEngine


def test_json_chunking_preserves_object_records(tmp_path):
    path = tmp_path / "events.json"
    path.write_text(json.dumps({"events": [
        {"id": "e1", "user": "Alice", "action": "created invoice"},
        {"id": "e2", "user": "Bob", "action": "approved invoice"},
    ]}))

    chunks = structured.chunk_json(path, path.read_text(), chunk_size=30, overlap=0)

    assert chunks
    assert chunks[0]["metadata"]["chunk_type"] == "json_records"
    assert "events[0]" in chunks[0]["text"]
    assert "user: Alice" in chunks[0]["text"]


def test_jsonl_chunking_preserves_line_metadata(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"id":"m1","text":"first message"}\n{"id":"m2","text":"second message"}\n')

    chunks = structured.chunk_jsonl(path, path.read_text(), chunk_size=20, overlap=0)

    assert chunks[0]["metadata"]["chunk_type"] == "jsonl_records"
    assert "line 1" in chunks[0]["text"]
    assert "first message" in chunks[0]["text"]


def test_csv_and_tsv_chunking_preserve_rows(tmp_path):
    csv_path = tmp_path / "people.csv"
    csv_path.write_text("name,email,role\nAlice,alice@example.com,Engineer\nBob,bob@example.com,Designer\n")
    tsv_path = tmp_path / "people.tsv"
    tsv_path.write_text("name\temail\trole\nCarol\tcarol@example.com\tPM\n")

    csv_chunks = structured.chunk_delimited(csv_path, csv_path.read_text(), chunk_size=20, overlap=0, delimiter=",")
    tsv_chunks = structured.chunk_delimited(tsv_path, tsv_path.read_text(), chunk_size=20, overlap=0, delimiter="\t")

    assert csv_chunks[0]["metadata"]["chunk_type"] == "csv_rows"
    assert "email: alice@example.com" in csv_chunks[0]["text"]
    assert tsv_chunks[0]["metadata"]["chunk_type"] == "tsv_rows"
    assert "role: PM" in tsv_chunks[0]["text"]


def test_engine_ingest_file_uses_structured_data_chunks(tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("name,email\nAlice,alice@example.com\n")
    engine = RetrievalEngine.create(tmp_path / "kb")

    chunks = engine.ingest_file(path, chunk_size=20, overlap=0)

    assert chunks
    assert chunks[0].metadata["chunk_type"] == "csv_rows"
    assert chunks[0].metadata.get("record_count") == "1"


def test_cli_discovers_and_preprocesses_structured_files(tmp_path):
    (tmp_path / "data.jsonl").write_text('{"id":"x","value":"needle"}\n')
    (tmp_path / "table.csv").write_text("id,value\na,needle\n")

    discovered = {p.name for p in _discover_files(tmp_path)}
    result = _preprocess_file(
        tmp_path / "data.jsonl",
        tokenizer_name="cl100k_base",
        chunk_size=20,
        overlap=0,
        max_file_bytes=0,
        max_file_tokens=0,
    )

    assert {"data.jsonl", "table.csv"}.issubset(discovered)
    assert result["status"] == "ok"
    assert result["token_items"][0]["metadata"]["chunk_type"] == "jsonl_records"
