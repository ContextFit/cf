from contextfit.cli import _discover_files, _preprocess_file
from contextfit.extractors import calendar, email
from contextfit.retrieval.engine import RetrievalEngine


def test_ingest_file_populates_session_id_for_session_retrieval(tmp_path):
    eml = tmp_path / "message.eml"
    eml.write_text(
        "From: Alice <alice@example.com>\n"
        "To: Bob <bob@example.com>\n"
        "Subject: Coffee\n"
        "Date: Wed, 13 May 2026 10:00:00 -0500\n"
        "Content-Type: text/plain; charset=utf-8\n"
        "\n"
        "I switched to a new coffee shop this morning.\n"
    )
    txt = tmp_path / "memo.txt"
    txt.write_text("The new shipment of supplies arrived on Friday.\n")
    engine = RetrievalEngine.create(tmp_path / "kb")

    eml_chunks = engine.ingest_file(eml)
    txt_chunks = engine.ingest_file(txt)

    assert eml_chunks
    assert txt_chunks
    assert all("session_id" in chunk.metadata for chunk in eml_chunks)
    assert all("session_id" in chunk.metadata for chunk in txt_chunks)
    assert {chunk.metadata["session_id"] for chunk in eml_chunks}.isdisjoint(
        {chunk.metadata["session_id"] for chunk in txt_chunks}
    )


def test_email_chunking_preserves_headers_and_body(tmp_path):
    path = tmp_path / "message.eml"
    path.write_text(
        "From: Alice <alice@example.com>\n"
        "To: Bob <bob@example.com>\n"
        "Subject: Project Update\n"
        "Date: Wed, 13 May 2026 10:00:00 -0500\n"
        "Content-Type: text/plain; charset=utf-8\n"
        "\n"
        "The deployment is complete.\n"
    )

    chunks = email.chunk_email(path, path.read_text(), chunk_size=40, overlap=0)

    assert chunks
    assert chunks[0]["metadata"]["chunk_type"] == "email_message"
    assert chunks[0]["metadata"]["subject"] == "Project Update"
    assert "From: Alice" in chunks[0]["text"]
    assert "deployment is complete" in chunks[0]["text"]


def test_ics_chunking_preserves_event_metadata(tmp_path):
    path = tmp_path / "calendar.ics"
    path.write_text(
        "BEGIN:VCALENDAR\n"
        "VERSION:2.0\n"
        "BEGIN:VEVENT\n"
        "UID:event-1\n"
        "SUMMARY:Board Meeting\n"
        "DTSTART:20260513T150000Z\n"
        "DTEND:20260513T160000Z\n"
        "LOCATION:Conference Room\n"
        "ATTENDEE;CN=Alice:mailto:alice@example.com\n"
        "END:VEVENT\n"
        "END:VCALENDAR\n"
    )

    chunks = calendar.chunk_ics(path, path.read_text(), chunk_size=40, overlap=0)

    assert chunks
    assert chunks[0]["metadata"]["chunk_type"] == "ics_events"
    assert chunks[0]["metadata"]["summary"] == "Board Meeting"
    assert "start: 20260513T150000Z" in chunks[0]["text"]
    assert "attendees:" in chunks[0]["text"]


def test_engine_ingest_file_uses_email_and_calendar_chunks(tmp_path):
    eml = tmp_path / "message.eml"
    eml.write_text("From: Alice <alice@example.com>\nSubject: Launch\n\nLaunch note.")
    ics = tmp_path / "calendar.ics"
    ics.write_text("BEGIN:VCALENDAR\nBEGIN:VEVENT\nUID:u1\nSUMMARY:Launch Review\nDTSTART:20260513T150000Z\nEND:VEVENT\nEND:VCALENDAR\n")
    engine = RetrievalEngine.create(tmp_path / "kb")

    email_chunks = engine.ingest_file(eml, chunk_size=30, overlap=0)
    calendar_chunks = engine.ingest_file(ics, chunk_size=30, overlap=0)

    assert email_chunks[0].metadata["chunk_type"] == "email_message"
    assert calendar_chunks[0].metadata["chunk_type"] == "ics_events"


def test_cli_discovers_and_preprocesses_ics(tmp_path):
    path = tmp_path / "calendar.ics"
    path.write_text("BEGIN:VCALENDAR\nBEGIN:VEVENT\nUID:u1\nSUMMARY:Planning\nDTSTART:20260513T150000Z\nEND:VEVENT\nEND:VCALENDAR\n")

    discovered = {p.name for p in _discover_files(tmp_path)}
    result = _preprocess_file(
        path,
        tokenizer_name="cl100k_base",
        chunk_size=30,
        overlap=0,
        max_file_bytes=0,
        max_file_tokens=0,
    )

    assert "calendar.ics" in discovered
    assert result["status"] == "ok"
    assert result["token_items"][0]["metadata"]["chunk_type"] == "ics_events"
