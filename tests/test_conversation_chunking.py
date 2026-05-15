from contextfit.extractors.conversation import chunk_conversation, conversation_to_text
from contextfit.retrieval.engine import RetrievalEngine


def test_conversation_chunking_preserves_turn_metadata():
    turns = [
        {"role": "user", "content": "I like jazz."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Remind me tomorrow."},
    ]

    chunks = chunk_conversation("s1", "2026-05-12", turns, chunk_size=20, overlap=0)

    assert chunks
    assert all(c["metadata"]["chunk_type"] == "conversation_turns" for c in chunks)
    assert chunks[0]["metadata"]["session_id"] == "s1"
    assert chunks[0]["metadata"]["turn_start"] == "1"
    assert "Session ID: s1" in chunks[0]["text"]
    assert "I like jazz" in chunks[0]["text"]


def test_conversation_chunking_does_not_leak_answer_marker_by_default():
    turns = [{"role": "user", "content": "The answer is blue.", "has_answer": True}]

    chunks = chunk_conversation("s1", "2026-05-12", turns)

    assert "HAS_ANSWER" not in chunks[0]["text"]


def test_conversation_parent_text_does_not_leak_answer_marker_by_default():
    turns = [{"role": "user", "content": "The answer is blue.", "has_answer": True}]

    text = conversation_to_text("s1", "2026-05-12", turns)

    assert "HAS_ANSWER" not in text
    assert "Turn 1 (user): The answer is blue." in text


def test_engine_ingest_conversation_uses_turn_chunks(tmp_path):
    turns = [
        {"role": "user", "content": "I decided to use Postgres."},
        {"role": "assistant", "content": "Good choice."},
    ]
    engine = RetrievalEngine.create(tmp_path / "kb")

    chunks = engine.ingest_conversation("s1", "2026-05-12", turns, chunk_size=30, overlap=0)

    assert chunks
    assert chunks[0].metadata["chunk_type"] == "conversation_turns"
    assert chunks[0].metadata["turn_start"] == "1"
    assert chunks[0].metadata["turn_end"] == "2"


def test_engine_ingest_conversation_can_add_parent_chunk(tmp_path):
    turns = [
        {"role": "user", "content": "I decided to use Postgres."},
        {"role": "assistant", "content": "Good choice."},
    ]
    engine = RetrievalEngine.create(tmp_path / "kb")

    chunks = engine.ingest_conversation(
        "s1",
        "2026-05-12",
        turns,
        chunk_size=30,
        overlap=0,
        include_parent=True,
    )

    assert [c.metadata["chunk_type"] for c in chunks].count("conversation_session_parent") == 1
    parent = [c for c in chunks if c.metadata["chunk_type"] == "conversation_session_parent"][0]
    assert parent.metadata["parent_id"] == "s1"
    assert parent.metadata["child_chunk_count"] == "1"
