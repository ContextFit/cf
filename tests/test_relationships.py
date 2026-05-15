from contextfit.retrieval.engine import RetrievalEngine
from contextfit.retrieval.relationships import RelationshipIndex, extract_entities, extract_relationship_edges


def test_extract_relationship_edges_with_evidence_chunk_id():
    edges = extract_relationship_edges("Bob Chen works at Acme AI. Bob Chen founded Sequoia Labs.", chunk_id=42)

    triples = {(e.subject, e.predicate, e.object, e.chunk_id) for e in edges}
    assert ("Bob Chen", "works_at", "Acme AI", 42) in triples
    assert ("Bob Chen", "founded", "Sequoia Labs", 42) in triples


def test_relationship_index_finds_neighbor_chunks():
    idx = RelationshipIndex()
    idx.add_text(1, "Bob Chen works at Acme AI.")
    idx.add_text(2, "Bob Chen is a distributed systems engineer who likes Rust.")

    matches = idx.related_matches("Who works at Acme AI?")
    by_id = {m.chunk_id: m for m in matches}

    assert 1 in by_id
    assert 2 in by_id
    assert by_id[1].score > by_id[2].score
    assert "neighbor" in by_id[2].reason


def test_engine_relationship_boost_can_append_backlinked_chunk(tmp_path):
    engine = RetrievalEngine.create(tmp_path)
    engine.ingest_text("Bob Chen works at Acme AI.", metadata={"source": "facts"})
    engine.ingest_text("Bob Chen is a distributed systems engineer who likes Rust.", metadata={"source": "profile"})
    engine.ingest_text("Carol builds mobile games at River Studio.", metadata={"source": "other"})

    result = engine.query("Who works at Acme AI?", top_k=2, relationship_boost=1.5, expand_query=False)
    texts = [engine.tokenizer.decode(chunk.tokens.tolist()) for chunk in result.chunks]

    assert any("Bob Chen works at Acme AI" in text for text in texts)
    assert any("distributed systems engineer" in text for text in texts)


def test_relationship_index_persists_with_engine(tmp_path):
    engine = RetrievalEngine.create(tmp_path)
    engine.ingest_text("Bob Chen works at Acme AI.")
    engine.save(tmp_path)

    loaded = RetrievalEngine.load(tmp_path)
    matches = loaded.relationships.related_matches("Acme AI")

    assert matches
    assert loaded.relationships.evidence_for_chunk(matches[0].chunk_id)["edges"]
