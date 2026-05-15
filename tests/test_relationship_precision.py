from contextfit.retrieval.relationships import RelationshipIndex


def test_query_without_specific_entity_does_not_match_common_concepts():
    idx = RelationshipIndex()
    for cid in range(1, 31):
        idx.add_text(cid, f"Session {cid}: Previous Occupation was discussed with someone.")

    assert idx.query_entities("What was my previous occupation?") == []
    assert idx.related_matches("What was my previous occupation?") == []


def test_common_entity_is_not_selective_in_larger_corpus():
    idx = RelationshipIndex()
    for cid in range(1, 41):
        idx.add_text(cid, "Acme AI appears in many generic notes.")
    idx.add_text(100, "Bob Chen works at Acme AI.")

    assert idx.query_entities("Who works at Acme AI?") == []


def test_selective_relationship_query_still_finds_neighbor():
    idx = RelationshipIndex()
    for cid in range(1, 30):
        idx.add_text(cid, f"Generic note {cid} about cooking and travel.")
    idx.add_text(100, "Bob Chen works at Acme AI.")
    idx.add_text(101, "Bob Chen is a distributed systems engineer who likes Rust.")

    matches = idx.related_matches("Who works at Acme AI?")
    ids = {m.chunk_id for m in matches}

    assert 100 in ids
    assert 101 in ids


def test_non_relationship_query_does_not_walk_neighbors():
    idx = RelationshipIndex()
    idx.add_text(1, "Bob Chen works at Acme AI.")
    idx.add_text(2, "Bob Chen is a distributed systems engineer who likes Rust.")

    matches = idx.related_matches("Tell me about Acme AI")
    ids = {m.chunk_id for m in matches}

    assert 1 in ids
    assert 2 not in ids
