from contextfit.retrieval.memory_atoms import augment_query_for_memory_atoms, episode_context_text, episode_relevance_score, extract_memory_atoms, query_memory_intents


def test_extracts_domain_neutral_user_memory_atoms_only_from_user_turns():
    turns = [
        {"role": "assistant", "content": "I recommend hiking boots."},
        {"role": "user", "content": "I love trail running. I am planning to run a 10k next month."},
    ]

    atoms = extract_memory_atoms(turns, source_id="s1", source_date="2026-05-09")

    assert {a.atom_type for a in atoms} >= {"user_preference", "user_goal"}
    assert all(a.source_id == "s1" for a in atoms)
    assert all("hiking boots" not in a.text for a in atoms)


def test_recommendation_question_becomes_user_interest_atom():
    atoms = extract_memory_atoms([
        {"role": "user", "content": "Can you recommend books about practical robotics?"},
    ])

    assert any(a.atom_type == "user_interest" for a in atoms)
    assert "practical robotics" in atoms[0].text


def test_query_memory_intents_are_general_not_topical():
    assert query_memory_intents("What should I watch tonight?") == {"user_preference", "user_interest", "entity_fact"}
    assert "open_loop" in query_memory_intents("What should I follow up on next week?")
    assert "temporal_update" in query_memory_intents("Which editor do I use now?")
    assert "entity_fact" in query_memory_intents("What should I cook for dinner?")


def test_memory_atom_query_augmentation_adds_only_general_intent_hints():
    query = augment_query_for_memory_atoms("What should I watch tonight?")

    assert "personal context" in query
    assert "user preference" in query
    assert "comedy" not in query.lower()
    assert "movie" not in query.lower().replace("what should i watch tonight?", "")


def test_episode_relevance_score_prefers_aligned_context():
    good = [{"role": "user", "content": "My brother is getting into woodworking and hand planes."}]
    bad = [{"role": "user", "content": "What are common birthday gift ideas under $50?"}]

    assert episode_relevance_score("What gift would fit my brother?", good) > episode_relevance_score("What gift would fit my brother?", bad)


def test_episode_context_text_keeps_whole_user_episode_traceable():
    text = episode_context_text([
        {"role": "user", "content": "My brother is getting into woodworking and hand planes."},
        {"role": "assistant", "content": "Decorative gifts are nice."},
    ], source_id="s1", source_date="2026-05-09")

    assert "Episode memory" in text
    assert "Source session: s1" in text
    assert "woodworking" in text
    assert "Decorative gifts" not in text


def test_extracts_entity_fact_atoms_for_general_context():
    atoms = extract_memory_atoms([
        {"role": "user", "content": "I bought a compact power bank for travel."},
    ])
    assert any(a.atom_type == "entity_fact" for a in atoms)


def test_atom_index_text_is_traceable_and_general():
    atom = extract_memory_atoms([
        {"role": "user", "content": "My favorite editor is Vim."},
    ], source_id="abc", source_date="2026-05-09")[0]

    text = atom.to_index_text()
    assert "Memory atom type: user_preference" in text
    assert "Source session: abc" in text
    assert "User memory: My favorite editor is Vim." in text
