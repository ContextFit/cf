from contextfit.retrieval.extractive import extract_evidence, format_evidence_compact, format_evidence_handles


def test_extract_tmd_rows_returns_matching_row_fields():
    text = """# Campaign

campaign[video-01]: headline=Intro, status=idea, depends_on=tagline-01
campaign[video-04]: headline="Give Claude Desktop local memory", status=blocked, target_date=2026-05-25, depends_on=task-01, notes="Blocked until PyPI release"
campaign[video-08]: headline=Citations, status=idea
"""

    evidence = extract_evidence(
        text,
        "video-04 status dependency target date blocker",
        mode="auto",
        metadata={"source": "campaign.tmd", "domain": "tmd"},
    )

    assert evidence
    assert evidence[0].kind == "tmd_row"
    assert evidence[0].row_id == "video-04"
    assert evidence[0].fields["status"] == "blocked"
    assert evidence[0].fields["depends_on"] == "task-01"
    assert "PyPI" in evidence[0].text


def test_format_evidence_compact_uses_tmd_like_metadata():
    text = "campaign[video-04]: status=blocked, depends_on=task-01"
    evidence = extract_evidence(
        text,
        "video-04 blocked",
        mode="rows",
        metadata={"source": "/tmp/workspace/marketing/campaign.tmd", "domain": "tmd"},
    )

    compact = format_evidence_compact(
        evidence,
        metadata={"source": "/tmp/workspace/marketing/campaign.tmd"},
        chunk_id=7,
        chunk_score=12.3456,
    )

    assert compact.startswith("c[7]")
    assert "e[1] k=row" in compact
    assert "src=workspace/marketing/campaign.tmd" in compact
    assert "r=video-04" in compact
    assert "rs=12.35" in compact
    assert "campaign[video-04]: status=blocked" in compact
    assert '"kind"' not in compact


def test_format_evidence_handles_returns_expiring_sidecar():
    text = "campaign[video-04]: status=blocked, depends_on=task-01"
    evidence = extract_evidence(
        text,
        "video-04 blocked",
        mode="rows",
        metadata={"source": "/tmp/workspace/marketing/campaign.tmd", "domain": "tmd"},
    )

    prompt, refs = format_evidence_handles(
        evidence,
        metadata={"source": "/tmp/workspace/marketing/campaign.tmd"},
        chunk_id=7,
        chunk_score=12.3456,
        expires_at="2026-05-14T22:00:00+00:00",
    )

    assert prompt.startswith("@r1 campaign[video-04]:")
    assert "src=" not in prompt
    assert refs["@r1"]["source"] == "workspace/marketing/campaign.tmd"
    assert refs["@r1"]["chunk_id"] == 7
    assert refs["@r1"]["row_id"] == "video-04"
    assert refs["@r1"]["table"] == "campaign"
    assert refs["@r1"]["expires_at"] == "2026-05-14T22:00:00+00:00"
    assert refs["@r1"]["rank_score"] == 12.3456


def test_extract_bullets_selects_relevant_lines_without_llm():
    text = """# Runtime validation

- ContextFit service restarted successfully.
- OpenAPI reports version 0.1.1 and status is ok.
- The loaded KBs are memory, email, and tax.
- Unrelated campaign note.
"""

    evidence = extract_evidence(text, "version loaded KBs", mode="auto", max_items=3)

    joined = "\n".join(item.text for item in evidence)
    assert "0.1.1" in joined
    assert "memory, email, and tax" in joined
    assert len(joined) < len(text)


def test_extract_evidence_offsets_lines_from_chunk_metadata():
    text = "first line\n- target status is blocked\nthird line"

    evidence = extract_evidence(
        text,
        "blocked status",
        mode="bullets",
        metadata={"source": "notes.md", "line_start": 40},
    )

    assert evidence[0].line_start == 41
    assert evidence[0].line_end == 41


def test_extract_spans_respects_budget():
    text = "alpha " * 100 + "needle commit 90e4f52 version 0.1.1 " + "omega " * 100

    evidence = extract_evidence(text, "commit 90e4f52 version", mode="spans", max_chars=120)

    assert evidence
    assert sum(len(item.text) for item in evidence) <= 121
    assert "90e4f52" in evidence[0].text
