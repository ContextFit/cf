from contextfit.cli import _looks_like_email, _preprocess_file
from contextfit.core.tokenizer import Tokenizer

SAMPLE_MD = (
    "# Notizen Köln\n"
    "\n"
    "Grüße aus der Straße: ökonomisch, äußerst übermäßig — ß bleibt ß.\n"
    "Beyond Latin-1: 日本語のテキスト und emoji ✅.\n"
)


def test_plain_markdown_is_not_treated_as_email(tmp_path):
    path = tmp_path / "note.md"
    path.write_text(SAMPLE_MD, encoding="utf-8")
    assert not _looks_like_email(path, path.read_text(encoding="utf-8"))


def test_eml_and_markdown_email_are_treated_as_email(tmp_path):
    assert _looks_like_email(tmp_path / "mail.eml", "From: a@b.c\n\nbody\n")
    assert _looks_like_email(
        tmp_path / "mail.md", "# Email: Update\n**From:** a@b.c\n\n## Content\nhi\n"
    )
    assert _looks_like_email(
        tmp_path / "raw.txt", "From: Alice <a@b.c>\nSubject: hi\n\nbody\n"
    )


def test_markdown_ingest_preserves_non_ascii(tmp_path):
    path = tmp_path / "note.md"
    path.write_text(SAMPLE_MD, encoding="utf-8")

    result = _preprocess_file(
        path,
        tokenizer_name="cl100k_base",
        chunk_size=64,
        overlap=0,
        max_file_bytes=0,
        max_file_tokens=0,
    )

    assert result["status"] == "ok"
    tokenizer = Tokenizer.load("cl100k_base")
    text = "\n".join(tokenizer.decode(item["tokens"]) for item in result["token_items"])
    for needle in ("Köln", "Grüße", "Straße", "äußerst", "日本語", "✅"):
        assert needle in text
