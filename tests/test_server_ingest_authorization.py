from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

import server


def _set_roots(monkeypatch: pytest.MonkeyPatch, roots: list[Path]) -> None:
    monkeypatch.setattr(server, "_ingest_roots", [root.resolve() for root in roots])


def test_resolve_authorized_ingest_path_allows_files_under_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    note = allowed / "note.md"
    note.write_text("# Safe\n", encoding="utf-8")
    _set_roots(monkeypatch, [allowed])

    assert server._resolve_authorized_ingest_path(str(note)) == note.resolve()


def test_resolve_authorized_ingest_path_rejects_absolute_os_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    _set_roots(monkeypatch, [allowed])

    with pytest.raises(HTTPException) as exc:
        server._resolve_authorized_ingest_path("/etc/passwd")

    assert exc.value.status_code == 403


def test_resolve_authorized_ingest_path_rejects_traversal_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("do not ingest\n", encoding="utf-8")
    _set_roots(monkeypatch, [allowed])

    with pytest.raises(HTTPException) as exc:
        server._resolve_authorized_ingest_path(str(allowed / ".." / "outside" / "secret.txt"))

    assert exc.value.status_code == 403


def test_resolve_authorized_ingest_path_rejects_symlink_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("do not ingest\n", encoding="utf-8")
    link = allowed / "linked-secret.txt"
    link.symlink_to(secret)
    _set_roots(monkeypatch, [allowed])

    with pytest.raises(HTTPException) as exc:
        server._resolve_authorized_ingest_path(str(link))

    assert exc.value.status_code == 403


def test_ingest_rejects_unauthorized_batch_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    kb = tmp_path / "kb"
    allowed.mkdir()
    kb.mkdir()
    note = allowed / "note.md"
    note.write_text("# Safe\n", encoding="utf-8")
    _set_roots(monkeypatch, [allowed])
    monkeypatch.setitem(server._kb_paths, "memory", kb)

    with pytest.raises(HTTPException) as exc:
        server.ingest(server.IngestRequest(kb="memory", paths=[str(note), "/etc/passwd"]))

    assert exc.value.status_code == 403


def test_access_key_allows_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_access_key", "secret")

    server._require_access(authorization="Bearer secret")


def test_access_key_rejects_missing_or_wrong_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "_access_key", "secret")

    with pytest.raises(HTTPException) as exc:
        server._require_access()

    assert exc.value.status_code == 401

    with pytest.raises(HTTPException) as exc:
        server._require_access(x_contextfit_access_key="wrong")

    assert exc.value.status_code == 401


def test_public_bind_requires_access_key() -> None:
    assert server._is_local_bind("127.0.0.1")
    assert server._is_local_bind("localhost")
    assert not server._is_local_bind("0.0.0.0")
