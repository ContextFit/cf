from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "longmemeval_contextfit.py"
    spec = importlib.util.spec_from_file_location("longmemeval_contextfit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_targeted_expansion_preserves_strong_anchors_and_fills_tail() -> None:
    mod = _load_module()
    session_texts = [
        ("s_anchor", "Turn 1 (user): I am planning Project Orion with Northwind."),
        ("s_keep", "Turn 1 (user): Project Orion deadline is next Friday."),
        ("s_tail_a", "Turn 1 (user): General planning notes."),
        ("s_tail_b", "Turn 1 (user): Generic reminder."),
        ("s_companion", "Turn 1 (user): Northwind owns the Project Orion budget constraint."),
    ]

    ranked = mod.targeted_expansion_sessions(
        "What should I consider for Project Orion and Northwind?",
        session_texts,
        ["s_anchor", "s_keep", "s_tail_a", "s_tail_b", "s_companion"],
        top_k=4,
        protected_k=2,
    )

    assert ranked[:2] == ["s_anchor", "s_keep"]
    assert "s_companion" in ranked
