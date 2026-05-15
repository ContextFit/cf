"""
Learned SID generator: train token → Semantic-ID associations.

This is the first learned query-to-SID layer for ContextFit. It is deliberately
small and dependency-free: no embeddings, no neural framework, and no text
round-trips. Training builds discrete associative tables from chunk token IDs to
SID tokens, then inference performs beam search over valid SID prefixes.

It is effectively a sparse token-native classifier over SID buckets.
"""

from __future__ import annotations

import io
import json
import os
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from contextfit.sid.generator import SIDPrediction
from contextfit.sid.semantic import SemanticIDIndex


class LearnedSIDGenerator:
    """
    Predict SID prefixes from query tokens using trained token associations.

    Training signal:
        For each chunk, every unique chunk token votes for every SID token in
        that chunk's SID path.

    Inference:
        Query tokens score allowed SID tokens at each level. Beam search keeps
        only valid hierarchical prefixes observed during training.
    """

    def __init__(
        self,
        sid_index: SemanticIDIndex,
        prior_weight: int = 1,
        token_weight: int = 4,
        max_token_votes: int = 4,
    ):
        self.sid_index = sid_index
        self.prior_weight = prior_weight
        self.token_weight = token_weight
        self.max_token_votes = max_token_votes

        self.level_priors: list[Counter[int]] = [Counter() for _ in range(sid_index.depth)]
        self.token_votes: list[dict[int, Counter[int]]] = [
            defaultdict(Counter) for _ in range(sid_index.depth)
        ]
        self.prefix_children: dict[tuple[int, ...], set[int]] = defaultdict(set)
        self.trained_chunks = 0

    def fit(self, chunk_tokens: dict[int, Iterable[int] | np.ndarray]) -> LearnedSIDGenerator:
        """Train from chunk_id → token sequence mappings."""
        self.level_priors = [Counter() for _ in range(self.sid_index.depth)]
        self.token_votes = [defaultdict(Counter) for _ in range(self.sid_index.depth)]
        self.prefix_children = defaultdict(set)
        self.trained_chunks = 0

        for chunk_id, tokens in chunk_tokens.items():
            sid = self.sid_index.get(chunk_id)
            if sid is None:
                continue

            if isinstance(tokens, np.ndarray):
                unique_tokens = set(int(t) for t in tokens.tolist())
            else:
                unique_tokens = set(int(t) for t in tokens)

            if not unique_tokens:
                continue

            self.trained_chunks += 1
            prefix: tuple[int, ...] = ()
            for level, sid_token in enumerate(sid.tokens):
                self.level_priors[level][sid_token] += 1
                self.prefix_children[prefix].add(sid_token)

                votes_for_level = self.token_votes[level]
                for token in unique_tokens:
                    votes_for_level[token][sid_token] += 1

                prefix = (*prefix, sid_token)

        return self

    def _score_sid_token(self, level: int, sid_token: int, query_tokens: set[int]) -> int:
        """Integer score for a SID token at one level."""
        score = self.prior_weight * self.level_priors[level].get(sid_token, 0)
        for token in query_tokens:
            count = self.token_votes[level].get(token, {}).get(sid_token, 0)
            score += self.token_weight * min(count, self.max_token_votes)
        return score

    def predict(
        self,
        query_tokens: list[int] | np.ndarray,
        top_k: int = 5,
        beam_width: int = 16,
        max_prefix_depth: int | None = None,
    ) -> list[SIDPrediction]:
        """Predict likely SID prefixes using beam search."""
        if self.trained_chunks == 0:
            return []

        if isinstance(query_tokens, np.ndarray):
            query_token_set = set(int(t) for t in query_tokens.tolist())
        else:
            query_token_set = set(int(t) for t in query_tokens)

        if not query_token_set:
            return []

        max_prefix_depth = max_prefix_depth or self.sid_index.depth
        max_prefix_depth = min(max_prefix_depth, self.sid_index.depth)

        beams: list[tuple[tuple[int, ...], int]] = [((), 0)]
        predictions: list[SIDPrediction] = []

        for level in range(max_prefix_depth):
            next_beams: list[tuple[tuple[int, ...], int]] = []
            for prefix, prefix_score in beams:
                allowed = self.prefix_children.get(prefix)
                if not allowed:
                    continue

                scored_children = [
                    (sid_token, self._score_sid_token(level, sid_token, query_token_set))
                    for sid_token in allowed
                ]
                scored_children.sort(key=lambda x: (-x[1], x[0]))

                for sid_token, child_score in scored_children[:beam_width]:
                    new_prefix = (*prefix, sid_token)
                    score = prefix_score + child_score
                    next_beams.append((new_prefix, score))

            next_beams.sort(key=lambda x: (-x[1], x[0]))
            beams = next_beams[:beam_width]

            for prefix, score in beams:
                resolved = self.sid_index.query_sid(prefix, min_prefix_depth=len(prefix))
                candidate_chunks = tuple(chunk_id for chunk_id, _ in resolved)
                if not candidate_chunks:
                    continue
                predictions.append(
                    SIDPrediction(
                        prefix=prefix,
                        score=float(score),
                        depth=len(prefix),
                        support=len(candidate_chunks),
                        candidate_chunks=candidate_chunks,
                    )
                )

        # Prefer deeper predictions when scores tie: full SID paths are more precise.
        predictions.sort(key=lambda p: (-p.score, -p.depth, -p.support, p.prefix))
        return predictions[:top_k]

    def retrieve(
        self,
        query_tokens: list[int] | np.ndarray,
        top_k: int = 5,
        prediction_k: int | None = None,
        beam_width: int = 16,
    ) -> list[tuple[int, float]]:
        """Predict SID prefixes and resolve them to scored chunks."""
        prediction_k = prediction_k or max(top_k, 5)
        predictions = self.predict(
            query_tokens,
            top_k=prediction_k,
            beam_width=max(beam_width, top_k * 2),
        )

        chunk_scores: dict[int, float] = defaultdict(float)
        for prediction in predictions:
            prefix_score = prediction.score * (prediction.depth / self.sid_index.depth)
            for chunk_id, trie_score in self.sid_index.query_sid(
                prediction.prefix,
                min_prefix_depth=prediction.depth,
            ):
                chunk_scores[chunk_id] += prefix_score * trie_score

            for chunk_id in prediction.candidate_chunks:
                chunk_scores[chunk_id] += prefix_score * 0.1

        results = sorted(chunk_scores.items(), key=lambda x: (-x[1], x[0]))
        return results[:top_k]

    # ------------------------------------------------------------------
    # Persistence — binary npz (fast) + JSON fallback
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Persist learned tables.

        Writes two files:
        * ``learned_sid_generator.npz``  — fast binary format (preferred).
        * ``learned_sid_generator.json`` — kept for backwards compatibility.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._save_npz(path)
        self._save_json(path)   # keep JSON for tools that read it directly

    def _save_npz(self, path: Path) -> None:
        """Save as a compact numpy archive.

        Layout per level (arrays are flattened and stored with a ``_L{n}`` suffix):
          priors_keys_L{n}    int32  — SID-token keys from level_priors[n]
          priors_vals_L{n}    int32  — corresponding counts
          tv_qtok_L{n}        int32  — query-token IDs  (row)
          tv_stok_L{n}        int32  — SID-token IDs    (col)
          tv_count_L{n}       int32  — vote counts       (value)

        Prefix children are stored as two parallel flat arrays:
          pc_prefix_flat      int32  — concatenated prefix tuples
          pc_prefix_lens      int32  — length of each prefix tuple
          pc_child_flat       int32  — corresponding children (one block per prefix)
          pc_child_lens       int32  — number of children for each prefix

        Scalars stored as 0-d arrays:
          trained_chunks, prior_weight, token_weight, max_token_votes, depth
        """
        arrays: dict[str, np.ndarray] = {}
        arrays["trained_chunks"] = np.array(self.trained_chunks, dtype=np.int64)
        arrays["prior_weight"] = np.array(self.prior_weight, dtype=np.int32)
        arrays["token_weight"] = np.array(self.token_weight, dtype=np.int32)
        arrays["max_token_votes"] = np.array(self.max_token_votes, dtype=np.int32)
        arrays["depth"] = np.array(len(self.level_priors), dtype=np.int32)

        for lvl, prior in enumerate(self.level_priors):
            if prior:
                k, v = zip(*prior.items())
                arrays[f"priors_keys_L{lvl}"] = np.array(k, dtype=np.int32)
                arrays[f"priors_vals_L{lvl}"] = np.array(v, dtype=np.int32)
            else:
                arrays[f"priors_keys_L{lvl}"] = np.empty(0, dtype=np.int32)
                arrays[f"priors_vals_L{lvl}"] = np.empty(0, dtype=np.int32)

        for lvl, tv in enumerate(self.token_votes):
            qtoks, stoks, counts = [], [], []
            for qtok, counter in tv.items():
                for stok, cnt in counter.items():
                    qtoks.append(qtok); stoks.append(stok); counts.append(cnt)
            arrays[f"tv_qtok_L{lvl}"] = np.array(qtoks, dtype=np.int32)
            arrays[f"tv_stok_L{lvl}"] = np.array(stoks, dtype=np.int32)
            arrays[f"tv_count_L{lvl}"] = np.array(counts, dtype=np.int32)

        pc_prefix_flat, pc_prefix_lens, pc_child_flat, pc_child_lens = [], [], [], []
        for prefix, children in self.prefix_children.items():
            pc_prefix_flat.extend(prefix)
            pc_prefix_lens.append(len(prefix))
            pc_child_flat.extend(sorted(children))
            pc_child_lens.append(len(children))
        arrays["pc_prefix_flat"] = np.array(pc_prefix_flat, dtype=np.int32)
        arrays["pc_prefix_lens"] = np.array(pc_prefix_lens, dtype=np.int32)
        arrays["pc_child_flat"] = np.array(pc_child_flat, dtype=np.int32)
        arrays["pc_child_lens"] = np.array(pc_child_lens, dtype=np.int32)

        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        tmp = path / "learned_sid_generator.npz.tmp"
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, path / "learned_sid_generator.npz")

    def _save_json(self, path: Path) -> None:
        """Legacy JSON save (kept for backwards compat)."""
        data = {
            "prior_weight": self.prior_weight,
            "token_weight": self.token_weight,
            "max_token_votes": self.max_token_votes,
            "trained_chunks": self.trained_chunks,
            "level_priors": [dict(counter) for counter in self.level_priors],
            "token_votes": [
                {str(token): dict(counter) for token, counter in level_votes.items()}
                for level_votes in self.token_votes
            ],
            "prefix_children": {
                ",".join(str(part) for part in prefix): sorted(children)
                for prefix, children in self.prefix_children.items()
            },
        }
        tmp_path = path / "learned_sid_generator.json.tmp"
        tmp_path.write_text(json.dumps(data, separators=(",", ":")))
        os.replace(tmp_path, path / "learned_sid_generator.json")

    @classmethod
    def load(cls, path: Path | str, sid_index: SemanticIDIndex) -> LearnedSIDGenerator:
        """Load learned tables — uses fast npz if available, falls back to JSON."""
        path = Path(path)
        if (path / "learned_sid_generator.npz").exists():
            return cls._load_npz(path, sid_index)
        return cls._load_json(path, sid_index)

    @classmethod
    def _load_npz(cls, path: Path, sid_index: SemanticIDIndex) -> LearnedSIDGenerator:
        """Fast load from numpy archive."""
        npz = np.load(path / "learned_sid_generator.npz", allow_pickle=False)
        model = cls(
            sid_index=sid_index,
            prior_weight=int(npz["prior_weight"]),
            token_weight=int(npz["token_weight"]),
            max_token_votes=int(npz["max_token_votes"]),
        )
        model.trained_chunks = int(npz["trained_chunks"])
        depth = int(npz["depth"])

        model.level_priors = []
        for lvl in range(depth):
            keys = npz[f"priors_keys_L{lvl}"]
            vals = npz[f"priors_vals_L{lvl}"]
            model.level_priors.append(Counter(dict(zip(keys.tolist(), vals.tolist()))))

        model.token_votes = []
        for lvl in range(depth):
            qtoks = npz[f"tv_qtok_L{lvl}"]
            stoks = npz[f"tv_stok_L{lvl}"]
            counts = npz[f"tv_count_L{lvl}"]
            tv: dict[int, Counter[int]] = defaultdict(Counter)
            for qt, st, cnt in zip(qtoks.tolist(), stoks.tolist(), counts.tolist()):
                tv[qt][st] = cnt
            model.token_votes.append(tv)

        model.prefix_children = defaultdict(set)
        pf = npz["pc_prefix_flat"].tolist()
        pl = npz["pc_prefix_lens"].tolist()
        cf = npz["pc_child_flat"].tolist()
        cl = npz["pc_child_lens"].tolist()
        pi, ci = 0, 0
        for plen, clen in zip(pl, cl):
            prefix = tuple(pf[pi: pi + plen])
            children = set(cf[ci: ci + clen])
            model.prefix_children[prefix] = children
            pi += plen; ci += clen
        return model

    @classmethod
    def _load_json(cls, path: Path, sid_index: SemanticIDIndex) -> LearnedSIDGenerator:
        """Legacy JSON load (fallback)."""
        data = json.loads((path / "learned_sid_generator.json").read_text())
        model = cls(
            sid_index=sid_index,
            prior_weight=data["prior_weight"],
            token_weight=data["token_weight"],
            max_token_votes=data["max_token_votes"],
        )
        model.trained_chunks = data["trained_chunks"]
        model.level_priors = [
            Counter({int(k): v for k, v in level.items()})
            for level in data["level_priors"]
        ]
        model.token_votes = []
        for level_votes in data["token_votes"]:
            restored: dict[int, Counter[int]] = defaultdict(Counter)
            for token, counter in level_votes.items():
                restored[int(token)] = Counter({int(k): v for k, v in counter.items()})
            model.token_votes.append(restored)
        model.prefix_children = defaultdict(set)
        for prefix_str, children in data["prefix_children"].items():
            prefix = tuple(int(part) for part in prefix_str.split(",") if part)
            model.prefix_children[prefix] = set(int(child) for child in children)
        return model

    def stats(self) -> dict:
        """Return learned model statistics."""
        return {
            "trained_chunks": self.trained_chunks,
            "levels": len(self.level_priors),
            "token_features": sum(len(level_votes) for level_votes in self.token_votes),
            "prefixes": len(self.prefix_children),
            "prior_weight": self.prior_weight,
            "token_weight": self.token_weight,
            "max_token_votes": self.max_token_votes,
        }
