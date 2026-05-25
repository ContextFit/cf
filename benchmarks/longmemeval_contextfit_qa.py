#!/usr/bin/env python3
"""Generate and judge LongMemEval QA answers from ContextFit retrieval artifacts.

This is the end-to-end QA companion to ``longmemeval_contextfit.py``. It keeps
retrieval fixed by consuming a saved ContextFit retrieval artifact, then asks an
LLM to answer from the retrieved sessions and uses the LongMemEval GPT judge
prompts to score the resulting hypotheses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from contextfit.retrieval.evidence_compiler import (
    ACTION_RE,
    DATE_RE,
    NUMBER_RE,
    PREFERENCE_QUERY_RE,
    TEMPORAL_QUERY_RE,
    TEMPORAL_RE,
    UPDATE_RE,
    build_count_list_ledger,
    build_deterministic_aggregation_assembly,
    build_evidence_packet,
    build_fusion_evidence_map,
    build_multi_session_evidence_ledger,
    build_multi_session_evidence_set,
    build_token_evidence_table,
    effective_aggregation_assembly_mode,
    effective_fusion_evidence_map_mode,
    is_count_list_question,
    compact_fact,
    question_keywords,
    scrub_turn,
    should_use_aggregation_assembly,
    should_use_evidence_packet,
    should_use_fusion_evidence_map,
    split_fact_candidates,
)
from contextfit.retrieval.memory_atoms import build_preference_support_view

DEFAULT_GENERATION_MODEL = "gpt-4o-2024-08-06"
DEFAULT_JUDGE_MODEL = "gpt-4o-2024-08-06"
OPENCLAW_MODEL_PREFIX = "openclaw:"
OPENAI_COMPATIBLE_MODEL_PREFIX = "openai-compatible:"
UNANSWERABLE_RESPONSE = "The information is not available in the provided history."
MISSING_EVIDENCE_MARKERS = (
    "not mentioned",
    "no mention",
    "no information",
    "not available",
    "not provided",
    "does not mention",
    "doesn't mention",
    "cannot determine",
    "can't determine",
    "insufficient",
)
STRUCTURED_FALLBACK_MARKERS = (
    "not available",
    "not provided",
    "cannot determine",
    "can't determine",
    "not enough information",
    "insufficient information",
    "unknown",
)
ANSWERER_ROUTER_TYPES = {
    "question_type_gpt5mini_temporal_preference_multi": {
        "temporal-reasoning",
        "single-session-preference",
        "multi-session",
    },
}


def should_use_evidence_packet_for_item(item: dict[str, Any], args: argparse.Namespace) -> bool:
    if (
        item.get("question_type") == "temporal-reasoning"
        and getattr(args, "temporal_evidence_packet", "inherit") == "off"
    ):
        return False
    return should_use_evidence_packet(item, args.evidence_packet)


def effective_top_k_context(item: dict[str, Any], args: argparse.Namespace) -> int:
    if item["question_type"] == "multi-session" and args.multi_session_top_k_context:
        return args.multi_session_top_k_context
    if item["question_type"] == "temporal-reasoning" and args.temporal_top_k_context:
        return args.temporal_top_k_context
    return args.top_k_context


def should_use_multi_session_evidence_set_selector(
    report: dict[str, Any],
    mode: str,
    *,
    min_confidence: float,
) -> bool:
    """Gate broad-pool multi-session selectors on deterministic confidence."""
    if not report.get("coherent"):
        return False
    if mode != "confidence_source_select":
        return True
    return (
        float(report.get("confidence", 0.0)) >= min_confidence
        and int(report.get("novel_decisive_group_count", 0)) > 0
        and int(report.get("marginal_utility", 0)) > 0
    )


def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 0.0,
    api_key: str | None = None,
    retries: int = 6,
    seed: int | None = None,
    metadata_out: dict[str, Any] | None = None,
) -> str:
    if model.startswith(OPENCLAW_MODEL_PREFIX):
        text = openclaw_model_completion(model.removeprefix(OPENCLAW_MODEL_PREFIX), messages, retries=retries)
        if metadata_out is not None:
            metadata_out.update({"provider": "openclaw", "model": model.removeprefix(OPENCLAW_MODEL_PREFIX)})
        return text
    if model.startswith(OPENAI_COMPATIBLE_MODEL_PREFIX):
        base_url, local_model = parse_openai_compatible_model(model)
        return openai_compatible_completion(
            base_url,
            local_model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=retries,
            seed=seed,
            metadata_out=metadata_out,
        )

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    request_payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if model.startswith("gpt-5"):
        request_payload["max_completion_tokens"] = max_tokens
    else:
        request_payload["temperature"] = temperature
        request_payload["max_tokens"] = max_tokens
    if seed is not None:
        request_payload["seed"] = seed
    payload = json.dumps(request_payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if metadata_out is not None:
                metadata_out.update(
                    {
                        "provider": "openai",
                        "id": data.get("id"),
                        "model": data.get("model", model),
                        "created": data.get("created"),
                        "system_fingerprint": data.get("system_fingerprint"),
                        "seed": seed,
                        "usage": data.get("usage"),
                    }
                )
            return str(data["choices"][0]["message"]["content"]).strip()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                body = exc.read().decode("utf-8", errors="replace")[:1000]
                raise RuntimeError(f"OpenAI HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"OpenAI request failed after {retries} retries: {last_error}") from last_error


def strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    text = re.sub(r"<think>.*", "", text, flags=re.I | re.S)
    return text.strip()


CHAT_ROLE_ECHO_RE = re.compile(r"^\s*(?:<\|im_start\|>)?\s*(?:system|user|assistant)\s*:?\s*$", re.I)
CHAT_TEMPLATE_TOKEN_RE = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>")


def sanitize_openai_compatible_text(text: str) -> str:
    """Trim local chat-template echoes without relying on server-side stop tokens."""
    text = strip_think_blocks(text)
    lines: list[str] = []
    saw_content = False
    for raw_line in text.splitlines():
        line = CHAT_TEMPLATE_TOKEN_RE.sub("", raw_line).rstrip()
        if CHAT_ROLE_ECHO_RE.match(line):
            if saw_content:
                break
            continue
        if line.strip():
            saw_content = True
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    return cleaned or text.strip()


def parse_openai_compatible_model(model: str) -> tuple[str, str]:
    spec = model.removeprefix(OPENAI_COMPATIBLE_MODEL_PREFIX)
    if "|" not in spec:
        raise RuntimeError(
            "openai-compatible model spec must be openai-compatible:<base_url>|<model>, "
            "for example openai-compatible:http://192.168.1.113:11435/v1|gwopus2-tools:latest"
        )
    base_url, local_model = spec.rsplit("|", 1)
    return base_url.rstrip("/"), local_model


def openai_compatible_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    retries: int = 6,
    seed: int | None = None,
    metadata_out: dict[str, Any] | None = None,
) -> str:
    request_payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if seed is not None:
        request_payload["seed"] = seed
    payload = json.dumps(request_payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    last_error: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            message = data["choices"][0]["message"]
            text = str(message.get("content") or "")
            if not text and message.get("reasoning"):
                text = str(message["reasoning"])
            if metadata_out is not None:
                metadata_out.update(
                    {
                        "provider": "openai-compatible",
                        "id": data.get("id"),
                        "model": data.get("model", model),
                        "created": data.get("created"),
                        "system_fingerprint": data.get("system_fingerprint"),
                        "seed": seed,
                        "usage": data.get("usage"),
                    }
                )
            return sanitize_openai_compatible_text(text)
        except urllib.error.HTTPError as exc:
            last_error = exc
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"OpenAI-compatible HTTP {exc.code}: {body}") from exc
            last_error = RuntimeError(f"OpenAI-compatible HTTP {exc.code}: {body}")
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"OpenAI-compatible request failed after {retries} retries: {last_error}") from last_error


def openclaw_model_completion(model: str, messages: list[dict[str, str]], *, retries: int = 6) -> str:
    """Run a model turn through the local OpenClaw gateway auth layer."""
    prompt = "\n\n".join(
        f"{message.get('role', 'user').upper()}:\n{message.get('content', '')}"
        for message in messages
    )
    env = os.environ.copy()
    config_path = env.get("OPENCLAW_CONFIG_PATH") or str(default_openclaw_config_path())
    env["OPENCLAW_CONFIG_PATH"] = config_path
    if not env.get("OPENCLAW_GATEWAY_TOKEN"):
        config = Path(config_path)
        if config.exists():
            try:
                token = json.loads(config.read_text()).get("gateway", {}).get("auth", {}).get("token")
            except json.JSONDecodeError:
                token = None
            if token:
                env["OPENCLAW_GATEWAY_TOKEN"] = str(token)

    cmd = [
        "openclaw",
        "infer",
        "model",
        "run",
        "--gateway",
        "--model",
        model,
        "--prompt",
        prompt,
        "--json",
    ]
    last_error: Exception | None = None
    for attempt in range(retries):
        proc = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=180, check=False)
        if proc.returncode == 0:
            try:
                data = json.loads(proc.stdout)
                outputs = data.get("outputs") or []
                if outputs:
                    return str(outputs[0].get("text", "")).strip()
            except json.JSONDecodeError as exc:
                last_error = exc
            else:
                last_error = RuntimeError(f"OpenClaw model run returned no text: {proc.stdout[:500]}")
        else:
            last_error = RuntimeError(proc.stderr[:1000] or proc.stdout[:1000])
        time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"OpenClaw model run failed after {retries} retries: {last_error}") from last_error


def default_openclaw_config_path() -> Path:
    home_config = Path.home() / ".openclaw" / "openclaw.json"
    if home_config.exists():
        return home_config
    home_parts = Path.home().parts
    if ".openclaw" in home_parts:
        idx = home_parts.index(".openclaw")
        state_config = Path(*home_parts[: idx + 1]) / "openclaw.json"
        if state_config.exists():
            return state_config
    return home_config


def completion_seed_for_purpose(args: argparse.Namespace, purpose: str) -> int | None:
    if purpose == "judge":
        return args.judge_seed
    if purpose == "answerability":
        return args.answerability_seed if args.answerability_seed is not None else args.judge_seed
    return args.generation_seed


def completion_cache_key(
    *,
    purpose: str,
    qid: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    seed: int | None,
) -> str:
    payload = {
        "version": 1,
        "purpose": purpose,
        "question_id": qid,
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def jsonable_arg_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [jsonable_arg_value(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable_arg_value(item) for item in value]
    return str(value)


def sanitized_args_dict(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {"api_key"}
    return {
        key: jsonable_arg_value(value)
        for key, value in sorted(vars(args).items())
        if key not in excluded and not key.endswith("_key")
    }


def cache_metadata_summary(cache_dir: Path | None, qids: set[str]) -> dict[str, Any]:
    if cache_dir is None or not cache_dir.exists():
        return {"cache_dir": str(cache_dir) if cache_dir else None, "present": False}

    by_purpose: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "models": set(),
            "seeds": set(),
            "system_fingerprints": set(),
        }
    )
    matched = 0
    for path in cache_dir.glob("*.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        qid = row.get("question_id")
        if qids and qid not in qids:
            continue
        purpose = str(row.get("purpose", "unknown"))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        bucket = by_purpose[purpose]
        bucket["count"] += 1
        matched += 1
        if row.get("model"):
            bucket["models"].add(str(row["model"]))
        if row.get("seed") is not None:
            bucket["seeds"].add(str(row["seed"]))
        if metadata.get("system_fingerprint"):
            bucket["system_fingerprints"].add(str(metadata["system_fingerprint"]))

    serializable = {
        purpose: {
            "count": values["count"],
            "models": sorted(values["models"]),
            "seeds": sorted(values["seeds"]),
            "system_fingerprints": sorted(values["system_fingerprints"]),
        }
        for purpose, values in sorted(by_purpose.items())
    }
    return {
        "cache_dir": str(cache_dir),
        "present": True,
        "matched_files": matched,
        "by_purpose": serializable,
    }


def build_run_provenance(args: argparse.Namespace, judged: list[dict[str, Any]]) -> dict[str, Any]:
    qids = {str(row["question_id"]) for row in judged if "question_id" in row}
    script_path = Path(__file__).resolve()
    compiler_path = script_path.parents[1] / "src" / "contextfit" / "retrieval" / "evidence_compiler.py"
    input_hashes = {
        "data": sha256_file(args.data),
        "retrieval_artifact": sha256_file(args.retrieval_artifact),
        "question_id_file": sha256_file(args.question_id_file),
        "supporting_retrieval_artifact": sha256_file(getattr(args, "supporting_retrieval_artifact", None)),
        "primary_retrieval_artifact": sha256_file(getattr(args, "primary_retrieval_artifact", None)),
    }
    code_hashes = {
        "benchmark_script": sha256_file(script_path),
        "evidence_compiler": sha256_file(compiler_path),
    }
    git_diff = git_output(["diff", "--", "benchmarks/longmemeval_contextfit_qa.py", "src/contextfit/retrieval/evidence_compiler.py", "tests/test_longmemeval_qa_routing.py", "tests/test_evidence_compiler.py"])
    git_status = git_output(["status", "--short"])
    args_payload = sanitized_args_dict(args)
    route_settings = {
        "top_k_context": args.top_k_context,
        "multi_session_top_k_context": args.multi_session_top_k_context,
        "temporal_top_k_context": args.temporal_top_k_context,
        "preference_support_packet": args.preference_support_packet,
        "preference_support_max_items": args.preference_support_max_items,
        "preference_support_per_source": args.preference_support_per_source,
        "multi_session_evidence_set": args.multi_session_evidence_set,
        "multi_session_evidence_set_min_confidence": args.multi_session_evidence_set_min_confidence,
        "evidence_packet": args.evidence_packet,
        "temporal_evidence_packet": args.temporal_evidence_packet,
        "fusion_evidence_map": args.fusion_evidence_map,
        "profile_event_ledger": getattr(args, "profile_event_ledger", "off"),
        "aggregation_assembly": args.aggregation_assembly,
        "count_list_ledger": getattr(args, "count_list_ledger", "off"),
        "source_aware": args.source_aware,
        "source_sufficiency": args.source_sufficiency,
    }
    run_hash_payload = {
        "args": args_payload,
        "input_hashes": input_hashes,
        "code_hashes": code_hashes,
        "git_diff_sha256": sha256_text(git_diff or ""),
        "question_ids": sorted(qids),
    }
    return {
        "run_hash": sha256_text(json.dumps(run_hash_payload, sort_keys=True, separators=(",", ":"))),
        "args": args_payload,
        "route_settings": route_settings,
        "inputs": input_hashes,
        "code": code_hashes,
        "git": {
            "head": git_output(["rev-parse", "HEAD"]),
            "status_short": git_status,
            "status_short_sha256": sha256_text(git_status or ""),
            "diff_sha256": sha256_text(git_diff or ""),
        },
        "outputs": {
            "hypotheses_out": str(args.hypotheses_out),
            "judged_out": str(args.judged_out),
            "summary_out": str(args.summary_out),
            "answerability_out": str(args.answerability_out),
            "extract_out": str(args.extract_out),
        },
        "models": {
            "generation_model": args.generation_model,
            "answerer_router": getattr(args, "answerer_router", "off"),
            "routed_generation_model": getattr(args, "routed_generation_model", ""),
            "routed_extraction_model": getattr(args, "routed_extraction_model", ""),
            "routed_answerability_model": getattr(args, "routed_answerability_model", ""),
            "judge_model": args.judge_model,
            "answerability_model": args.answerability_model,
            "extraction_model": args.extraction_model,
            "generation_seed": args.generation_seed,
            "judge_seed": args.judge_seed,
            "answerability_seed": args.answerability_seed,
        },
        "cache": cache_metadata_summary(args.completion_cache_dir, qids),
    }


def cached_chat_completion(
    args: argparse.Namespace,
    *,
    purpose: str,
    qid: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    seed = completion_seed_for_purpose(args, purpose)
    cache_dir = args.completion_cache_dir
    key = completion_cache_key(
        purpose=purpose,
        qid=qid,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
    cache_path = cache_dir / f"{key}.json" if cache_dir else None
    if cache_path and cache_path.exists():
        data = json.loads(cache_path.read_text())
        return str(data["text"])

    metadata: dict[str, Any] = {}
    text = chat_completion(
        model,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=args.api_key,
        seed=seed,
        metadata_out=metadata,
    )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "text": text,
                    "metadata": metadata,
                    "cache_key": key,
                    "purpose": purpose,
                    "question_id": qid,
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "seed": seed,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return text


def use_routed_answerer(args: argparse.Namespace, item: dict[str, Any]) -> bool:
    return item.get("question_type") in ANSWERER_ROUTER_TYPES.get(args.answerer_router, set())


def generation_model_for_item(args: argparse.Namespace, item: dict[str, Any]) -> str:
    if use_routed_answerer(args, item):
        return args.routed_generation_model
    return args.generation_model


def extraction_model_for_item(args: argparse.Namespace, item: dict[str, Any]) -> str:
    if use_routed_answerer(args, item):
        return args.routed_extraction_model or args.routed_generation_model
    return args.extraction_model


def answerability_model_for_item(args: argparse.Namespace, item: dict[str, Any]) -> str:
    if use_routed_answerer(args, item):
        return args.routed_answerability_model or args.routed_generation_model
    return args.answerability_model


def generation_max_tokens_for_item(args: argparse.Namespace, item: dict[str, Any]) -> int:
    if use_routed_answerer(args, item) and args.routed_generation_max_tokens:
        return args.routed_generation_max_tokens
    return args.generation_max_tokens


def extraction_max_tokens_for_item(args: argparse.Namespace, item: dict[str, Any]) -> int:
    if use_routed_answerer(args, item) and args.routed_extraction_max_tokens:
        return args.routed_extraction_max_tokens
    return args.extraction_max_tokens


def answerability_max_tokens_for_item(args: argparse.Namespace, item: dict[str, Any]) -> int:
    if use_routed_answerer(args, item) and args.routed_answerability_max_tokens:
        return args.routed_answerability_max_tokens
    return args.answerability_max_tokens


def parse_yes_no_label(response: str) -> bool:
    match = re.search(r"\b(yes|no)\b", response.strip(), flags=re.I)
    if match:
        return match.group(1).lower() == "yes"
    return "yes" in response.lower()



def session_to_text(session_id: str, date: str, turns: list[dict[str, Any]], max_chars: int) -> str:
    lines = [f"Session ID: {session_id}", f"Session Date: {date}", "Session Content:"]
    for turn in turns:
        clean = scrub_turn(turn)
        if clean["content"]:
            lines.append(f"{clean['role']}: {clean['content']}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[: max_chars - 80].rstrip() + "\n[session truncated for context budget]"
    return text


def build_multi_session_compiler_prompt(
    item: dict[str, Any],
    retrieved_sessions: list[str],
    *,
    top_k: int,
    max_session_chars: int,
    max_ledger_rows: int,
    max_ledger_chars: int,
) -> str:
    date_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_dates"], strict=True))
    turns_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True))
    selected = [sid for sid in retrieved_sessions[:top_k] if sid in turns_by_sid]
    selected.sort(key=lambda sid: date_by_sid.get(sid, ""))
    context_parts = []
    for i, sid in enumerate(selected, start=1):
        context_parts.append(
            f"### Retrieved Session {i}\n"
            + session_to_text(sid, date_by_sid[sid], turns_by_sid[sid], max_session_chars)
        )
    context = "\n\n".join(context_parts)
    ledger = build_multi_session_evidence_ledger(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        max_rows=max_ledger_rows,
        max_chars=max_ledger_chars,
    )
    return (
        "Answer the multi-session question using a strict evidence compilation contract.\n\n"
        "Rules:\n"
        "- Use the Multi-Session Evidence Ledger as the constrained working set.\n"
        "- You may inspect the History Chats only to verify ledger rows, not to introduce uncited facts.\n"
        "- For counts/lists/totals, enumerate Candidate Set, then Deduped Set, then Final Answer.\n"
        "- For comparisons, include every requested side and cite the ledger rows for each side.\n"
        "- For temporal/current/latest wording, preserve dates and prefer the latest supported state.\n"
        "- If the ledger and history do not contain enough evidence for a complete answer, the Final Answer must be exactly: "
        f"{UNANSWERABLE_RESPONSE}\n"
        "- Every included candidate in Candidate Set or Deduped Set must cite one or more E# ledger ids.\n\n"
        f"{ledger}\n\n"
        f"History Chats:\n\n{context}\n\n"
        f"Current Date: {item.get('question_date', '')}\n"
        f"Question: {item['question']}\n"
        "Candidate Set, Deduped Set, Missing/Excluded Evidence, and Final Answer:"
    )


def build_multi_session_guided_prompt(
    item: dict[str, Any],
    retrieved_sessions: list[str],
    *,
    top_k: int,
    max_session_chars: int,
    max_ledger_rows: int,
    max_ledger_chars: int,
) -> str:
    date_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_dates"], strict=True))
    turns_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True))
    selected = [sid for sid in retrieved_sessions[:top_k] if sid in turns_by_sid]
    selected.sort(key=lambda sid: date_by_sid.get(sid, ""))
    context_parts = []
    for i, sid in enumerate(selected, start=1):
        context_parts.append(
            f"### Retrieved Session {i}\n"
            + session_to_text(sid, date_by_sid[sid], turns_by_sid[sid], max_session_chars)
        )
    context = "\n\n".join(context_parts)
    ledger = build_multi_session_evidence_ledger(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        max_rows=max_ledger_rows,
        max_chars=max_ledger_chars,
    )
    return (
        "Answer the multi-session question from the retrieved chat history.\n\n"
        "Use the Multi-Session Evidence Ledger as a checklist for likely evidence, but the full History Chats are authoritative. "
        "If a relevant fact appears in History Chats but not in the ledger, you may use it and cite the retrieved session.\n\n"
        "Rules:\n"
        "- First write Source Notes for every retrieved session containing relevant evidence.\n"
        "- For counts/lists/totals, write Candidate Set and Deduped Set before Final Answer.\n"
        "- For comparisons, include every requested side and cite the supporting sessions.\n"
        "- For temporal/current/latest wording, preserve dates and prefer the latest supported state.\n"
        "- Do not stop after the first matching session; scan all retrieved sessions.\n"
        "- If the retrieved history does not contain enough evidence for a complete answer, say that the information is not available in the provided history.\n\n"
        f"{ledger}\n\n"
        f"History Chats:\n\n{context}\n\n"
        f"Current Date: {item.get('question_date', '')}\n"
        f"Question: {item['question']}\n"
        "Source Notes, Candidate Set, Deduped Set, and Final Answer:"
    )





def _section_after(label: str, text: str) -> str:
    match = re.search(rf"(?:^|\n)\s*(?:#+\s*)?{re.escape(label)}\s*:?\s*(?:\n|$)(.*)", text, re.I | re.S)
    return match.group(1).strip() if match else ""


def final_answer_text(hypothesis: str) -> str:
    final = _section_after("Final Answer", hypothesis)
    if not final:
        return hypothesis.strip()
    next_section = re.search(r"\n\s*(?:#+\s*)?[A-Z][A-Za-z /-]{2,40}:?\s*(?:\n|$)", final)
    if next_section:
        final = final[: next_section.start()]
    return final.strip()


def source_notes_text(hypothesis: str) -> str:
    notes = _section_after("Source Notes", hypothesis)
    if not notes:
        notes = _section_after("Candidate Evidence", hypothesis)
    if not notes:
        return ""
    final_match = re.search(r"\n\s*(?:#+\s*)?Final Answer\s*:?\s*(?:\n|$)", notes, re.I)
    if final_match:
        notes = notes[: final_match.start()]
    return notes.strip()


def answer_says_unavailable(text: str) -> bool:
    return any(marker in text.lower() for marker in MISSING_EVIDENCE_MARKERS)


def deterministic_verification_report(item: dict[str, Any], hypothesis: str) -> dict[str, Any]:
    """Flag answer shapes that often need a correction pass.

    This verifier is deliberately conservative and source-local. It does not
    know the gold answer and does not judge correctness; it only asks whether
    the generated response has enough self-contained support to be trusted.
    """
    question = item["question"]
    notes = source_notes_text(hypothesis)
    final = final_answer_text(hypothesis)
    notes_lower = notes.lower()
    final_lower = final.lower()
    flags: list[str] = []

    if not notes and re.search(r"\b(source notes|retrieved session)\b", hypothesis, re.I):
        flags.append("missing_source_notes")
    if notes and answer_says_unavailable(notes) and not answer_says_unavailable(final):
        flags.append("final_answer_despite_missing_source_evidence")
    if is_count_list_question(question):
        has_number = bool(NUMBER_RE.search(final))
        has_list_shape = bool(re.search(r"[,;]|\band\b", final_lower))
        has_aggregation_trace = bool(re.search(r"\b(?:dedup|candidate|count|total|distinct|source)\b", notes_lower))
        if not answer_says_unavailable(final) and not (has_number or has_list_shape):
            flags.append("count_list_final_lacks_count_or_list")
        if not answer_says_unavailable(final) and not has_aggregation_trace:
            flags.append("count_list_lacks_aggregation_trace")
    if TEMPORAL_QUERY_RE.search(question):
        support_text = f"{notes}\n{final}"
        has_temporal_support = bool(DATE_RE.search(support_text) or TEMPORAL_RE.search(support_text))
        if not answer_says_unavailable(final) and not has_temporal_support:
            flags.append("temporal_answer_lacks_ordering_evidence")
    if not answer_says_unavailable(final) and re.search(r"\bnot (?:in|available|provided|mentioned)\b", notes_lower):
        flags.append("source_notes_negate_final_answer")

    return {
        "needs_correction": bool(flags),
        "flags": flags,
        "final_answer": final,
    }


def build_correction_prompt(
    item: dict[str, Any],
    retrieved_sessions: list[str],
    original_hypothesis: str,
    verifier_report: dict[str, Any],
    *,
    top_k: int,
    max_session_chars: int,
    evidence_packet: bool,
    evidence_packet_max_items: int,
    evidence_packet_max_chars: int,
) -> str:
    base_prompt = build_answer_prompt(
        item,
        retrieved_sessions,
        top_k=top_k,
        max_session_chars=max_session_chars,
        cot=False,
        source_aware=True,
        source_sufficiency=True,
        token_evidence=False,
        evidence_packet=evidence_packet,
        fusion_evidence_map=False,
        fusion_evidence_map_max_items=0,
        fusion_evidence_map_max_chars=0,
        evidence_packet_max_items=evidence_packet_max_items,
        evidence_packet_max_chars=evidence_packet_max_chars,
        token_evidence_max_lines=0,
        token_evidence_signals=False,
        count_list_mode=is_count_list_question(item["question"]),
        strict_missing_final_answer=True,
    )
    return (
        "A previous answer may be weakly supported by its source notes.\n"
        "Use the verifier flags to correct the answer. Do not preserve the previous answer if the retrieved history does not support it.\n"
        "If required evidence is missing, the Final Answer must be exactly: "
        f"{UNANSWERABLE_RESPONSE}\n\n"
        f"Previous Answer:\n{original_hypothesis}\n\n"
        f"Verifier Flags: {', '.join(verifier_report['flags']) or '(none)'}\n\n"
        f"{base_prompt}"
    )


def build_answer_prompt(
    item: dict[str, Any],
    retrieved_sessions: list[str],
    *,
    top_k: int,
    max_session_chars: int,
    cot: bool,
    source_aware: bool,
    source_sufficiency: bool,
    token_evidence: bool,
    evidence_packet: bool,
    fusion_evidence_map: bool,
    fusion_evidence_map_max_items: int,
    fusion_evidence_map_max_chars: int,
    evidence_packet_max_items: int,
    evidence_packet_max_chars: int,
    token_evidence_max_lines: int,
    token_evidence_signals: bool,
    count_list_mode: bool,
    strict_missing_final_answer: bool,
    preference_support_packet: str = "off",
    preference_support_max_items: int = 10,
    preference_support_per_source: int = 2,
    profile_event_ledger: str = "off",
    profile_event_ledger_max_rows: int = 36,
    profile_event_ledger_max_chars: int = 10_000,
) -> str:
    date_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_dates"], strict=True))
    turns_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True))
    selected = [sid for sid in retrieved_sessions[:top_k] if sid in turns_by_sid]
    selected_retrieval_order = list(selected)

    # LongMemEval's generation script sorts selected context chronologically.
    selected.sort(key=lambda sid: date_by_sid.get(sid, ""))

    context_parts = []
    for i, sid in enumerate(selected, start=1):
        context_parts.append(
            f"### Retrieved Session {i}\n"
            + session_to_text(sid, date_by_sid[sid], turns_by_sid[sid], max_session_chars)
        )
    context = "\n\n".join(context_parts)
    evidence_table = ""
    if token_evidence:
        evidence_table = (
            build_token_evidence_table(
                item,
                selected,
                date_by_sid,
                turns_by_sid,
                max_lines=token_evidence_max_lines,
                include_signals=token_evidence_signals,
            )
            + "\n\n"
        )
    if fusion_evidence_map:
        fusion_map = build_fusion_evidence_map(
            item,
            selected,
            date_by_sid,
            turns_by_sid,
            max_items=fusion_evidence_map_max_items,
            max_chars=fusion_evidence_map_max_chars,
        )
        if fusion_map:
            evidence_table += fusion_map + "\n\n"
    if evidence_packet:
        evidence_table += (
            build_evidence_packet(
                item,
                selected,
                date_by_sid,
                turns_by_sid,
                max_items_per_section=evidence_packet_max_items,
                max_chars=evidence_packet_max_chars,
            )
            + "\n\n"
        )
    if preference_support_packet != "off" and item.get("question_type") == "single-session-preference":
        support_sources = [
            {
                "source_id": sid,
                "date": date_by_sid.get(sid, ""),
                "turns": turns_by_sid[sid],
            }
            for sid in selected_retrieval_order
        ]
        support_view = build_preference_support_view(
            item["question"],
            support_sources,
            max_items=preference_support_max_items,
            per_source=preference_support_per_source,
        )
        if support_view:
            evidence_table += (
                support_view
                + "\n"
                + "Preference support is a compact view over the already retrieved sessions. "
                + "Use it to identify transferable personal context, but verify the final answer against History Chats.\n\n"
            )
    if should_use_profile_event_ledger(item, profile_event_ledger):
        profile_ledger = build_profile_event_ledger(
            item,
            selected,
            date_by_sid,
            turns_by_sid,
            max_rows=profile_event_ledger_max_rows,
            max_chars=profile_event_ledger_max_chars,
        )
        if profile_ledger:
            evidence_table += profile_ledger + "\n\n"
    use_count_list_mode = count_list_mode and is_count_list_question(item["question"])
    if use_count_list_mode:
        answer_instruction = (
            "Answer the question based on the provided chat history. "
            "This is a count/list aggregation question. Treat each retrieved session as a separate evidence source. "
            "First write Candidate Evidence: inspect every retrieved session and list each concrete candidate item, event, amount, date, or occurrence that could affect the answer, with its source session. "
            "Then write Deduped Set: merge repeated mentions of the same real-world item/event, keep distinct items/events separate, and explain any exclusion. "
            "For temporal wording such as current, last month, before, after, latest, or currently, only include candidates that satisfy that time constraint. "
            "If the question asks for a count, the Final Answer must be the count implied by the Deduped Set. "
            "If the question asks for a list, the Final Answer must list exactly the deduped items. "
            "Do not stop after the first matching session. Do not estimate from general knowledge. "
            "If the provided history truly does not contain enough relevant information, say that the information is not available in the provided history."
        )
        answer_label = "Candidate Evidence, Deduped Set, and Final Answer:"
    elif source_sufficiency:
        answer_instruction = (
            "Answer the question based on the provided chat history. "
            "Treat each retrieved session as a separate evidence source. "
            "First write Required Evidence: identify the exact kind of facts needed to answer the question, such as dates, counts, quantities, places, roles, entities, or comparison sides. "
            "Then write Source Notes: inspect every retrieved session and list all concrete facts that could help answer the question. Do not omit relevant numeric amounts, dates, named entities, or candidate items. "
            "For multi-session, counting, total, list, and comparison questions, scan all retrieved sessions and combine distinct facts across sources; do not stop after the first matching session. "
            "For when/date questions, preserve the most specific date expression available, including holidays such as Valentine's Day when that identifies a date. "
            "Use dates and temporal wording to decide whether facts are current, previous, updated, or superseded. "
            "If several sessions mention the same item, count it once unless the question asks for mentions or events. "
            "After Source Notes, write exactly one line: Sufficiency: sufficient or Sufficiency: insufficient. "
            "Mark Sufficiency: insufficient if the source notes do not contain every required entity, value, date, place, role, event, or side of a comparison needed to answer the question. "
            "For comparison questions, if any requested side is missing, mark insufficient. "
            "For questions about a specific role, place, course, item, or event, mark insufficient when the notes only support a similar but different role, place, course, item, or event. "
            "If Sufficiency is insufficient, the Final Answer must be exactly: The information is not available in the provided history. "
            "If Sufficiency is sufficient, write Final Answer with the concise answer. "
            "For recommendation or preference questions, Sufficiency is sufficient when the source notes contain personal context that can reasonably support the recommendation."
        )
        answer_label = "Required Evidence, Source Notes, Sufficiency, and Final Answer:"
    elif source_aware:
        answer_instruction = (
            "Answer the question based on the provided chat history. "
            "Treat each retrieved session as a separate evidence source. "
            "If a Token Evidence Table or Deterministic Evidence Packet is provided, use it as a deterministic hint layer for correlations, counts, dates, updates, and repeated facts, but verify the final answer against the full retrieved sessions. "
            "If a Canonical Profile/Event Ledger is provided, treat it as the candidate working set: include accepted ledger rows in Source Notes, reject out-of-scope rows explicitly, use latest-wins only when a later row supersedes the same entity/state, and do not answer from a vague match when the ledger shows a more specific missing requirement. "
            "First write Source Notes: for every retrieved session that contains relevant evidence, list the concrete facts from that session. "
            "For multi-session, counting, total, list, and comparison questions, scan all retrieved sessions and combine distinct facts across sources; do not stop after the first matching session. "
            "Use dates and temporal wording to decide whether facts are current, previous, updated, or superseded. "
            "For temporal questions, identify the two or more dated events or states needed for the calculation or ordering before writing the answer. "
            "If several sessions mention the same item, count it once unless the question asks for mentions or events. "
            "Then write Final Answer with the concise answer. "
            "The Final Answer must be one concise line and must not introduce facts absent from Source Notes. "
            "For recommendation or preference questions, use indirect personal context such as the user's interests, tools, constraints, goals, and prior activities. "
            "If the provided history truly does not contain enough relevant information, say that the information is not available in the provided history."
        )
        if strict_missing_final_answer:
            answer_instruction += (
                " If your Source Notes explicitly say that a required value, entity, date, place, role, event, or side of a comparison is missing or not mentioned, the Final Answer must be exactly: The information is not available in the provided history."
            )
        answer_label = "Source Notes and Final Answer:"
    elif cot:
        answer_instruction = (
            "Answer the question based on the relevant chat history. "
            "First extract the relevant information from the provided sessions, then reason over that information to get the answer. "
            "For recommendation or preference questions, use indirect personal context such as the user's interests, tools, constraints, goals, and prior activities. "
            "If the provided history truly does not contain enough relevant information, say that the information is not available in the provided history."
        )
        answer_label = "Answer (relevant information, reasoning, final answer):"
    else:
        answer_instruction = (
            "Answer the question based only on the provided relevant chat history. "
            "If the provided history does not contain enough information to answer, say that the information is not available in the provided history. "
            "Be concise and give the final answer directly."
        )
        answer_label = "Answer:"
    return (
        "I will give you several history chats between you and a user. "
        f"{answer_instruction}\n\n"
        f"{evidence_table}"
        f"History Chats:\n\n{context}\n\n"
        f"Current Date: {item.get('question_date', '')}\n"
        f"Question: {item['question']}\n"
        f"{answer_label}"
    )


def should_use_profile_event_ledger(item: dict[str, Any], mode: str) -> bool:
    if mode == "off":
        return False
    if mode == "all":
        return True
    if mode == "general":
        question = item["question"]
        return (
            item["question_type"]
            in {"knowledge-update", "multi-session", "single-session-preference", "temporal-reasoning"}
            or bool(TEMPORAL_QUERY_RE.search(question))
            or bool(UPDATE_RE.search(question))
            or bool(PREFERENCE_QUERY_RE.search(question))
            or is_count_list_question(question)
        )
    raise ValueError(f"unknown profile event ledger mode: {mode}")


PROFILE_EVENT_RE = re.compile(
    r"\b("
    r"like|likes|liked|love|loves|loved|enjoy|enjoys|prefer|prefers|favorite|favourite|"
    r"avoid|avoids|hate|hates|want|wants|trying|goal|need|needs|must|budget|deadline|"
    r"decided|chose|picked|selected|went with|switched|started|stopped|currently|current|"
    r"latest|recently|no longer|used to"
    r")\b",
    re.I,
)


def build_profile_event_ledger(
    item: dict[str, Any],
    selected: list[str],
    date_by_sid: dict[str, str],
    turns_by_sid: dict[str, list[dict[str, Any]]],
    *,
    max_rows: int,
    max_chars: int,
) -> str:
    """Build a compact canonical profile/event ledger from retrieved sessions.

    The ledger is deterministic and source-linked. It gives the reader a stable
    working set for the failure modes that plain source-aware prompts often
    mishandle: indirect preference support, cross-session aggregation,
    superseded state, and date/window anchoring.
    """
    keywords = question_keywords(item["question"])
    rows: list[tuple[int, str, int, int, str]] = []
    seen: set[str] = set()
    count_list = is_count_list_question(item["question"])
    preference_like = item.get("question_type") == "single-session-preference" or bool(
        PREFERENCE_QUERY_RE.search(item["question"])
    )

    for source_idx, sid in enumerate(selected, start=1):
        date = date_by_sid.get(sid, "")
        for turn_idx, turn in enumerate(turns_by_sid[sid], start=1):
            clean = scrub_turn(turn)
            role = clean["role"]
            content = clean["content"]
            if not content:
                continue
            for sentence in split_fact_candidates(content):
                compact = compact_fact(sentence, max_chars=260)
                if not compact:
                    continue
                lower = compact.lower()
                q_hits = sorted(word for word in keywords if word in lower)
                facets = profile_event_facets(compact, role=role)
                has_number = bool(NUMBER_RE.search(compact))
                has_date = bool(DATE_RE.search(compact) or TEMPORAL_RE.search(compact) or date)
                keep = bool(
                    q_hits
                    or facets
                    or (count_list and (has_number or ACTION_RE.search(compact)))
                    or (preference_like and role == "user" and re.search(r"\b(my|i|me|we|our)\b", lower))
                )
                if not keep:
                    continue
                key = f"{sid}|{turn_idx}|{re.sub(r'[^a-z0-9]+', ' ', lower).strip()}"
                if key in seen:
                    continue
                seen.add(key)
                weight = (
                    5 * len(q_hits)
                    + 3 * len(facets)
                    + int(has_number)
                    + int(has_date)
                    + (2 if role == "user" else 0)
                )
                facet_text = ",".join(facets) if facets else "context"
                q_text = ",".join(q_hits[:8]) if q_hits else "-"
                rows.append(
                    (
                        weight,
                        date,
                        source_idx,
                        turn_idx,
                        f"source=S{source_idx} sid={sid} date={date} turn={turn_idx} "
                        f"role={role} facets={facet_text} query_terms={q_text} fact={compact}",
                    )
                )

    if not rows:
        return ""
    selected_rows = sorted(rows, key=lambda row: (-row[0], row[1], row[2], row[3], row[4]))[:max_rows]
    selected_rows = sorted(selected_rows, key=lambda row: (row[1], row[2], row[3], row[4]))
    lines = [
        "Canonical Profile/Event Ledger:",
        "- Built deterministically from the retrieved sessions only; rows are evidence candidates, not final conclusions.",
        "- For preference/recommendation questions, use preference/goal/constraint rows as transferable personal context.",
        "- For temporal/current/latest questions, order rows by date and apply latest-wins only to the same entity/state.",
        "- For count/list/total questions, form Candidate Set and Deduped Set from included rows; cite L# ids.",
        "- Reject adjacent, hypothetical, generic, or out-of-window rows before the final answer.",
    ]
    for idx, (_weight, _date, _source_idx, _turn_idx, row) in enumerate(selected_rows, start=1):
        lines.append(f"L{idx}: {row}")
    ledger = "\n".join(lines)
    if len(ledger) > max_chars:
        return ledger[: max_chars - 63].rstrip() + "\n[canonical profile/event ledger truncated]"
    return ledger


def profile_event_facets(sentence: str, *, role: str) -> list[str]:
    text = sentence.lower()
    facets: list[str] = []
    if role == "user" and re.search(r"\b(love|like|enjoy|prefer|favorite|favourite|hate|avoid|fan of|into)\b", text):
        facets.append("preference")
    if role == "user" and re.search(r"\b(can'?t|cannot|must|need to|have to|budget|deadline|allergy|limit|under\s+\$?\d+)\b", text):
        facets.append("constraint")
    if role == "user" and re.search(r"\b(want to|trying to|goal|aim|improve|working on|planning to|hope to|looking to)\b", text):
        facets.append("goal")
    if re.search(r"\b(current|currently|latest|recent|recently|upcoming|next|this week|changed|switched|started|stopped|no longer|used to)\b", text):
        facets.append("temporal_state")
    if re.search(r"\b(decided|chose|picked|selected|went with|settled on|committed)\b", text):
        facets.append("decision")
    if DATE_RE.search(sentence) or TEMPORAL_RE.search(sentence):
        facets.append("dated_event")
    if NUMBER_RE.search(sentence):
        facets.append("quantity")
    if ACTION_RE.search(sentence) or UPDATE_RE.search(sentence) or PROFILE_EVENT_RE.search(sentence):
        facets.append("event")
    return list(dict.fromkeys(facets))


def build_aggregation_assembly_prompt(
    item: dict[str, Any],
    retrieved_sessions: list[str],
    *,
    top_k: int,
    max_session_chars: int,
    aggregation_max_candidates: int,
    aggregation_max_chars: int,
) -> tuple[str, dict[str, Any]]:
    date_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_dates"], strict=True))
    turns_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True))
    selected = [sid for sid in retrieved_sessions[:top_k] if sid in turns_by_sid]
    selected.sort(key=lambda sid: date_by_sid.get(sid, ""))
    assembly = build_deterministic_aggregation_assembly(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        max_candidates=aggregation_max_candidates,
        max_chars=aggregation_max_chars,
    )

    context_parts = []
    for i, sid in enumerate(selected, start=1):
        context_parts.append(
            f"### Retrieved Session {i}\n"
            + session_to_text(sid, date_by_sid[sid], turns_by_sid[sid], max_session_chars)
        )
    context = "\n\n".join(context_parts)
    prompt = (
        "I will give you a deterministic aggregation assembly plus full retrieved chat history.\n\n"
        "Use the aggregation assembly as the working set for count/list/cross-thread synthesis. "
        "First write Candidate Review: include or exclude each relevant C# candidate and explain dedupe decisions. "
        "Then write Deduped Set: one bullet per distinct included real-world item/event/value with C# citations. "
        "For pickup/return questions, prefer rows explicitly marked as pickup/return obligations, keep pickup and return obligations separate when both are stated, and exclude third-party returns such as someone returning borrowed clothing. "
        "For count questions, the Final Answer must be the count implied by the Deduped Set. "
        "For list questions, the Final Answer must list exactly the included deduped items. "
        "Use the full History Chats only to verify or clarify the C# candidates, not to invent uncited candidates. "
        "If the assembly lacks enough evidence, the Final Answer must be exactly: "
        f"{UNANSWERABLE_RESPONSE}\n\n"
        f"{assembly['text']}\n\n"
        f"History Chats:\n\n{context}\n\n"
        f"Current Date: {item.get('question_date', '')}\n"
        f"Question: {item['question']}\n"
        "Candidate Review, Deduped Set, and Final Answer:"
    )
    report = {
        "coherent": assembly["coherent"],
        "candidate_count": assembly["candidate_count"],
        "dedupe_group_count": assembly["dedupe_group_count"],
    }
    return prompt, report


def build_count_list_ledger_prompt(
    item: dict[str, Any],
    retrieved_sessions: list[str],
    *,
    top_k: int,
    max_session_chars: int,
    max_candidates: int,
    max_chars: int,
    typed_only: bool = False,
    semantic_counting: bool = False,
) -> tuple[str, dict[str, Any]]:
    date_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_dates"], strict=True))
    turns_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True))
    selected = [sid for sid in retrieved_sessions[:top_k] if sid in turns_by_sid]
    selected.sort(key=lambda sid: date_by_sid.get(sid, ""))
    ledger = build_count_list_ledger(
        item,
        selected,
        date_by_sid,
        turns_by_sid,
        max_candidates=max_candidates,
        max_chars=max_chars,
        typed_only=typed_only,
        semantic_counting=semantic_counting,
    )

    context_parts = []
    for i, sid in enumerate(selected, start=1):
        context_parts.append(
            f"### Retrieved Session {i}\n"
            + session_to_text(sid, date_by_sid[sid], turns_by_sid[sid], max_session_chars)
        )
    context = "\n\n".join(context_parts)
    computed_answer = ledger.get("computed_answer")
    if computed_answer is not None:
        count_rule = (
            f"The Python computed answer is {computed_answer}. "
            "Use that answer as the Final Answer unless the accepted groups below clearly include an out-of-scope candidate."
        )
    elif ledger.get("computed_count") is not None:
        count_rule = (
            f"The Python computed count is {ledger['computed_count']}. "
            "Use that count as the Final Answer unless the accepted groups below clearly include an out-of-scope candidate."
        )
    else:
        count_rule = "Python emitted candidate groups but did not compute a numeric answer for this question type; use the accepted groups as the working set."
    prompt = (
        "I will give you a Python Count/List Ledger plus full retrieved chat history.\n\n"
        "The ledger is the primary working set for count/list questions. "
        "Python has already extracted candidate rows and deduplicated repeated mentions into groups. "
        f"{count_rule} "
        "Do not count raw sessions. Do not recount repeated mentions. "
        "Use History Chats only to verify or reject listed groups, not to invent uncited groups. "
        "First write Accepted Groups: include each group you keep with its G#/C# citations. "
        "Then write Excluded Groups if any accepted ledger group is out of scope. "
        "Then write Final Answer as one concise line. "
        "If the ledger lacks enough evidence, the Final Answer must be exactly: "
        f"{UNANSWERABLE_RESPONSE}\n\n"
        f"{ledger['text']}\n\n"
        f"History Chats:\n\n{context}\n\n"
        f"Current Date: {item.get('question_date', '')}\n"
        f"Question: {item['question']}\n"
        "Accepted Groups, Excluded Groups, and Final Answer:"
    )
    report = {
        "coherent": ledger["coherent"],
        "candidate_count": ledger["candidate_count"],
        "dedupe_group_count": ledger["dedupe_group_count"],
        "computed_count": ledger["computed_count"],
        "computed_answer": ledger.get("computed_answer"),
        "computed_answer_kind": ledger["computed_answer_kind"],
        "typed_ledger": ledger.get("typed_ledger"),
        "high_precision_count": ledger["high_precision_count"],
    }
    return prompt, report


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from a model response that may include light wrapping."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def normalize_extracted_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = NUMBER_RE.search(value.replace(",", ""))
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def format_computed_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_agent_ledger_answer(answer: str | None, unit: str) -> str:
    if answer is None:
        return UNANSWERABLE_RESPONSE
    unit = unit.strip()
    if unit.upper() in {"USD", "$"}:
        number = normalize_extracted_number(answer)
        if number is not None:
            return f"${int(number):,}" if abs(number - round(number)) < 1e-9 else f"${number:,.2f}"
    if unit and not answer.lower().endswith(unit.lower()):
        return f"{answer} {unit}"
    return answer


DIMENSIONAL_MODIFIER_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*[- ]\s*"
    r"(gallons?|gal|liters?|litres?|l|ounces?|oz|pounds?|lbs?|inches?|inch|feet|foot|ft|"
    r"centimeters?|centimetres?|cm|millimeters?|millimetres?|mm|kilograms?|kg|quarts?|cups?|ml)\b",
    re.I,
)


def normalize_dimensional_modifier_text(text: str) -> set[str]:
    modifiers: set[str] = set()
    unit_aliases = {
        "gallon": "gallon",
        "gallons": "gallon",
        "gal": "gallon",
        "liter": "liter",
        "liters": "liter",
        "litre": "liter",
        "litres": "liter",
        "l": "liter",
        "ounce": "ounce",
        "ounces": "ounce",
        "oz": "ounce",
        "pound": "pound",
        "pounds": "pound",
        "lb": "pound",
        "lbs": "pound",
        "inch": "inch",
        "inches": "inch",
        "foot": "foot",
        "feet": "foot",
        "ft": "foot",
        "centimeter": "centimeter",
        "centimeters": "centimeter",
        "centimetre": "centimeter",
        "centimetres": "centimeter",
        "cm": "centimeter",
        "millimeter": "millimeter",
        "millimeters": "millimeter",
        "millimetre": "millimeter",
        "millimetres": "millimeter",
        "mm": "millimeter",
        "kilogram": "kilogram",
        "kilograms": "kilogram",
        "kg": "kilogram",
        "quart": "quart",
        "quarts": "quart",
        "cup": "cup",
        "cups": "cup",
        "ml": "milliliter",
    }
    for match in DIMENSIONAL_MODIFIER_RE.finditer(text):
        number = format_computed_number(float(match.group(1)))
        unit = unit_aliases.get(match.group(2).lower(), match.group(2).lower())
        modifiers.add(f"{number} {unit}")
    return modifiers


def build_count_list_agent_extraction_prompt(
    item: dict[str, Any],
    retrieved_sessions: list[str],
    *,
    top_k: int,
    max_session_chars: int,
    include_calculation_plan: bool = False,
) -> tuple[str, dict[str, str]]:
    date_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_dates"], strict=True))
    turns_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True))
    selected = [sid for sid in retrieved_sessions[:top_k] if sid in turns_by_sid]
    selected.sort(key=lambda sid: date_by_sid.get(sid, ""))

    source_ids: dict[str, str] = {}
    context_parts = []
    for i, sid in enumerate(selected, start=1):
        source_id = f"S{i}"
        source_ids[source_id] = sid
        context_parts.append(
            f"### {source_id}\n"
            + session_to_text(sid, date_by_sid[sid], turns_by_sid[sid], max_session_chars)
        )
    context = "\n\n".join(context_parts)
    calculation_shape = (
        ',\n  "calculation": {"function": "count_rows|sum_values|difference|average_values|list_entities", "rows": ["A1", "A2"]}\n'
        if include_calculation_plan
        else "\n"
    )
    calculation_rules = (
        "- Optionally include calculation to tell Python how to reduce accepted rows. Rows are addressed as A1, A2, etc. in accepted-list order.\n"
        "- calculation may only use: count_rows, sum_values, difference, average_values, or list_entities.\n"
        if include_calculation_plan
        else ""
    )
    prompt = (
        "Extract a structured count/list ledger from the retrieved chat history.\n"
        "Do not answer in prose. Return JSON only.\n\n"
        "JSON shape:\n"
        "{\n"
        '  "operation": "count|sum|average|difference|duration|list|unknown",\n'
        '  "unit": "short unit or empty string",\n'
        '  "answerability": "sufficient|insufficient",\n'
        '  "accepted": [\n'
        '    {"entity": "deduped real-world item/event/value", "value": 1, "source": "S1", "evidence": "short quote or paraphrase"}\n'
        "  ],\n"
        '  "rejected": [\n'
        '    {"source": "S2", "reason": "why this candidate is out of scope"}\n'
        "  ],\n"
        '  "dedupe_notes": ["..."],\n'
        '  "missing_or_uncertain": ["..."]'
        + calculation_shape
        + "}\n\n"
        "Rules:\n"
        "- Extract only candidates directly supported by the retrieved sessions.\n"
        "- Use source ids exactly as shown: S1, S2, etc.\n"
        "- For count/list questions, one accepted row should represent one distinct included real-world item/event unless the question asks for repeated occurrences.\n"
        "- For sum/average questions, every accepted row must include a numeric value.\n"
        "- For remaining-to-go questions such as points still needed, use operation=difference with the current value and target value.\n"
        "- For combined totals such as page count of multiple books or total views, use operation=sum.\n"
        "- For elapsed-time questions, use operation=duration and one accepted row with the computed numeric value and unit.\n"
        + calculation_rules
        + "- Apply temporal filters in the question such as current, latest, before, after, last month, or currently.\n"
        "- Put ambiguous or out-of-scope evidence in rejected or missing_or_uncertain, not accepted.\n"
        "- If the retrieved history lacks enough evidence, set answerability to insufficient and accepted to [].\n\n"
        f"History Chats:\n\n{context}\n\n"
        f"Current Date: {item.get('question_date', '')}\n"
        f"Question: {item['question']}\n"
        "JSON:"
    )
    return prompt, source_ids


def validate_count_list_agent_extraction(
    raw_text: str,
    source_ids: dict[str, str],
    *,
    allow_calculation_plan: bool = False,
    derive_schema_operation: bool = False,
    question: str = "",
) -> dict[str, Any]:
    data = extract_json_object(raw_text)
    if data is None:
        return {"coherent": False, "reason": "invalid_json", "raw_text": raw_text[:1000]}

    operation = str(data.get("operation", "unknown")).strip().lower()
    if operation not in {"count", "sum", "average", "difference", "duration", "list", "unknown"}:
        operation = "unknown"
    answerability = str(data.get("answerability", "")).strip().lower()
    accepted_raw = data.get("accepted")
    accepted_rows = accepted_raw if isinstance(accepted_raw, list) else []
    rejected_raw = data.get("rejected")
    rejected_rows = rejected_raw if isinstance(rejected_raw, list) else []

    accepted: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    allowed_sources = set(source_ids)
    for row in accepted_rows:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", "")).strip()
        entity = str(row.get("entity", "")).strip()
        row_sources = [part.strip() for part in re.split(r"[,;/]", source) if part.strip()]
        if not row_sources and isinstance(row.get("sources"), list):
            row_sources = [str(part).strip() for part in row["sources"] if str(part).strip()]
        if not row_sources or any(part not in allowed_sources for part in row_sources) or not entity:
            continue
        source_key = ",".join(row_sources)
        evidence = str(row.get("evidence", "")).strip()[:300]
        key = (
            source_key,
            re.sub(r"\s+", " ", entity.lower()),
            re.sub(r"\s+", " ", evidence.lower()),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        value = normalize_extracted_number(row.get("value"))
        accepted.append(
            {
                "entity": entity,
                "source": source_key,
                "session_id": ",".join(source_ids[part] for part in row_sources),
                "value": value,
                "evidence": evidence,
            }
        )

    coherent = answerability == "sufficient" and operation != "unknown" and bool(accepted)
    values = [row["value"] for row in accepted if row["value"] is not None]
    computed_answer: str | None = None
    calculation_report: dict[str, Any] | None = None
    if coherent and allow_calculation_plan:
        calculation_report = execute_count_list_calculation_plan(data.get("calculation"), accepted)
        if calculation_report.get("coherent"):
            computed_answer = str(calculation_report.get("computed_answer"))
    schema_report: dict[str, Any] | None = None
    if coherent and derive_schema_operation:
        schema_report = derive_count_list_schema_calculation(
            operation=operation,
            unit=str(data.get("unit", "")).strip(),
            accepted=accepted,
            question=question,
        )
        if schema_report.get("coherent"):
            operation = str(schema_report.get("operation", operation))
            computed_answer = str(schema_report.get("computed_answer"))
    if operation == "count" and len(accepted) == 1 and values and str(data.get("unit", "")).lower() in {
        "day",
        "days",
        "hour",
        "hours",
        "week",
        "weeks",
        "month",
        "months",
    }:
        operation = "duration"
    if computed_answer is not None:
        pass
    elif coherent and operation == "count":
        if values and len(values) == len(accepted) and any(abs(value - 1.0) > 1e-9 for value in values):
            computed_answer = format_computed_number(sum(values))
        else:
            computed_answer = str(len(accepted))
    elif coherent and operation == "list":
        computed_answer = ", ".join(row["entity"] for row in accepted)
    elif coherent and operation == "sum" and len(values) == len(accepted):
        computed_answer = format_computed_number(sum(values))
    elif coherent and operation == "average" and len(values) == len(accepted):
        computed_answer = format_computed_number(sum(values) / len(values))
    elif coherent and operation == "difference" and len(values) == len(accepted) and len(values) == 1:
        computed_answer = format_computed_number(values[0])
    elif coherent and operation == "difference" and len(values) == len(accepted) and len(values) >= 2:
        computed_answer = format_computed_number(max(values) - min(values))
    elif coherent and operation == "duration" and len(values) == 1:
        computed_answer = format_computed_number(values[0])
    elif coherent:
        coherent = False

    if coherent and len(accepted) > 30:
        coherent = False

    return {
        "coherent": coherent,
        "reason": "ok" if coherent else "validation_failed",
        "question": question,
        "operation": operation,
        "unit": str(data.get("unit", "")).strip()[:80],
        "answerability": answerability,
        "accepted": accepted,
        "rejected": rejected_rows[:20],
        "dedupe_notes": data.get("dedupe_notes") if isinstance(data.get("dedupe_notes"), list) else [],
        "missing_or_uncertain": data.get("missing_or_uncertain") if isinstance(data.get("missing_or_uncertain"), list) else [],
        "calculation": calculation_report,
        "schema_calculation": schema_report,
        "computed_answer": computed_answer,
        "accepted_count": len(accepted),
    }


def execute_count_list_calculation_plan(
    calculation: Any,
    accepted: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(calculation, dict):
        return {"coherent": False, "reason": "missing_calculation"}
    function = str(calculation.get("function", "")).strip().lower()
    if function not in {"count_rows", "sum_values", "difference", "average_values", "list_entities"}:
        return {"coherent": False, "reason": "unknown_function", "function": function}

    raw_rows = calculation.get("rows")
    if isinstance(raw_rows, list) and raw_rows:
        row_indexes: list[int] = []
        for raw_ref in raw_rows:
            match = re.fullmatch(r"A(\d+)", str(raw_ref).strip(), flags=re.I)
            if not match:
                return {"coherent": False, "reason": "bad_row_ref", "function": function}
            index = int(match.group(1)) - 1
            if index < 0 or index >= len(accepted):
                return {"coherent": False, "reason": "row_ref_out_of_range", "function": function}
            row_indexes.append(index)
    else:
        row_indexes = list(range(len(accepted)))

    rows = [accepted[index] for index in row_indexes]
    values = [row["value"] for row in rows if row.get("value") is not None]
    computed_answer: str | None = None
    if function == "count_rows":
        computed_answer = str(len(rows))
    elif function == "sum_values" and len(values) == len(rows):
        computed_answer = format_computed_number(sum(values))
    elif function == "difference" and len(values) == len(rows) and len(values) == 1:
        computed_answer = format_computed_number(values[0])
    elif function == "difference" and len(values) == len(rows) and len(values) >= 2:
        computed_answer = format_computed_number(max(values) - min(values))
    elif function == "average_values" and len(values) == len(rows) and values:
        computed_answer = format_computed_number(sum(values) / len(values))
    elif function == "list_entities":
        computed_answer = ", ".join(row["entity"] for row in rows)
    if computed_answer is None:
        return {"coherent": False, "reason": "execution_failed", "function": function}
    return {
        "coherent": True,
        "reason": "ok",
        "function": function,
        "rows": [f"A{index + 1}" for index in row_indexes],
        "computed_answer": computed_answer,
    }


def derive_count_list_schema_calculation(
    *,
    operation: str,
    unit: str,
    accepted: list[dict[str, Any]],
    question: str,
) -> dict[str, Any]:
    operation = operation.strip().lower()
    unit_norm = unit.strip().lower()
    question_norm = question.strip().lower()
    values = [row["value"] for row in accepted if row.get("value") is not None]
    if not accepted:
        return {"coherent": False, "reason": "no_rows"}

    if operation in {"sum", "average", "difference", "duration"}:
        return {"coherent": False, "reason": "operation_already_specific"}

    if operation not in {"count", "list"}:
        return {"coherent": False, "reason": "unsupported_operation"}

    if len(values) == len(accepted) and values:
        additive_unit = unit_norm in {
            "day",
            "days",
            "hour",
            "hours",
            "minute",
            "minutes",
            "week",
            "weeks",
            "month",
            "months",
            "page",
            "pages",
            "point",
            "points",
            "ride",
            "rides",
            "view",
            "views",
            "usd",
            "$",
            "dollar",
            "dollars",
            "mile",
            "miles",
            "km",
            "kilometer",
            "kilometers",
        }
        asks_for_total = bool(
            re.search(
                r"\b(total|sum|combined|altogether|in all|overall)\b|"
                r"\bhow many\b.*\b(pages?|points?|rides?|views?|hours?|days?|minutes?|dollars?|miles?)\b",
                question_norm,
            )
        )
        if additive_unit or asks_for_total:
            return {
                "coherent": True,
                "reason": "numeric_rows_additive_schema",
                "operation": "sum",
                "function": "sum_values",
                "computed_answer": format_computed_number(sum(values)),
            }

    if operation == "count" and (not values or all(abs(value - 1.0) < 1e-9 for value in values)):
        return {
            "coherent": True,
            "reason": "distinct_rows_count_schema",
            "operation": "count",
            "function": "count_rows",
            "computed_answer": str(len(accepted)),
        }

    return {"coherent": False, "reason": "no_schema_rule"}


def build_count_list_agent_hypothesis(report: dict[str, Any]) -> str:
    accepted_lines = []
    for idx, row in enumerate(report["accepted"], start=1):
        value = "" if row["value"] is None else f", value={format_computed_number(float(row['value']))}"
        evidence = f", evidence={row['evidence']}" if row["evidence"] else ""
        accepted_lines.append(
            f"- A{idx}: source={row['source']} session={row['session_id']}, entity={row['entity']}{value}{evidence}"
        )
    rejected = report.get("rejected") or []
    rejected_lines = []
    for idx, row in enumerate(rejected[:8], start=1):
        if isinstance(row, dict):
            rejected_lines.append(
                f"- R{idx}: source={row.get('source', '')}, reason={row.get('reason', '')}"
            )
    final = format_agent_ledger_answer(report.get("computed_answer"), str(report.get("unit", "")))
    return (
        "Accepted Rows:\n"
        + ("\n".join(accepted_lines) if accepted_lines else "- None")
        + "\n\nRejected Rows:\n"
        + ("\n".join(rejected_lines) if rejected_lines else "- None")
        + f"\n\nFinal Answer: {final}"
    )


def is_safe_count_list_agent_route(report: dict[str, Any]) -> bool:
    if not report.get("coherent"):
        return False
    if report.get("missing_or_uncertain"):
        return False

    operation = str(report.get("operation", "")).strip().lower()
    unit = str(report.get("unit", "")).strip().lower()
    accepted_count = int(report.get("accepted_count") or 0)

    if operation == "list":
        return False
    if operation == "count" and accepted_count <= 1:
        return False
    if operation == "count" and re.search(r"\bitems?\b", unit):
        return False
    if operation == "sum" and re.search(r"\bitems?\b", unit):
        return False
    if operation == "sum" and unit in {"usd", "$", "dollar", "dollars"} and accepted_count >= 4:
        return False
    question_modifiers = normalize_dimensional_modifier_text(str(report.get("question", "")))
    if question_modifiers:
        accepted_text = " ".join(
            f"{row.get('entity', '')} {row.get('evidence', '')}" for row in report.get("accepted", [])
        )
        accepted_modifiers = normalize_dimensional_modifier_text(accepted_text)
        if not question_modifiers.issubset(accepted_modifiers):
            return False
    return True


def is_unsupported_count_list_ledger_question(question: str) -> bool:
    question_norm = question.strip().lower()
    if not re.search(r"\bpercent(?:age)?\b|%", question_norm):
        return False
    if re.search(r"\b(?:what|which)\s+(?:percent|percentage)\s+of\b", question_norm):
        return True
    if re.search(r"\b(?:percent|percentage)\s+of\b", question_norm):
        return True
    if re.search(
        r"\b(?:higher|lower|greater|less|more|fewer|compare|compared|versus|vs\.?|than)\b",
        question_norm,
    ):
        return True
    return False


def is_safe_count_list_agent_plan_route(report: dict[str, Any]) -> bool:
    if not is_safe_count_list_agent_route(report):
        return False
    calculation = report.get("calculation")
    if not isinstance(calculation, dict) or not calculation.get("coherent"):
        return False
    function = str(calculation.get("function", "")).strip().lower()
    if function == "count_rows":
        return False
    if function == "difference" and str(calculation.get("computed_answer")) in {"0", "0.0"}:
        return False
    return True


def is_safe_count_list_agent_schema_route(report: dict[str, Any]) -> bool:
    if not is_safe_count_list_agent_route(report):
        return False
    schema_calculation = report.get("schema_calculation")
    if not isinstance(schema_calculation, dict) or not schema_calculation.get("coherent"):
        return True
    function = str(schema_calculation.get("function", "")).strip().lower()
    if function == "count_rows":
        unit = str(report.get("unit", "")).strip().lower()
        if re.search(r"\bitems?\b", unit):
            return False
    return True


def is_safe_count_list_agent_blend_route(report: dict[str, Any]) -> bool:
    """Stricter schema route used after high-precision typed ledgers abstain."""
    if not is_safe_count_list_agent_schema_route(report):
        return False
    schema_calculation = report.get("schema_calculation")
    if not isinstance(schema_calculation, dict) or not schema_calculation.get("coherent"):
        return True
    function = str(schema_calculation.get("function", "")).strip().lower()
    if function != "count_rows":
        return True

    unit = str(report.get("unit", "")).strip().lower()
    if re.search(r"\btimes?\b", unit):
        return False

    accepted = report.get("accepted") if isinstance(report.get("accepted"), list) else []
    sources = {
        source
        for row in accepted
        for source in str(row.get("source", "")).split(",")
        if source.strip()
    }
    if len(accepted) >= 5 and len(accepted) > len(sources) + 2:
        return False
    if len(accepted) >= 4 and len(sources) <= 1:
        return False
    return True


def build_count_list_agent_answer_prompt(
    item: dict[str, Any],
    report: dict[str, Any],
) -> str:
    accepted_lines = []
    for idx, row in enumerate(report["accepted"], start=1):
        value = "" if row["value"] is None else f", value={format_computed_number(float(row['value']))}"
        evidence = f", evidence={row['evidence']}" if row["evidence"] else ""
        accepted_lines.append(
            f"A{idx}: source={row['source']} session={row['session_id']}, entity={row['entity']}{value}{evidence}"
        )
    ledger = "\n".join(accepted_lines)
    computed = report.get("computed_answer")
    return (
        "Answer the count/list aggregation question from the validated structured ledger.\n\n"
        "Python has validated the extractor output against source ids and computed the result. "
        "Do not inspect or recount raw history. Use the computed answer unless an accepted row is obviously out of scope. "
        "If it is out of scope, say the information is not available rather than inventing a new answer.\n\n"
        f"Operation: {report.get('operation')}\n"
        f"Unit: {report.get('unit')}\n"
        f"Computed Answer: {computed}\n"
        f"Accepted Rows:\n{ledger}\n\n"
        f"Question: {item['question']}\n"
        "Accepted Rows, Exclusions if any, and Final Answer:"
    )


def build_multi_session_evidence_set_prompt(
    item: dict[str, Any],
    retrieved_sessions: list[str],
    *,
    top_k: int,
    pool_k: int,
    max_session_chars: int,
    max_candidates: int,
    max_chars: int,
) -> tuple[str, dict[str, Any]]:
    date_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_dates"], strict=True))
    turns_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True))
    candidate_pool = [sid for sid in retrieved_sessions[:pool_k] if sid in turns_by_sid]
    evidence_set = build_multi_session_evidence_set(
        item,
        candidate_pool,
        date_by_sid,
        turns_by_sid,
        base_top_k=top_k,
        max_candidates=max_candidates,
        max_chars=max_chars,
    )

    context_ids: list[str] = []
    for sid in evidence_set["selected_source_ids"]:
        if sid in turns_by_sid and sid not in context_ids:
            context_ids.append(sid)
    for sid in retrieved_sessions[:top_k]:
        if sid in turns_by_sid and sid not in context_ids:
            context_ids.append(sid)
    context_ids.sort(key=lambda sid: date_by_sid.get(sid, ""))

    context_parts = []
    for i, sid in enumerate(context_ids, start=1):
        context_parts.append(
            f"### Evidence Session {i}\n"
            + session_to_text(sid, date_by_sid[sid], turns_by_sid[sid], max_session_chars)
        )
    context = "\n\n".join(context_parts)
    prompt = (
        "I will give you a compact multi-session evidence set plus verifying chat history.\n\n"
        "Use the Multi-Session Evidence Set as a coverage checklist for count/list/cross-thread synthesis. "
        "Do not spend tokens reviewing every excluded row. "
        "First write Evidence Used: only the C# rows that directly answer the question, with a short reason for inclusion. "
        "Then write Deduped Set: one bullet per distinct included real-world item/event/value with C# citations. "
        "For count questions, the Final Answer must be the count implied by the Deduped Set. "
        "For list questions, the Final Answer must list exactly the included deduped items. "
        "Ignore rows that are only adjacent, hypothetical, generic advice, questions, or unrelated examples. "
        "Use History Chats to verify or clarify C# candidates and to reject irrelevant rows. "
        "If the evidence set and verifying chats lack enough evidence, the Final Answer must be exactly: "
        f"{UNANSWERABLE_RESPONSE}\n\n"
        f"{evidence_set['text']}\n\n"
        f"History Chats:\n\n{context}\n\n"
        f"Current Date: {item.get('question_date', '')}\n"
        f"Question: {item['question']}\n"
        "Evidence Used, Deduped Set, and Final Answer:"
    )
    report = {
        "coherent": evidence_set["coherent"],
        "candidate_count": evidence_set["candidate_count"],
        "dedupe_group_count": evidence_set["dedupe_group_count"],
        "base_group_count": evidence_set["base_group_count"],
        "pool_added_group_count": evidence_set["pool_added_group_count"],
        "novel_decisive_group_count": evidence_set["novel_decisive_group_count"],
        "added_distraction_risk": evidence_set["added_distraction_risk"],
        "marginal_utility": evidence_set["marginal_utility"],
        "selected_source_ids": evidence_set["selected_source_ids"],
        "selected_source_ranks": evidence_set["selected_source_ranks"],
        "confidence": evidence_set["confidence"],
        "confidence_reasons": evidence_set["confidence_reasons"],
        "strong_candidate_count": evidence_set["strong_candidate_count"],
        "top_candidate_weight": evidence_set["top_candidate_weight"],
        "pool_k": pool_k,
    }
    return prompt, report


def build_multi_session_source_select_prompt(
    item: dict[str, Any],
    retrieved_sessions: list[str],
    *,
    top_k: int,
    pool_k: int,
    max_session_chars: int,
    max_candidates: int,
    max_chars: int,
) -> tuple[str, dict[str, Any]]:
    date_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_dates"], strict=True))
    turns_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True))
    candidate_pool = [sid for sid in retrieved_sessions[:pool_k] if sid in turns_by_sid]
    evidence_set = build_multi_session_evidence_set(
        item,
        candidate_pool,
        date_by_sid,
        turns_by_sid,
        base_top_k=top_k,
        max_candidates=max_candidates,
        max_chars=max_chars,
    )
    selected_retrieved: list[str] = []
    for sid in evidence_set["selected_source_ids"]:
        if sid in turns_by_sid and sid not in selected_retrieved:
            selected_retrieved.append(sid)
    for sid in retrieved_sessions[:top_k]:
        if sid in turns_by_sid and sid not in selected_retrieved:
            selected_retrieved.append(sid)

    prompt = build_answer_prompt(
        item,
        selected_retrieved,
        top_k=len(selected_retrieved),
        max_session_chars=max_session_chars,
        cot=False,
        source_aware=True,
        source_sufficiency=False,
        token_evidence=False,
        evidence_packet=False,
        fusion_evidence_map=False,
        fusion_evidence_map_max_items=0,
        fusion_evidence_map_max_chars=0,
        evidence_packet_max_items=0,
        evidence_packet_max_chars=0,
        token_evidence_max_lines=0,
        token_evidence_signals=False,
        count_list_mode=False,
        strict_missing_final_answer=False,
        preference_support_packet="off",
        preference_support_max_items=0,
        preference_support_per_source=0,
    )
    report = {
        "coherent": evidence_set["coherent"],
        "candidate_count": evidence_set["candidate_count"],
        "dedupe_group_count": evidence_set["dedupe_group_count"],
        "base_group_count": evidence_set["base_group_count"],
        "pool_added_group_count": evidence_set["pool_added_group_count"],
        "novel_decisive_group_count": evidence_set["novel_decisive_group_count"],
        "added_distraction_risk": evidence_set["added_distraction_risk"],
        "marginal_utility": evidence_set["marginal_utility"],
        "selected_source_ids": evidence_set["selected_source_ids"],
        "selected_source_ranks": evidence_set["selected_source_ranks"],
        "confidence": evidence_set["confidence"],
        "confidence_reasons": evidence_set["confidence_reasons"],
        "strong_candidate_count": evidence_set["strong_candidate_count"],
        "top_candidate_weight": evidence_set["top_candidate_weight"],
        "selected_context_count": len(selected_retrieved),
        "pool_k": pool_k,
    }
    return prompt, report


def _selected_session_texts(
    item: dict[str, Any],
    retrieved_sessions: list[str],
    *,
    top_k: int,
    max_session_chars: int,
) -> list[tuple[int, str, str]]:
    date_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_dates"], strict=True))
    turns_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True))
    selected = [sid for sid in retrieved_sessions[:top_k] if sid in turns_by_sid]
    selected.sort(key=lambda sid: date_by_sid.get(sid, ""))
    return [
        (i, sid, session_to_text(sid, date_by_sid[sid], turns_by_sid[sid], max_session_chars))
        for i, sid in enumerate(selected, start=1)
    ]


def build_evidence_contract_prompt(
    item: dict[str, Any],
    primary_sessions: list[str],
    supporting_sessions: list[str],
    *,
    primary_top_k: int,
    supporting_top_k: int,
    max_session_chars: int,
) -> str:
    primary = _selected_session_texts(
        item,
        primary_sessions,
        top_k=primary_top_k,
        max_session_chars=max_session_chars,
    )
    primary_ids = {sid for _, sid, _text in primary}
    supporting_candidates = [sid for sid in supporting_sessions if sid not in primary_ids]
    supporting = _selected_session_texts(
        item,
        supporting_candidates,
        top_k=supporting_top_k,
        max_session_chars=max_session_chars,
    )

    primary_context = "\n\n".join(
        f"### Primary Evidence Session {i}\n{text}" for i, _sid, text in primary
    )
    supporting_context = "\n\n".join(
        f"### Supporting Evidence Session {i}\n{text}" for i, _sid, text in supporting
    )
    if not supporting_context:
        supporting_context = "(No separate supporting evidence sessions were selected.)"

    return (
        "I will give you chat history retrieved through two evidence lanes.\n\n"
        "Primary Evidence is the high-precision lane selected by explicit temporal/query metadata filters. "
        "For questions involving dates, currentness, updates, latest/previous state, or a requested time window, use Primary Evidence as the authority for the time-constrained answer.\n\n"
        "Supporting Evidence is broader companion context from normal retrieval. Use it only to clarify entities, fill non-conflicting background, recover companion facts, or decide that Primary Evidence is incomplete. "
        "Do not let Supporting Evidence override the date window or current/previous state established by Primary Evidence unless Primary Evidence is clearly insufficient.\n\n"
        "Write Source Notes with separate Primary and Supporting bullets. Then write Final Answer with the concise answer. "
        "If Primary Evidence and Supporting Evidence together still do not contain enough information, say that the information is not available in the provided history. "
        "For recommendation or preference questions, use indirect personal context such as interests, tools, constraints, goals, and prior activities.\n\n"
        f"Primary Evidence:\n\n{primary_context}\n\n"
        f"Supporting Evidence:\n\n{supporting_context}\n\n"
        f"Current Date: {item.get('question_date', '')}\n"
        f"Question: {item['question']}\n"
        "Source Notes and Final Answer:"
    )


def build_source_set_aware_prompt(
    item: dict[str, Any],
    primary_sessions: list[str],
    expanded_sessions: list[str],
    *,
    primary_top_k: int,
    companion_top_k: int,
    max_session_chars: int,
) -> str:
    primary = _selected_session_texts(
        item,
        primary_sessions,
        top_k=primary_top_k,
        max_session_chars=max_session_chars,
    )
    primary_ids = {sid for _i, sid, _text in primary}
    companion_candidates = [sid for sid in expanded_sessions if sid not in primary_ids]
    companions = _selected_session_texts(
        item,
        companion_candidates,
        top_k=companion_top_k,
        max_session_chars=max_session_chars,
    )

    primary_context = "\n\n".join(
        f"### Primary Source {i}\n{text}" for i, _sid, text in primary
    )
    companion_context = "\n\n".join(
        f"### Added Companion Source {i}\n{text}" for i, _sid, text in companions
    )
    if not companion_context:
        companion_context = "(No added companion sources were selected.)"

    return (
        "I will give you chat history retrieved through two source sets.\n\n"
        "Primary Sources are the original high-confidence retrieved sessions. "
        "Treat these as the authority for the baseline answer.\n\n"
        "Added Companion Sources come from a targeted expansion pass. "
        "They are not automatically authoritative. Use them only when they add a concrete "
        "candidate item, entity, event, date, count component, or companion fact that is "
        "needed to answer the question. Ignore companion sources that only repeat, distract, "
        "or loosely match the question.\n\n"
        "First write Companion Audit: for each added companion source, state "
        "changes_candidate_set: yes or changes_candidate_set: no, with a short reason.\n"
        "Then write Source Notes with separate Primary and Companion bullets. "
        "For count, list, total, cross-thread, and comparison questions, write Candidate Set "
        "and Deduped Set before the final answer. Count each real-world item/event once unless "
        "the question asks for mentions. For temporal wording, preserve ordering and current "
        "versus previous state. If companions do not change the candidate set, answer from "
        "Primary Sources only. If the combined source sets still lack required evidence, say "
        "that the information is not available in the provided history.\n\n"
        f"Primary Sources:\n\n{primary_context}\n\n"
        f"Added Companion Sources:\n\n{companion_context}\n\n"
        f"Current Date: {item.get('question_date', '')}\n"
        f"Question: {item['question']}\n"
        "Companion Audit, Source Notes, Candidate Set/Deduped Set if needed, and Final Answer:"
    )


def build_structured_extraction_prompt(
    item: dict[str, Any],
    retrieved_sessions: list[str],
    *,
    top_k: int,
    max_session_chars: int,
    include_question_type: bool,
) -> str:
    date_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_dates"], strict=True))
    turns_by_sid = dict(zip(item["haystack_session_ids"], item["haystack_sessions"], strict=True))
    selected = [sid for sid in retrieved_sessions[:top_k] if sid in turns_by_sid]
    selected.sort(key=lambda sid: date_by_sid.get(sid, ""))

    context_parts = []
    for i, sid in enumerate(selected, start=1):
        context_parts.append(
            f"### Retrieved Session {i}\n"
            + session_to_text(sid, date_by_sid[sid], turns_by_sid[sid], max_session_chars)
        )
    context = "\n\n".join(context_parts)
    question_type_hint = f"Question Type: {item['question_type']}\n" if include_question_type else ""
    return (
        "Extract source-grounded facts needed to answer the question from retrieved chat history.\n"
        "Do not answer the question yet.\n\n"
        "Return compact JSON only with this shape:\n"
        "{\n"
        '  "question_kind": "infer from the question text",\n'
        '  "required_facts": ["..."],\n'
        '  "facts": [\n'
        '    {"source": "Retrieved Session N", "date": "...", "fact": "...", "value": "...", "entity": "...", "event": "...", "currentness": "current|previous|unknown"}\n'
        "  ],\n"
        '  "dedupe_notes": ["..."],\n'
        '  "temporal_order": ["..."],\n'
        '  "missing_or_uncertain": ["..."]\n'
        "}\n\n"
        "Rules:\n"
        "- Extract only facts supported by the retrieved sessions.\n"
        "- For counts/lists/totals, include every candidate item/event/value that may affect the result.\n"
        "- For temporal/current/latest questions, preserve dates and note whether facts are current, previous, or superseded.\n"
        "- For comparisons, extract both sides and the attributes being compared.\n"
        "- For preference/recommendation questions, extract user-specific preferences, interests, constraints, goals, and prior activities.\n"
        "- If the evidence is incomplete, record the gap in missing_or_uncertain.\n\n"
        f"History Chats:\n\n{context}\n\n"
        f"Current Date: {item.get('question_date', '')}\n"
        f"{question_type_hint}"
        f"Question: {item['question']}\n"
        "JSON:"
    )


def build_structured_answer_prompt(
    item: dict[str, Any],
    structured_facts: str,
    *,
    include_question_type: bool,
) -> str:
    question_type_hint = f"Question Type: {item['question_type']}\n" if include_question_type else ""
    return (
        "Answer the question using the structured facts extracted from retrieved chat history.\n"
        "The structured facts may contain candidate facts, dedupe notes, temporal ordering, and missing/uncertain evidence.\n\n"
        "Rules:\n"
        "- Base the answer only on the structured facts.\n"
        "- For counts/lists/totals, dedupe repeated mentions of the same real-world item/event and count distinct included items/events.\n"
        "- For temporal/current/latest questions, use temporal_order and currentness to choose the correct fact.\n"
        "- For comparison questions, make sure every requested side is represented.\n"
        "- If missing_or_uncertain shows a required fact is absent, say: The information is not available in the provided history.\n"
        "- Give a concise final answer.\n\n"
        f"{question_type_hint}"
        f"Question: {item['question']}\n"
        f"Structured Facts:\n{structured_facts}\n\n"
        "Final Answer:"
    )


def should_use_temporal_hybrid(item: dict[str, Any], router: str) -> bool:
    if router == "question_type":
        return item["question_type"] == "temporal-reasoning"
    if router == "query":
        return bool(TEMPORAL_QUERY_RE.search(item["question"]))
    raise ValueError(f"unknown temporal router: {router}")


def should_fallback_from_structured_answer(item: dict[str, Any], hypothesis: str) -> bool:
    text = hypothesis.lower()
    if any(marker in text for marker in STRUCTURED_FALLBACK_MARKERS):
        return True
    question = item["question"].lower()
    # These query-shape fallbacks are benchmark-ablation guardrails, not product
    # routing logic. They protect known weak structured-extraction shapes while
    # we measure whether the path is worth turning into a general query router.
    if re.search(r"\b(?:ago|since)\b", question):
        return True
    if re.search(r"\bhow many\b.+\bbefore\b", question):
        return True
    return False


def get_anscheck_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    if not abstention:
        if task in {"single-session-user", "single-session-assistant", "multi-session"}:
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the answer, answer no. \n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the answer, answer no. "
                "In addition, do not penalize off-by-one errors for the number of days. "
                "If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response from a model. "
                "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                "The model does not need to reflect all the points in the rubric. "
                "The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
                "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        raise NotImplementedError(task)
    template = (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. "
        "The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\n"
        "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    )
    return template.format(question, answer, response)


def get_answerability_prompt(item: dict[str, Any], hypothesis: str) -> str:
    return (
        "You are checking whether a model's answer is sufficiently supported by its own cited source notes.\n"
        "The model was asked to answer from retrieved chat history only. It may include Source Notes and a Final Answer.\n\n"
        "Answer YES only if the response contains concrete source-backed evidence for the requested answer. "
        "Answer NO if the response says the information is unavailable, if the source notes do not contain the requested fact, "
        "or if the final answer relies on inference not grounded in the source notes.\n\n"
        "For counting/list/total questions, answer YES only if the source notes enumerate enough distinct items/events/facts to support the count or list. "
        "For temporal questions, answer YES only if the source notes include the dates or update ordering needed for the answer. "
        "For comparison questions involving two or more named entities, events, locations, roles, or quantities, answer NO if any requested side is missing. "
        "If the notes say one requested entity/event/value is not mentioned, answer NO even if the model still gives a final answer. "
        "If the question asks about a specific role, place, course, item, or event, answer NO when the notes only support a similar but different role, place, course, item, or event. "
        "For recommendation/preference questions, answer YES if the source notes contain personal context that could reasonably support the recommendation.\n\n"
        f"Question: {item['question']}\n\n"
        f"Model Response:\n{hypothesis}\n\n"
        "Do the source notes sufficiently support an answer to the question? Answer YES or NO only."
    )


def answerability_label(args: argparse.Namespace, item: dict[str, Any], hypothesis: str) -> dict[str, Any]:
    model = answerability_model_for_item(args, item)
    response = cached_chat_completion(
        args,
        purpose="answerability",
        qid=item["question_id"],
        model=model,
        messages=[{"role": "user", "content": get_answerability_prompt(item, hypothesis)}],
        max_tokens=answerability_max_tokens_for_item(args, item),
    )
    return {
        "model": model,
        "answerable": parse_yes_no_label(response),
        "raw": response,
    }


def should_run_answerability_gate(args: argparse.Namespace, hypothesis: str) -> bool:
    if not args.answerability_trigger_missing:
        return True
    text = hypothesis.lower()
    return any(marker in text for marker in MISSING_EVIDENCE_MARKERS)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        print(json.dumps(row, ensure_ascii=False), file=f, flush=True)


def generate_hypotheses(args: argparse.Namespace) -> None:
    data = json.loads(args.data.read_text())
    data = filter_items(data, args)
    retrieval = json.loads(args.retrieval_artifact.read_text())
    rows_by_qid = {row["question_id"]: row for row in retrieval["rows"]}
    aggregation_assembly_mode = effective_aggregation_assembly_mode(args)
    fusion_evidence_map_mode = effective_fusion_evidence_map_mode(args)
    supporting_rows_by_qid: dict[str, dict[str, Any]] = {}
    primary_rows_by_qid: dict[str, dict[str, Any]] = {}
    if args.evidence_contract:
        if not args.supporting_retrieval_artifact:
            raise RuntimeError("--evidence-contract requires --supporting-retrieval-artifact")
        supporting_retrieval = json.loads(args.supporting_retrieval_artifact.read_text())
        supporting_rows_by_qid = {row["question_id"]: row for row in supporting_retrieval["rows"]}
    if args.source_set_aware:
        if not args.primary_retrieval_artifact:
            raise RuntimeError("--source-set-aware requires --primary-retrieval-artifact")
        primary_retrieval = json.loads(args.primary_retrieval_artifact.read_text())
        primary_rows_by_qid = {row["question_id"]: row for row in primary_retrieval["rows"]}

    done = {row["question_id"] for row in read_jsonl(args.hypotheses_out)}
    extracted_done = {row["question_id"]: row for row in read_jsonl(args.extract_out)}

    for i, item in enumerate(data, start=1):
        qid = item["question_id"]
        if qid in done:
            continue
        retrieval_row = rows_by_qid[qid]
        retrieved = retrieval_row["retrieved_sessions"]
        top_k_context = effective_top_k_context(item, args)
        route = "source_aware"
        aggregation_report: dict[str, Any] | None = None
        count_list_ledger_report: dict[str, Any] | None = None
        multi_session_evidence_set_report: dict[str, Any] | None = None
        if args.source_set_aware:
            primary_row = primary_rows_by_qid.get(qid)
            if not primary_row:
                raise RuntimeError(f"missing primary retrieval row for {qid}")
            route = "source_set_aware"
            prompt = build_source_set_aware_prompt(
                item,
                primary_row["retrieved_sessions"],
                retrieved,
                primary_top_k=top_k_context,
                companion_top_k=args.source_set_companion_top_k,
                max_session_chars=args.max_session_chars,
            )
            hypothesis = cached_chat_completion(
                args,
                purpose="generation",
                qid=qid,
                model=generation_model_for_item(args, item),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=generation_max_tokens_for_item(args, item),
            )
            append_jsonl(args.hypotheses_out, {"question_id": qid, "hypothesis": hypothesis, "route": route})
            print(f"[generate {i}/{len(data)}] {qid} {item['question_type']} route={route}", flush=True)
            continue
        if (
            args.multi_session_evidence_set in {"count_list", "source_select", "confidence_source_select"}
            and item["question_type"] == "multi-session"
            and is_count_list_question(item["question"])
        ):
            build_evidence_set_prompt = (
                build_multi_session_source_select_prompt
                if args.multi_session_evidence_set in {"source_select", "confidence_source_select"}
                else build_multi_session_evidence_set_prompt
            )
            prompt, multi_session_evidence_set_report = build_evidence_set_prompt(
                item,
                retrieved,
                top_k=top_k_context,
                pool_k=args.multi_session_evidence_set_pool_k,
                max_session_chars=args.max_session_chars,
                max_candidates=args.multi_session_evidence_set_max_candidates,
                max_chars=args.multi_session_evidence_set_max_chars,
            )
            if should_use_multi_session_evidence_set_selector(
                multi_session_evidence_set_report,
                args.multi_session_evidence_set,
                min_confidence=args.multi_session_evidence_set_min_confidence,
            ):
                route = f"multi_session_evidence_set_{args.multi_session_evidence_set}"
                hypothesis = cached_chat_completion(
                    args,
                    purpose="generation",
                    qid=qid,
                    model=generation_model_for_item(args, item),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=generation_max_tokens_for_item(args, item),
                )
                append_jsonl(
                    args.hypotheses_out,
                    {
                        "question_id": qid,
                        "hypothesis": hypothesis,
                        "route": route,
                        "multi_session_evidence_set_report": multi_session_evidence_set_report,
                    },
                )
                print(f"[generate {i}/{len(data)}] {qid} {item['question_type']} route={route}", flush=True)
                continue
        typed_numeric_question = bool(re.search(r"\baverage\b.*\bage\b|\bage\b.*\baverage\b", item["question"], re.I))
        if (
            args.count_list_ledger in {"python", "typed_python", "semantic_python", "agent_blend"}
            and item["question_type"] == "multi-session"
            and (is_count_list_question(item["question"]) or typed_numeric_question)
            and not is_unsupported_count_list_ledger_question(item["question"])
        ):
            prompt, count_list_ledger_report = build_count_list_ledger_prompt(
                item,
                retrieved,
                top_k=top_k_context,
                max_session_chars=args.max_session_chars,
                max_candidates=args.count_list_ledger_max_candidates,
                max_chars=args.count_list_ledger_max_chars,
                typed_only=args.count_list_ledger in {"typed_python", "semantic_python", "agent_blend"},
                semantic_counting=args.count_list_ledger == "semantic_python",
            )
            if count_list_ledger_report["coherent"]:
                route = (
                    "count_list_ledger_typed_python"
                    if args.count_list_ledger == "agent_blend"
                    else f"count_list_ledger_{args.count_list_ledger}"
                )
                hypothesis = cached_chat_completion(
                    args,
                    purpose="generation",
                    qid=qid,
                    model=generation_model_for_item(args, item),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=generation_max_tokens_for_item(args, item),
                )
                append_jsonl(
                    args.hypotheses_out,
                    {
                        "question_id": qid,
                        "hypothesis": hypothesis,
                        "route": route,
                        "count_list_ledger_report": count_list_ledger_report,
                    },
                )
                print(f"[generate {i}/{len(data)}] {qid} {item['question_type']} route={route}", flush=True)
                continue
        if (
            args.count_list_ledger in {"agent_llm", "agent_plan", "agent_schema", "agent_blend"}
            and item["question_type"] == "multi-session"
            and (is_count_list_question(item["question"]) or typed_numeric_question)
            and not is_unsupported_count_list_ledger_question(item["question"])
        ):
            if qid in extracted_done and "count_list_agent_report" in extracted_done[qid]:
                count_list_agent_report = extracted_done[qid]["count_list_agent_report"]
            else:
                extract_prompt, source_ids = build_count_list_agent_extraction_prompt(
                    item,
                    retrieved,
                    top_k=top_k_context,
                    max_session_chars=args.max_session_chars,
                    include_calculation_plan=args.count_list_ledger == "agent_plan",
                )
                raw_extraction = cached_chat_completion(
                    args,
                    purpose="count_list_extraction",
                    qid=qid,
                    model=extraction_model_for_item(args, item),
                    messages=[{"role": "user", "content": extract_prompt}],
                    max_tokens=extraction_max_tokens_for_item(args, item),
                )
                count_list_agent_report = validate_count_list_agent_extraction(
                    raw_extraction,
                    source_ids,
                    allow_calculation_plan=args.count_list_ledger == "agent_plan",
                    derive_schema_operation=args.count_list_ledger in {"agent_schema", "agent_blend"},
                    question=item["question"],
                )
                extract_row = {
                    "question_id": qid,
                    "count_list_agent_raw": raw_extraction,
                    "count_list_agent_report": count_list_agent_report,
                }
                append_jsonl(args.extract_out, extract_row)
                extracted_done[qid] = extract_row
            safe_agent_route = (
                is_safe_count_list_agent_plan_route(count_list_agent_report)
                if args.count_list_ledger == "agent_plan"
                else is_safe_count_list_agent_blend_route(count_list_agent_report)
                if args.count_list_ledger == "agent_blend"
                else is_safe_count_list_agent_schema_route(count_list_agent_report)
                if args.count_list_ledger == "agent_schema"
                else is_safe_count_list_agent_route(count_list_agent_report)
            )
            if safe_agent_route:
                route = (
                    "count_list_ledger_agent_schema"
                    if args.count_list_ledger == "agent_blend"
                    else f"count_list_ledger_{args.count_list_ledger}"
                )
                hypothesis = build_count_list_agent_hypothesis(count_list_agent_report)
                append_jsonl(
                    args.hypotheses_out,
                    {
                        "question_id": qid,
                        "hypothesis": hypothesis,
                        "route": route,
                        "count_list_agent_report": count_list_agent_report,
                    },
                )
                print(f"[generate {i}/{len(data)}] {qid} {item['question_type']} route={route}", flush=True)
                continue
        if should_use_aggregation_assembly(item, aggregation_assembly_mode):
            prompt, aggregation_report = build_aggregation_assembly_prompt(
                item,
                retrieved,
                top_k=top_k_context,
                max_session_chars=args.max_session_chars,
                aggregation_max_candidates=args.aggregation_max_candidates,
                aggregation_max_chars=args.aggregation_max_chars,
            )
            if aggregation_report["coherent"]:
                route = "aggregation_assembly"
                hypothesis = cached_chat_completion(
                    args,
                    purpose="generation",
                    qid=qid,
                    model=generation_model_for_item(args, item),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=generation_max_tokens_for_item(args, item),
                )
                append_jsonl(
                    args.hypotheses_out,
                    {
                        "question_id": qid,
                        "hypothesis": hypothesis,
                        "route": route,
                        "aggregation_report": aggregation_report,
                    },
                )
                print(f"[generate {i}/{len(data)}] {qid} {item['question_type']} route={route}", flush=True)
                continue
        if args.multi_session_evidence_compiler != "off" and item["question_type"] == "multi-session":
            route = f"multi_session_evidence_compiler_{args.multi_session_evidence_compiler}"
            build_prompt = (
                build_multi_session_compiler_prompt
                if args.multi_session_evidence_compiler == "strict"
                else build_multi_session_guided_prompt
            )
            prompt = build_prompt(
                item,
                retrieved,
                top_k=top_k_context,
                max_session_chars=args.max_session_chars,
                max_ledger_rows=args.multi_session_ledger_max_rows,
                max_ledger_chars=args.multi_session_ledger_max_chars,
            )
            hypothesis = cached_chat_completion(
                args,
                purpose="generation",
                qid=qid,
                model=generation_model_for_item(args, item),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=generation_max_tokens_for_item(args, item),
            )
            append_jsonl(args.hypotheses_out, {"question_id": qid, "hypothesis": hypothesis, "route": route})
            print(f"[generate {i}/{len(data)}] {qid} {item['question_type']} route={route}", flush=True)
            continue
        if args.evidence_contract and retrieval_row.get("structured_temporal_filters"):
            supporting_row = supporting_rows_by_qid.get(qid)
            if not supporting_row:
                raise RuntimeError(f"missing supporting retrieval row for {qid}")
            route = "evidence_contract"
            prompt = build_evidence_contract_prompt(
                item,
                retrieved,
                supporting_row["retrieved_sessions"],
                primary_top_k=args.evidence_contract_primary_top_k,
                supporting_top_k=args.evidence_contract_supporting_top_k,
                max_session_chars=args.max_session_chars,
            )
            hypothesis = cached_chat_completion(
                args,
                purpose="generation",
                qid=qid,
                model=generation_model_for_item(args, item),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=generation_max_tokens_for_item(args, item),
            )
            append_jsonl(args.hypotheses_out, {"question_id": qid, "hypothesis": hypothesis, "route": route})
            print(f"[generate {i}/{len(data)}] {qid} {item['question_type']} route={route}", flush=True)
            continue
        if args.temporal_hybrid and should_use_temporal_hybrid(item, args.temporal_hybrid_router):
            route = "structured_temporal"
            if qid in extracted_done:
                structured_facts = extracted_done[qid]["structured_facts"]
            else:
                extract_prompt = build_structured_extraction_prompt(
                    item,
                    retrieved,
                    top_k=top_k_context,
                    max_session_chars=args.max_session_chars,
                    include_question_type=args.include_question_type_in_prompts,
                )
                structured_facts = cached_chat_completion(
                    args,
                    purpose="extraction",
                    qid=qid,
                    model=extraction_model_for_item(args, item),
                    messages=[{"role": "user", "content": extract_prompt}],
                    max_tokens=extraction_max_tokens_for_item(args, item),
                )
                extract_row = {"question_id": qid, "structured_facts": structured_facts}
                append_jsonl(args.extract_out, extract_row)
                extracted_done[qid] = extract_row
            prompt = build_structured_answer_prompt(
                item,
                structured_facts,
                include_question_type=args.include_question_type_in_prompts,
            )
            hypothesis = cached_chat_completion(
                args,
                purpose="generation",
                qid=qid,
                model=generation_model_for_item(args, item),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=generation_max_tokens_for_item(args, item),
            )
            if should_fallback_from_structured_answer(item, hypothesis):
                structured_hypothesis = hypothesis
                fallback_prompt = build_answer_prompt(
                    item,
                    retrieved,
                    top_k=top_k_context,
                    max_session_chars=args.max_session_chars,
                    cot=False,
                    source_aware=True,
                    source_sufficiency=False,
                    token_evidence=False,
                    evidence_packet=False,
                    fusion_evidence_map=False,
                    fusion_evidence_map_max_items=args.fusion_evidence_map_max_items,
                    fusion_evidence_map_max_chars=args.fusion_evidence_map_max_chars,
                    evidence_packet_max_items=args.evidence_packet_max_items,
                    evidence_packet_max_chars=args.evidence_packet_max_chars,
                    token_evidence_max_lines=args.token_evidence_max_lines,
                    token_evidence_signals=False,
                    count_list_mode=False,
                    strict_missing_final_answer=False,
                    preference_support_packet=args.preference_support_packet,
                    preference_support_max_items=args.preference_support_max_items,
                    preference_support_per_source=args.preference_support_per_source,
                    profile_event_ledger=args.profile_event_ledger,
                    profile_event_ledger_max_rows=args.profile_event_ledger_max_rows,
                    profile_event_ledger_max_chars=args.profile_event_ledger_max_chars,
                )
                hypothesis = cached_chat_completion(
                    args,
                    purpose="generation",
                    qid=qid,
                    model=generation_model_for_item(args, item),
                    messages=[{"role": "user", "content": fallback_prompt}],
                    max_tokens=generation_max_tokens_for_item(args, item),
                )
                route = "source_aware_fallback_after_structured_unavailable"
                append_jsonl(
                    args.hypotheses_out,
                    {
                        "question_id": qid,
                        "hypothesis": hypothesis,
                        "route": route,
                        "structured_hypothesis": structured_hypothesis,
                    },
                )
            else:
                append_jsonl(args.hypotheses_out, {"question_id": qid, "hypothesis": hypothesis, "route": route})
            print(f"[generate {i}/{len(data)}] {qid} {item['question_type']} route={route}", flush=True)
            continue
        if args.structured_extract:
            route = "structured"
            if qid in extracted_done:
                structured_facts = extracted_done[qid]["structured_facts"]
            else:
                extract_prompt = build_structured_extraction_prompt(
                    item,
                    retrieved,
                    top_k=top_k_context,
                    max_session_chars=args.max_session_chars,
                    include_question_type=args.include_question_type_in_prompts,
                )
                structured_facts = cached_chat_completion(
                    args,
                    purpose="extraction",
                    qid=qid,
                    model=extraction_model_for_item(args, item),
                    messages=[{"role": "user", "content": extract_prompt}],
                    max_tokens=extraction_max_tokens_for_item(args, item),
                )
                extract_row = {"question_id": qid, "structured_facts": structured_facts}
                append_jsonl(args.extract_out, extract_row)
                extracted_done[qid] = extract_row
            prompt = build_structured_answer_prompt(
                item,
                structured_facts,
                include_question_type=args.include_question_type_in_prompts,
            )
        else:
            prompt = build_answer_prompt(
                item,
                retrieved,
                top_k=top_k_context,
                max_session_chars=args.max_session_chars,
                cot=args.cot,
                source_aware=args.source_aware,
                source_sufficiency=args.source_sufficiency,
                token_evidence=args.token_evidence,
                evidence_packet=should_use_evidence_packet_for_item(item, args),
                fusion_evidence_map=should_use_fusion_evidence_map(item, fusion_evidence_map_mode),
                fusion_evidence_map_max_items=args.fusion_evidence_map_max_items,
                fusion_evidence_map_max_chars=args.fusion_evidence_map_max_chars,
                evidence_packet_max_items=args.evidence_packet_max_items,
                evidence_packet_max_chars=args.evidence_packet_max_chars,
                token_evidence_max_lines=args.token_evidence_max_lines,
                token_evidence_signals=args.token_evidence_signals,
                count_list_mode=args.count_list_mode,
                strict_missing_final_answer=args.strict_missing_final_answer,
                preference_support_packet=args.preference_support_packet,
                preference_support_max_items=args.preference_support_max_items,
                preference_support_per_source=args.preference_support_per_source,
                profile_event_ledger=args.profile_event_ledger,
                profile_event_ledger_max_rows=args.profile_event_ledger_max_rows,
                profile_event_ledger_max_chars=args.profile_event_ledger_max_chars,
            )
            if not args.source_aware:
                route = "standard"
        hypothesis = cached_chat_completion(
            args,
            purpose="generation",
            qid=qid,
            model=generation_model_for_item(args, item),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=generation_max_tokens_for_item(args, item),
        )
        row: dict[str, Any] = {"question_id": qid, "hypothesis": hypothesis, "route": route}
        if aggregation_report is not None:
            row["aggregation_report"] = aggregation_report
        if args.auto_correct == "verifier":
            verifier_report = deterministic_verification_report(item, hypothesis)
            row["auto_verifier"] = verifier_report
            if verifier_report["needs_correction"]:
                correction_prompt = build_correction_prompt(
                    item,
                    retrieved,
                    hypothesis,
                    verifier_report,
                    top_k=top_k_context,
                    max_session_chars=args.max_session_chars,
                    evidence_packet=should_use_evidence_packet_for_item(item, args),
                    evidence_packet_max_items=args.evidence_packet_max_items,
                    evidence_packet_max_chars=args.evidence_packet_max_chars,
                )
                corrected = cached_chat_completion(
                    args,
                    purpose="correction",
                    qid=qid,
                    model=generation_model_for_item(args, item),
                    messages=[{"role": "user", "content": correction_prompt}],
                    max_tokens=generation_max_tokens_for_item(args, item),
                )
                row["original_hypothesis"] = hypothesis
                row["hypothesis"] = corrected
                row["route"] = f"{route}_auto_corrected"
                row["auto_verifier_after"] = deterministic_verification_report(item, corrected)
        append_jsonl(args.hypotheses_out, row)
        print(f"[generate {i}/{len(data)}] {qid} {item['question_type']} route={row['route']}", flush=True)


def judge_hypotheses(args: argparse.Namespace) -> None:
    data = json.loads(args.data.read_text())
    data = filter_items(data, args)
    by_qid = {item["question_id"]: item for item in data}
    hypotheses = read_jsonl(args.hypotheses_out)
    allowed_qids = set(by_qid)
    hypotheses = [row for row in hypotheses if row["question_id"] in allowed_qids]
    done = {row["question_id"] for row in read_jsonl(args.judged_out)}
    answerability_done = {row["question_id"]: row for row in read_jsonl(args.answerability_out)}

    for i, row in enumerate(hypotheses, start=1):
        qid = row["question_id"]
        if qid in done:
            continue
        item = by_qid[qid]
        row_for_judge = dict(row)
        if args.answerability_check and should_run_answerability_gate(args, row["hypothesis"]):
            if qid in answerability_done:
                gate = answerability_done[qid]["answerability_check"]
            else:
                gate = answerability_label(args, item, row["hypothesis"])
                answerability_row = {"question_id": qid, "answerability_check": gate}
                append_jsonl(args.answerability_out, answerability_row)
                answerability_done[qid] = answerability_row
            row_for_judge["answerability_check"] = gate
            if not gate["answerable"]:
                row_for_judge["original_hypothesis"] = row_for_judge["hypothesis"]
                row_for_judge["hypothesis"] = UNANSWERABLE_RESPONSE
        prompt = get_anscheck_prompt(
            item["question_type"],
            item["question"],
            item["answer"],
            row_for_judge["hypothesis"],
            abstention=qid.endswith("_abs"),
        )
        response = cached_chat_completion(
            args,
            purpose="judge",
            qid=qid,
            model=args.judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=args.judge_max_tokens,
        )
        judged = dict(row_for_judge)
        judged["autoeval_label"] = {
            "model": args.judge_model,
            "label": parse_yes_no_label(response),
            "raw": response,
        }
        append_jsonl(args.judged_out, judged)
        print(f"[judge {i}/{len(hypotheses)}] {qid} label={judged['autoeval_label']['label']}", flush=True)


def summarize(judged_path: Path, data_path: Path, args: argparse.Namespace | None = None) -> dict[str, Any]:
    judged = read_jsonl(judged_path)
    data = {item["question_id"]: item for item in json.loads(data_path.read_text())}
    type2acc: dict[str, list[int]] = defaultdict(list)
    route_counts: dict[str, int] = defaultdict(int)
    abstention: list[int] = []
    all_acc: list[int] = []
    for row in judged:
        item = data[row["question_id"]]
        val = 1 if row["autoeval_label"]["label"] else 0
        type2acc[item["question_type"]].append(val)
        route_counts[row.get("route", "unknown")] += 1
        all_acc.append(val)
        if row["question_id"].endswith("_abs"):
            abstention.append(val)
    by_type = {
        qt: {"n": len(vals), "accuracy": (sum(vals) / len(vals) if vals else 0.0)}
        for qt, vals in sorted(type2acc.items())
    }
    non_empty_types = [v["accuracy"] for v in by_type.values() if v["n"]]
    summary = {
        "n": len(all_acc),
        "overall_accuracy": sum(all_acc) / len(all_acc) if all_acc else 0.0,
        "task_averaged_accuracy": sum(non_empty_types) / len(non_empty_types) if non_empty_types else 0.0,
        "abstention_accuracy": sum(abstention) / len(abstention) if abstention else None,
        "abstention_n": len(abstention),
        "by_type": by_type,
        "route_counts": dict(sorted(route_counts.items())),
    }
    if args is not None:
        summary["provenance"] = build_run_provenance(args, judged)
    return summary


def filter_items(data: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.question_id_file:
        allowed_qids = {
            line.strip()
            for line in args.question_id_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        data = [item for item in data if item["question_id"] in allowed_qids]
    if args.abstention_only:
        data = [item for item in data if item["question_id"].endswith("_abs")]
    if args.non_abstention_only:
        data = [item for item in data if not item["question_id"].endswith("_abs")]
    if args.question_type:
        allowed = set(args.question_type)
        data = [item for item in data if item["question_type"] in allowed]
    if args.count_list_only:
        data = [item for item in data if is_count_list_question(item["question"])]
    if args.limit:
        data = data[: args.limit]
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("benchmarks/data/longmemeval_s_cleaned.json"))
    ap.add_argument("--retrieval-artifact", type=Path, default=Path("benchmarks/longmemeval_token_only_leaderboard_run_20260516.json"))
    ap.add_argument("--hypotheses-out", type=Path, default=Path("benchmarks/longmemeval_contextfit_token_only_qa_hypotheses_20260516.jsonl"))
    ap.add_argument("--judged-out", type=Path, default=Path("benchmarks/longmemeval_contextfit_token_only_qa_judged_20260516.jsonl"))
    ap.add_argument("--generation-model", default=DEFAULT_GENERATION_MODEL)
    ap.add_argument("--extraction-model", default=DEFAULT_GENERATION_MODEL)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--answerability-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument(
        "--answerer-router",
        choices=("off", "question_type_gpt5mini_temporal_preference_multi"),
        default="off",
        help=(
            "route answerer/extractor/answerability models for selected question types; "
            "question_type_gpt5mini_temporal_preference_multi routes temporal, preference, and multi-session rows"
        ),
    )
    ap.add_argument("--routed-generation-model", default="gpt-5-mini")
    ap.add_argument("--routed-extraction-model")
    ap.add_argument("--routed-answerability-model")
    ap.add_argument("--generation-max-tokens", type=int, default=500)
    ap.add_argument("--judge-max-tokens", type=int, default=10)
    ap.add_argument("--extraction-max-tokens", type=int, default=1200)
    ap.add_argument("--answerability-max-tokens", type=int, default=10)
    ap.add_argument("--routed-generation-max-tokens", type=int, default=0)
    ap.add_argument("--routed-extraction-max-tokens", type=int, default=0)
    ap.add_argument("--routed-answerability-max-tokens", type=int, default=0)
    ap.add_argument(
        "--generation-seed",
        type=int,
        default=None,
        help="best-effort deterministic seed for generation, extraction, and correction calls",
    )
    ap.add_argument(
        "--judge-seed",
        type=int,
        default=None,
        help="best-effort deterministic seed for judge calls",
    )
    ap.add_argument(
        "--answerability-seed",
        type=int,
        default=None,
        help="best-effort deterministic seed for answerability gate calls; defaults to --judge-seed",
    )
    ap.add_argument(
        "--completion-cache-dir",
        type=Path,
        help="cache LLM outputs by prompt/model/seed hash so exact reruns reuse identical answer and judge text",
    )
    ap.add_argument("--structured-extract", action="store_true", help="run a two-pass structured fact extraction before answering")
    ap.add_argument(
        "--temporal-hybrid",
        action="store_true",
        help="experimental: use structured extraction for temporal rows, falling back to source-aware answers when structured output is unavailable",
    )
    ap.add_argument(
        "--temporal-hybrid-router",
        choices=("question_type", "query"),
        default="question_type",
        help="router for --temporal-hybrid; question_type reproduces LongMemEval-label experiments, query avoids dataset-label routing",
    )
    ap.add_argument(
        "--include-question-type-in-prompts",
        action="store_true",
        help="include LongMemEval question_type in structured prompts; off by default to avoid label leakage in cleaner runs",
    )
    ap.add_argument(
        "--extract-out",
        type=Path,
        default=Path("benchmarks/longmemeval_contextfit_token_only_qa_extract_20260516.jsonl"),
    )
    ap.add_argument("--cot", action="store_true", help="use extract-then-reason answer prompt")
    ap.add_argument("--source-aware", action="store_true", help="use source-notes plus final-answer prompt")
    ap.add_argument("--source-sufficiency", action="store_true", help="use source notes plus an in-answer sufficiency decision")
    ap.add_argument(
        "--expert-ensemble",
        choices=("off", "moe"),
        default="off",
        help=(
            "enable a conservative mixture-of-experts preset; moe currently adds knowledge-update/temporal "
            "fusion-map hints without changing preference, broad multi-session, or aggregation rows"
        ),
    )
    ap.add_argument(
        "--evidence-contract",
        action="store_true",
        help="for rows with structured filters, keep filtered evidence primary and broad retrieval evidence as a separate supporting lane",
    )
    ap.add_argument(
        "--supporting-retrieval-artifact",
        type=Path,
        help="broad retrieval artifact used as supporting evidence for --evidence-contract",
    )
    ap.add_argument(
        "--evidence-contract-primary-top-k",
        type=int,
        default=10,
        help="number of filtered sessions to include as primary evidence",
    )
    ap.add_argument(
        "--evidence-contract-supporting-top-k",
        type=int,
        default=5,
        help="number of non-duplicate broad sessions to include as supporting evidence",
    )
    ap.add_argument(
        "--source-set-aware",
        action="store_true",
        help="keep primary retrieved sessions separate from targeted-expansion companion sessions",
    )
    ap.add_argument(
        "--primary-retrieval-artifact",
        type=Path,
        help="baseline retrieval artifact used as Primary Sources for --source-set-aware",
    )
    ap.add_argument(
        "--source-set-companion-top-k",
        type=int,
        default=5,
        help="number of non-primary targeted-expansion sessions to include as companion sources",
    )
    ap.add_argument(
        "--aggregation-assembly",
        choices=("off", "count_list", "multi_session_count_list"),
        default="off",
        help="use deterministic candidate/dedupe assembly for count/list aggregation questions",
    )
    ap.add_argument(
        "--aggregation-max-candidates",
        type=int,
        default=36,
        help="max candidate rows in deterministic aggregation assembly",
    )
    ap.add_argument(
        "--aggregation-max-chars",
        type=int,
        default=12_000,
        help="max characters in deterministic aggregation assembly",
    )
    ap.add_argument(
        "--count-list-ledger",
        choices=(
            "off",
            "python",
            "typed_python",
            "semantic_python",
            "agent_llm",
            "agent_plan",
            "agent_schema",
            "agent_blend",
        ),
        default="off",
        help="use a structured candidate/dedupe/count ledger for multi-session count/list rows",
    )
    ap.add_argument(
        "--count-list-ledger-max-candidates",
        type=int,
        default=36,
        help="max candidate rows in Python count/list ledger",
    )
    ap.add_argument(
        "--count-list-ledger-max-chars",
        type=int,
        default=12_000,
        help="max characters in Python count/list ledger",
    )
    ap.add_argument("--count-list-mode", action="store_true", help="use a specialized candidate/deduped-set prompt for count/list questions")
    ap.add_argument("--token-evidence", action="store_true", help="prepend a deterministic token evidence table to the answer prompt")
    ap.add_argument(
        "--evidence-packet",
        choices=("off", "all", "general"),
        default="off",
        help=(
            "prepend deterministic event/update/count/temporal/preference evidence candidates; "
            "general routes by query shape but skips multi-session and preference rows"
        ),
    )
    ap.add_argument(
        "--evidence-packet-max-items",
        type=int,
        default=16,
        help="max candidate rows per deterministic evidence packet section",
    )
    ap.add_argument(
        "--evidence-packet-max-chars",
        type=int,
        default=10_000,
        help="max total characters for the deterministic evidence packet",
    )
    ap.add_argument(
        "--fusion-evidence-map",
        choices=("off", "all", "general", "targeted", "moe", "temporal"),
        default="off",
        help=(
            "prepend a lightweight coverage/timeline/update map while keeping fused source-aware context authoritative; "
            "moe is the conservative expert mode for knowledge-update and temporal rows; temporal isolates temporal rows only"
        ),
    )
    ap.add_argument(
        "--fusion-evidence-map-max-items",
        type=int,
        default=12,
        help="max candidate rows per fusion evidence map section",
    )
    ap.add_argument(
        "--fusion-evidence-map-max-chars",
        type=int,
        default=5_000,
        help="max total characters for the lightweight fusion evidence map",
    )
    ap.add_argument("--token-evidence-max-lines", type=int, default=5, help="max fact lines per retrieved source in the token evidence table")
    ap.add_argument("--token-evidence-signals", action="store_true", help="include raw date/number/temporal token dumps in the token evidence table")
    ap.add_argument(
        "--preference-support-packet",
        choices=("off", "retrieved"),
        default="off",
        help="prepend compact source-ordered preference support extracted only from already retrieved sessions",
    )
    ap.add_argument(
        "--preference-support-max-items",
        type=int,
        default=10,
        help="max compact preference-support rows to include",
    )
    ap.add_argument(
        "--preference-support-per-source",
        type=int,
        default=2,
        help="max compact preference-support rows per retrieved source",
    )
    ap.add_argument(
        "--profile-event-ledger",
        choices=("off", "general", "all"),
        default="off",
        help="prepend a deterministic canonical profile/event ledger for preference, temporal, update, and aggregation rows",
    )
    ap.add_argument(
        "--profile-event-ledger-max-rows",
        type=int,
        default=36,
        help="max source-linked rows in the canonical profile/event ledger",
    )
    ap.add_argument(
        "--profile-event-ledger-max-chars",
        type=int,
        default=10_000,
        help="max characters in the canonical profile/event ledger",
    )
    ap.add_argument(
        "--strict-missing-final-answer",
        action="store_true",
        help="force unavailable final answers when source notes explicitly say required evidence is missing",
    )
    ap.add_argument(
        "--auto-correct",
        choices=("off", "verifier"),
        default="off",
        help="run a deterministic verifier and rerun a stricter correction prompt for flagged answers",
    )
    ap.add_argument(
        "--multi-session-evidence-compiler",
        choices=("off", "strict", "guided"),
        default="off",
        help="use a narrow deterministic evidence ledger plus strict answer contract for multi-session rows",
    )
    ap.add_argument(
        "--multi-session-ledger-max-rows",
        type=int,
        default=40,
        help="max rows in the deterministic multi-session evidence ledger",
    )
    ap.add_argument(
        "--multi-session-ledger-max-chars",
        type=int,
        default=12_000,
        help="max characters in the deterministic multi-session evidence ledger",
    )
    ap.add_argument(
        "--multi-session-evidence-set",
        choices=("off", "count_list", "source_select", "confidence_source_select"),
        default="off",
        help="use a compact broad-pool evidence set or source selector for multi-session count/list rows",
    )
    ap.add_argument(
        "--multi-session-evidence-set-min-confidence",
        type=float,
        default=0.72,
        help="minimum deterministic selector confidence for --multi-session-evidence-set confidence_source_select",
    )
    ap.add_argument(
        "--multi-session-evidence-set-pool-k",
        type=int,
        default=30,
        help="retrieved-session pool size scanned by --multi-session-evidence-set",
    )
    ap.add_argument(
        "--multi-session-evidence-set-max-candidates",
        type=int,
        default=16,
        help="max compact candidate rows in the multi-session evidence set",
    )
    ap.add_argument(
        "--multi-session-evidence-set-max-chars",
        type=int,
        default=8_000,
        help="max characters in the compact multi-session evidence set",
    )
    ap.add_argument("--answerability-check", action="store_true", help="gate unsupported answers before judging")
    ap.add_argument(
        "--answerability-trigger-missing",
        action="store_true",
        help="only run the answerability gate when the response itself mentions missing evidence",
    )
    ap.add_argument(
        "--answerability-out",
        type=Path,
        default=Path("benchmarks/longmemeval_contextfit_token_only_qa_answerability_20260516.jsonl"),
    )
    ap.add_argument(
        "--question-type",
        action="append",
        help="only run rows with this LongMemEval question_type; may be repeated",
    )
    ap.add_argument("--question-id-file", type=Path, help="only run question ids listed in this newline-delimited file")
    ap.add_argument("--abstention-only", action="store_true")
    ap.add_argument("--non-abstention-only", action="store_true")
    ap.add_argument("--count-list-only", action="store_true", help="only run questions matching count/list wording")
    ap.add_argument("--top-k-context", type=int, default=5)
    ap.add_argument(
        "--multi-session-top-k-context",
        type=int,
        default=0,
        help="override --top-k-context for multi-session questions; 0 keeps the global value",
    )
    ap.add_argument(
        "--temporal-top-k-context",
        type=int,
        default=0,
        help="override --top-k-context for temporal-reasoning questions; 0 keeps the global value",
    )
    ap.add_argument(
        "--temporal-evidence-packet",
        choices=("inherit", "off"),
        default="inherit",
        help="override --evidence-packet for temporal-reasoning rows",
    )
    ap.add_argument("--max-session-chars", type=int, default=16_000)
    ap.add_argument("--limit", type=int, default=0, help="0 means all rows")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--summary-out", type=Path, default=Path("benchmarks/longmemeval_contextfit_token_only_qa_summary_20260516.json"))
    args = ap.parse_args()
    args.limit = args.limit if args.limit > 0 else 0
    args.api_key = os.environ.get("OPENAI_API_KEY")
    uses_openclaw_generation = args.generation_model.startswith(OPENCLAW_MODEL_PREFIX)
    uses_openclaw_judge = args.judge_model.startswith(OPENCLAW_MODEL_PREFIX)
    uses_openai_compatible_generation = args.generation_model.startswith(OPENAI_COMPATIBLE_MODEL_PREFIX)
    uses_openai_compatible_judge = args.judge_model.startswith(OPENAI_COMPATIBLE_MODEL_PREFIX)
    needs_openai_key = (
        (not args.skip_generate and not (uses_openclaw_generation or uses_openai_compatible_generation))
        or (not args.skip_judge and not (uses_openclaw_judge or uses_openai_compatible_judge))
    )
    if not args.api_key and needs_openai_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    if not args.skip_generate:
        generate_hypotheses(args)
    if not args.skip_judge:
        judge_hypotheses(args)
    summary = summarize(args.judged_out, args.data, args)
    args.summary_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
