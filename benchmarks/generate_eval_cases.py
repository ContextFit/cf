#!/usr/bin/env python3
"""Generate diverse agent-memory eval cases via GPT-4o-mini.

Produces cases following the same schema as agent_memory_eval.json:
  - behavior: one of 8 types
  - question: natural query an agent might receive
  - answer_session_ids: 1-2 gold sessions
  - sessions: 4-6 sessions (gold + distractors)

Usage:
  python benchmarks/generate_eval_cases.py --n 420 --out benchmarks/data/generated_cases.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request
from pathlib import Path

# ── Behavior templates ────────────────────────────────────────────────────────

BEHAVIORS = {
    "preference_informs_recommendation": {
        "desc": "User expressed a preference in one session; query asks for a recommendation that should be informed by it.",
        "question_examples": [
            "What show should I watch tonight?",
            "Can you recommend a book for me?",
            "What wine should I get for dinner?",
            "What kind of music would I enjoy?",
            "What sport should I try?",
        ],
        "gold_hint": "A session where the user expressed a clear preference, taste, or 'go-to' choice.",
        "distractor_hint": "Unrelated sessions on different topics — travel plans, work tasks, tech questions.",
        "n_sessions": "4 to 6",
        "n_gold": 1,
    },
    "constraint_informs_advice": {
        "desc": "User described a hard constraint; query asks for advice that must respect it.",
        "question_examples": [
            "What should I bring to the party?",
            "Help me plan something for next weekend.",
            "What should I order for the team lunch?",
            "What can I make with what I have at home?",
        ],
        "gold_hint": "A session where the user stated a budget, dietary restriction, time limit, or other hard requirement.",
        "distractor_hint": "Sessions on unrelated topics or sessions that mention loosely related topics without the actual constraint.",
        "n_sessions": "4 to 6",
        "n_gold": 1,
    },
    "goal_informs_advice": {
        "desc": "User stated a goal or aspiration; query asks how to make progress or what to do next.",
        "question_examples": [
            "What should I focus on this week?",
            "How should I spend my free Saturday?",
            "What's a good next step for me?",
            "Help me move forward on what I've been working on.",
        ],
        "gold_hint": "A session where the user explicitly stated a goal, aspiration, project, or learning intention.",
        "distractor_hint": "Sessions on unrelated topics, or sessions where the user asked generic questions rather than stating their own goal.",
        "n_sessions": "4 to 6",
        "n_gold": 1,
    },
    "temporal_supersession": {
        "desc": "User's state changed between two sessions; query asks about current state and should retrieve the *newer* session.",
        "question_examples": [
            "What am I currently using?",
            "What's my current situation with X?",
            "Which Y do I use these days?",
            "What did I switch to?",
        ],
        "gold_hint": "The NEWER session where the user describes a change (switched, now using, recently changed, no longer, etc.).",
        "distractor_hint": "The OLDER session (describing the previous state) plus unrelated sessions.",
        "n_sessions": "4 to 6",
        "n_gold": 1,
    },
    "decision_retrieval": {
        "desc": "User made a specific decision; query asks what they decided.",
        "question_examples": [
            "What did I decide about X?",
            "Which option did I go with?",
            "What was my conclusion on Y?",
        ],
        "gold_hint": "A session where the user commits to a specific choice ('I decided', 'we went with', 'I chose', 'I'll use').",
        "distractor_hint": "A session where options are discussed but no decision is made, plus unrelated sessions.",
        "n_sessions": "4 to 6",
        "n_gold": 1,
    },
    "open_loop_retrieval": {
        "desc": "User created a pending action or reminder; query asks what they need to do or follow up on.",
        "question_examples": [
            "Is there something I was supposed to follow up on?",
            "Do I have any outstanding tasks I haven't finished?",
            "What did I need to take care of?",
            "Is there anything I was supposed to do this week?",
        ],
        "gold_hint": "A session where the user said 'remind me', 'todo', 'I need to follow up', 'I haven't done X yet', or similar pending-action language.",
        "distractor_hint": "Completed tasks, general questions without pending actions, unrelated topics.",
        "n_sessions": "4 to 6",
        "n_gold": 1,
    },
    "episodic_interest_inference": {
        "desc": "User mentioned a context, interest, or situation in one session; a later vague advice query should retrieve that episode even though the query vocabulary doesn't match directly.",
        "question_examples": [
            "What should I get as a gift?",
            "What fun thing should I do this weekend?",
            "What activity would I enjoy?",
            "What should I make for dinner?",
            "What would be a good surprise for someone close to me?",
        ],
        "gold_hint": "A session where the user mentioned a specific context (hobby, interest, situation, resource, or relationship) that makes one answer clearly better than generic advice.",
        "distractor_hint": "Generic sessions (asking about common topics), sessions that have topical overlap but no personal context, unrelated sessions.",
        "n_sessions": "4 to 6",
        "n_gold": 1,
    },
    "multi_session_synthesis": {
        "desc": "Evidence is spread across two sessions; both are needed to fully address the query.",
        "question_examples": [
            "Help me plan for X.",
            "What context do I have around Y?",
            "Help me get ready for Z.",
            "What should I think about for W?",
        ],
        "gold_hint": "TWO sessions that together cover different aspects of the query (e.g., one about a goal and one about a constraint, or one about a situation and one about a preference).",
        "distractor_hint": "Sessions that cover only one dimension or are unrelated.",
        "n_sessions": "4 to 6",
        "n_gold": 2,
    },
}

DOMAINS = [
    "cooking and food", "travel and trips", "fitness and health", "music and entertainment",
    "technology and software", "home and family", "career and work", "finance and budgeting",
    "education and learning", "hobbies and crafts", "pets and animals", "gardening and outdoors",
    "reading and books", "social events and relationships", "parenting and children",
    "photography and art", "gaming and sports", "fashion and shopping", "sleep and wellness",
    "cars and transportation", "home renovation and DIY", "cooking equipment", "language learning",
    "meditation and mental health", "podcasts and media", "writing and creative projects",
]

SYSTEM_PROMPT = """You generate evaluation cases for an AI agent memory retrieval benchmark.
Each case tests whether a retrieval system can find the right past session(s) given a natural query.

Output ONLY valid JSON — no markdown fences, no commentary.
Follow the schema EXACTLY as shown in the example."""

def make_prompt(behavior: str, domain: str, existing_ids: set[str]) -> str:
    b = BEHAVIORS[behavior]
    example_q = random.choice(b["question_examples"])
    n_gold = b["n_gold"]
    n_sessions = "5"  # fixed for consistency

    return f"""Generate one eval case for the behavior type: {behavior}

Description: {b['desc']}

Domain/topic: {domain}
Example question style: {example_q!r}

Rules:
- session_ids must be short snake_case strings, unique across all cases (prefix them with a 3-letter random slug to avoid collisions)
- Use {n_sessions} sessions total: {n_gold} gold session(s) + distractor sessions
- Gold session hint: {b['gold_hint']}
- Distractor hint: {b['distractor_hint']}
- {"answer_session_ids has EXACTLY 2 ids (both gold sessions)" if n_gold == 2 else "answer_session_ids has EXACTLY 1 id"}
- Each session has 2-4 turns with realistic role/content
- The question should NOT contain vocabulary from the gold session (test indirect retrieval)
- Make the distractors realistic and tempting (same domain, plausible content)
- Do NOT use any of these existing session_ids: {sorted(existing_ids)[:20]}

Output this JSON structure:
{{
  "id": "short_snake_case_unique_id",
  "behavior": "{behavior}",
  "question": "natural query string",
  "answer_session_ids": ["s_gold1"{', "s_gold2"' if n_gold == 2 else ''}],
  "sessions": [
    {{
      "session_id": "s_gold1",
      "date": "2026/MM/DD (Weekday) HH:MM",
      "turns": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}}
      ]
    }},
    ... (total {n_sessions} sessions)
  ]
}}"""


def call_gpt(prompt: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
        "max_tokens": 2000,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def parse_case(raw: str) -> dict | None:
    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])
    try:
        case = json.loads(raw)
    except json.JSONDecodeError:
        return None
    # Validate schema
    required = {"id", "behavior", "question", "answer_session_ids", "sessions"}
    if not required.issubset(case):
        return None
    if not case["answer_session_ids"] or not case["sessions"]:
        return None
    for s in case["sessions"]:
        if not all(k in s for k in ("session_id", "date", "turns")):
            return None
    return case


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=420, help="Number of new cases to generate")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/data/generated_cases.json"))
    ap.add_argument("--existing", type=Path, default=Path("benchmarks/data/agent_memory_eval.json"))
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY required")

    existing = json.loads(args.existing.read_text()) if args.existing.exists() else []
    existing_ids: set[str] = set()
    for case in existing:
        existing_ids.add(case["id"])
        for s in case.get("sessions", []):
            existing_ids.add(s["session_id"])

    # Resume from partial output
    generated: list[dict] = []
    if args.out.exists():
        generated = json.loads(args.out.read_text())
        print(f"Resuming from {len(generated)} existing generated cases")
        for c in generated:
            existing_ids.add(c["id"])
            for s in c.get("sessions", []):
                existing_ids.add(s["session_id"])

    behaviors_list = list(BEHAVIORS.keys())
    target = args.n
    failures = 0
    max_failures = 30

    while len(generated) < target and failures < max_failures:
        # Round-robin behaviors, random domains
        behavior = behaviors_list[len(generated) % len(behaviors_list)]
        domain = random.choice(DOMAINS)

        prompt = make_prompt(behavior, domain, existing_ids)
        try:
            raw = call_gpt(prompt, api_key, args.model)
            case = parse_case(raw)
            if case is None:
                print(f"  [parse error] behavior={behavior} domain={domain}")
                failures += 1
                continue
            if case["id"] in existing_ids:
                case["id"] = case["id"] + f"_{len(generated)}"
            existing_ids.add(case["id"])
            for s in case.get("sessions", []):
                existing_ids.add(s["session_id"])
            generated.append(case)
            print(f"[{len(generated)}/{target}] {case['id']} ({behavior}, {domain})")
            failures = 0  # reset on success
            # Save incrementally
            if len(generated) % 10 == 0:
                args.out.write_text(json.dumps(generated, indent=2))
        except Exception as e:
            print(f"  [error] {e}")
            failures += 1
            time.sleep(2)

    args.out.write_text(json.dumps(generated, indent=2))
    print(f"\nGenerated {len(generated)} cases → {args.out}")

    # Behavior distribution
    from collections import Counter
    dist = Counter(c["behavior"] for c in generated)
    print("\nDistribution:")
    for b, n in sorted(dist.items()):
        print(f"  {b}: {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
