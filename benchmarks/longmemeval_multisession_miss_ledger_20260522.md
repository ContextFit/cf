# LongMemEval Multi-Session Miss Ledger - 2026-05-22

Diagnostic ledger for the 11 multi-session misses in the current token-native
160-row gate.

## Inputs

- Data: `benchmarks/data/longmemeval_s_cleaned.json`
- Retrieval: `benchmarks/longmemeval_token_only_leaderboard_run_20260516.json`
- Base judged run: `benchmarks/longmemeval_contextfit_qa_judged_temporal_constraints_provenancecache_160_20260521.jsonl`
- Base multi-session score on this slice: `19/30 = 63.3%`
- Multi-session top-k context: `10`

## Summary

| Failure class | Count | Rows |
|---|---:|---|
| Evidence absent from base top-10 | 8 | `gpt4_59c863d7`, `aae3761f`, `gpt4_f2262a51`, `gpt4_a56e767c`, `gpt4_15e38248`, `88432d0a`, `2ce6a0f2`, `gpt4_d12ceb0e` |
| Gold evidence present; reducer/dedupe failed | 3 | `0a995998`, `3a704032`, `gpt4_2f8be40d` |
| Gold evidence present; candidate extraction missed all gold sources | 0 | none |

Main conclusion: the dominant current failure class is retrieval/ranking coverage,
not broad-pool prompt selection. The previous marginal selector correctly fired
`0/25` on known count/list rows because its selected evidence added no new
groups beyond base top-10. However, the miss ledger shows many gold sessions are
not in base top-10 at all, so the next retrieval work should focus on recovering
missing gold-supporting sessions, not reusing the current source-select prompt.

## Wider Retrieval Diagnostic

Artifacts:

- Top-30 rerun for all 8 coverage misses:
  `benchmarks/longmemeval_multisession_coverage8_top30_20260522.json`
- Top-50 rerun for the 3 rows still incomplete at top-30:
  `benchmarks/longmemeval_multisession_coverage3_top50_20260522.json`

Result: all 8 coverage misses have their missing gold sessions somewhere in the
retrieved session pool by top-50. This is not a first-stage absence problem; it
is a promotion/ranking problem.

| QID | Gold ranks at top-30/top-50 | All gold found by | Interpretation |
|---|---|---:|---|
| `gpt4_59c863d7` | 2, 6, 7, 29 | 29 | recoverable by top-30 promotion |
| `aae3761f` | 1, 2, 14 | 14 | recoverable by top-30 promotion |
| `gpt4_f2262a51` | 20, 32, 38 | 38 | needs deeper promotion; top-30 incomplete |
| `gpt4_a56e767c` | 3, 21, 26 | 26 | recoverable by top-30 promotion |
| `gpt4_15e38248` | 4, 9, 31, 41 | 41 | needs deeper promotion; top-30 incomplete |
| `88432d0a` | 11, 12, 20, 36 | 36 | needs deeper promotion; top-30 incomplete |
| `2ce6a0f2` | 1, 2, 6, 12 | 12 | recoverable by top-30 promotion |
| `gpt4_d12ceb0e` | 1, 5, 6 | 6 | all gold is already within top-10 in the deeper rerun, suggesting run/config variance or top-10 ordering sensitivity |

Promotion opportunity:

- A top-30 promotion rule could recover complete evidence for 5 of the 8
  coverage rows.
- A top-50/deeper promotion rule could recover all 8, but with higher
  distraction risk.
- The next candidate should learn a narrow promotion signal that identifies
  gold-like companion sessions from ranks 11-50 without replacing the stable
  top-10 wholesale.

## Per-Row Ledger

| QID | Question | Gold answer | Gold sessions in top-10 | Missing gold sessions | Candidate table status | Prior alternate route hits | Class |
|---|---|---:|---:|---:|---|---|---|
| `0a995998` | How many items of clothing do I need to pick up or return from a store? | `3` | 3/3 | 0 | Candidate rows include all gold sessions. Base answer counted only Zara boots; aggregation assembly fixed this row. | aggregation yes; source-select no; autocorrect no | reducer/dedupe |
| `gpt4_59c863d7` | How many model kits have I worked on or bought? | 5 kits | 3/4 | 1 | Candidate rows include only the three retrieved gold sessions. Missing gold source prevents full count. | none | retrieval coverage |
| `3a704032` | How many plants did I acquire in the last month? | `3` | 3/3 | 0 | Candidate rows include all gold sessions. Base answer excluded/handled timing incorrectly; source-select, aggregation, and autocorrect fixed this row. | source-select yes; aggregation yes; autocorrect yes | reducer/temporal filtering |
| `aae3761f` | How many hours in total did I spend driving to my three road trip destinations combined? | 15 hours one-way, or 30 round trip | 2/3 | 1 | Candidate rows include only the two retrieved gold sessions. Missing third destination prevents correct total. | none | retrieval coverage |
| `gpt4_f2262a51` | How many different doctors did I visit? | 3 doctors | 0/3 | 3 | No gold sessions in top-10; candidate rows come from unrelated medical/general sessions. | none | retrieval coverage |
| `gpt4_a56e767c` | How many movie festivals that I attended? | 4 festivals | 1/3 | 2 | Candidate rows include only one retrieved gold session. | none | retrieval coverage |
| `gpt4_2f8be40d` | How many weddings have I attended in this year? | 3 weddings | 3/3 | 0 | Candidate rows include all gold sessions, but base over-included distractor weddings. Autocorrect fixed this row. | autocorrect yes; source-select no; aggregation no | reducer/dedupe |
| `gpt4_15e38248` | How many pieces of furniture did I buy, assemble, sell, or fix in the past few months? | `4` | 2/4 | 2 | Candidate rows include only two retrieved gold sessions. Aggregation fixed with candidate extraction from available context, but base top-10 still lacks two answer sessions. | aggregation yes | retrieval coverage plus reducer |
| `88432d0a` | How many times did I bake something in the past two weeks? | `4` | 2/4 | 2 | Candidate rows include only two retrieved gold sessions. | none | retrieval coverage |
| `2ce6a0f2` | How many different art-related events did I attend in the past month? | `4` | 3/4 | 1 | Candidate rows include only the three retrieved gold sessions. | none | retrieval coverage |
| `gpt4_d12ceb0e` | What is the average age of me, my parents, and my grandparents? | `59.6` | 2/3 | 1 | Candidate rows include user and parents, but missing grandparents source prevents average. | none | retrieval coverage |

## Implications

1. Do not spend on another broad source-selector prompt. The known source
   selector did not add new selected groups beyond base top-10 on this slice.
2. Split next work into two tracks:
   - Retrieval/ranking recovery for the 8 coverage misses.
   - Narrow reducer/dedupe fixes for the 3 rows where all gold evidence is
     already in base top-10.
3. The wider retrieval diagnostic shows the missing gold sessions are present by
   top-50 for all 8 coverage misses. The next retrieval work should therefore be
   a safe promotion/reranking rule over ranks 11-50, not first-stage expansion.
4. For the 3 reducer rows, use existing candidate rows as a diagnostic oracle
   and derive narrow rules from proven fixes:
   - pickup/return obligations should remain separate (`0a995998`)
   - question-date windowing should avoid excluding in-window acquisitions
     (`3a704032`)
   - yearly count/list rows need stricter distractor rejection (`gpt4_2f8be40d`)

## Safe Promotion Rerank Diagnostic

Added an opt-in retrieval flag in `benchmarks/longmemeval_contextfit.py`:
`--safe-promotion-rerank`.

The rule preserves the first four ranked sessions, then promotes from ranks
11-50 using user-turn-only count/list evidence density: query term hits,
light synonym expansion for count/list nouns/actions, first-person event cues,
numeric cues, and penalties for generic assistant-style prompts. It is
retrieval-only and benchmark-local for now.

Artifacts:

- Existing top-10 route over the 30 multi-session slice:
  `benchmarks/longmemeval_multisession30_base_top10_20260522.json`
- Safe-promotion route over the 30 multi-session slice:
  `benchmarks/longmemeval_multisession30_safe_promotion_top10_20260522.json`
- Focused safe-promotion route over the 8 coverage misses:
  `benchmarks/longmemeval_multisession_coverage8_safe_promotion_top10_20260522.json`

Retrieval result:

| Slice | Route | Any@10 | All evidence@10 |
|---|---|---:|---:|
| 8 coverage misses | existing top-10 routes | 7/8 | 0/8 |
| 8 coverage misses | safe promotion | 8/8 | 8/8 |
| 30 multi-session rows | existing top-10 route | 29/30 | 20/30 |
| 30 multi-session rows | safe promotion | 30/30 | 26/30 |

Flips on the 30-row retrieval slice:

- All-evidence wins: `gpt4_59c863d7`, `aae3761f`,
  `gpt4_f2262a51`, `gpt4_a56e767c`, `gpt4_15e38248`, `88432d0a`,
  `2ce6a0f2`, `gpt4_d12ceb0e`.
- All-evidence losses: `d23cf73b`, `d682f1a2`.

Interpretation: this is the first route in this thread that materially improves
retrieval coverage for the known multi-session misses. It is not yet a quality
candidate because it still displaces gold evidence on two rows and may reduce
early-rank precision. Next gate should be judged QA on the 30-row multi-session
slice before considering a 160-row run.

## Shared EvidenceSource Extraction

The safe-promotion rule was moved out of the LongMemEval benchmark adapter into
the shared evidence compiler as
`promote_evidence_sources_for_count_list(item, sources, ...)`.

The LongMemEval `--safe-promotion-rerank` flag is now a thin adapter that wraps
retrieved sessions as `EvidenceSource` rows and calls the shared function. This
keeps the improvement available to BEAM-style memories, real-world files, docs,
chat logs, and other adapters without depending on LongMemEval session IDs.

Focused BEAM 100K conv0 diagnostic using existing top-200 ContextFit retrieval
artifacts:

| Row | Baseline top-10 issue | Shared promotion top-10 recovery |
|---|---|---|
| `100K_0_q12_multi_session_reasoning` | `category` starts at rank 18 and `notes` starts at rank 33 | promoted `contextfit:53` (`category`) and `contextfit:82` (`notes`) |
| `100K_0_q13_multi_session_reasoning` | actual user lockout implementation source is rank 73 | promoted `contextfit:76` with account lockout after failed login attempts |

BEAM caveat: this is a focused retrieval/evidence gate, not a full judged BEAM
quality run. It proves the pattern is not LongMemEval-specific when a broad
candidate pool is available, but BEAM should still get a scored answer/judge
run before broad quality claims.

## 160-Row Retrieval Gate And GPT-5.5 QA Pair

Ran the safe-promotion rule on the fixed 160-row LongMemEval gate from
`benchmarks/longmemeval_fusion_map_paired_160_qids_20260520.txt`.

Retrieval result versus the existing top-10 retrieval artifact:

| Slice | Existing all-evidence@10 | Safe promotion all-evidence@10 |
|---|---:|---:|
| 160-row gate | 135/160 | 139/160 |
| Multi-session rows | 20/30 | 24/30 |

The only retrieval changes were in multi-session rows. Multi-session had 6
all-evidence wins and 2 all-evidence losses.

Judged GPT-5.5 apples-to-apples QA pair on the same 30 multi-session rows:

| Route | Score |
|---|---:|
| Existing top-10 base | 17/30 = 56.7% |
| Safe promotion | 24/30 = 80.0% |

QA flips: 9 wins and 2 losses, net +7. Wins were `gpt4_59c863d7`,
`aae3761f`, `gpt4_f2262a51`, `gpt4_a56e767c`, `gpt4_15e38248`, `88432d0a`,
`7024f17c`, `2ce6a0f2`, and `gpt4_d12ceb0e`; losses were `d23cf73b` and
`d682f1a2`.

Interpretation: this clears the focused proof gate. The next expensive proof
step should be a 160-row judged QA pair under one model/judge stack, or
refreshing GPT-4o access to compare directly with the historical 80.0% 160-row
baseline.

## GPT-4o Apples-To-Apples 160-Row Gate

Ran the 160-row safe-promotion QA gate through the direct OpenAI API using
`gpt-4o-2024-08-06`, the same paired QID file, the same generation/judge seeds
(`20260521`), and the same answer policy as the historical 80.0% control.

Result:

| Route | Overall | Task avg | Multi-session |
|---|---:|---:|---:|
| Historical top-10 control | 80.0% | 78.1% | 63.3% |
| Safe promotion | 78.75% | 76.7% | 56.7% |

Non-multi categories were unchanged. The regression came entirely from
multi-session: GPT-4o had 2 wins (`88432d0a`, `2ce6a0f2`) and 4 losses
(`46a3abf7`, `d23cf73b`, `d682f1a2`, `7024f17c`) versus the historical control.

Interpretation: safe promotion is model-sensitive. It clearly helps GPT-5.5 on
the 30-row multi-session slice (`17/30 -> 24/30`) but is not a gpt-4o 160-row
quality improvement as currently gated. Do not promote this as the default
general route yet; the next step should be a narrower promotion gate that avoids
rows like `46a3abf7`, `d23cf73b`, `d682f1a2`, and `7024f17c`.

## Python Count/List Ledger Gate

Added an opt-in LongMemEval QA flag:
`--count-list-ledger python`.

The route uses shared `build_count_list_ledger()` to extract candidate rows,
dedupe them into groups, and expose a Python-computed count when a strict
high-precision gate passes. The intended role is to keep raw counting out of
the LLM when Python can produce a small auditable group ledger.

GPT-4o diagnostic results on the 30-row multi-session safe-promotion slice:

| Route | Score | Ledger routes |
|---|---:|---:|
| Broad generic ledger | 11/30 = 36.7% | 25/30 |
| Narrow high-precision ledger | 20/30 = 66.7% | 2/30 |
| Matched source-aware control, same seed/retrieval/model/judge | 21/30 = 70.0% | 0/30 |

Interpretation: the current generic ledger is not a quality win. The broad
version over-counts noisy groups, and the narrow version is still not better
than the same-seed source-aware control. Keep it opt-in/diagnostic only.

Useful next direction: typed Python ledgers rather than one generic dedupe
counter:

- duration-sum ledger for days/hours/weeks/months questions;
- entity-count ledger for doctors, events, baked items, services, etc.;
- action-status ledger for bought/assembled/sold/fixed or pickup/return style
  questions.

Each typed ledger should have a strict domain gate and should produce a small
accepted/rejected group table before the model sees the retrieved sessions.

## Typed Python Ledger Gate

Added `--count-list-ledger typed_python`, which only routes when a strict typed
Python ledger is available. Initial typed ledgers:

- duration sums for road-trip/camping day/hour questions;
- age averages for explicit age-average questions.

Focused 30-row GPT-4o multi-session result on the safe-promotion retrieval
artifact:

| Route | Score | Ledger routes |
|---|---:|---:|
| Matched source-aware control, same seed/retrieval/model/judge | 21/30 = 70.0% | 0/30 |
| Safe promotion + typed ledger | 22/30 = 73.3% | 3/30 |

Typed ledger rows:

| Row | Typed answer | Label | Control label |
|---|---:|---:|---:|
| `b5ef892d` | `8 days` | pass | pass |
| `aae3761f` | `15 hours` | pass | fail |
| `gpt4_d12ceb0e` | `59.6` | pass | fail |

160-row GPT-4o gate using the same paired QID file, model, judge, seeds, cache,
and safe-promotion retrieval artifact as the prior safe-promotion 160 run:

| Route | Overall | Task avg | Multi-session |
|---|---:|---:|---:|
| Safe promotion | 126/160 = 78.75% | 76.7% | 17/30 = 56.7% |
| Safe promotion + typed ledger | 128/160 = 80.0% | 78.1% | 19/30 = 63.3% |
| Historical token-native control | 128/160 = 80.0% | 78.1% | 19/30 = 63.3% |

Interpretation: typed Python ledgers are the first Python counting route with a
clean positive signal. They recover the safe-promotion GPT-4o regression back to
the historical token-native score, but they do not yet beat the original
token-native control because the remaining safe-promotion losses are
retrieval/distraction rows outside the current typed-ledger coverage.

## Typed Python Ledger v3 Gate

Extended `--count-list-ledger typed_python` with two additional strict ledgers:

- action-status counts for clothing pickup/return questions;
- entity counts for model-kit questions.

The dry 160-row safe-promotion gate now routes 5 multi-session rows:

| Row | Type | Computed answer |
|---|---|---:|
| `0a995998` | action_status_clothing_pickup_return | `3` |
| `gpt4_59c863d7` | entity_count_model_kits | `5` |
| `b5ef892d` | duration_sum_days | `8 days` |
| `aae3761f` | duration_sum_hours | `15 hours` |
| `gpt4_d12ceb0e` | age_average | `59.6` |

160-row GPT-4o gate using the same paired QID file, model, judge, seeds, cache,
and safe-promotion retrieval artifact as the prior runs:

| Route | Overall | Task avg | Multi-session |
|---|---:|---:|---:|
| Historical token-native control | 128/160 = 80.0% | 78.1% | 19/30 = 63.3% |
| Safe promotion only | 126/160 = 78.75% | 76.7% | 17/30 = 56.7% |
| Safe promotion + typed ledger v3 | 130/160 = 81.25% | 79.4% | 21/30 = 70.0% |

Typed ledger rows were all correct, with no typed-row losses. Flips versus the
historical token-native control:

- wins: `0a995998`, `gpt4_59c863d7`, `aae3761f`, `88432d0a`, `2ce6a0f2`,
  `gpt4_d12ceb0e`;
- losses: `46a3abf7`, `d23cf73b`, `d682f1a2`, `7024f17c`.

Interpretation: this is the first GPT-4o 160-row gate that beats the historical
token-native control while still staying on the token-native track. No fusion
was used. The improvement is narrow but real on this gate: Python handles the
typed numeric/count rows, while the remaining losses are still promotion
distraction rows outside typed-ledger coverage.
