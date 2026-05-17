# LongMemEval-S QA Evidence Run - Source-Aware 20k

Date: 2026-05-16

## Purpose

This run evaluates an official-style LongMemEval-S QA bridge on top of the saved ContextFit token-only retrieval artifact. It is distinct from the retrieval/evidence-ranking run documented in `longmemeval_token_only_leaderboard_evidence_20260516.md`.

The goal was to test whether stronger answer synthesis improves the weakest slice, multi-session QA, without changing the retrieval artifact.

## Inputs

- Dataset: `benchmarks/data/longmemeval_s_cleaned.json`
- Retrieval artifact: `benchmarks/longmemeval_token_only_leaderboard_run_20260516.json`
- Retrieval artifact SHA-256: `09f7f8bbd6c1622749cb3961f39cbbf6bcd673b318ad3bd63e88d9d1441a0ca2`
- QA runner: `benchmarks/longmemeval_contextfit_qa.py`
- Generation model: `gpt-4o-2024-08-06`
- Judge model: `gpt-4o-2024-08-06`

## Best Current QA Configuration

- Top-k retrieved sessions supplied to the answerer: 10
- Per-session context budget: 20,000 characters
- Prompt mode: source-aware
- Generation max tokens: 1,200
- Judge prompt: LongMemEval-style yes/no answer-check prompt
- Retrieval code: unchanged

The source-aware prompt asks the answer model to list source notes from each relevant retrieved session, combine distinct facts across sessions, use dates for temporal updates, and then provide a concise final answer.

## Command Shape

```bash
OPENAI_API_KEY=... /tmp/cf-structure-venv/bin/python benchmarks/longmemeval_contextfit_qa.py \
  --top-k-context 10 \
  --max-session-chars 20000 \
  --source-aware \
  --generation-max-tokens 1200 \
  --hypotheses-out benchmarks/longmemeval_contextfit_token_only_qa_hypotheses_source_20k_20260516.jsonl \
  --judged-out benchmarks/longmemeval_contextfit_token_only_qa_judged_source_20k_20260516.jsonl \
  --summary-out benchmarks/longmemeval_contextfit_token_only_qa_summary_source_20k_20260516.json
```

## Results

| Metric | Score |
|---|---:|
| Rows judged | 500 |
| Overall QA accuracy | **81.8%** |
| Task-averaged QA accuracy | **83.5%** |
| Abstention accuracy | 80.0% |
| Abstention rows | 30 |

## Results by Type

| Question type | n | Accuracy |
|---|---:|---:|
| knowledge-update | 78 | 88.5% |
| multi-session | 133 | **71.4%** |
| single-session-assistant | 56 | 98.2% |
| single-session-preference | 30 | 70.0% |
| single-session-user | 70 | 97.1% |
| temporal-reasoning | 133 | 75.9% |

## Comparison to Prior QA Bridge

Prior top-10 CoT prompt:

- Overall QA accuracy: 78.6%
- Task-averaged QA accuracy: 80.8%
- Multi-session QA accuracy: 60.9%
- Abstention accuracy: 90.0%

Source-aware 20k prompt:

- Overall QA accuracy: 81.8%
- Task-averaged QA accuracy: 83.5%
- Multi-session QA accuracy: 71.4%
- Abstention accuracy: 80.0%

The largest improvement is multi-session QA: 60.9% -> 71.4%.

## Sufficiency Gate Probe

A post-hoc answerability/sufficiency gate was tested to recover abstention behavior.

Triggered sufficiency gate full run:

- Overall QA accuracy: 81.6%
- Task-averaged QA accuracy: 83.4%
- Multi-session QA accuracy: 69.9%
- Abstention accuracy: 83.3%

The gate is not the current headline configuration. It recovered some abstention behavior, but incorrectly blocked valid multi-session answers and reduced net overall accuracy. The right next fix is to make the answerer itself abstain when its own source notes show missing required evidence, rather than adding a broad post-hoc gate.

## Token Evidence Reducer Probe

An experimental deterministic token evidence reducer was added behind `--token-evidence`. It keeps retrieval fixed, builds a compact source-linked evidence table from retrieved sessions, and prepends that table to the source-aware answer prompt.

Two 40-row multi-session probes were run against the same first 40 multi-session rows:

| Configuration | Accuracy | Notes |
|---|---:|---|
| Source-aware baseline on same rows | 67.5% | 27/40 |
| Token evidence with raw token signals | 55.0% | 22/40; +2 wins, -7 regressions |
| Token evidence facts-only | 60.0% | 24/40; +2 wins, -5 regressions |

The simple reducer is not good enough for the headline path. It can find some new wins, but it currently introduces more regressions than improvements. Keep it experimental/off and do not run the full 500-row evaluation until the reducer is more selective.

Artifacts:

- `benchmarks/longmemeval_contextfit_token_only_qa_summary_token_evidence_multisession_probe_20260516.json`
- `benchmarks/longmemeval_contextfit_token_only_qa_summary_token_evidence_facts_multisession_probe_20260516.json`

## Failure Cluster Analysis

The current source-aware 20k QA run has 91 misses out of 500 rows. A local failure-cluster analysis found:

| Cluster | Miss count |
|---|---:|
| temporal wording/reasoning | 47 |
| count/list/total wording | 41 |
| missing all gold evidence in top 10 | 29 |
| retrieval miss at top 10 | 12 |
| preference/recommendation | 10 |
| abstention | 6 |
| comparison | 5 |

Among synthesis-like misses where the retrieval artifact already had a gold session in top 5 and all gold sessions in top 10, there were 42 misses:

| Synthesis-like cluster | Miss count |
|---|---:|
| count/list/total wording | 21 |
| temporal wording/reasoning | 19 |
| preference/recommendation | 5 |
| comparison | 2 |

This makes count/list and temporal-update behavior the best next targets. The analysis metadata is stored in:

- `benchmarks/longmemeval_contextfit_qa_failure_clusters_20260516.json`

## Count/List Mode Probe

A narrow `--count-list-mode` prompt was added to use a candidate-evidence and deduped-set contract only for count/list questions. It was tested on the first 60 count/list rows:

| Configuration | Accuracy |
|---|---:|
| Source-aware baseline on same rows | 70.0% |
| Count/list candidate+dedupe mode | 68.3% |

The count/list mode created real wins, but not enough to offset regressions:

- Wins over baseline: 5
- Regressions versus baseline: 6

Readout: do not enable count/list mode broadly. The right next version needs a router/confidence rule, or a narrower trigger for high-risk aggregation rows, before a full 500-row evaluation.

Artifact:

- `benchmarks/longmemeval_contextfit_token_only_qa_summary_count_list_mode_probe_20260516.json`

## Structured Extraction Probe

A two-pass `--structured-extract` mode was added:

1. Extract source-grounded structured facts from retrieved sessions.
2. Answer from the extracted facts.

This is still LLM-based, but it separates bookkeeping from final answer generation. A targeted probe was built from hard synthesis-like rows plus a matched set of currently-correct hard rows:

- Probe id file: `benchmarks/longmemeval_contextfit_qa_structured_hard_probe_qids_20260516.txt`
- Rows: 56
- Composition: 28 source-aware misses where retrieval already had enough evidence, plus 28 source-aware correct hard rows.

Results:

| Slice | Source-aware baseline | Structured two-pass |
|---|---:|---:|
| All hard probe rows | 50.0% | **67.9%** |
| Multi-session hard rows | 68.3% | 65.9% |
| Temporal-reasoning hard rows | 0.0% | **63.6%** |

Flip analysis:

- Wins over baseline: 15
- Regressions versus baseline: 5

Readout: this is the first materially positive synthesis signal. It is especially strong for temporal reasoning, but should not be applied broadly to multi-session/count rows yet because that slice regressed slightly. The next experiment should be a larger temporal-only probe before any full 500-row run.

Artifacts:

- `benchmarks/longmemeval_contextfit_token_only_qa_extract_structured_hard_probe_20260516.jsonl`
- `benchmarks/longmemeval_contextfit_token_only_qa_hypotheses_structured_hard_probe_20260516.jsonl`
- `benchmarks/longmemeval_contextfit_token_only_qa_judged_structured_hard_probe_20260516.jsonl`
- `benchmarks/longmemeval_contextfit_token_only_qa_summary_structured_hard_probe_20260516.json`

## Structured Temporal Probe

The larger temporal-only probe tested `--structured-extract` on every temporal-reasoning row:

- Rows: 133
- Source-aware baseline on same rows: 101/133 = 75.9%
- Structured two-pass on same rows: 94/133 = 70.7%
- Wins over baseline: 10
- Regressions versus baseline: 17
- Net: -7

Readout: structured extraction is not safe as a blanket temporal route. It fixes real temporal reasoning misses, but it also over-abstains or picks the wrong date on rows the source-aware baseline already answered correctly.

Artifacts:

- `benchmarks/longmemeval_contextfit_token_only_qa_extract_structured_temporal_20260516.jsonl`
- `benchmarks/longmemeval_contextfit_token_only_qa_hypotheses_structured_temporal_20260516.jsonl`
- `benchmarks/longmemeval_contextfit_token_only_qa_judged_structured_temporal_20260516.jsonl`
- `benchmarks/longmemeval_contextfit_token_only_qa_summary_structured_temporal_20260516.json`
- `benchmarks/longmemeval_contextfit_token_only_qa_structured_temporal_comparison_20260516.json`

## Hybrid Temporal Fallback Candidate

A deterministic hybrid candidate was then assembled from already-judged artifacts:

- Use source-aware 20k for all non-temporal rows.
- For temporal rows, use the structured two-pass answer only when it does not say the information is unavailable.
- If the structured answer says the information is unavailable, fall back to the source-aware answer.

This produced the first full-run material improvement signal:

| Metric | Source-aware 20k | Hybrid temporal fallback |
|---|---:|---:|
| Overall QA accuracy | 81.8% | **82.8%** |
| Task-averaged QA accuracy | 83.5% | **84.2%** |
| Temporal-reasoning accuracy | 75.9% | **79.7%** |
| Multi-session accuracy | 71.4% | 71.4% |
| Abstention accuracy | 80.0% | 80.0% |

Flip analysis against source-aware 20k:

- Wins over baseline: 10
- Regressions versus baseline: 5
- Net: +5

Route counts:

- Structured temporal answers used: 107
- Source-aware fallback after structured unavailable: 26
- Source-aware non-temporal rows: 367

Readout: this is promising, but it should be treated as a candidate configuration until rerun as one coherent pipeline or verified with the official evaluator. It combines two already-judged local artifacts using a deterministic route based only on the generated answer text, not on labels.

Artifacts:

- `benchmarks/longmemeval_contextfit_token_only_qa_hypotheses_hybrid_temporal_structured_fallback_20260516.jsonl`
- `benchmarks/longmemeval_contextfit_token_only_qa_judged_hybrid_temporal_structured_fallback_20260516.jsonl`
- `benchmarks/longmemeval_contextfit_token_only_qa_summary_hybrid_temporal_structured_fallback_20260516.json`
- `benchmarks/longmemeval_contextfit_token_only_qa_hybrid_temporal_structured_fallback_comparison_20260516.json`
- `benchmarks/official/longmemeval_hybrid_temporal_structured_fallback_submission_20260516.jsonl`

## Coherent Temporal Hybrid Full Run

The hybrid route was then implemented as a single coherent runner path behind `--temporal-hybrid`.

Important caveat: the first coherent run still used LongMemEval's `question_type` field for the temporal router. This is acceptable as a benchmark ablation, but it should not be treated as product routing or as the cleanest generalization claim. The runner now supports a cleaner query-text router via `--temporal-hybrid-router query`, and structured prompts do not include `question_type` unless `--include-question-type-in-prompts` is explicitly passed.

Command shape:

```bash
OPENAI_API_KEY=... /tmp/cf-structure-venv/bin/python benchmarks/longmemeval_contextfit_qa.py \
  --source-aware \
  --temporal-hybrid \
  --top-k-context 10 \
  --max-session-chars 20000 \
  --generation-max-tokens 1200 \
  --extraction-max-tokens 1200 \
  --hypotheses-out benchmarks/longmemeval_contextfit_token_only_qa_hypotheses_temporal_hybrid_full_20260516.jsonl \
  --judged-out benchmarks/longmemeval_contextfit_token_only_qa_judged_temporal_hybrid_full_20260516.jsonl \
  --extract-out benchmarks/longmemeval_contextfit_token_only_qa_extract_temporal_hybrid_full_20260516.jsonl \
  --summary-out benchmarks/longmemeval_contextfit_token_only_qa_summary_temporal_hybrid_full_20260516.json
```

Results:

| Metric | Source-aware 20k | Coherent temporal hybrid |
|---|---:|---:|
| Overall QA accuracy | 81.8% | **82.2%** |
| Task-averaged QA accuracy | 83.5% | 82.7% |
| Temporal-reasoning accuracy | 75.9% | **79.7%** |
| Multi-session accuracy | 71.4% | **72.2%** |
| Abstention accuracy | 80.0% | **83.3%** |

Route counts:

- Source-aware rows: 367
- Structured temporal rows: 73
- Source-aware fallback after structured unavailable/guardrail: 60

Readout: the coherent run preserves the temporal lift but is a smaller net win than the assembled artifact candidate: 81.8% -> 82.2%, with 23 wins and 21 regressions versus source-aware. Treat this as an experimental synthesis signal, not the stable headline. The stable QA claim remains source-aware 20k at 81.8% until a no-label query-router run holds up.

Artifacts:

- `benchmarks/longmemeval_contextfit_token_only_qa_extract_temporal_hybrid_full_20260516.jsonl`
- `benchmarks/longmemeval_contextfit_token_only_qa_hypotheses_temporal_hybrid_full_20260516.jsonl`
- `benchmarks/longmemeval_contextfit_token_only_qa_judged_temporal_hybrid_full_20260516.jsonl`
- `benchmarks/longmemeval_contextfit_token_only_qa_summary_temporal_hybrid_full_20260516.json`
- `benchmarks/official/longmemeval_temporal_hybrid_full_submission_20260516.jsonl`

## No-Label Temporal Hybrid Guardrail

A cleaner anti-overfit guardrail was run with:

- Query-text temporal routing: `--temporal-hybrid-router query`
- No `question_type` in structured prompts
- Same source-aware fallback behavior
- Same retrieval artifact, top-k, context budget, generation model, and judge model

This tests whether the temporal hybrid improvement generalizes without LongMemEval task-label routing.

Results:

| Metric | Source-aware 20k | No-label query-router hybrid |
|---|---:|---:|
| Overall QA accuracy | **81.8%** | 81.2% |
| Task-averaged QA accuracy | **83.5%** | 82.8% |
| Temporal-reasoning accuracy | 75.9% | 77.4% |
| Multi-session accuracy | **71.4%** | 68.4% |
| Abstention accuracy | **80.0%** | 76.7% |

Route counts:

- Source-aware rows: 263
- Structured temporal rows: 142
- Source-aware fallback after structured unavailable/guardrail: 95

Readout: the no-label guardrail did not hold as an improvement. It confirms that the temporal hybrid is useful research but not a defensible headline configuration yet. Keep source-aware 20k as the stable QA result.

Artifacts:

- `benchmarks/longmemeval_contextfit_token_only_qa_extract_temporal_hybrid_query_nolabel_20260516.jsonl`
- `benchmarks/longmemeval_contextfit_token_only_qa_hypotheses_temporal_hybrid_query_nolabel_20260516.jsonl`
- `benchmarks/longmemeval_contextfit_token_only_qa_judged_temporal_hybrid_query_nolabel_20260516.jsonl`
- `benchmarks/longmemeval_contextfit_token_only_qa_summary_temporal_hybrid_query_nolabel_20260516.json`

## Official-Format Submission Package

The LongMemEval repository documents the QA output format as JSONL with two fields per line:

- `question_id`
- `hypothesis`

The ungated source-aware hypotheses were repackaged into that exact schema:

- Official-format package: `benchmarks/official/longmemeval_source_aware_qa_submission_20260516.jsonl`
- SHA-256: `54789b620c917c82dea2c72600ba1d7dd9f8da9cdce0e7a53ab0625089c64972`
- Rows: 500
- Schema check: every row contains exactly `question_id` and `hypothesis`
- Reference check: packaged `question_id` set matches `benchmarks/data/longmemeval_s_cleaned.json`

Official-style aggregation over the judged artifact reproduces the headline result:

| Metric | Score |
|---|---:|
| Overall QA accuracy | **81.8%** |
| Task-averaged QA accuracy | **83.5%** |
| Abstention accuracy | 80.0% |

Parity metadata is recorded in:

- `benchmarks/official/longmemeval_source_aware_qa_submission_20260516.parity.json`

This package is ready to be evaluated with LongMemEval's official `src/evaluation/evaluate_qa.py` command shape. The local judged artifact already uses the same GPT-4o model id and answer-check prompt templates from the official evaluator, but the official evaluator has not been rerun as a separate 500-call judge pass in this step.

## Official Evaluator Fresh Judge Pass

The official LongMemEval evaluator was cloned locally from the upstream repository and run against the stable source-aware submission package as a fresh 500-call judge pass.

Command shape:

```bash
cd /tmp/LongMemEval-official/src/evaluation
OPENAI_API_KEY=... /tmp/longmemeval-lite-py312/bin/python evaluate_qa.py \
  gpt-4o \
  /Users/christophe/.openclaw/tools/cf/benchmarks/official/longmemeval_source_aware_qa_submission_20260516.jsonl \
  /tmp/LongMemEval-official/data/longmemeval_s_cleaned.json
```

Official evaluator output:

| Metric | Score |
|---|---:|
| Rows judged | 500 |
| Overall QA accuracy | **81.8%** |
| knowledge-update | 88.5% |
| multi-session | 70.7% |
| single-session-assistant | 98.2% |
| single-session-preference | 70.0% |
| single-session-user | 97.1% |
| temporal-reasoning | 76.7% |

The official fresh judge pass reproduces the stable headline overall score, while some per-type slices differ slightly from the earlier local judged artifact due to judge-pass variance.

Official evaluator log:

- `benchmarks/official/longmemeval_source_aware_qa_submission_20260516.jsonl.eval-results-gpt-4o`
- Rows: 500
- SHA-256: `e1b10cc582aaae4f4749c9d3571107447dad6a0582fd158f98ff20c95854eed2`

## Interpretation

The result supports the hypothesis that ContextFit's retrieval was already strong enough for many missed multi-session questions, and that answer synthesis was the main bottleneck. A source-aware synthesis prompt plus a larger per-session context budget improved end-to-end QA without changing retrieval.

This should still be framed carefully:

- This is an internal reproduction-style QA bridge, not an official LongMemEval leaderboard submission.
- The margin versus Supermemory's public overall figure is very small and judge-sensitive.
- The multi-session score approximately matches Supermemory's published multi-session figure rather than clearly exceeding it.
- Abstention at 80.0% remains a real caveat and product-trust issue.

## Artifacts

- Hypotheses: `benchmarks/longmemeval_contextfit_token_only_qa_hypotheses_source_20k_20260516.jsonl`
- Judged rows: `benchmarks/longmemeval_contextfit_token_only_qa_judged_source_20k_20260516.jsonl`
- Summary: `benchmarks/longmemeval_contextfit_token_only_qa_summary_source_20k_20260516.json`
- Official-format package: `benchmarks/official/longmemeval_source_aware_qa_submission_20260516.jsonl`
- Official-format parity: `benchmarks/official/longmemeval_source_aware_qa_submission_20260516.parity.json`
- Triggered-gate summary: `benchmarks/longmemeval_contextfit_token_only_qa_summary_source_20k_gate_trigger_20260516.json`

Generated root-level JSON/JSONL QA artifacts are intentionally ignored by git. The `benchmarks/official/` package is the clean, submission-shaped artifact for this run. This Markdown evidence file records the current best local QA result and its caveats.
