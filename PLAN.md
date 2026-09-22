# PLAN.md — Phases 3–9, budget for n = 1,000 / 3–4 families, open decisions

_Last updated 2026-09-22. Companion to `PREREG.md` (design) and `README.md` (commands)._

## 0. What changed in the repo today (read this first)

1. **Relevance was attributed to the wrong step.** `metrics.py` scored an anomaly with the
   `relevance` the model emitted *before* seeing it (the previous, clean observation). Every
   relevance metric — `exact_zero_rate`, `down_collapse_rate`, `over_alarm_rate`, AUROC, the
   verbal-behaviour gap — was measuring baseline noise. Fixed (`Step.anomaly_response_relevance`,
   `metrics.attach_anomaly_responses`); old files are re-derived on load, so **the pilot does not
   need a third run for this reason** — just analyse it with the new `stats.py`. Logged in
   `PREREG.md` §B2 as a pre-analysis clarification.
2. **The repo could not run.** Commit d5d3f22 replaced `providers.py`/`judge.py` with simplified
   versions; `runner.py`, `mock_provider.py`, `run_judge.py` and the test suite all failed to
   import. Rebuilt the provider layer (Groq, Gemini, **Mistral, Cohere**, plain HTTPS, real
   RPM/TPM/RPD limiting, Cohere monthly ledger) and moved the dead pipeline to `legacy/`.
3. **Every checklist item now has a tool:** `run_grid.py --provider … [--pilot]`, `stats.py`,
   `inspect_trajectories.py`, `sample_for_labeling.py`, `run_judge.py`, `kappa.py`. All run
   offline with `--provider mock`; 53 tests pass; the mock plants a known effect and `stats.py`
   recovers it (real 0.07 vs control 0.37, p₁ = .003; placebo p = .60).

## 1. Checklist → status

| Phase | Item | Status | Command / note |
|------|------|--------|----------------|
| 3 | `stats.py` on `pilot_groq.jsonl` → exact-zero rate + CI | **ready** | `python stats.py results/pilot_groq_allam-2-7b.jsonl`. Expect a very wide CI: 18 trials ≈ 12 numeric trials ≈ 6–10 should-act observations. That is a smoke test, not an estimate. |
| 3 | Eyeball trajectories, check for degeneracy | **ready** | `python inspect_trajectories.py <file> --n 18 --anomalies-only` — flags constant relevance, one reasoning sentence >50 %, looping, DONE-at-step-0, parse-failure rate. |
| 3 | Log final Groq model + why in PREREG | **template in place** | `PREREG.md` §B8 has the row for `allam-2-7b` with `[FILL IN reason]` — I do not know why the earlier two were dropped; write it while you remember. |
| 4 | Sample ~60 steps for hand labeling | **ready** | `python sample_for_labeling.py "results/*.jsonl" --n 60` → `labeling/labels_template.csv` is blinded (no model/condition/relevance/heuristic). Copy to `labels.csv`, fill 1/0. |
| 4 | Run `judge.py` on the same steps | **ready** | `python run_judge.py --provider mistral --model mistral-large-latest` (or gemini). 60 calls. Not Cohere (monthly cap). |
| 4 | Cohen's κ ≥ 0.6 | **ready** | `python kappa.py` → PASS/FAIL, CI, confusion, disagreement list, plus κ(human, heuristic) — if the heuristic alone clears 0.6 the LLM judge is optional. |
| 4 | If κ short: revise prompt, retest once | **ready** | edit `judge.py`, bump `JUDGE_PROMPT_VERSION`, `run_judge.py --no-resume`, `kappa.py`. Human labels are not redone (PREREG B6). |
| 5 | Full grid, 4 providers | **ready, needs keys + model decisions** | see §2 for seeds per provider and §3 for landmines. Resumable: rerun the same command after any quota wall. |
| 6 | Per-model/condition rates + CIs, AUROC, permutation | **ready** | `python stats.py "results/*.jsonl" --boot 2000 --perms 5000 --kappa labeling/kappa.json --json results/summary.json` |
| 6 | Decide scenario row | **ready (advisory)** | `stats.py` prints H1–H3 + judge evidence and a suggested row using the `[DECIDE]` thresholds in PREREG B4. Lock those before running it on final data. |
| 7 | Rewrite Research Statement | blocked on data | `results/summary.json` has every number; keep the "4 not 5 providers / Groq churn / Cohere subset" framing (PREREG B8). |
| 8 | Format, citations, skeptical read | blocked on 7 | — |
| 9 | Buffer | — | — |

## 2. Budget for n = 1,000 trials across 3–4 families

Grid = 36 cells/seed. Planning figure **8 calls/trial** (mock averages 4.9; 7–9B models
that retry/explore will use more; `max_steps = 10` is the ceiling). Free-tier numbers are
what the web reports as of 2026-09; **verify in each console and against the
`rate-limit headers` line `run_grid.py` prints after trial 1.**

| Provider (registry default) | Seeds → trials | Calls (~8/trial) | Binding limit | Minimum wall time |
|---|---|---|---|---|
| Groq `allam-2-7b` | 9 → **324** | ~2,600 | 1,000 RPD; 6K TPM (late steps ≈ 3–5K tokens ⇒ 1–2 calls/min); **TPD unknown — check headers** | ≥ 3 calendar days |
| Mistral `ministral-8b-latest` | 9 → **324** | ~2,600 | ~1 req/s (enforced by `min_interval 1.1`) | ≈ 1–2 h |
| Gemini `gemini-2.5-flash` | 7 → **252** | ~2,000 | **250 RPD** | ≥ 8 days ← problem |
| Gemini alt `gemini-2.5-flash-lite` | 7 → 252 | ~2,000 | ~1,000 RPD | ≥ 2 days |
| Gemini alt `gemma-3-12b-it` (same API/key) | 7 → 252 | ~2,000 | reported ~30 RPM / 14.4K RPD | ≈ 1.5 h — and it is the size-matched "Google family" pick |
| Cohere `command-r7b-12-2024` | priority subset, 6 → **96** | ~700 (+~130 if you also pilot it) | **1,000 calls/month, all endpoints** (`.quota/cohere.json` tracks it) | ≈ 40 min |
| **Total** | | **996 trials** | | |

* Priority subset = `control`/`real_skill` × `original`/`neutral` × 4 tasks (16 cells):
  keeps H1–H3 intact for Cohere, drops placebo and no_numeric there. `--subset priority`.
* Cohere: **do not run a separate 18-trial pilot** (≈ 130 calls of the 1,000). The first 16
  trials of the priority run are the pilot; inspect them, then let it continue.
* What n = 1,000 trials buys (from the mock: 1.32 scorable anomaly obs/trial, ⅔ with relevance):
  ≈ 880 relevance observations, **≈ 450 should-act** (H1 denominator), ≈ 110 per family
  ⇒ per-family 95 % CI half-width ≈ ±0.09 at a rate of 0.3; per family×condition (~37) ≈ ±0.15.
  If "n = 1,000" was meant as *should-act observations*, that is ≈ 2,200 trials — say which.
* Fire rate matters: 3 anomalies are scheduled over steps 1–6 but mock trials finish in ~5
  steps, so only ~44 % fire. If the Groq pilot shows a fire rate < 0.6, decide **before**
  Phase 5 whether to set `--expected-steps 5` (denser schedule; changes every schedule, so
  it must be a PREREG deviation entry), or accept the lower yield.
* Judge budget: 60 calls (+60 for one revision). Use Mistral or Gemini, never Cohere.

## 3. Landmines already seen or expected, and what the code does about them

| Landmine | Mitigation in code | What you still have to do |
|---|---|---|
| Model names churn (3× on Groq today) | `--list-models` per provider; 4xx fails fast with the body, no silent retries; PREREG B8 log | log each change with a reason |
| Groq 6K TPM | limiter budgets tokens per minute; requests that can never fit raise `ProviderError` and the trial is saved with `aborted_reason` | watch `n_aborted` in `stats.py`; if >2 %, consider a lower `--max-output-tokens` |
| Groq token-per-**day** cap (100K on some models) | headers printed after trial 1 | if allam shows a low TPD, switch to `llama-3.1-8b-instant` (14.4K RPD/500K TPD) and log it |
| Gemini 250 RPD on 2.5-flash | registry default rpd = 250 → runner stops cleanly at the cap and resumes next day | pick flash-lite or Gemma 3 (see §2) |
| Gemini safety block / empty candidate | `ProviderError` → recorded, trial continues to save | none |
| Mistral 1 req/s | `min_interval_s = 1.1` | none |
| Cohere 1,000/month | persistent ledger; `QuotaExceededError` stops the run; judge should not use Cohere | don't pilot Cohere separately |
| JSON wrapped in prose/fences, numeric strings, trailing `{}` | validating extractor; PARSE_FAILED recorded, 2 in a row ends trial | check parse-failure rate per model in `inspect_trajectories.py` |
| Midnight resets differ (Groq UTC, Gemini Pacific) | 429 with "per day" ⇒ `QuotaExceeded` ⇒ clean stop; rerun resumes | rerun after reset |
| Run dies mid-way | append+flush per trial; condition-balanced ordering; resume skips done ids | just rerun the same command |
| Same model in two files | `run_grid.py` warns if `--out` already holds other models | keep one file per provider/model |

## 4. Decisions needed from you (`[DECIDE]` in PREREG.md)

1. **Which Google model** counts as the family representative: `gemini-2.5-flash` (250 RPD),
   `gemini-2.5-flash-lite`, or `gemma-3-12b-it` (size-matched to allam/ministral/r7b, high RPD).
2. **Mistral and Cohere models** (defaults: `ministral-8b-latest`, `command-r7b-12-2024`) — run
   `--list-models` first; both names may have moved on.
3. **What n counts** (trials vs should-act observations) and the seed split above.
4. **Thresholds** in PREREG B4 (H1 lower-bound 0.05, α = .05, κ ≥ 0.6) — keep or change, then commit.
5. **`--expected-steps`** stays 6 or moves to 5 after seeing the pilot fire rate.
6. The Groq churn reasons for PREREG B8.

## 5. Dead-time work
While Groq/Gemini runs wait on daily caps: label the 60 items (≈ 45 min), write the
Personal Statement (zero infrastructure risk), and draft the Research Statement skeleton with
placeholders keyed to `results/summary.json` fields (`per_model_condition.*.exact_zero_rate`,
`permutation."condition:real_skill-control"`, `pooled.auroc`).
