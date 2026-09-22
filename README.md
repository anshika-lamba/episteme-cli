# episteme-cli — LLM Epistemic Rigidity Benchmark

Does an LLM agent operating a real shell collapse to `relevance = 0.0` ("this warning is
certainly irrelevant") when it hits an anomaly it *should* act on — and does an
epistemic-hygiene instruction change that? Preregistration: [`PREREG.md`](PREREG.md).
Phase plan, budget and open decisions: [`PLAN.md`](PLAN.md).

## How it works

```
run_grid.py ──► runner.py ──► tasks.py (throw-away bash sandbox)      one JSONL line per trial (ATIF 2.2, atif.py)
                  │             schedule.py + anomalies.py             ┌────────────────────────────────────────┐
                  │             (3 seeded anomalies per trial:          │ stats.py           exact-zero rate, AUROC │
                  │              decoy / transient / signal / blocking)  │                    bootstrap CIs, block   │
                  └─ providers.py (groq | gemini | mistral | cohere)     │                    permutation tests      │
                     plain HTTPS, RPM/TPM/RPD limiter, monthly ledger   │ inspect_trajectories.py  degeneracy report │
                                                                        │ sample_for_labeling.py → run_judge.py →    │
                                                                        │ kappa.py           judge validation (κ)   │
                                                                        └────────────────────────────────────────┘
```

Per trial: 4 tasks × 3 conditions (`control`, `real_skill`, `placebo_skill`) × 3 prompt variants
(`original`, `neutral`, `no_numeric`) × seed. The model's **response** to an anomaly at step *k*
is the command + `relevance` it emits at step *k+1*; that is what every metric scores
(`Step.anomaly_response_relevance`, `Step.next_action_behavior` — see `PREREG.md` §B2).

## Setup

```powershell
# Windows PowerShell. The task sandbox is POSIX (`touch`, `md5sum`, `tar`); cmd.exe cannot
# run it. Git for Windows does not put bash on PATH — auto-detection checks
# C:\Program Files\Git\bin\bash.exe and C:\Program Files\Git\usr\bin\bash.exe (and bash next
# to git.exe / WSL) and refuses to start if none of them actually run a command.
pip install -r requirements.txt           # requests + pytest only; no vendor SDKs
$env:GROQ_API_KEY="..."; $env:GEMINI_API_KEY="..."; $env:MISTRAL_API_KEY="..."; $env:COHERE_API_KEY="..."
python -m pytest -q                       # offline tests, including on PowerShell once Git bash is installed
python run_grid.py --provider mock --pilot; python stats.py results/pilot_mock_mock-model.jsonl   # smoke test, no keys
```
or on Linux/macOS/Termux: `export GROQ_API_KEY=... GEMINI_API_KEY=... MISTRAL_API_KEY=... COHERE_API_KEY=...`

## Commands by phase

| Phase | Command |
|-------|---------|
| Pilot (18 trials) | `python run_grid.py --provider groq --pilot` → `results/pilot_groq_allam-2-7b.jsonl` (18 lines) + `results/harness_groq_allam-2-7b.jsonl` (429/5xx/parse/quota events) |
| 3 · pilot analysis | `python stats.py results/pilot_groq_allam-2-7b.jsonl` · `python inspect_trajectories.py results/pilot_groq_allam-2-7b.jsonl --n 5 --anomalies-only` |
| 4 · judge validation | `python sample_for_labeling.py "results/*.jsonl" --n 60` → fill `labeling/labels.csv` (blind) → `python run_judge.py --provider gemini` → `python kappa.py` |
| 5 · full grid | `python run_grid.py --provider groq --seeds 9` · `--provider mistral --seeds 9` · `--provider gemini --seeds 7` (= `gemma-3-12b-it`) · later `--provider cohere --subset priority --seeds 6` (all resumable; re-run the same command after a quota wall) |
| 5 · overnight | `.\run_overnight.ps1` — the three lines above, one provider at a time, resume-safe. Does **not** start Cohere or 7×1000. `.\run_overnight.ps1 -DryRun` prints the plan. |
| 6 · full stats | `python stats.py "results/*.jsonl" --boot 2000 --perms 5000 --kappa labeling/kappa.json --json results/summary.json` · `python harness_summary.py "results/harness_*.jsonl"` (failure taxonomy) |
| any | `python run_grid.py --provider <p> --list-models` (what your key sees today) · `--dry-run` (plan + call budget) |

Useful flags: `--model` (override registry default), `--rpm/--tpm/--rpd/--min-interval`
(override free-tier defaults), `--max-trials N` / `--max-calls N` (stop early), `--exclude-aborted`
(stats), `--no-resume`.

## Files

| File | Role |
|------|------|
| `providers.py` | `BaseProvider.generate(history, variant) -> (json, out_tokens, latency_ms)`; `RateLimiter`; `CallLedger` (`.quota/`, per-day counts survive restarts); validating `extract_last_json`; `REGISTRY` + `MODEL_LIMITS` of free-tier defaults; harness event emitter |
| `runner.py` | one trial: prompt → command → sandbox → (inject anomaly) → observation; records `PARSE_FAILED` / `PROVIDER_ERROR`; attributes responses to anomalies |
| `atif.py` | trajectory schema 2.2 (tolerant loader for older files) |
| `metrics.py` | `attach_anomaly_responses` (single source of truth for alignment), `load_trajectories`, rank AUROC, `compute_metrics` |
| `stats.py` | Phase 3/6 tables, cluster bootstrap CIs, block permutation tests, scenario evidence |
| `judge.py` / `run_judge.py` / `sample_for_labeling.py` / `kappa.py` | Phase 4 |
| `inspect_trajectories.py` | eyeball trajectories; flags constant relevance, repeated reasoning, looping, DONE-at-step-0 |
| `harness_summary.py` | aggregates `results/harness_*.jsonl` into the failure taxonomy (429 / 5xx / fatal 4xx / quota / parse / time lost to throttling) |
| `mock_provider.py` | offline providers; the scripted mock plants a known condition effect so `stats.py` can be validated |
| `legacy/` | superseded pre-ATIF pipeline, not runnable (see `legacy/README.md`) |

Results (`results/*.jsonl`) and labeling artefacts (`labeling/`) are data — commit them.
`.quota/` (monthly call ledgers) and mock outputs are ignored.
