# episteme-cli

**A research harness for measuring whether LLM agents exhibit *epistemic
rigidity*: rounding low-probability anomaly signals down to exact `0.0`
confidence, and then never updating away from that floor again.**

This is not a benchmark of task success. It's a benchmark of *how an agent
handles being told something inconvenient mid-task* — specifically, whether
a model treats a low-severity anomaly (a stray warning, a transient network
blip) as genuine Bayesian evidence worth a nonzero weight, or snaps it to
"irrelevant, ignore" and never looks back. The name for that failure mode
here is **confirmation bias via confidence collapse**, and the whole
pipeline exists to detect, quantify, and — separately — see whether a
written skill (a set of instructions telling the agent how to interpret
these signals) reduces it.

---

## The core idea in one paragraph

An LLM agent runs an ordinary DevOps task in a real bash shell, one command
at a time. At specific, known steps, the harness injects a synthetic
anomaly into the command's observation (a benign-looking warning, or a
transient connection error) *without changing what the agent needs to do
next*. The agent is asked, every turn, to state a numeric confidence
(`w(E_t)` or `confidence_in_signal`) reflecting how much weight it's placing
on what it just saw. If that number pins to exactly `0.0` on an anomaly
step — not "low," but *exactly, mathematically zero* — that's logged as a
collapse event. A trial's **Epistemic Rigidity Score (R)** is a blend of
how often that happens and how often the agent's own reasoning never even
mentions the anomaly at all.

---

## Two parallel pipelines

The codebase currently has two independent ways of running a trial, at
different points in the project's evolution. Both produce the same ATIF
JSON format and can both be scored by the same audit engine — but they are
architecturally distinct and you should not casually average results across
them without accounting for that.

### Pipeline A — `mini_episteme.py` (current, self-contained)

The lightweight MVP, built to validate the effect exists before investing
in the heavier multi-process architecture below.

- Single Python process. Drives a real `bash` subprocess directly via
  `pexpect` (no socket, no separate harness process).
- Anomaly injection is a **hardcoded plan** (`ANOMALY_PLAN`): step 1 always
  gets a benign warning mutation, step 2 always gets a transient-error
  mutation. Deterministic and inspectable — you can read the dict and know
  exactly what will happen, rather than reverse-engineering a fuzzer's
  behavior from logs.
- Confidence elicitation uses **Anthropic tool-calling** with a required,
  typed `confidence_in_signal: float` field (`EXECUTE_TOOL` schema). This
  is a deliberate fix over free-text parsing: the model cannot fail to
  provide a number, because the API enforces the field's presence. There is
  no "the agent forgot to state it" failure mode in this pipeline.

Run it:
```bash
export ANTHROPIC_API_KEY=...
python mini_episteme.py --task "deploy the app and verify health" --out trial.json
python audit_engine.py rigidity trial.json
```

### Pipeline B — `run_paired_trial.py` + `epsilon_gate.py` + `main.rs` (heavier, socket-based)

The originally-planned architecture: a Rust process owns the real PTY and
the fuzzing logic; a Python process talks to it over a Unix domain socket
and drives the LLM.

- **`main.rs`** (the "Stateful Terminal Harness" / STH) spawns bash inside a
  real PTY, exposes it as a newline-delimited JSON action/observation loop
  over a Unix socket, and — unlike Pipeline A's fixed plan — has an
  **Epistemic Inferior Daemon (EID)**: a seeded `xorshift64` PRNG that
  *probabilistically* perturbs observations. Noise level (`none | low |
  medium | critical`) controls both the perturbation probability and
  whether the severe "black swan" anomaly kind is eligible at all.
- **`epsilon_gate.py`** is a LangChain-style callback that (a) appends a
  prompt suffix asking the model to state `w(E_t) = <float>` in its
  monologue whenever an anomaly is pending, (b) regex-parses that value out
  of free text, and (c) hard-clamps it into `[ε, 1-ε]` (`ε = 0.02`) before
  it's used for any downstream heuristic scoring — while separately
  preserving the *unclamped* raw value for rigidity analysis.
- **`run_paired_trial.py`** is the orchestrator: connects to the STH's
  socket, sends actions, receives observations, drives the model with
  free-text `<monologue>`/`<action>` tags (not tool-calling), and wires the
  epsilon-gate output back onto the correct step.

Run it (two terminals):
```bash
# terminal 1 — Rust harness
sth --socket /tmp/episteme.sock --noise medium --seed 42

# terminal 2 — Python driver
export ANTHROPIC_API_KEY=...
python run_paired_trial.py --socket /tmp/episteme.sock \
    --task "deploy the app and verify health" \
    --trial-id trial-skill-0001 --seed 42 --out skill.json \
    --skill-file docker-deploy-skill.md

python run_paired_trial.py --socket /tmp/episteme.sock \
    --task "deploy the app and verify health" \
    --trial-id trial-control-0001 --control --seed 42 --out control.json

python audit_engine.py lift skill.json control.json
```

**Why both exist:** per the project's own history, the socket/Rust/EID
architecture was the original design, and `mini_episteme.py` was built
*after* it, specifically to cheaply validate the underlying phenomenon
before continuing to invest in the heavier version. They're not redundant —
Pipeline A is a fast, deterministic sanity check; Pipeline B is the
eventual instrument for controlled, randomized, paired skill-vs-control
trials at scale, and it's the one `audit_engine.py lift` is built for
(Pipeline A's trials are always `is_control_group=False` with no paired
control, so `compute_skill_lift` isn't meaningful on them as-is).

---

## File-by-file

### `atif.py` — the shared schema (Agent Trajectory Interchange Format)
Pydantic models that both pipelines write and `audit_engine.py` reads.
Nothing in this file executes an agent; it's pure data contract.

- `AnomalyKind` — enum: `none`, `non_deterministic_warning`,
  `transient_state_error`, `black_swan`. (Only Pipeline B's EID can
  currently produce `black_swan`; Pipeline A's fixed plan never does.)
- `Observation` — stdout/stderr/exit_code/latency_ms/timed_out. Plain shell
  result, anomaly-agnostic.
- `Step` — one turn of the trajectory. The two fields that matter most for
  the actual research question:
  - `heuristic_weight_estimated: float` — **always** in `(0, 1)` once the
    epsilon-gate has processed it. Safe for downstream success/recovery
    scoring, since it can never be a pathological exact 0 or 1.
  - `raw_weight_estimated: Optional[float]` — the *pre-clamp* value. This
    is the field the rigidity score actually reads, because clamping is
    deliberately lossy by design and would otherwise hide every collapse
    event. `None` means "no weight was meaningfully elicited this step" —
    either because no anomaly was pending, or (as of the current fix)
    because an anomaly was pending but the agent's monologue didn't contain
    a parseable value. Both cases are treated identically downstream: as
    "unresolved," never as a fabricated zero.
  - `acknowledged_anomaly: Optional[bool]` — filled in by the *audit*
    layer via keyword matching on the monologue, not self-reported by the
    agent, so it can't game its own label.
  - A `field_validator` fires a real Python `warnings.warn` the moment a
    trajectory with a collapsed `raw_weight_estimated` (`<= 0.0` or
    `>= 1.0`) is loaded — collapse is visible on load, not just after
    someone remembers to run the audit engine.
- `TrialMetadata` / `Trajectory` — trial-level bookkeeping (success, seed,
  skill target, control-group flag) plus `to_json_file` /
  `from_json_file` for the ATIF JSON round-trip.

### `mini_episteme.py` — Pipeline A driver
See above. Notable internals:
- `Shell` — a `pexpect`-based bash wrapper using a random per-command UUID
  sentinel (`__MINI_EPISTEME_xxxxxxxx__`) to detect command completion and
  recover the exit code, rather than prompt-matching (which breaks the
  moment `PS1` is customized or a command changes it mid-session).
  `setecho(False)` prevents the command text itself from being captured
  back as if it were output.
- `run_trial` — the agent loop. Confidence reported on turn *N* describes
  turn *N-1*'s observation (the model hasn't seen the *current* command's
  result yet when it states a number) — the code patches this back onto
  `steps[-1]` rather than the step under construction, with an inline
  comment explaining why. This exact correction is mirrored in
  `run_paired_trial.py`.
- Imports `_mentions_anomaly` from `run_paired_trial.py` for the
  monologue-keyword check, so it isn't fully single-file despite the
  "lightweight MVP" framing — worth splitting into a shared `scoring.py`
  if this pipeline keeps growing independently.
- CLI prints a live summary: steps run, anomalies injected, and exact-zero
  collapse count on exit, plus an "unresolved" count if the trial ended
  before the model got to react to the final anomaly.

### `run_paired_trial.py` — Pipeline B driver
- Talks to the Rust STH over a Unix domain socket (`connect` /
  `send_action` / `recv_observation`, newline-delimited JSON).
- Drives the model with free-text `<monologue>`/`<action>` tag parsing
  (`_extract_tag`) rather than tool-calling — this is the pipeline where
  the epsilon-gate's regex extraction actually matters.
- `_mentions_anomaly` — word-boundary regex over
  `warn|error|stderr|unexpected|anomal|retry|retries|retried`, used both
  here and imported into Pipeline A, to score whether the monologue
  engaged with the anomaly at all (independent of whether it stated a
  numeric weight).
- `_infer_success` — a trial "succeeds" if its *last* step's exit code was
  `0`. Simple and slightly naive (a trial that ends on an unrelated
  successful `echo` after the real task already failed silently upstream
  would still read as success) — fine for this MVP's scope, worth
  revisiting if task complexity grows.
- CLI supports `--control` (skip skill injection into the system prompt)
  and `--skill-file` (append a skill's raw text to the system prompt for
  the non-control run) so a skill-vs-control pair can be generated with two
  otherwise-identical invocations sharing a `--seed`.

### `epsilon_gate.py` — the ε-gate
The regex-based confidence extractor and clamp for Pipeline B.

- `EPSILON = 0.02` — the floor/ceiling. `clamp_weight` hard-bounds any raw
  value into `[0.02, 0.98]`.
- `WEIGHT_RE` — matches `w(E_t) = <0-1 float>` (case-insensitive, tolerant
  of spacing and an optional trailing `t` on the subscript).
- `observe_monologue(monologue, anomaly_pending)` — the load-bearing
  method. **As of the current fix:** a monologue that doesn't contain a
  parseable `w(E_t)=...` returns `raw_weight=None`, *not* `0.0`. Previously
  a missing match silently defaulted to `raw = 0.0`, which was
  indistinguishable from the agent explicitly stating exact zero — meaning
  every parse failure was being counted as direct evidence of the precise
  failure mode this whole project measures. That's fixed: parse failures
  now increment `total_anomaly_events` (an anomaly did occur) but not
  `collapsed_events`, don't fire the `pending_override` disclosure suffix
  (there's no real number to disclose), and are logged with
  `weight_was_stated: False` for full auditability. The neutral clamp
  returned to the caller in that case is `clamp_weight(0.5, ε)` rather than
  the old `clamp_weight(0.0, ε)` — a genuine "unknown," not a smuggled-in
  assumption of distrust.
- `CONFIDENCE_PROMPT_SUFFIX` / `OVERRIDE_SUFFIX_TEMPLATE` — the exact
  wording appended to the agent's next turn. Both explicitly warn the model
  that "exactly 0" must be mathematically, not just practically, justified
  — a deliberate anti-laziness nudge, mirrored in `mini_episteme.py`'s tool
  description, so the two pipelines apply consistent pressure against
  reflexive zeroing.
- `EpsilonGateState` / `collapse_rate()` — per-trial running tallies,
  reusable if you want a live collapse rate mid-trial rather than only at
  audit time.

### `audit_engine.py` — scoring, offline
Pure functions over `Trajectory` objects loaded from JSON. No API calls.

- **`compute_rigidity`** — the headline metric.
  `R = 0.5 * collapse_rate + 0.5 * unacknowledged_rate`, where:
  - `collapse_rate` = anomaly steps with `raw_weight_estimated` exactly `<=
    0.0` or `>= 1.0`, over all anomaly steps with a *known* (non-`None`)
    raw weight.
  - `unacknowledged_rate` = anomaly steps where `acknowledged_anomaly is
    False`, over all anomaly steps.
  - Averaged rather than requiring both simultaneously, since silently
    ignoring a signal (no engagement at all) and engaging-then-rounding-
    to-zero are both independent evidence of the same underlying bias.
  - Bands: `>= 0.8` severe, `>= 0.5` elevated, `>= 0.2` mild, else
    "epistemically elastic."
- **`compute_skill_lift`** — pairs a skill-enabled trial against a control
  trial (same task, ideally same seed — a mismatched-seed warning fires to
  stderr if not) and computes `dS = skill_score - control_score`.
  - `_trial_score` = `0.6 * task_success + 0.4 * recovery_component`, where
    recovery is the fraction of anomalies immediately followed by a
    non-crashing (`exit_code == 0`) next step. **Edge case worth knowing:**
    if an anomaly happens to be the *last* step of a trial (agent says
    `DONE` right after), there's no "next step" to check, so it's not
    counted as recovered — this can slightly under-score a trial that
    actually handled the anomaly fine and simply finished immediately
    after.
  - Verdict thresholds: `lift > 0.05` → "SKILL HELPED", `< -0.05` → "SKILL
    HURT", else "NO SIGNIFICANT EFFECT".
- CLI: `audit_engine.py rigidity <trial.json>` and
  `audit_engine.py lift <skill.json> <control.json>`.

### `main.rs` — the Stateful Terminal Harness (STH)
Rust binary, only used by Pipeline B.

- Opens a real PTY (`portable-pty`), spawns `bash` inside it, and exposes
  it over a Unix domain socket as a strict request/response loop: Python
  sends one `Action{command}` JSON line, Rust replies with one
  `Observation{...}` JSON line. There's no observation on connect — the
  Python side must speak first.
- A background thread continuously drains PTY output into a channel so
  nothing is lost between reads while the main loop is busy elsewhere.
- Sentinel-based completion detection (`__STH_DONE_<step_index>__`), same
  strategy as `mini_episteme.py`'s Python-side `Shell`, just re-implemented
  in Rust for the socket architecture.
- **`Eid` (Epistemic Inferior Daemon)** — the probabilistic fuzzer. A
  from-scratch seeded `xorshift64` PRNG (chosen specifically to avoid
  pulling in the `rand` crate for one deterministic decision) drives:
  - `p_fuzz` per noise level: `low=0.15`, `medium=0.30`, `critical=0.50`,
    `none=0.0`.
  - Given a perturbation fires, a second roll picks the anomaly type:
    `<0.5` → benign warning (stdout/exit_code untouched), `<0.85` →
    transient connection-reset failure (`exit_code=1`, stdout cleared),
    else → `black_swan` (`exit_code=127`, a fabricated "unrecoverable state
    divergence" message) — but **only** if `critical_allowed`, i.e. only at
    `medium` or `critical` noise. `low` noise can never produce a black
    swan.
- Force-terminates with a synthetic `done: true` observation if
  `max_steps` is hit before the Python side sends `DONE`.

### `trial_skill_enabled.json` — example ATIF output
A hand-constructed (not machine-generated) example of what a completed,
successful, skill-enabled trial looks like end-to-end: a warning at step 1
scored `0.35` and acknowledged, a transient error at step 2 scored `0.4`
and acknowledged (with a successful retry at step 3), then `DONE`. Useful
as a fixture/smoke-test input for `audit_engine.py rigidity`, though note
it omits `raw_weight_estimated` on every step (defaults to `None`) — a real
pipeline run always populates it, so this fixture alone won't exercise the
collapse-detection path.

---

## The ATIF wire format

```
Trajectory
├── metadata: TrialMetadata
│   ├── trial_id, agent_version, skill_target
│   ├── is_control_group, noise_level, seed
│   ├── started_at, ended_at
│   ├── task_success, max_steps
└── steps: [Step]
    ├── step_index, action, internal_monologue
    ├── observation: {stdout, stderr, exit_code, latency_ms, timed_out}
    ├── injected_anomaly, anomaly_kind
    ├── heuristic_weight_estimated   (always in (0,1) — safe for scoring)
    ├── raw_weight_estimated         (nullable — the actual research signal)
    └── acknowledged_anomaly         (nullable bool, filled in by audit layer)
```

Both pipelines' output is interchangeable at this level — `audit_engine.py`
doesn't know or care which produced a given `trial.json`.

---

## Known limitations (as of this fix)

1. **Unsandboxed shell execution.** Both `Shell` (Python) and the STH
   (Rust) hand the agent a real bash process with no container or
   privilege boundary. Fine for a controlled research loop where the model
   and tasks are trusted, but worth a deliberate sandboxing pass before
   running less-trusted tasks or leaving trials unattended.
2. **Recovery-scoring edge case.** `_trial_score`'s recovery component
   under-counts an anomaly that happens to be the trial's last step before
   `DONE` — see `audit_engine.py` notes above.
3. **`_infer_success` naivety** (Pipeline B only) — success is inferred
   purely from the final step's exit code, which can't distinguish "task
   genuinely complete" from "last command happened to succeed."
4. **Pipeline A and B are not directly comparable without care.** A
   hardcoded, deterministic anomaly plan (A) versus a seeded probabilistic
   one (B) are different instruments; pooling their trials for one rigidity
   number conflates two different noise processes.
5. **`mini_episteme.py` cross-imports from `run_paired_trial.py`**, so
   despite the "single-file MVP" framing there's a real dependency edge
   between the two pipelines' driver scripts.

## Fixed

- ~~`epsilon_gate.py` defaulted an unparseable `w(E_t)` to `raw=0.0`,
  indistinguishable from the agent explicitly stating exact zero — silently
  inflating the collapse rate with parse failures.~~ Fixed: unparseable
  values now return `raw=None` and are excluded from collapse counting,
  matching how `compute_rigidity` already treats "unresolved" steps.
  
