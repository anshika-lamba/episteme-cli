#!/usr/bin/env python3
"""Run the experiment grid for one provider/model, resumably.

    python run_grid.py --provider groq --pilot                 # 18-trial pilot -> results/pilot_groq.jsonl
    python run_grid.py --provider gemini --seeds 7             # full grid: 36 cells x 7 seeds = 252 trials
    python run_grid.py --provider cohere --subset priority --seeds 8   # 16 cells x 8 = 128 trials (~900 calls)
    python run_grid.py --provider mistral --seeds 7 --dry-run  # print the plan + call budget, no API calls
    python run_grid.py --provider groq --list-models           # what does this key actually see today?

Design
------
* Grid = tasks x conditions x variants x seeds. Within a seed, cells are ordered
  round-robin over conditions, so a run killed half-way still leaves the
  conditions balanced (the primary comparison is control vs real_skill).
* Every finished trajectory is appended, flushed and fsync'd (JSONL). The file is
  never truncated. Re-running the same command reads --out first and skips
  trial_ids already logged, including trials whose only failure was a formatting
  collapse (that is a measurement, not a crash). A trailing partial line from a
  reboot is truncated. Trials aborted by a quota wall, a sandbox setup failure,
  or transport exhaustion ("gave up after") are dropped and re-queued; a finished
  trial that merely logged a permanent ProviderError (e.g. HTTP 400) stays skipped.
* QuotaExceededError (daily/monthly cap) stops the run cleanly; the partial
  trajectory is still written with metadata.aborted_reason set.
* Pilot = 2 tasks x 3 conditions x 3 variants x seed 0 = 18 trials, i.e. one of
  every prompt cell -- enough to sanity-check every prompt/provider combination.
"""
import argparse
import datetime as _dt
import itertools
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

from providers import make_provider, list_models, ProviderError, QuotaExceededError, REGISTRY, effective_limits
from harness_summary import JsonlSink, summarize_events
import tasks as tasks_mod
from tasks import TASKS, TaskEnv
from runner import run_trial

ALL_TASKS = list(TASKS.keys())                       # python_test, config_health, checksum_build, log_rotation
ALL_CONDITIONS = ["control", "real_skill", "placebo_skill"]
ALL_VARIANTS = ["original", "neutral", "no_numeric"]
PILOT_TASKS = ["python_test", "log_rotation"]
# Priority subset for capped providers (Cohere 1000/month): keeps the primary contrast
# (control vs real_skill) and both numeric variants across all tasks.
PRIORITY = dict(tasks=ALL_TASKS, conditions=["control", "real_skill"], variants=["original", "neutral"])
CALLS_PER_TRIAL_EST = 8  # planning figure: <= max_steps (10); typical 5-8


REQUIRED_TOOLS = ["md5sum", "tar", "sed", "printf", "touch", "grep", "cat"]  # used by the 4 task scripts (python is probed separately: python3|python|py)
OPTIONAL_TOOLS = ["pytest"]

# Git for Windows does not put bash.exe on PATH (only cmd\git.exe). Check both layouts
# explicitly — shutil.which("bash") is not enough on a default PowerShell install.
_GIT_BASH_OFF_PATH = (
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files\Git\usr\bin\bash.exe",
)


def resolve_sandbox_shell(choice: str) -> Tuple[Optional[list], Optional[str]]:
    """-> (shell_prefix | None, error_message).

    On Windows, auto is WSL only. cmd.exe fails every Linux command, and Git bash is
    not a substitute: the trial user must not have passwordless sudo. TaskEnv uses the
    same prepare_wsl_sandbox() path, so a direct TaskEnv agrees with this CLI.
    """
    if choice == "native":
        if os.name == "nt":
            return None, "native shell on Windows is cmd.exe; it cannot run the task scripts. Leave --sandbox-shell at auto (WSL)."
        return None, None
    if choice in ("auto", "wsl"):
        if os.name != "nt":
            return None, None
        try:
            return tasks_mod.prepare_wsl_sandbox(), None
        except Exception as e:
            return None, str(e)
    pref = [choice, "-c"]
    if (os.path.sep in choice or "/" in choice) and not os.path.isfile(choice):
        return None, f"--sandbox-shell {choice!r} does not exist"
    if not tasks_mod._probe_shell(pref):
        return None, f"--sandbox-shell {choice!r} does not work (failed the echo probe)"
    return pref, None


def preflight() -> list:
    """Cheap, quota-free checks that the sandbox works HERE, before burning 2,000 API calls.
    On Windows this is what catches 'cmd.exe cannot run the POSIX task scripts'."""
    problems, warns = [], []
    env = TaskEnv(TASKS["log_rotation"])
    probe = lambda cmd: env.run_cmd(cmd, timeout=30)
    try:
        env.setup()
        out, err, code, _ = probe("printf ok > p.txt && [ -s p.txt ] && echo OK")
        if "OK" not in out:
            problems.append(f"sandbox shell cannot run POSIX commands (exit {code}, stderr: {err.strip()[:160]})")
        for tool in REQUIRED_TOOLS + OPTIONAL_TOOLS:
            out, _e, code, _ = probe(f"command -v {tool} >/dev/null 2>&1 && echo FOUND")
            if "FOUND" not in out:
                (warns if tool in OPTIONAL_TOOLS else problems).append(f"missing tool: {tool}")
        out, _e, _c, _ = probe('for p in python3 python py; do command -v "$p" >/dev/null 2>&1 && { echo FOUND; break; }; done')
        if "FOUND" not in out:
            problems.append("no python reachable (python3 | python | py) - the harness scripts need an interpreter")
        for t in TASKS.values():  # every task must set up and not trivially succeed
            te = TaskEnv(t)
            try:
                te.setup()
                if te.check_success():
                    warns.append(f"task {t.name!r} SUCCEEDS right after setup (env problem? a model acing it proves nothing)")
            except Exception as e:
                problems.append(f"task {t.name!r} setup failed: {str(e)[:160]}")
            finally:
                te.cleanup()
    finally:
        env.cleanup()
    for w in warns:
        print(f"[preflight][warn] {w}", file=sys.stderr)
    return problems


def build_plan(tasks: List[str], conditions: List[str], variants: List[str], seeds: List[int]) -> List[Tuple[str, str, str, int]]:
    plan = []
    for seed in seeds:
        by_cond: Dict[str, List[Tuple[str, str]]] = {}
        for task, variant in itertools.product(tasks, variants):
            for cond in conditions:
                by_cond.setdefault(cond, []).append((task, variant))
        rng = random.Random(f"order|{seed}")
        for cond in conditions:
            rng.shuffle(by_cond[cond])
        for i in range(len(tasks) * len(variants)):
            for cond in conditions:
                task, variant = by_cond[cond][i]
                plan.append((task, cond, variant, seed))
    return plan


def note_trial_abort(consecutive_gave_up: int, reason: Optional[str], limit: int) -> Tuple[int, Optional[str]]:
    """Count consecutive transport-exhaustion aborts. After `limit` in a row, stop this
    provider (the trials are re-queued on resume) so an overnight run can move on.
    Any other outcome resets the counter. limit <= 0 disables the breaker."""
    if reason and "gave up after" in reason:
        consecutive_gave_up += 1
        if limit > 0 and consecutive_gave_up >= limit:
            return consecutive_gave_up, (
                f"{consecutive_gave_up} trials in a row exhausted transport retries; "
                "stopping this provider so the night can continue. Re-run to resume these trials."
            )
        return consecutive_gave_up, None
    return 0, None


def is_retryable_abort(reason: Optional[str]) -> bool:
    """Trials that did not produce a usable record and should be re-run on resume.

    Quota walls, a missing sandbox, and transport exhaustion ("gave up after N
    attempts") are harness failures, not measurements. A generic ProviderError
    (bad model, HTTP 400) is kept so a permanent rejection is not retried all night.
    """
    reason = reason or ""
    if not reason:
        return False
    if reason.startswith("QuotaExceeded") or reason.startswith("sandbox_setup_failed"):
        return True
    return "gave up after" in reason


def repair_jsonl_tail(path: str) -> bool:
    """Truncate a torn last line (process killed mid-write). Returns True if bytes were dropped.

    A line that ends in a newline is kept even if it does not parse — that is corruption
    in the middle of a finished write, and silently deleting it would hide data loss.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    with open(path, "rb") as f:
        data = f.read()
    if data.endswith(b"\n"):
        return False
    last_nl = data.rfind(b"\n")
    keep = data[: last_nl + 1] if last_nl >= 0 else b""
    tmp = path + ".tail.tmp"
    with open(tmp, "wb") as f:
        f.write(keep)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return True


def compact_retryable(path: str) -> int:
    """Drop trials whose latest line is a retryable abort, so a resumed run does not
    leave two copies of the same trial_id for stats.py. Returns how many trials were dropped."""
    if not os.path.exists(path):
        return 0
    repair_jsonl_tail(path)
    latest: Dict[str, Tuple[str, bool]] = {}
    order: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                meta = json.loads(raw).get("metadata") or {}
            except json.JSONDecodeError:
                continue
            tid = meta.get("trial_id") or ""
            if tid not in latest:
                order.append(tid)
            latest[tid] = (raw, not is_retryable_abort(meta.get("aborted_reason")))
    dropped = sum(1 for tid in order if not latest[tid][1])
    if not dropped:
        return 0
    tmp = path + ".compact.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for tid in order:
            raw, complete = latest[tid]
            if complete and tid:
                f.write(raw + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return dropped


def existing_trial_ids(path: str) -> Tuple[Set[str], Set[str]]:
    ids, models = set(), set()
    if not os.path.exists(path):
        return ids, models
    repair_jsonl_tail(path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                meta = json.loads(line).get("metadata", {})
            except json.JSONDecodeError:
                continue
            if is_retryable_abort(meta.get("aborted_reason")):
                continue  # re-run quota walls, sandbox setup failures, transport exhaustion
            ids.add(meta.get("trial_id", ""))
            models.add(meta.get("agent_version", ""))
    return ids, models


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True, choices=list(REGISTRY) + ["mock"])
    ap.add_argument("--model", default=None, help="override the registry default model for this provider")
    ap.add_argument("--out", default=None, help="results JSONL (default results/<pilot_|>{provider}[_{model}].jsonl)")
    ap.add_argument("--pilot", action="store_true", help="18-trial pilot: 2 tasks x 3 conditions x 3 variants x seed 0")
    ap.add_argument("--subset", choices=["full", "priority"], default="full")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (grid replicates)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--tasks", nargs="+", default=None, choices=ALL_TASKS)
    ap.add_argument("--conditions", nargs="+", default=None, choices=ALL_CONDITIONS)
    ap.add_argument("--variants", nargs="+", default=None, choices=ALL_VARIANTS)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--expected-steps", type=int, default=6, help="anomaly schedule horizon (do not change mid-experiment)")
    ap.add_argument("--rpm", type=int, default=None)
    ap.add_argument("--tpm", type=int, default=None)
    ap.add_argument("--rpd", type=int, default=None)
    ap.add_argument("--min-interval", type=float, default=None, help="seconds between calls (Mistral ~1.1)")
    ap.add_argument("--transport-retries", type=int, default=8,
                    help="retries after the first attempt for HTTP 429/503/network (default 8; "
                         "daily/monthly quota 429s are not retried)")
    ap.add_argument("--gave-up-stop", type=int, default=3,
                    help="stop this provider after N trials in a row exhaust transport retries "
                         "(0 disables; those trials are re-queued on resume)")
    ap.add_argument("--monthly-cap", type=int, default=None, help="persistent per-month call cap (.quota/)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=400)
    ap.add_argument("--json-mode", action="store_true", help="ask the API for JSON-mode output (asymmetric across providers; off by default)")
    ap.add_argument("--max-trials", type=int, default=None, help="stop after this many NEW trials this run")
    ap.add_argument("--max-calls", type=int, default=None, help="stop once this many API calls were made this run")
    ap.add_argument("--sleep-between", type=float, default=0.0, help="extra pause between trials")
    ap.add_argument("--no-resume", action="store_true", help="do not skip trial_ids already in --out")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--mock-seed", type=int, default=0)
    ap.add_argument("--harness-log", default=None, help="JSONL of harness-level events (429/5xx/parse failures/quota); default results/harness_{provider}_{model}.jsonl")
    ap.add_argument("--sandbox-shell", default="auto",
                    help="auto|wsl|native|<path to bash.exe>. On Windows, auto is WSL "
                         "(wsl -l -v must show a default distro marked *). cmd.exe is refused. "
                         "Trials run as user episteme; sudo is disabled.")
    ap.add_argument("--no-preflight", action="store_true", help="skip the 2-second sandbox sanity checks (not recommended)")
    args = ap.parse_args()

    if args.list_models:
        for m in list_models(args.provider):
            print(m)
        return 0

    if args.pilot:
        tasks, conditions, variants, seeds = PILOT_TASKS, ALL_CONDITIONS, ALL_VARIANTS, [args.seed_start]
    elif args.subset == "priority":
        tasks, conditions, variants = PRIORITY["tasks"], PRIORITY["conditions"], PRIORITY["variants"]
        seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    else:
        tasks, conditions, variants = ALL_TASKS, ALL_CONDITIONS, ALL_VARIANTS
        seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    tasks = args.tasks or tasks
    conditions = args.conditions or conditions
    variants = args.variants or variants
    plan = build_plan(tasks, conditions, variants, seeds)

    model_tag = (args.model or (REGISTRY[args.provider]["cls"].default_model if args.provider in REGISTRY else "mock-model"))
    safe_model = model_tag.replace("/", "-").replace(":", "-")
    out = args.out or os.path.join("results", f"{'pilot_' if args.pilot else ''}{args.provider}_{safe_model}.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    if not args.no_resume and os.path.exists(out):
        dropped = compact_retryable(out)
        if dropped:
            print(f"[resume] dropped {dropped} retryable abort(s) from {out} so they will be re-run", file=sys.stderr)
    done_ids, done_models = (set(), set()) if args.no_resume else existing_trial_ids(out)
    if done_models and any(m and m != model_tag for m in done_models):
        print(f"[warn] {out} already contains other models: {sorted(done_models)} (use --out to separate them)", file=sys.stderr)
    todo = [p for p in plan if f"{p[0]}_{p[1]}_{p[2]}_{p[3]}" not in done_ids]
    if args.max_trials is not None:
        todo = todo[: args.max_trials]

    print(f"provider={args.provider} model={model_tag} out={out}", file=sys.stderr)
    print(f"grid: {len(tasks)} tasks x {len(conditions)} conditions x {len(variants)} variants x {len(seeds)} seeds = {len(plan)} trials; "
          f"{len(plan) - len(todo)} already done, {len(todo)} to run (~{len(todo) * CALLS_PER_TRIAL_EST} calls at {CALLS_PER_TRIAL_EST}/trial)", file=sys.stderr)
    if args.provider in REGISTRY:
        spec = effective_limits(args.provider, args.model)
        rpm = args.rpm or spec["rpm"]
        rpd = args.rpd or spec["rpd"]
        est_calls = len(todo) * CALLS_PER_TRIAL_EST
        print(f"limits: rpm={rpm} tpm={args.tpm or spec['tpm']} rpd={rpd} min_interval={args.min_interval or spec['min_interval_s']}s "
              f"monthly_cap={args.monthly_cap or spec.get('monthly_cap')} | {REGISTRY[args.provider]['notes']}", file=sys.stderr)
        print(f"budget: >= {est_calls / rpm:.0f} min at the RPM ceiling; >= {est_calls / rpd:.1f} days at the RPD cap", file=sys.stderr)
    if not args.no_preflight:
        shell, err = resolve_sandbox_shell(args.sandbox_shell)
        if err:
            print(f"[preflight] FAIL: {err}", file=sys.stderr)
            print("Fix: install WSL, set a default distro (`wsl -l -v` must show *), or run on Linux. Do not use cmd.exe.", file=sys.stderr)
            return 1
        if shell:
            print(f"[preflight] routing sandbox commands through: {shell[0]}", file=sys.stderr)
        tasks_mod.set_shell(shell)
        try:
            problems = preflight()
        except Exception as e:  # e.g. TaskEnv spawn machinery itself unusable
            problems = [f"sandbox unusable: {str(e)[:200]}"]
        if problems:
            print("[preflight] FAIL — refusing to start (nothing wasted):", file=sys.stderr)
            for pr in problems:
                print(f"  - {pr}", file=sys.stderr)
            print("Fix: `wsl -l -v` must show a default distro (marked *). The harness runs trials as the unprivileged user 'episteme' so sudo cannot install packages into the distro. "
                  "Only bypass with --no-preflight if you know what you are doing.", file=sys.stderr)
            return 1
        print("[preflight] sandbox OK (shell + tools + all 4 task setups)", file=sys.stderr)

    if args.dry_run:
        for i, (task, cond, var, seed) in enumerate(todo[:12]):
            print(f"  {i:3d} {task:15s} {cond:14s} {var:10s} seed={seed}")
        if len(todo) > 12:
            print(f"  ... {len(todo) - 12} more")
        return 0
    if not todo:
        print("nothing to do", file=sys.stderr)
        return 0

    overrides = dict(rpm=args.rpm, tpm=args.tpm, rpd=args.rpd, min_interval_s=args.min_interval, monthly_cap=args.monthly_cap,
                     temperature=args.temperature, max_output_tokens=args.max_output_tokens, json_mode=args.json_mode or None,
                     transport_retries=args.transport_retries, seed=args.mock_seed)
    try:
        provider = make_provider(args.provider, args.model, **overrides)
    except ProviderError as e:
        print(f"[fatal] {e}", file=sys.stderr)
        return 2
    harness_log = args.harness_log or os.path.join("results", f"harness_{args.provider}_{safe_model}.jsonl")
    sink = JsonlSink(harness_log, run_meta={"provider": args.provider, "model": model_tag, "out": out})
    if hasattr(provider, "event_sink"):
        provider.event_sink = sink
    if getattr(provider, "ledger", None):
        led = provider.ledger
        print(f"ledger (.quota/{args.provider}.json): {led.daily_used()} calls today (UTC) -> limiter starts at {provider.limiter.daily_calls}/{provider.limiter.rpd_limit}; "
              f"month {led.used()}" + (f"/{led.monthly_cap}" if led.monthly_cap else ""), file=sys.stderr)
    print(f"harness events -> {harness_log}", file=sys.stderr)

    t_start = time.time()
    n_done = n_aborted = 0
    stop_reason = None
    stop_code = 0
    consecutive_gave_up = 0
    with open(out, "a", encoding="utf-8") as f:
        try:
            for i, (task, cond, var, seed) in enumerate(todo):
                calls_before = getattr(provider, "total_calls", 0)
                traj = run_trial(provider, task, cond, var, seed, expected_steps=args.expected_steps, max_steps=args.max_steps)
                f.write(traj.to_json() + "\n")
                f.flush()
                n_done += 1
                fired = sum(1 for s in traj.steps if s.injected_anomaly)
                parse_failed = sum(1 for s in traj.steps if s.action == "PARSE_FAILED")
                rels = [s.anomaly_response_relevance for s in traj.steps if s.injected_anomaly]
                calls = getattr(provider, "total_calls", 0) - calls_before
                status = "ABORTED" if traj.metadata.aborted_reason else ("ok" if traj.metadata.task_success else "fail")
                print(f"[{n_done}/{len(todo)}] {traj.metadata.trial_id:45s} steps={len(traj.steps):2d} fired={fired}/{len(traj.metadata.scheduled_anomalies)} "
                      f"parse_fail={parse_failed} rel={rels} task={status} calls={calls} elapsed={time.time() - t_start:.0f}s", file=sys.stderr)
                if n_done == 1 and getattr(provider, "last_ratelimit_headers", None):
                    print(f"    rate-limit headers from the API: {provider.last_ratelimit_headers}", file=sys.stderr)
                if traj.metadata.aborted_reason:
                    n_aborted += 1
                    print(f"    aborted: {traj.metadata.aborted_reason[:200]}", file=sys.stderr)
                    if traj.metadata.aborted_reason.startswith("QuotaExceeded"):
                        stop_reason = "quota exceeded -- re-run later; this trial will be retried automatically"
                        stop_code = 3
                        break
                    consecutive_gave_up, breaker = note_trial_abort(
                        consecutive_gave_up, traj.metadata.aborted_reason, args.gave_up_stop)
                    if breaker:
                        stop_reason = breaker
                        stop_code = 4
                        break
                else:
                    consecutive_gave_up = 0
                if args.max_calls is not None and getattr(provider, "total_calls", 0) >= args.max_calls:
                    stop_reason = f"--max-calls {args.max_calls} reached"
                    stop_code = 3
                    break
                if args.sleep_between:
                    time.sleep(args.sleep_between)
        except KeyboardInterrupt:
            stop_reason = "interrupted (partial results are on disk; re-run to resume)"
            stop_code = 3

    summary = {"provider": args.provider, "model": model_tag, "out": out, "trials_written": n_done, "aborted": n_aborted,
               "remaining": len(todo) - n_done, "elapsed_s": round(time.time() - t_start), "stop_reason": stop_reason,
               "usage": provider.usage_summary() if hasattr(provider, "usage_summary") else None,
               "harness_events": summarize_events(sink.events), "harness_log": harness_log,
               "finished_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    sink.close(summary)
    print(json.dumps(summary, indent=2), file=sys.stderr)
    try:
        total_lines = sum(1 for line in open(out, encoding="utf-8") if line.strip())
        print(f"{out}: {total_lines} lines", file=sys.stderr)
    except OSError:
        pass
    return stop_code


if __name__ == "__main__":
    sys.exit(main())
