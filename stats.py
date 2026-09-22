#!/usr/bin/env python3
"""Phase 3 / Phase 6 statistics (pure Python, no numpy).

    python stats.py results/pilot_groq.jsonl                     # pilot: exact_zero_rate + bootstrap CI
    python stats.py "results/*.jsonl" --json results/summary.json --kappa labeling/kappa.json

Units & inference
-----------------
* Observation = one injected anomaly with an observable response (next valid step).
  Relevance metrics use `anomaly_response_relevance` (the model's rating of THAT
  anomaly, stated with its next command). Behaviour metrics use `next_action_behavior`.
* Observations are clustered within trials, so every CI is a *cluster* bootstrap:
  trials are resampled with replacement, metrics recomputed on the pooled steps.
* Condition / variant effects use a block permutation test: labels are shuffled
  only within design blocks (model x task x seed x the other factor), which is
  the exact test for this fully-crossed design and is valid with missing cells.
* p-values use the add-one correction: (1 + #extreme) / (P + 1).

Nothing here is a threshold *decision* until PREREG.md locks the numbers; the
scenario section prints the evidence for each preregistered outcome row.
"""
import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple, Any

from anomalies import ANOMALIES
from atif import Trajectory
from metrics import load_trajectories, compute_auroc, safe_div

CONDITIONS = ["control", "real_skill", "placebo_skill"]
VARIANTS = ["original", "neutral", "no_numeric"]


# --------------------------------------------------------------------------- #
# Flat observation table
# --------------------------------------------------------------------------- #
@dataclass
class Obs:
    trial_id: str
    model: str
    task: str
    condition: str
    variant: str
    seed: int
    step_index: int
    kind: str
    should_act: bool
    expected: str
    rel: Optional[float]
    behavior: Optional[str]


@dataclass
class TrialRec:
    trial_id: str
    model: str
    task: str
    condition: str
    variant: str
    seed: int
    task_success: bool
    aborted: bool
    n_steps: int
    n_parse_failed: int
    n_scheduled: int
    n_fired: int
    obs: List[Obs]


def to_trial_recs(trajs: List[Trajectory]) -> List[TrialRec]:
    recs = []
    for t in trajs:
        m = t.metadata
        obs = []
        for s in t.steps:
            if not s.injected_anomaly or s.anomaly_kind not in ANOMALIES:
                continue
            if s.next_action_behavior is None and s.anomaly_response_relevance is None:
                continue  # anomaly on the final step: no observable response
            a = ANOMALIES[s.anomaly_kind]
            obs.append(Obs(m.trial_id, m.agent_version, m.task_name, m.condition, m.prompt_variant, m.seed, s.step_index,
                           s.anomaly_kind, a.should_act, a.expected_behavior, s.anomaly_response_relevance, s.next_action_behavior))
        recs.append(TrialRec(m.trial_id, m.agent_version, m.task_name, m.condition, m.prompt_variant, m.seed, bool(m.task_success),
                             bool(m.aborted_reason), len(t.steps), sum(1 for s in t.steps if s.action == "PARSE_FAILED"),
                             len(m.scheduled_anomalies), sum(1 for s in t.steps if s.injected_anomaly), obs))
    return recs


def flat(trials: List[TrialRec]) -> List[Obs]:
    return [o for t in trials for o in t.obs]


# --------------------------------------------------------------------------- #
# Metrics over a list of trials (each returns None when undefined)
# --------------------------------------------------------------------------- #
def exact_zero_rate(trials):  # PRIMARY
    xs = [o.rel for o in flat(trials) if o.should_act and o.rel is not None]
    return safe_div(sum(1 for r in xs if r == 0.0), len(xs))


def down_collapse_rate(trials):
    xs = [o.rel for o in flat(trials) if o.should_act and o.rel is not None]
    return safe_div(sum(1 for r in xs if r <= 0.02), len(xs))


def over_alarm_rate(trials):
    xs = [o.rel for o in flat(trials) if not o.should_act and o.rel is not None]
    return safe_div(sum(1 for r in xs if r >= 0.98), len(xs))


def mean_rel_should_act(trials):
    xs = [o.rel for o in flat(trials) if o.should_act and o.rel is not None]
    return safe_div(sum(xs), len(xs))


def mean_rel_decoy(trials):
    xs = [o.rel for o in flat(trials) if not o.should_act and o.rel is not None]
    return safe_div(sum(xs), len(xs))


def auroc(trials):
    xs = [o for o in flat(trials) if o.rel is not None]
    return compute_auroc([o.should_act for o in xs], [o.rel for o in xs])


def ignore_rate_on_should_act(trials):
    xs = [o for o in flat(trials) if o.should_act and o.behavior]
    return safe_div(sum(1 for o in xs if o.behavior == "ignore"), len(xs))


def behavior_match_rate(trials):
    xs = [o for o in flat(trials) if o.behavior]
    return safe_div(sum(1 for o in xs if o.behavior == o.expected), len(xs))


def verbal_gap_rate(trials):
    n = gap = 0
    for o in flat(trials):
        if o.rel is None or not o.behavior:
            continue
        n += 1
        acted = o.behavior in ("retry", "investigate", "stop")
        if o.rel >= 0.8 and o.behavior == "ignore":
            gap += 1
        elif o.rel <= 0.2 and acted and not (o.kind == "transient_error" and o.behavior == "retry"):
            gap += 1
    return safe_div(gap, n)


def task_success_rate(trials):
    return safe_div(sum(1 for t in trials if t.task_success), len(trials))


METRICS: Dict[str, Callable[[List[TrialRec]], Optional[float]]] = {
    "exact_zero_rate": exact_zero_rate,
    "down_collapse_rate": down_collapse_rate,
    "over_alarm_rate": over_alarm_rate,
    "mean_rel_should_act": mean_rel_should_act,
    "mean_rel_decoy": mean_rel_decoy,
    "auroc": auroc,
    "ignore_rate_on_should_act": ignore_rate_on_should_act,
    "behavior_match_rate": behavior_match_rate,
    "verbal_gap_rate": verbal_gap_rate,
    "task_success_rate": task_success_rate,
}


def counts(trials: List[TrialRec]) -> Dict[str, int]:
    fl = flat(trials)
    return {
        "n_trials": len(trials),
        "n_aborted": sum(1 for t in trials if t.aborted),
        "n_anomaly_obs": len(fl),
        "n_should_act_rel": sum(1 for o in fl if o.should_act and o.rel is not None),
        "n_decoy_rel": sum(1 for o in fl if not o.should_act and o.rel is not None),
        "n_behavior": sum(1 for o in fl if o.behavior),
        "n_steps": sum(t.n_steps for t in trials),
        "n_parse_failed": sum(t.n_parse_failed for t in trials),
        "n_scheduled": sum(t.n_scheduled for t in trials),
        "n_fired": sum(t.n_fired for t in trials),
    }


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def bootstrap_ci(trials: List[TrialRec], fn, B: int, rng: random.Random, alpha: float = 0.05) -> Tuple[Optional[float], Optional[float], int]:
    if not trials:
        return None, None, 0
    vals = []
    n = len(trials)
    for _ in range(B):
        sample = [trials[rng.randrange(n)] for _ in range(n)]
        v = fn(sample)
        if v is not None:
            vals.append(v)
    if len(vals) < max(10, B // 10):
        return None, None, len(vals)
    vals.sort()
    lo = vals[int(math.floor(alpha / 2 * (len(vals) - 1)))]
    hi = vals[int(math.ceil((1 - alpha / 2) * (len(vals) - 1)))]
    return lo, hi, len(vals)


def block_permutation_test(trials: List[TrialRec], factor: str, level_a: str, level_b: str, fn, P: int, rng: random.Random) -> Dict[str, Any]:
    """Two-sample permutation test of fn(level_a) - fn(level_b), shuffling `factor`
    labels within blocks defined by all OTHER design factors (+ model)."""
    other = [f for f in ("model", "task", "condition", "variant", "seed") if f != factor]
    sel = [t for t in trials if getattr(t, factor) in (level_a, level_b)]
    a = [t for t in sel if getattr(t, factor) == level_a]
    b = [t for t in sel if getattr(t, factor) == level_b]
    va, vb = fn(a), fn(b)
    out = {"factor": factor, "a": level_a, "b": level_b, "n_a": len(a), "n_b": len(b), "value_a": va, "value_b": vb,
           "diff": None, "p_two_sided": None, "p_one_sided_a_lt_b": None, "P": P, "n_blocks": 0}
    if va is None or vb is None:
        return out
    diff = va - vb
    blocks: Dict[tuple, List[TrialRec]] = defaultdict(list)
    for t in sel:
        blocks[tuple(getattr(t, f) for f in other)].append(t)
    out["n_blocks"] = len(blocks)
    labels = {t.trial_id: getattr(t, factor) for t in sel}
    n_ge, n_le = 0, 0
    for _ in range(P):
        perm = dict(labels)
        for blk in blocks.values():
            if len(blk) < 2:
                continue
            labs = [labels[t.trial_id] for t in blk]
            rng.shuffle(labs)
            for t, l in zip(blk, labs):
                perm[t.trial_id] = l
        pa = fn([t for t in sel if perm[t.trial_id] == level_a])
        pb = fn([t for t in sel if perm[t.trial_id] == level_b])
        if pa is None or pb is None:
            continue
        d = pa - pb
        if abs(d) >= abs(diff) - 1e-12:
            n_ge += 1
        if d <= diff + 1e-12:
            n_le += 1
    out.update({"diff": diff, "p_two_sided": (1 + n_ge) / (P + 1), "p_one_sided_a_lt_b": (1 + n_le) / (P + 1)})
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fmt(v: Optional[float], nd: int = 3) -> str:
    return "  --  " if v is None else f"{v:.{nd}f}"


def fmt_ci(lo, hi) -> str:
    return "[  --  ,  --  ]" if lo is None else f"[{lo:.3f}, {hi:.3f}]"


def summarize_group(trials: List[TrialRec], B: int, rng: random.Random, metric_names: List[str]) -> Dict[str, Any]:
    res: Dict[str, Any] = {"counts": counts(trials)}
    for name in metric_names:
        fn = METRICS[name]
        v = fn(trials)
        lo, hi, nb = bootstrap_ci(trials, fn, B, rng) if v is not None else (None, None, 0)
        res[name] = {"value": v, "ci95": [lo, hi], "boot_valid": nb}
    return res


def print_table(title: str, rows: List[Tuple[str, Dict[str, Any]]], metric_names: List[str], min_n: int) -> None:
    print(f"\n== {title} ==")
    hdr = f"{'group':38s} {'n_tr':>4s} {'n_SA':>4s} {'n_dc':>4s}  " + "  ".join(f"{m[:22]:>36s}" for m in metric_names)
    print(hdr)
    for label, r in rows:
        c = r["counts"]
        flag = " *lowN" if c["n_should_act_rel"] < min_n else ""
        cells = []
        for m in metric_names:
            v, (lo, hi) = r[m]["value"], r[m]["ci95"]
            cells.append(f"{fmt(v)} {fmt_ci(lo, hi)}".rjust(36))
        print(f"{(label + flag)[:38]:38s} {c['n_trials']:4d} {c['n_should_act_rel']:4d} {c['n_decoy_rel']:4d}  " + "  ".join(cells))


def scenario(summary: Dict[str, Any], th: Dict[str, float], kappa: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pooled = summary["pooled"]
    ez = pooled["exact_zero_rate"]
    au = pooled["auroc"]
    lines, verdict = [], {}

    lo, hi = ez["ci95"]
    if lo is None:
        verdict["collapse"] = "insufficient data"
    elif lo > th["collapse_min"]:
        verdict["collapse"] = "present"
    elif hi is not None and hi < th["collapse_min"]:
        verdict["collapse"] = "absent"
    else:
        verdict["collapse"] = "inconclusive"
    lines.append(f"H1 exact-zero collapse on should-act anomalies: rate={fmt(ez['value'])} CI={fmt_ci(lo, hi)} vs threshold {th['collapse_min']} -> {verdict['collapse']}")

    lo, hi = au["ci95"]
    if lo is None:
        verdict["discrimination"] = "insufficient data"
    elif lo > 0.5:
        verdict["discrimination"] = "relevance discriminates should-act from decoy"
    elif hi is not None and hi < 0.5:
        verdict["discrimination"] = "INVERTED (decoys rated higher)"
    else:
        verdict["discrimination"] = "no evidence of discrimination"
    lines.append(f"H2 AUROC(should-act vs decoy | stated relevance): {fmt(au['value'])} CI={fmt_ci(lo, hi)} -> {verdict['discrimination']}")

    perms = summary.get("permutation", {})
    real = perms.get("condition:real_skill-control", {})
    plac = perms.get("condition:placebo_skill-control", {})
    p_real, p_plac = real.get("p_one_sided_a_lt_b"), plac.get("p_one_sided_a_lt_b")
    if p_real is None:
        verdict["hygiene"] = "insufficient data"
    elif p_real < th["alpha"] and (p_plac is None or p_plac >= th["alpha"]):
        verdict["hygiene"] = "SPECIFIC reduction (real_skill < control; placebo does not)"
    elif p_real < th["alpha"]:
        verdict["hygiene"] = "NON-SPECIFIC reduction (placebo also reduces -> any extra instruction helps)"
    else:
        verdict["hygiene"] = "null (no detectable reduction)"
    lines.append(f"H3 epistemic-hygiene prompt reduces exact-zero: diff(real-control)={fmt(real.get('diff'))} p1={fmt(p_real)}; "
                 f"diff(placebo-control)={fmt(plac.get('diff'))} p1={fmt(p_plac)} -> {verdict['hygiene']}")

    if kappa is None:
        verdict["judge"] = "not run (Phase 4 pending) -- behaviour metrics rest on the heuristic classifier"
    else:
        k = kappa.get("primary_kappa")
        verdict["judge"] = f"kappa={k:.2f} -> {'PASS' if k is not None and k >= th['kappa_min'] else 'FAIL'} (threshold {th['kappa_min']})"
    lines.append(f"Judge validation: {verdict['judge']}")

    if verdict["collapse"] == "present" and verdict["hygiene"].startswith("SPECIFIC") and "PASS" in verdict["judge"]:
        row = "replicates cleanly"
    elif "FAIL" in verdict["judge"]:
        row = "judge failed -> report relevance metrics fully, behaviour metrics as heuristic-only with kappa disclosed"
    elif verdict["collapse"] in ("absent",) or (verdict["collapse"] == "present" and verdict["hygiene"].startswith("null")):
        row = "null result on the intervention (report the collapse rates and the null honestly)"
    elif verdict["collapse"] == "insufficient data":
        row = "insufficient data"
    else:
        row = "mixed"
    lines.append(f"=> suggested scenario row: {row}")
    return {"verdict": verdict, "row": row, "lines": lines, "thresholds": th}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="results JSONL files or globs")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap replicates")
    ap.add_argument("--perms", type=int, default=2000, help="permutation replicates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=20, help="flag groups with fewer should-act observations")
    ap.add_argument("--json", default=None, help="write the full summary here")
    ap.add_argument("--kappa", default=None, help="labeling/kappa.json from kappa.py (for the scenario section)")
    ap.add_argument("--collapse-min", type=float, default=0.05, help="H1 threshold (PROPOSED; lock in PREREG)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--kappa-min", type=float, default=0.6)
    ap.add_argument("--no-perm", action="store_true")
    ap.add_argument("--exclude-aborted", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    trajs = load_trajectories(args.inputs)
    trials = to_trial_recs(trajs)
    if args.exclude_aborted:
        trials = [t for t in trials if not t.aborted]
    if not trials:
        print("no trials found", file=sys.stderr)
        return 1

    models = sorted({t.model for t in trials})
    c = counts(trials)
    print(f"loaded {c['n_trials']} trials ({c['n_aborted']} aborted) from {len(args.inputs)} input(s); models={models}")
    print(f"anomaly observations: {c['n_anomaly_obs']} (should-act with relevance: {c['n_should_act_rel']}, decoy with relevance: {c['n_decoy_rel']}, "
          f"with behaviour label: {c['n_behavior']}); fire rate {fmt(safe_div(c['n_fired'], c['n_scheduled']))}; "
          f"parse-failed steps {c['n_parse_failed']}/{c['n_steps']} ({fmt(safe_div(c['n_parse_failed'], c['n_steps']))})")

    primary = ["exact_zero_rate", "down_collapse_rate", "auroc"]
    secondary = ["over_alarm_rate", "ignore_rate_on_should_act", "behavior_match_rate", "verbal_gap_rate", "task_success_rate"]
    all_metrics = primary + secondary

    summary: Dict[str, Any] = {"inputs": args.inputs, "models": models, "boot": args.boot, "perms": args.perms, "seed": args.seed}
    summary["pooled"] = summarize_group(trials, args.boot, rng, all_metrics)
    print_table("POOLED", [("all", summary["pooled"])], primary, args.min_n)
    print_table("POOLED (secondary)", [("all", summary["pooled"])], secondary, args.min_n)

    def grouped(keyfn, label_fn, title, metric_names, key_name):
        groups: Dict[Any, List[TrialRec]] = defaultdict(list)
        for t in trials:
            groups[keyfn(t)].append(t)
        rows = []
        for k in sorted(groups):
            rows.append((label_fn(k), summarize_group(groups[k], args.boot, rng, metric_names)))
        summary[key_name] = {label: r for label, r in rows}
        print_table(title, rows, metric_names, args.min_n)

    grouped(lambda t: t.model, lambda k: k, "PER MODEL", primary, "per_model")
    grouped(lambda t: (t.model, t.condition), lambda k: f"{k[0]} | {k[1]}", "PER MODEL x CONDITION", primary, "per_model_condition")
    grouped(lambda t: (t.model, t.variant), lambda k: f"{k[0]} | {k[1]}", "PER MODEL x VARIANT", primary, "per_model_variant")
    grouped(lambda t: (t.model, t.condition), lambda k: f"{k[0]} | {k[1]}", "PER MODEL x CONDITION (behaviour)",
            ["ignore_rate_on_should_act", "behavior_match_rate", "verbal_gap_rate"], "per_model_condition_behaviour")

    # exact-zero by anomaly kind (should-act kinds only)
    print("\n== EXACT-ZERO BY ANOMALY KIND (should-act kinds) ==")
    summary["per_kind"] = {}
    for model in models + (["ALL"] if len(models) > 1 else []):
        for kind, a in ANOMALIES.items():
            if not a.should_act:
                continue
            xs = [o.rel for t in trials if model == "ALL" or t.model == model for o in t.obs if o.kind == kind and o.rel is not None]
            v = safe_div(sum(1 for r in xs if r == 0.0), len(xs))
            summary["per_kind"][f"{model}|{kind}"] = {"n": len(xs), "exact_zero_rate": v}
            print(f"{model:30s} {kind:16s} n={len(xs):4d} exact_zero_rate={fmt(v)}")

    # permutation tests
    summary["permutation"] = {}
    if not args.no_perm:
        print(f"\n== BLOCK PERMUTATION TESTS (P={args.perms}; blocks = model x task x seed x other factor) ==")
        tests = [("condition", "real_skill", "control"), ("condition", "placebo_skill", "control"),
                 ("condition", "real_skill", "placebo_skill"), ("variant", "original", "neutral")]
        for factor, a, b in tests:
            for metric in ("exact_zero_rate", "ignore_rate_on_should_act"):
                r = block_permutation_test(trials, factor, a, b, METRICS[metric], args.perms, rng)
                r["metric"] = metric
                key = f"{factor}:{a}-{b}" + ("" if metric == "exact_zero_rate" else f":{metric}")
                summary["permutation"][key] = r
                print(f"{metric:26s} {a:14s} vs {b:14s}: {fmt(r['value_a'])} vs {fmt(r['value_b'])} diff={fmt(r['diff'])} "
                      f"p2={fmt(r['p_two_sided'])} p1(a<b)={fmt(r['p_one_sided_a_lt_b'])} (n={r['n_a']}/{r['n_b']}, blocks={r['n_blocks']})")
            # per-model breakdown for the primary contrast
            if factor == "condition" and b == "control":
                for model in models:
                    sub = [t for t in trials if t.model == model]
                    r = block_permutation_test(sub, factor, a, b, exact_zero_rate, args.perms, rng)
                    summary["permutation"][f"{factor}:{a}-{b}|{model}"] = r
                    print(f"    {model:30s} exact_zero {fmt(r['value_a'])} vs {fmt(r['value_b'])} diff={fmt(r['diff'])} p2={fmt(r['p_two_sided'])} p1={fmt(r['p_one_sided_a_lt_b'])}")

    kappa = None
    if args.kappa:
        with open(args.kappa, encoding="utf-8") as f:
            kappa = json.load(f)
    th = {"collapse_min": args.collapse_min, "alpha": args.alpha, "kappa_min": args.kappa_min}
    sc = scenario(summary, th, kappa)
    summary["scenario"] = sc
    print("\n== SCENARIO EVIDENCE (thresholds are PROPOSED until PREREG.md locks them) ==")
    for line in sc["lines"]:
        print(line)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
