#!/usr/bin/env python3
"""Phase 3 sanity check: read trajectories by eye and flag degenerate models.

    python inspect_trajectories.py results/pilot_groq.jsonl            # degeneracy report + 3 trajectories
    python inspect_trajectories.py results/pilot_groq.jsonl --n 18     # print all of them
    python inspect_trajectories.py results/pilot_groq.jsonl --anomalies-only --trial-id python_test_control_original_0

The report answers: is `relevance` a real signal or a constant? Is the reasoning
the same sentence every time? Did the model DONE out immediately, loop on one
command, or fail to produce JSON? Does relevance move when an anomaly is shown?
"""
import argparse
import sys
from collections import Counter, defaultdict

from metrics import load_trajectories

FLAGS = []


def degeneracy_report(trajs) -> None:
    by_model = defaultdict(list)
    for t in trajs:
        by_model[t.metadata.agent_version].append(t)
    for model, ts in sorted(by_model.items()):
        steps = [s for t in ts for s in t.steps]
        valid = [s for s in steps if s.is_valid_action]
        n_steps = len(steps)
        parse_failed = sum(1 for s in steps if s.action == "PARSE_FAILED")
        prov_err = sum(1 for s in steps if s.action == "PROVIDER_ERROR")
        rels = [s.stated_relevance for s in valid if s.stated_relevance is not None]
        rel_vals = Counter(round(r, 3) for r in rels)
        reasons = Counter((s.internal_monologue or "").strip()[:80] for s in valid)
        cmds = Counter(s.action.strip()[:40] for s in valid)
        done_first = sum(1 for t in ts if t.steps and t.steps[0].action.strip().upper() == "DONE")
        max_steps_hit = sum(1 for t in ts if len(t.steps) >= 10 and t.steps[-1].action.strip().upper() != "DONE")
        anom_resp = [s.anomaly_response_relevance for t in ts for s in t.steps if s.injected_anomaly and s.anomaly_response_relevance is not None]
        clean_rel = [s.stated_relevance for t in ts for i, s in enumerate(t.steps)
                     if s.stated_relevance is not None and i > 0 and not t.steps[i - 1].injected_anomaly and t.steps[i - 1].is_valid_action]
        succ = sum(1 for t in ts if t.metadata.task_success)
        fired = sum(1 for t in ts for s in t.steps if s.injected_anomaly)
        sched = sum(len(t.metadata.scheduled_anomalies) for t in ts)
        print(f"\n### {model}: {len(ts)} trials, {n_steps} steps ({n_steps / len(ts):.1f}/trial), task success {succ}/{len(ts)}, "
              f"anomalies fired {fired}/{sched}, aborted {sum(1 for t in ts if t.metadata.aborted_reason)}")
        print(f"  parse failures: {parse_failed}/{n_steps} ({parse_failed / max(1, n_steps):.1%}); provider errors: {prov_err}; "
              f"DONE as first action: {done_first}; hit max steps without DONE: {max_steps_hit}")
        flags = []
        if n_steps and parse_failed / n_steps > 0.2:
            flags.append("HIGH PARSE FAILURE RATE")
        if rels:
            top_rel, top_rel_n = rel_vals.most_common(1)[0]
            print(f"  relevance: {len(rels)} values, {len(rel_vals)} distinct; most common {top_rel} x{top_rel_n} ({top_rel_n / len(rels):.0%}); "
                  f"mean on clean obs {sum(clean_rel) / len(clean_rel):.2f} (n={len(clean_rel)})" if clean_rel else
                  f"  relevance: {len(rels)} values, {len(rel_vals)} distinct; most common {top_rel} x{top_rel_n}")
            if anom_resp:
                print(f"  relevance in response to anomalies: mean {sum(anom_resp) / len(anom_resp):.2f}, exact 0.0 in {sum(1 for r in anom_resp if r == 0.0)}/{len(anom_resp)} (n={len(anom_resp)})")
            if len(rel_vals) <= 2 and len(rels) >= 8:
                flags.append("RELEVANCE NEARLY CONSTANT")
            if clean_rel and anom_resp and abs(sum(anom_resp) / len(anom_resp) - sum(clean_rel) / len(clean_rel)) < 0.02 and len(anom_resp) >= 8:
                flags.append("RELEVANCE DOES NOT MOVE ON ANOMALIES")
        else:
            print("  relevance: none (all no_numeric or all parse-failed?)")
        if reasons:
            top_r, top_r_n = reasons.most_common(1)[0]
            print(f"  reasoning: {len(reasons)} distinct of {len(valid)}; most common ({top_r_n}x, {top_r_n / max(1, len(valid)):.0%}): {top_r!r}")
            if len(valid) >= 8 and top_r_n / len(valid) > 0.5:
                flags.append("REASONING IS THE SAME SENTENCE >50% OF THE TIME")
        top_c, top_c_n = cmds.most_common(1)[0] if cmds else ("", 0)
        print(f"  commands: {len(cmds)} distinct; top: {cmds.most_common(4)}")
        if len(valid) >= 8 and top_c_n / len(valid) > 0.6:
            flags.append("ONE COMMAND >60% OF STEPS (looping?)")
        if done_first > len(ts) / 3:
            flags.append("DONE-ON-FIRST-STEP IN >1/3 OF TRIALS")
        print("  FLAGS: " + (", ".join(flags) if flags else "none -- looks non-degenerate"))


def print_trajectory(t, anomalies_only: bool) -> None:
    m = t.metadata
    print(f"\n=== {m.trial_id} | model={m.agent_version} | success={m.task_success} | schedule={m.scheduled_anomalies} | aborted={m.aborted_reason}")
    for i, s in enumerate(t.steps):
        if anomalies_only and not (s.injected_anomaly or (i > 0 and t.steps[i - 1].injected_anomaly)):
            continue
        tag = f"<< {s.anomaly_kind} injected" if s.injected_anomaly else ""
        print(f"  [{s.step_index}] rel={s.stated_relevance!s:5} cmd={s.action[:90]!r}")
        if s.internal_monologue:
            print(f"       why: {s.internal_monologue[:220]!r}")
        o = s.observation
        if o.stdout.strip():
            print(f"       out: {o.stdout.strip()[:160]!r}")
        if o.stderr.strip():
            print(f"       err: {o.stderr.strip()[:200]!r}  exit={o.exit_code} {tag}")
        elif tag:
            print(f"       {tag}")
        if s.injected_anomaly:
            print(f"       -> response: behavior={s.next_action_behavior} relevance={s.anomaly_response_relevance}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--n", type=int, default=3, help="how many trajectories to print")
    ap.add_argument("--trial-id", default=None)
    ap.add_argument("--anomalies-only", action="store_true", help="only print anomaly steps and the response to them")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    trajs = load_trajectories(args.inputs)
    print(f"{len(trajs)} trajectories")
    if not args.no_report:
        degeneracy_report(trajs)
    shown = [t for t in trajs if t.metadata.trial_id == args.trial_id] if args.trial_id else trajs[: args.n]
    for t in shown:
        print_trajectory(t, args.anomalies_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
