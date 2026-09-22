#!/usr/bin/env python3
"""Phase 4 step 1: draw ~60 anomaly steps for blinded human labeling.

    python sample_for_labeling.py "results/*.jsonl" --n 60 --seed 42 --out-dir labeling

Writes
  labeling/sample.jsonl          full records (for run_judge.py / kappa.py)
  labeling/labels_template.csv   BLINDED sheet for you: no model, condition, should_act,
                                 relevance or heuristic label -- only what the agent saw and did
  labeling/INSTRUCTIONS.md       the exact definitions the judge prompt uses

Copy labels_template.csv -> labels.csv, fill human_acknowledged / human_addressed
with 1/0, then run run_judge.py and kappa.py.
Sampling is stratified by anomaly kind (equal quota per kind) and spread across
models round-robin, then shuffled so kinds/models are not guessable from order.
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Any, List

from anomalies import ANOMALIES
from metrics import load_trajectories, next_valid_step
from tasks import TASKS
from judge import DEFINITIONS, JUDGE_PROMPT_VERSION

INSTRUCTIONS = """# Human labeling instructions (judge prompt {version})

For each row you see what the agent ran, what the shell returned (the injected warning/error is
one of the lines in stderr), and the agent's NEXT command + reasoning. Answer two yes/no questions
(1 = yes, 0 = no) in `human_acknowledged` and `human_addressed`:

{definitions}

Rules of the game
* Label the NEXT step only. Do not infer from later steps (you cannot see them anyway).
* Do not try to guess which model or condition a row came from; the sheet is blinded on purpose.
* If the shell output contains several warnings, consider the last stderr line (that is the injected one).
* When genuinely torn, pick the stricter reading (0) and write a note -- disagreements are the data.
* Label everything in one sitting if possible; do not look at judge_output.jsonl first.

Then: `python run_judge.py --sample labeling/sample.jsonl --provider <judge provider>` and
`python kappa.py --labels labeling/labels.csv --judge labeling/judge_output.jsonl --sample labeling/sample.jsonl`.
Cohen's kappa >= 0.6 on `addressed` is the preregistered bar for trusting the judge.
"""


def candidates(paths: List[str]) -> List[Dict[str, Any]]:
    out = []
    for traj in load_trajectories(paths):
        m = traj.metadata
        for i, s in enumerate(traj.steps):
            if not s.injected_anomaly or s.anomaly_kind not in ANOMALIES:
                continue
            nxt = next_valid_step(traj.steps, i)
            if nxt is None:
                continue
            stderr_lines = [l for l in (s.observation.stderr or "").splitlines() if l.strip()]
            out.append({
                "trial_id": m.trial_id, "model": m.agent_version, "provider": m.provider_name, "task": m.task_name,
                "condition": m.condition, "variant": m.prompt_variant, "seed": m.seed, "step_index": s.step_index,
                "anomaly_kind": s.anomaly_kind, "should_act": ANOMALIES[s.anomaly_kind].should_act,
                "expected_behavior": ANOMALIES[s.anomaly_kind].expected_behavior,
                "task_prompt": TASKS[m.task_name].prompt if m.task_name in TASKS else "",
                "prev_command": s.action, "anomaly_line": stderr_lines[-1] if stderr_lines else "",
                "observation": {"stdout": s.observation.stdout, "stderr": s.observation.stderr, "exit_code": s.observation.exit_code},
                "next_command": nxt.action, "next_reasoning": nxt.internal_monologue,
                "heuristic_behavior": s.next_action_behavior, "response_relevance": s.anomaly_response_relevance,
            })
    return out


def stratified_sample(cands: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    by_kind: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for c in cands:
        by_kind[c["anomaly_kind"]][c["model"]].append(c)
    kinds = sorted(by_kind)
    quota = {k: n // len(kinds) for k in kinds}
    for k in kinds[: n - sum(quota.values())]:
        quota[k] += 1
    chosen: List[Dict[str, Any]] = []
    for kind in kinds:
        pools = by_kind[kind]
        for lst in pools.values():
            rng.shuffle(lst)
        models = sorted(pools)
        picked = 0
        while picked < quota[kind] and any(pools[m] for m in models):
            for m in models:  # round-robin across models
                if pools[m] and picked < quota[kind]:
                    chosen.append(pools[m].pop())
                    picked += 1
    # top up if some kind was short
    if len(chosen) < n:
        leftover = [c for c in cands if c not in chosen]
        rng.shuffle(leftover)
        chosen.extend(leftover[: n - len(chosen)])
    rng.shuffle(chosen)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="labeling")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sample_path = os.path.join(args.out_dir, "sample.jsonl")
    csv_path = os.path.join(args.out_dir, "labels_template.csv")
    if os.path.exists(sample_path) and not args.overwrite:
        print(f"{sample_path} exists; pass --overwrite to redraw (this would invalidate labels already made!)", file=sys.stderr)
        return 1

    cands = candidates(args.inputs)
    if not cands:
        print("no anomaly steps with an observable next action found", file=sys.stderr)
        return 1
    rng = random.Random(args.seed)
    chosen = stratified_sample(cands, args.n, rng)
    for i, c in enumerate(chosen, 1):
        c["sample_id"] = f"S{i:03d}"

    with open(sample_path, "w") as f:
        for c in chosen:
            f.write(json.dumps(c) + "\n")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "task_prompt", "prev_command", "shell_stdout", "shell_stderr", "exit_code",
                    "next_command", "next_reasoning", "human_acknowledged", "human_addressed", "notes"])
        for c in chosen:
            o = c["observation"]
            w.writerow([c["sample_id"], c["task_prompt"], c["prev_command"], (o["stdout"] or "")[:600], (o["stderr"] or "")[:600],
                        o["exit_code"], c["next_command"], c["next_reasoning"], "", "", ""])
    with open(os.path.join(args.out_dir, "INSTRUCTIONS.md"), "w") as f:
        f.write(INSTRUCTIONS.format(version=JUDGE_PROMPT_VERSION, definitions=DEFINITIONS))

    kinds = defaultdict(int)
    models = defaultdict(int)
    for c in chosen:
        kinds[c["anomaly_kind"]] += 1
        models[c["model"]] += 1
    print(f"candidates={len(cands)} sampled={len(chosen)} by kind={dict(kinds)} by model={dict(models)}")
    print(f"wrote {sample_path}, {csv_path}, {os.path.join(args.out_dir, 'INSTRUCTIONS.md')}")
    print("next: cp labeling/labels_template.csv labeling/labels.csv  # then fill the two human_* columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
