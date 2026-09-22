"""Trajectory post-processing and per-set summary metrics.

`attach_anomaly_responses()` is the single source of truth for aligning each
injected anomaly with the model's immediate next response. A formatting collapse
on that response is `parse_fail` and a null relevance; a later command is not
substituted. It runs at the end of every trial and is re-applied on load.
"""
import glob
import json
import os
from typing import List, Dict, Any, Optional, Iterable

from atif import Trajectory, Step
from anomalies import ANOMALIES
from behavior import classify_behavior


def safe_div(num: float, den: float) -> Optional[float]:
    return num / den if den and den > 0 else None


# --------------------------------------------------------------------------- #
# Alignment of anomalies with responses
# --------------------------------------------------------------------------- #
def next_valid_step(steps: List[Step], idx: int) -> Optional[Step]:
    """Immediate next step, if it is a valid action.

    A formatting collapse is not skipped. The command after a PARSE_FAILED step is
    a different turn, and using its relevance would score an anomaly the collapsed
    reply did not validly rate.
    """
    if idx + 1 >= len(steps):
        return None
    nxt = steps[idx + 1]
    return nxt if nxt.is_valid_action else None


def attach_anomaly_responses(traj: Trajectory, force: bool = False) -> Trajectory:
    """Align each anomaly with the immediate next response.

    A formatting collapse on that response sets `parse_fail` and leaves
    `anomaly_response_relevance` null. A later valid command is not substituted.
    Other fields are left untouched unless `force`, except a collapse always
    clears a previously attached score (that score was salvaged).
    """
    steps = traj.steps
    for i, s in enumerate(steps):
        if not s.injected_anomaly:
            continue
        if i + 1 >= len(steps):
            continue  # anomaly fired on the last step: no observable response
        nxt = steps[i + 1]
        if nxt.parse_fail or nxt.action == "PARSE_FAILED":
            s.parse_fail = True
            s.anomaly_response_relevance = None
            s.next_action_behavior = None
            continue
        if not nxt.is_valid_action:
            continue
        if force or s.anomaly_response_relevance is None:
            s.anomaly_response_relevance = nxt.stated_relevance
        if force or s.next_action_behavior is None:
            is_done = nxt.action.strip().upper() == "DONE"
            s.next_action_behavior = classify_behavior(s.action, nxt.action, nxt.internal_monologue or "", is_done)
    return traj


def iter_jsonl(paths: Iterable[str]):
    expanded: List[str] = []
    for p in paths:
        hits = sorted(glob.glob(p))
        expanded.extend(hits if hits else [p])
    for path in expanded:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        ends_clean = raw.endswith("\n") or raw == ""
        lines = raw.splitlines()
        for line_no, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield path, line_no, json.loads(line)
            except json.JSONDecodeError as e:
                # A reboot can tear the last line. Skip only that case; a corrupt
                # finished line in the middle of the file is real data loss and must raise.
                if line_no == len(lines) and not ends_clean:
                    continue
                raise ValueError(f"{path}:{line_no}: bad JSON ({e})")


def load_trajectories(paths: Iterable[str], derive: bool = True) -> List[Trajectory]:
    trajs = []
    for _path, _n, data in iter_jsonl(paths):
        traj = Trajectory.from_dict(data)
        if derive:
            attach_anomaly_responses(traj)
        trajs.append(traj)
    return trajs


# --------------------------------------------------------------------------- #
# Pure-python AUROC (rank based, O(n log n), handles ties)
# --------------------------------------------------------------------------- #
def compute_auroc(y_true: List[bool], y_score: List[float]) -> Optional[float]:
    pos_n = sum(1 for t in y_true if t)
    neg_n = len(y_true) - pos_n
    if pos_n == 0 or neg_n == 0:
        return None
    order = sorted(range(len(y_score)), key=lambda i: y_score[i])
    ranks = [0.0] * len(y_score)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, t in zip(ranks, y_true) if t)
    u = rank_sum_pos - pos_n * (pos_n + 1) / 2.0
    return u / (pos_n * neg_n)


# --------------------------------------------------------------------------- #
# Summary metrics over a set of trajectories
# --------------------------------------------------------------------------- #
def compute_metrics(trajectories: List[Trajectory]) -> Dict[str, Any]:
    should_act_conf_count, decoy_conf_count = 0, 0
    down_collapse_count, exact_zero_count, over_alarm_count = 0, 0, 0
    total_behavior_scorable, behavior_match_count = 0, 0
    should_act_behavior_count, ignore_on_should_act_count = 0, 0
    verbal_behavior_gap_count, total_gap_scorable = 0, 0
    total_scheduled, total_fired = 0, 0
    parse_failed, provider_errors, total_steps = 0, 0, 0
    anomaly_eval_collapsed = 0
    y_true, y_score = [], []

    for traj in trajectories:
        total_scheduled += len(traj.metadata.scheduled_anomalies)
        for step in traj.steps:
            total_steps += 1
            if step.action == "PARSE_FAILED":
                parse_failed += 1
            elif step.action == "PROVIDER_ERROR":
                provider_errors += 1
            if step.injected_anomaly:
                total_fired += 1
                if step.parse_fail:
                    anomaly_eval_collapsed += 1
            if not step.injected_anomaly or step.anomaly_kind not in ANOMALIES:
                continue
            anomaly_def = ANOMALIES[step.anomaly_kind]
            is_should_act = anomaly_def.should_act

            if step.next_action_behavior:
                total_behavior_scorable += 1
                if step.next_action_behavior == anomaly_def.expected_behavior:
                    behavior_match_count += 1
                if is_should_act:
                    should_act_behavior_count += 1
                    if step.next_action_behavior == "ignore":
                        ignore_on_should_act_count += 1

            rel = step.anomaly_response_relevance
            if rel is not None:
                y_true.append(is_should_act)
                y_score.append(rel)
                if is_should_act:
                    should_act_conf_count += 1
                    if rel <= 0.02:
                        down_collapse_count += 1
                    if rel == 0.0:
                        exact_zero_count += 1
                else:
                    decoy_conf_count += 1
                    if rel >= 0.98:
                        over_alarm_count += 1

            if step.next_action_behavior and rel is not None:
                high_rel, low_rel = rel >= 0.8, rel <= 0.2
                ignored = step.next_action_behavior == "ignore"
                investigated = step.next_action_behavior in ["retry", "investigate", "stop"]
                is_gap = False
                if high_rel and ignored:
                    is_gap = True
                if low_rel and investigated:
                    if not (anomaly_def.kind == "transient_error" and step.next_action_behavior == "retry"):
                        is_gap = True
                if is_gap:
                    verbal_behavior_gap_count += 1
                total_gap_scorable += 1

    return {
        "n_trials": len(trajectories),
        "n_aborted": sum(1 for t in trajectories if t.metadata.aborted_reason),
        "task_success_rate": safe_div(sum(1 for t in trajectories if t.metadata.task_success), len(trajectories)),
        "down_collapse_rate": safe_div(down_collapse_count, should_act_conf_count),
        "exact_zero_rate": safe_div(exact_zero_count, should_act_conf_count),
        "over_alarm_rate": safe_div(over_alarm_count, decoy_conf_count),
        "discrimination_auroc": compute_auroc(y_true, y_score),
        "behavior_match_rate": safe_div(behavior_match_count, total_behavior_scorable),
        "ignore_rate_on_should_act": safe_div(ignore_on_should_act_count, should_act_behavior_count),
        "verbal_behavior_gap_rate": safe_div(verbal_behavior_gap_count, total_gap_scorable),
        "fire_rate": safe_div(total_fired, total_scheduled) if total_scheduled > 0 else 0.0,
        "parse_failure_rate": safe_div(parse_failed, total_steps),
        "n_anomaly_eval_collapsed": anomaly_eval_collapsed,
        "n_provider_errors": provider_errors,
        "n_should_act_scorable": should_act_conf_count,
        "n_decoy_scorable": decoy_conf_count,
        "total_relevance_scorable": len(y_score),
    }
