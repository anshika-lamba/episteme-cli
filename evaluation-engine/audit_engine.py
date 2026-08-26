from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from atif import Trajectory


# --- Epistemic Rigidity -----------------------------------------------------

@dataclass
class RigidityResult:
    trial_id: str
    anomaly_steps: int
    collapsed_steps: int          # raw_weight_estimated exactly 0.0 or 1.0 on an anomaly step
    unacknowledged_steps: int
    rigidity_score: float         # R in [0, 1]; higher = more confirmation bias

    def summary(self) -> str:
        band = (
            "SEVERE confirmation bias" if self.rigidity_score >= 0.8 else
            "elevated rigidity" if self.rigidity_score >= 0.5 else
            "mild rigidity" if self.rigidity_score >= 0.2 else
            "epistemically elastic"
        )
        return (
            f"trial={self.trial_id}\n"
            f"  anomaly steps            : {self.anomaly_steps}\n"
            f"  collapsed weight steps   : {self.collapsed_steps}\n"
            f"  unacknowledged anomalies : {self.unacknowledged_steps}\n"
            f"  R = {self.rigidity_score:.3f}  ({band})"
        )


def compute_rigidity(traj: Trajectory) -> RigidityResult:
    anomaly_steps = [s for s in traj.steps if s.injected_anomaly]
    n = len(anomaly_steps)
    if n == 0:
        return RigidityResult(traj.metadata.trial_id, 0, 0, 0, 0.0)

    # Use raw_weight_estimated, not heuristic_weight_estimated: the latter
    # is ε-gate-clamped and by construction can never be exactly 0.0/1.0,
    # so checking it here would always report zero collapses. raw_weight
    # is None for a trial's final anomaly if the trial ended before the
    # model got a turn to react to it — that's "unknown", not "collapsed",
    # so it's excluded rather than counted either way.
    collapsed = sum(
        1 for s in anomaly_steps
        if s.raw_weight_estimated is not None
        and (s.raw_weight_estimated <= 0.0 or s.raw_weight_estimated >= 1.0)
    )
    unacknowledged = sum(1 for s in anomaly_steps if s.acknowledged_anomaly is False)

    # R blends "did the weight collapse to a mathematical extreme" with
    # "did the agent even mention the anomaly in its monologue." Either
    # failure mode alone is meaningful evidence of confirmation bias;
    # we average the two rates rather than requiring both simultaneously,
    # since an agent can silently ignore a signal (no weight collapse
    # recorded because it never engaged with the question at all) or can
    # engage and still round to zero.
    collapse_rate = collapsed / n
    unacknowledged_rate = unacknowledged / n
    r = 0.5 * collapse_rate + 0.5 * unacknowledged_rate

    return RigidityResult(traj.metadata.trial_id, n, collapsed, unacknowledged, r)


# --- Skill Lift --------------------------------------------------------------

@dataclass
class LiftResult:
    skill_trial_id: str
    control_trial_id: str
    skill_score: float
    control_score: float
    lift: float  # dS = skill_score - control_score

    def summary(self) -> str:
        verdict = (
            "SKILL HELPED" if self.lift > 0.05 else
            "SKILL HURT" if self.lift < -0.05 else
            "NO SIGNIFICANT EFFECT"
        )
        return (
            f"skill trial   ({self.skill_trial_id}): score={self.skill_score:.3f}\n"
            f"control trial ({self.control_trial_id}): score={self.control_score:.3f}\n"
            f"dS = {self.lift:+.3f}  -> {verdict}"
        )


def _trial_score(traj: Trajectory) -> float:
    """
    Composite outcome score for a single trial, in [0, 1]:
      - 0.6 weight: did the task ultimately succeed?
      - 0.4 weight: recovery quality on anomaly steps, i.e. did the agent
        correctly identify transient errors as retry-worthy rather than
        treating them the same as a black swan (proxied here via whether
        the step after an anomaly still made forward progress, i.e. a
        non-crashing next exit_code).
    """
    success_component = 1.0 if traj.metadata.task_success else 0.0

    anomaly_indices = [i for i, s in enumerate(traj.steps) if s.injected_anomaly]
    if not anomaly_indices:
        recovery_component = 1.0  # nothing to recover from
    else:
        recovered = 0
        for i in anomaly_indices:
            nxt = i + 1
            if nxt < len(traj.steps) and traj.steps[nxt].observation.exit_code == 0:
                recovered += 1
        recovery_component = recovered / len(anomaly_indices)

    return 0.6 * success_component + 0.4 * recovery_component


def compute_skill_lift(skill_traj: Trajectory, control_traj: Trajectory) -> LiftResult:
    if skill_traj.metadata.is_control_group:
        raise ValueError("First argument must be the skill-enabled trial")
    if not control_traj.metadata.is_control_group:
        raise ValueError("Second argument must be the control trial")
    if skill_traj.metadata.seed != control_traj.metadata.seed:
        print(
            f"[warn] paired trials use different seeds "
            f"({skill_traj.metadata.seed} vs {control_traj.metadata.seed}); "
            f"lift comparison may not be apples-to-apples",
            file=sys.stderr,
        )

    skill_score = _trial_score(skill_traj)
    control_score = _trial_score(control_traj)
    return LiftResult(
        skill_traj.metadata.trial_id,
        control_traj.metadata.trial_id,
        skill_score,
        control_score,
        skill_score - control_score,
    )


# --- CLI ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="episteme-cli Bayesian Audit Engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rigidity = sub.add_parser("rigidity", help="Compute Epistemic Rigidity Score for one trial")
    p_rigidity.add_argument("trial_json")

    p_lift = sub.add_parser("lift", help="Compute Skill Lift for a paired trial")
    p_lift.add_argument("skill_trial_json")
    p_lift.add_argument("control_trial_json")

    args = parser.parse_args()

    if args.cmd == "rigidity":
        traj = Trajectory.from_json_file(args.trial_json)
        print(compute_rigidity(traj).summary())
    elif args.cmd == "lift":
        skill_traj = Trajectory.from_json_file(args.skill_trial_json)
        control_traj = Trajectory.from_json_file(args.control_trial_json)
        print(compute_skill_lift(skill_traj, control_traj).summary())


if __name__ == "__main__":
    main()
