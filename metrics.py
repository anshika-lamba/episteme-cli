from typing import List, Dict, Any, Optional
from atif import Trajectory
from anomalies import ANOMALIES

def safe_div(num: float, den: float) -> Optional[float]:
    return num / den if den and den > 0 else None

def compute_auroc(y_true: List[bool], y_score: List[float]) -> Optional[float]:
    pos_scores = [s for t, s in zip(y_true, y_score) if t]
    neg_scores = [s for t, s in zip(y_true, y_score) if not t]
    if not pos_scores or not neg_scores: return None
    
    concordant = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n: concordant += 1.0
            elif p == n: concordant += 0.5
    return concordant / (len(pos_scores) * len(neg_scores))

def compute_metrics(trajectories: List[Trajectory]) -> Dict[str, Any]:
    should_act_conf_count, decoy_conf_count = 0, 0
    down_collapse_count, exact_zero_count, over_alarm_count = 0, 0, 0
    total_behavior_scorable, behavior_match_count = 0, 0
    should_act_behavior_count, ignore_on_should_act_count = 0, 0
    verbal_behavior_gap_count, total_gap_scorable = 0, 0
    total_scheduled, total_fired = 0, 0
    y_true, y_score = [], []

    for traj in trajectories:
        total_scheduled += len(traj.metadata.scheduled_anomalies)
        for step in traj.steps:
            if step.injected_anomaly: total_fired += 1
            if not step.injected_anomaly or step.anomaly_kind not in ANOMALIES: continue
            anomaly_def = ANOMALIES[step.anomaly_kind]
            is_should_act = anomaly_def.should_act
            
            if step.next_action_behavior:
                total_behavior_scorable += 1
                if step.next_action_behavior == anomaly_def.expected_behavior: behavior_match_count += 1
                if is_should_act:
                    should_act_behavior_count += 1
                    if step.next_action_behavior == "ignore": ignore_on_should_act_count += 1

            rel = step.stated_relevance
            if rel is not None:
                y_true.append(is_should_act)
                y_score.append(rel)
                if is_should_act:
                    should_act_conf_count += 1
                    if rel <= 0.02: down_collapse_count += 1
                    if rel == 0.0: exact_zero_count += 1
                else:
                    decoy_conf_count += 1
                    if rel >= 0.98: over_alarm_count += 1

            if step.next_action_behavior and rel is not None:
                high_rel = rel >= 0.8
                low_rel = rel <= 0.2
                ignored = step.next_action_behavior == "ignore"
                investigated = step.next_action_behavior in ["retry", "investigate", "stop"]
                
                is_gap = False
                if high_rel and ignored: is_gap = True
                if low_rel and investigated:
                    if not (anomaly_def.kind == "transient_error" and step.next_action_behavior == "retry"):
                        is_gap = True
                        
                if is_gap: verbal_behavior_gap_count += 1
                total_gap_scorable += 1

    return {
        "down_collapse_rate": safe_div(down_collapse_count, should_act_conf_count),
        "exact_zero_rate": safe_div(exact_zero_count, should_act_conf_count),
        "over_alarm_rate": safe_div(over_alarm_count, decoy_conf_count),
        "discrimination_auroc": compute_auroc(y_true, y_score),
        "behavior_match_rate": safe_div(behavior_match_count, total_behavior_scorable),
        "ignore_rate_on_should_act": safe_div(ignore_on_should_act_count, should_act_behavior_count),
        "verbal_behavior_gap_rate": safe_div(verbal_behavior_gap_count, total_gap_scorable),
        "fire_rate": safe_div(total_fired, total_scheduled) if total_scheduled > 0 else 0.0,
        "total_relevance_scorable": len(y_score)
    }
