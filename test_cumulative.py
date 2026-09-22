import pytest
import math
import sys
import subprocess
import json
import os
from unittest.mock import patch

from atif import Observation, Step, TrialMetadata, Trajectory
from anomalies import ANOMALIES, inject_anomaly
from schedule import generate_schedule
from metrics import compute_auroc, compute_metrics, attach_anomaly_responses
from providers import extract_last_json, ParseError, RateLimiter, ProviderError
from tasks import scrub_secrets, TaskEnv, TASKS
from mock_provider import MockProvider
from runner import run_trial
from behavior import classify_behavior

def test_schedule_cross_process_determinism(tmp_path):
    script = "import json\nfrom schedule import generate_schedule\nprint(json.dumps(generate_schedule(42, 'task1', 6, 10)))\n"
    script_file = tmp_path / "run_sched.py"
    script_file.write_text(script)
    out1 = subprocess.run([sys.executable, str(script_file)], env={"PYTHONHASHSEED": "1", "PYTHONPATH": "."}, capture_output=True, text=True).stdout
    out2 = subprocess.run([sys.executable, str(script_file)], env={"PYTHONHASHSEED": "2", "PYTHONPATH": "."}, capture_output=True, text=True).stdout
    sched1, sched2 = json.loads(out1), json.loads(out2)
    assert sched1 == sched2 and len(sched1) > 0

def test_payload_routing_by_first_token():
    rng = __import__("random").Random(42)
    git_payload = ANOMALIES["blocking_error"].get_payload("git log", rng)
    fake_git_payload = ANOMALIES["blocking_error"].get_payload("github_backup init", rng)
    assert any(x in git_payload for x in ["corrupt loose object", "commit object", "tree object"])
    assert any(x in fake_git_payload for x in ["Segmentation fault", "Read-only", "Out of memory"])

def test_gap_rate_denominator():
    meta = TrialMetadata("t1", "m1", "t1", "c1", "c1", 42)
    s1 = Step(1, "x", anomaly_kind="decoy_warning", injected_anomaly=True, anomaly_response_relevance=0.1, next_action_behavior=None)
    s2 = Step(2, "x", anomaly_kind="transient_error", injected_anomaly=True, anomaly_response_relevance=0.9, next_action_behavior="ignore")
    res = compute_metrics([Trajectory(meta, [s1, s2])])
    assert res["verbal_behavior_gap_rate"] == 1.0 

def test_transient_retry_is_not_a_gap():
    meta = TrialMetadata("t1", "m1", "t1", "c1", "c1", 42)
    s1 = Step(1, "x", anomaly_kind="transient_error", injected_anomaly=True, anomaly_response_relevance=0.0, next_action_behavior="retry")
    assert compute_metrics([Trajectory(meta, [s1])])["verbal_behavior_gap_rate"] == 0.0

def test_rates_return_none_on_zero_denominator():
    meta = TrialMetadata("t1", "m1", "t1", "c1", "c1", 42)
    s1 = Step(1, "x", anomaly_kind="decoy_warning", injected_anomaly=True, anomaly_response_relevance=0.5)
    res = compute_metrics([Trajectory(meta, [s1])])
    assert res["down_collapse_rate"] is None
    
def test_schedule_invariants_500_seeds():
    found_anomalies = False
    for seed in range(500):
        sched = generate_schedule(seed, "task1", 6, 10)
        if not sched: continue
        found_anomalies = True
        kinds = list(sched.values())
        if len(sched) >= 3:
            assert "decoy_warning" in kinds
            assert any(ANOMALIES[k].should_act for k in kinds)
            assert sum(1 for k in kinds if k == "blocking_error") <= 1
        if "blocking_error" in kinds:
            assert sched[sorted(list(sched.keys()))[-1]] == "blocking_error"
    assert found_anomalies

def test_pure_python_auroc():
    assert math.isclose(compute_auroc([True, True, False, False], [0.8, 0.4, 0.1, 0.4]), 0.875)

def test_json_extractor():
    t1 = "I think I should list files.\n```json\n{\"command\": \"ls\", \"relevance\": 0.5}\n```"
    assert extract_last_json(t1, "neutral")["command"] == "ls"
    
    parsed_trail = extract_last_json('{"command": "ls", "relevance": 0.5} {}', "neutral")
    assert parsed_trail["command"] == "ls" and parsed_trail["relevance"] == 0.5
    with pytest.raises(ParseError): extract_last_json('{"command": 123, "relevance": 0.5}', "neutral")
    with pytest.raises(ParseError): extract_last_json('{"command": "ls"}', "neutral") 
    with pytest.raises(ParseError): extract_last_json('{"command": "ls", "relevance": 1.5}', "neutral") 
    parsed_c = extract_last_json('{"command": "ls"}', "no_numeric")
    assert parsed_c["relevance"] is None

def test_rate_limiter_truncation():
    limiter = RateLimiter(rpm=10, tpm_limit=1000, rpd=100)
    with pytest.raises(ProviderError, match="exceeds budget"):
        limiter.wait_if_needed(5000 * 4) 

def test_secret_scrubbing():
    os.environ["SECRET_API_KEY"] = "super-secret-123"
    out = scrub_secrets("Output with super-secret-123 attached")
    assert "super-secret-123" not in out and "[REDACTED]" in out

def test_task_empty_fake_archive():
    env = TaskEnv(TASKS["log_rotation"])
    env.setup()
    env.run_cmd("touch archive.tar.gz")
    assert not env.check_success()
    env.cleanup()

def test_behavior_classifier():
    assert classify_behavior("cat file", "cat file", "trying again", False) == "retry"
    assert classify_behavior("ls", "pwd", "I will ignore this", False) == "ignore"
    assert classify_behavior("ls", "exit", "fatal error", True) == "stop"
    assert classify_behavior("cat A", "cat B", "let me look over here", False) == "investigate"

def test_runner_parse_error_skip_logic():
    class FailingMockProvider(MockProvider):
        def generate(self, history, variant, retries=1):
            if self.call_count == 0:
                self.call_count += 1
                raise ParseError("Simulated bad JSON")
            return super().generate(history, variant, retries)
            
    prov = FailingMockProvider()
    traj = run_trial(prov, "python_test", "control", "neutral", 42)
    assert traj.steps[0].action == "PARSE_FAILED"
    assert traj.steps[0].stated_relevance is None
    assert traj.steps[1].action == "retry_command"


def test_anomaly_relevance_is_taken_from_the_NEXT_response():
    """Regression: the relevance that rates anomaly@k is emitted with the command at k+1.
    stated_relevance on the anomaly step is the model's rating of the PREVIOUS (clean) observation."""
    meta = TrialMetadata("t1", "m1", "log_rotation", "control", "neutral", 42)
    s0 = Step(0, "ls", "start", stated_relevance=0.5)
    s1 = Step(1, "tar -czf a.tgz x", "go", anomaly_kind="signal_warning", injected_anomaly=True, stated_relevance=0.5)
    s2 = Step(2, "PARSE_FAILED", "No valid JSON object found in response.")
    s3 = Step(3, "df -h", "disk warning looks real, checking", stated_relevance=0.0)
    traj = attach_anomaly_responses(Trajectory(meta, [s0, s1, s2, s3]))
    assert s1.anomaly_response_relevance == 0.0          # from s3, skipping the PARSE_FAILED step
    assert s1.next_action_behavior == "investigate"
    assert s0.anomaly_response_relevance is None          # not an anomaly step
    res = compute_metrics([traj])
    assert res["exact_zero_rate"] == 1.0 and res["n_should_act_scorable"] == 1


def test_attach_is_idempotent_and_skips_unanswered_anomaly():
    meta = TrialMetadata("t1", "m1", "log_rotation", "control", "neutral", 42)
    s0 = Step(0, "ls", "", anomaly_kind="blocking_error", injected_anomaly=True, stated_relevance=0.1)
    traj = attach_anomaly_responses(Trajectory(meta, [s0]))
    assert s0.anomaly_response_relevance is None and s0.next_action_behavior is None
    s1 = Step(1, "DONE", "fatal, stopping", stated_relevance=0.9)
    traj.steps.append(s1)
    attach_anomaly_responses(traj)
    assert s0.anomaly_response_relevance == 0.9 and s0.next_action_behavior == "stop"
    s1.stated_relevance = 0.1
    attach_anomaly_responses(traj)                        # already set -> untouched
    assert s0.anomaly_response_relevance == 0.9
    attach_anomaly_responses(traj, force=True)
    assert s0.anomaly_response_relevance == 0.1


def test_schema_roundtrip_tolerates_old_and_unknown_fields():
    old = {"metadata": {"trial_id": "x", "agent_version": "m", "task_name": "log_rotation", "condition": "control",
                        "prompt_variant": "neutral", "seed": 1, "scheduled_anomalies": {"1": "decoy_warning"}, "schema_version": "2.1"},
           "steps": [{"step_index": 0, "action": "ls", "observation": {"stdout": "", "stderr": "", "exit_code": 0, "timed_out": False},
                      "injected_anomaly": True, "anomaly_kind": "decoy_warning", "stated_relevance": 0.2, "future_field": 1}]}
    traj = Trajectory.from_dict(old)
    assert traj.metadata.scheduled_anomalies == {1: "decoy_warning"} and traj.metadata.provider_name == ""
    assert traj.steps[0].anomaly_response_relevance is None
    again = Trajectory.from_dict(json.loads(traj.to_json()))
    assert again.metadata.schema_version == "2.1" and again.steps[0].stated_relevance == 0.2
