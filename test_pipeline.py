"""Tests for the provider layer, grid runner, statistics and judge tooling.
Everything runs offline (fake HTTP sessions / mock providers)."""
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

import providers
from providers import (extract_last_json, ParseError, ProviderError, QuotaExceededError, RateLimiter, CallLedger,
                       GroqProvider, GeminiProvider, CohereProvider, MistralProvider, make_provider, REGISTRY)
from runner import run_trial
from run_grid import build_plan, existing_trial_ids, ALL_TASKS, ALL_CONDITIONS, ALL_VARIANTS
from metrics import compute_auroc
import stats
import tasks
from tasks import TASKS, TaskEnv
from stats import TrialRec, Obs, exact_zero_rate, bootstrap_ci, block_permutation_test
from kappa import cohen_kappa, parse_label
from judge import parse_judge_json, render_prompt


# --------------------------------------------------------------------------- #
# extract_last_json
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected_cmd,expected_rel", [
    ('{"command": "ls -la", "relevance": 0.25, "reasoning": "look"}', "ls -la", 0.25),
    ('Sure! Here you go:\n```json\n{"command": "pwd", "relevance": 0, "reasoning": "x"}\n```\nDone.', "pwd", 0.0),
    ('{"command": "cat a", "relevance": "0.5"}', "cat a", 0.5),                       # numeric string accepted
    ('{"command": "ls", "relevance": 1, "reasoning": {"nested": "{brace}"}}', "ls", 1.0),
    ('prefix {"command": "echo {}", "relevance": 0.1} trailing {not json}', "echo {}", 0.1),  # last span invalid -> fall back
    ('{"command": "ls", "relevance": 0.5} {}', "ls", 0.5),                            # trailing empty object is not an answer
    ('```json\n{"command": "df -h", "relevance": 0.0, "reasoning": "disk",}\n```', "df -h", 0.0),  # fence + trailing comma
    ("{'command': 'ls', 'relevance': 0.4, 'reasoning': 'ok'}", "ls", 0.4),            # single-quoted object
    ('{\u201ccommand\u201d: \u201cls\u201d, \u201crelevance\u201d: 0.2}', "ls", 0.2),  # smart quotes
    ('{"command": "ls", "relevance": 0.3', "ls", 0.3),                                 # one missing closer
    ('example {"command": "ls", "relevance": 0.9} final {"command": "df -h", "relevance": 0.0}', "df -h", 0.0),
    ('[{"command": "ls", "relevance": 0.9}, {"command": "pwd", "relevance": 0.1}]', "pwd", 0.1),
    ('{"command": "ls", "relevance": [0.0, 0.95, 1.0]}', "ls", 1.0),                  # array -> last in-range number
])
def test_extract_valid(text, expected_cmd, expected_rel):
    p = extract_last_json(text, "neutral")
    assert p["command"] == expected_cmd and p["relevance"] == expected_rel and isinstance(p["reasoning"], str)


@pytest.mark.parametrize("text,msg", [
    ("no json here", "No valid JSON"),
    ("[0.0, 0.95, 1.0]", "No valid JSON"),                          # a relevance array has no command; do not invent one
    ('{"command": "", "relevance": 0.5}', "command"),
    ('{"command": "ls"}', "missing 'relevance'"),
    ('{"command": "ls", "relevance": null}', "missing 'relevance'"),
    ('{"command": "ls", "relevance": true}', "between 0.0 and 1.0"),
    ('{"command": "ls", "relevance": -0.1}', "between 0.0 and 1.0"),
    ('{"command": "ls", "relevance": "high"}', "between 0.0 and 1.0"),
    ('{"command": "ls", "relevance": "95%"}', "between 0.0 and 1.0"),  # do not rescale a percent into 0.0
])
def test_extract_invalid(text, msg):
    with pytest.raises(ParseError, match=msg):
        extract_last_json(text, "original")


def test_no_numeric_forces_none_even_if_model_supplies_relevance():
    assert extract_last_json('{"command": "ls", "relevance": 0.7}', "no_numeric")["relevance"] is None


def test_array_relevance_is_flagged_and_missing_relevance_is_not_zero():
    coerced = extract_last_json('{"command": "ls", "relevance": [0.0, "0.4"]}', "neutral")
    assert coerced["relevance"] == 0.4 and coerced["_relevance_coerced"] == "array_last"
    with pytest.raises(ParseError, match="missing 'relevance'"):
        extract_last_json('{"command": "ls", "relevance": null}', "original")


# --------------------------------------------------------------------------- #
# RateLimiter / CallLedger
# --------------------------------------------------------------------------- #
def test_rate_limiter_rpm_window(monkeypatch):
    clock = [1000.0]
    sleeps = []
    monkeypatch.setattr(providers.time, "time", lambda: clock[0])
    monkeypatch.setattr(providers.time, "sleep", lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s)))
    lim = RateLimiter(rpm=2, tpm_limit=10000, rpd=100)
    for _ in range(2):
        lim.wait_if_needed(400)
        lim.record_call(100)
    lim.wait_if_needed(400)  # third call within the minute must wait ~60s
    assert sleeps and 59 < sum(sleeps) <= 62


def test_rate_limiter_tpm_budget(monkeypatch):
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(providers.time, "time", lambda: clock[0])
    monkeypatch.setattr(providers.time, "sleep", lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s)))
    lim = RateLimiter(rpm=100, tpm_limit=1000, rpd=100)
    lim.wait_if_needed(100)
    lim.record_call(900)
    lim.wait_if_needed(800)  # 900 + 200 > 1000 -> wait for the window
    assert sleeps


def test_rate_limiter_daily_cap():
    lim = RateLimiter(rpm=100, tpm_limit=10000, rpd=1)
    lim.wait_if_needed(10)
    lim.record_call(10)
    with pytest.raises(QuotaExceededError, match="DAILY_REQUEST_CAP"):
        lim.wait_if_needed(10)


def test_call_ledger_persists_and_caps(tmp_path):
    led = CallLedger("cohere", monthly_cap=2, ledger_dir=str(tmp_path))
    led.check()
    led.record()
    led.record()
    assert led.used() == 2 and led.remaining() == 0
    with pytest.raises(QuotaExceededError, match="MONTHLY_CALL_CAP"):
        CallLedger("cohere", monthly_cap=2, ledger_dir=str(tmp_path)).check()  # re-read from disk


# --------------------------------------------------------------------------- #
# HTTP layer with a fake session
# --------------------------------------------------------------------------- #
class FakeResp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.text = json.dumps(body) if isinstance(body, (dict, list)) else str(body)

    def json(self):
        if isinstance(self._body, (dict, list)):
            return self._body
        raise ValueError("not json")


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, json, headers))
        return self.responses.pop(0)


def _fast(monkeypatch):
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)


def _openai_body(text):
    return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 50, "completion_tokens": 20}}


def test_groq_happy_path_and_retry_on_429(monkeypatch, tmp_path):
    _fast(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    p = GroqProvider("allam-2-7b", ledger_dir=str(tmp_path))
    p.session = FakeSession([FakeResp(429, {"error": "slow down"}, {"retry-after": "1"}),
                             FakeResp(200, _openai_body('{"command":"ls","relevance":0.0,"reasoning":"r"}'))])
    parsed, out_tok, latency = p.generate([{"role": "user", "content": "hi"}], "neutral")
    assert parsed["command"] == "ls" and parsed["relevance"] == 0.0 and out_tok == 20
    assert p.session.calls[0][1]["model"] == "allam-2-7b" and p.session.calls[0][1]["temperature"] == 0.0
    assert p.total_calls == 1 and p.limiter.daily_calls == 2  # the 429 still counted against RPM/RPD


def test_bad_model_fails_fast_not_retried(monkeypatch, tmp_path):
    _fast(monkeypatch)
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    p = MistralProvider("ministral-8b-latest", ledger_dir=str(tmp_path))
    p.session = FakeSession([FakeResp(404, {"message": "model not found"})])
    with pytest.raises(ProviderError, match="404"):
        p.generate([{"role": "user", "content": "hi"}], "neutral")
    assert len(p.session.calls) == 1


def test_503_retries_eight_times_then_aborts_that_trial_only(monkeypatch, tmp_path):
    sleeps = []
    monkeypatch.setattr(providers.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(providers, "_sleep_backoff", lambda attempt, retry_after: 12.5)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    p = GroqProvider("allam-2-7b", ledger_dir=str(tmp_path), transport_retries=8)
    p.session = FakeSession([FakeResp(503, "unavailable")] * 9)
    with pytest.raises(ProviderError, match="gave up after 9"):
        p.generate([{"role": "user", "content": "hi"}], "neutral")
    assert len(p.session.calls) == 9 and sleeps == [12.5] * 8
    p.session = FakeSession([FakeResp(503, "unavailable")] * 8 + [FakeResp(200, _openai_body('{"command":"ls","relevance":0.0}'))])
    parsed, _, _ = p.generate([{"role": "user", "content": "hi"}], "neutral")
    assert parsed["command"] == "ls" and len(p.session.calls) == 9


def test_backoff_is_jittered_ten_to_thirty_and_honors_retry_after():
    samples = [providers._sleep_backoff(0, None) for _ in range(40)]
    assert all(10.0 <= s <= 30.0 for s in samples)
    assert len({round(s, 3) for s in samples}) > 1
    later = [providers._sleep_backoff(3, None) for _ in range(20)]
    assert all(s <= 180.0 for s in later) and max(later) > 30.0
    honored = providers._sleep_backoff(5, 4.0)
    assert 4.0 <= honored <= 6.0


def test_daily_quota_429_is_terminal(monkeypatch, tmp_path):
    _fast(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    p = GeminiProvider("gemini-2.5-flash", ledger_dir=str(tmp_path))
    p.session = FakeSession([FakeResp(429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded for GenerateRequestsPerDayPerProjectPerModel"}})])
    with pytest.raises(QuotaExceededError):
        p.generate([{"role": "user", "content": "hi"}], "neutral")


def test_gemini_message_conversion_and_parse(monkeypatch, tmp_path):
    _fast(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    p = GeminiProvider("gemini-2.5-flash", ledger_dir=str(tmp_path))
    p.session = FakeSession([FakeResp(200, {"candidates": [{"content": {"parts": [{"text": '{"command":"pwd","relevance":0.3}'}]}}],
                                            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}})])
    hist = [{"role": "user", "content": "sys"}, {"role": "assistant", "content": "{}"}, {"role": "user", "content": "obs"}]
    parsed, _, _ = p.generate(hist, "neutral")
    sent = p.session.calls[0][1]["contents"]
    assert [c["role"] for c in sent] == ["user", "model", "user"] and parsed["command"] == "pwd"
    assert p.session.calls[0][2]["x-goog-api-key"] == "k"


def test_cohere_v2_shape_and_monthly_ledger(monkeypatch, tmp_path):
    _fast(monkeypatch)
    monkeypatch.setenv("COHERE_API_KEY", "k")
    p = CohereProvider("command-r7b-12-2024", monthly_cap=1, ledger_dir=str(tmp_path))
    p.session = FakeSession([FakeResp(200, {"message": {"content": [{"type": "text", "text": '{"command":"ls","relevance":0.9}'}]},
                                            "usage": {"tokens": {"input_tokens": 7, "output_tokens": 3}}})])
    parsed, _, _ = p.generate([{"role": "user", "content": "hi"}], "neutral")
    assert parsed["relevance"] == 0.9 and "/v2/chat" in p.session.calls[0][0]
    with pytest.raises(QuotaExceededError, match="MONTHLY_CALL_CAP"):
        p.generate([{"role": "user", "content": "hi"}], "neutral")


def test_parse_error_propagates_without_retry(monkeypatch, tmp_path):
    _fast(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    p = GroqProvider("allam-2-7b", ledger_dir=str(tmp_path))
    p.session = FakeSession([FakeResp(200, _openai_body("I cannot comply."))])
    with pytest.raises(ParseError):
        p.generate([{"role": "user", "content": "hi"}], "neutral")
    assert len(p.session.calls) == 1


def test_registry_and_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        make_provider("groq")
    with pytest.raises(ProviderError, match="Unknown provider"):
        make_provider("openai")
    assert set(REGISTRY) == {"groq", "gemini", "mistral", "cohere"}


# --------------------------------------------------------------------------- #
# run_grid planning / resume
# --------------------------------------------------------------------------- #
def test_plan_is_condition_balanced_at_every_prefix_and_deterministic():
    plan = build_plan(ALL_TASKS, ALL_CONDITIONS, ALL_VARIANTS, [0, 1])
    assert len(plan) == 4 * 3 * 3 * 2 and len(set(plan)) == len(plan)
    for k in range(3, len(plan) + 1, 3):
        prefix = plan[:k]
        counts = {c: sum(1 for p in prefix if p[1] == c) for c in ALL_CONDITIONS}
        assert len(set(counts.values())) == 1, f"unbalanced prefix {k}: {counts}"
    assert plan == build_plan(ALL_TASKS, ALL_CONDITIONS, ALL_VARIANTS, [0, 1])


def test_transport_circuit_breaker_stops_after_three_and_resets():
    from run_grid import note_trial_abort
    n, why = note_trial_abort(0, "ProviderError: groq: gave up after 9 attempts: HTTP 503", 3)
    assert n == 1 and why is None
    n, why = note_trial_abort(n, "ProviderError: groq: gave up after 9 attempts: HTTP 503", 3)
    assert n == 2 and why is None
    n, why = note_trial_abort(n, "ProviderError: groq: gave up after 9 attempts: HTTP 503", 3)
    assert n == 3 and "exhausted transport" in why
    assert note_trial_abort(2, None, 3) == (0, None)
    assert note_trial_abort(5, "ProviderError: groq: gave up after 9 attempts", 0)[1] is None


def test_existing_trial_ids_skips_quota_aborted(tmp_path):
    f = tmp_path / "r.jsonl"
    f.write_text(json.dumps({"metadata": {"trial_id": "a", "agent_version": "m", "aborted_reason": None}}) + "\n"
                 + json.dumps({"metadata": {"trial_id": "b", "agent_version": "m", "aborted_reason": "QuotaExceeded: x"}}) + "\n"
                 + json.dumps({"metadata": {"trial_id": "c", "agent_version": "m", "aborted_reason": "ProviderError: 500"}}) + "\n")
    ids, models = existing_trial_ids(str(f))
    assert ids == {"a", "c"} and models == {"m"}


def test_resume_requeues_transport_exhaustion_and_drops_a_torn_tail(tmp_path):
    from run_grid import compact_retryable, repair_jsonl_tail, is_retryable_abort
    from metrics import load_trajectories
    f = tmp_path / "r.jsonl"
    rows = [
        {"metadata": {"trial_id": "a", "agent_version": "m", "task_name": "log_rotation", "condition": "control",
                      "prompt_variant": "neutral", "seed": 0, "schema_version": "2.2"}, "steps": []},
        {"metadata": {"trial_id": "b", "agent_version": "m", "task_name": "log_rotation", "condition": "control",
                      "prompt_variant": "neutral", "seed": 1, "schema_version": "2.2",
                      "aborted_reason": "ProviderError: groq: gave up after 9 attempts: HTTP 503"}, "steps": []},
        {"metadata": {"trial_id": "c", "agent_version": "m", "task_name": "log_rotation", "condition": "control",
                      "prompt_variant": "neutral", "seed": 2, "schema_version": "2.2",
                      "aborted_reason": "ProviderError: 500"}, "steps": []},
    ]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"metadata": {"trial_id": "torn"')
    assert repair_jsonl_tail(str(f))
    assert "torn" not in f.read_text() and f.read_text().endswith("\n")
    assert is_retryable_abort(rows[1]["metadata"]["aborted_reason"])
    assert not is_retryable_abort(rows[2]["metadata"]["aborted_reason"])
    ids, _ = existing_trial_ids(str(f))
    assert ids == {"a", "c"}
    assert compact_retryable(str(f)) == 1
    assert "gave up after" not in f.read_text()
    assert {t.metadata.trial_id for t in load_trajectories([str(f)])} == {"a", "c"}
    # a torn tail must not make stats.py raise
    f.write_bytes(f.read_bytes() + b'{"metadata": {"trial_id": "torn"')
    assert {t.metadata.trial_id for t in load_trajectories([str(f)])} == {"a", "c"}


def test_mock_pilot_end_to_end(tmp_path):
    out = tmp_path / "pilot.jsonl"
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    r = subprocess.run([sys.executable, "run_grid.py", "--provider", "mock", "--pilot", "--out", str(out)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 18 and all(l["metadata"]["schema_version"] == "2.2" for l in lines)
    r2 = subprocess.run([sys.executable, "run_grid.py", "--provider", "mock", "--pilot", "--out", str(out)], capture_output=True, text=True, env=env)
    assert "18 already done, 0 to run" in r2.stderr
    r3 = subprocess.run([sys.executable, "stats.py", str(out), "--boot", "50", "--perms", "50"], capture_output=True, text=True, env=env)
    assert r3.returncode == 0 and "exact_zero_rate" in r3.stdout and "SCENARIO" in r3.stdout


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _synthetic_trials(n_seeds: int, p_zero: dict, rng: random.Random):
    trials = []
    for seed in range(n_seeds):
        for task in ALL_TASKS:
            for variant in ("original", "neutral"):
                for cond in ALL_CONDITIONS:
                    obs = []
                    for j in range(2):
                        rel = 0.0 if rng.random() < p_zero[cond] else round(rng.uniform(0.2, 1.0), 2)
                        obs.append(Obs(f"{task}_{cond}_{variant}_{seed}", "m", task, cond, variant, seed, j, "signal_warning", True, "investigate", rel, "investigate"))
                    obs.append(Obs(f"{task}_{cond}_{variant}_{seed}", "m", task, cond, variant, seed, 5, "decoy_warning", False, "ignore", round(rng.uniform(0.0, 0.4), 2), "ignore"))
                    trials.append(TrialRec(f"{task}_{cond}_{variant}_{seed}", "m", task, cond, variant, seed, True, False, 6, 0, 3, 3, obs))
    return trials


def test_permutation_detects_planted_effect_and_not_placebo():
    rng = random.Random(0)
    trials = _synthetic_trials(6, {"control": 0.5, "placebo_skill": 0.5, "real_skill": 0.1}, rng)
    real = block_permutation_test(trials, "condition", "real_skill", "control", exact_zero_rate, 400, rng)
    plac = block_permutation_test(trials, "condition", "placebo_skill", "control", exact_zero_rate, 400, rng)
    assert real["diff"] < -0.25 and real["p_one_sided_a_lt_b"] < 0.01
    assert plac["p_two_sided"] > 0.05
    assert real["n_blocks"] == 6 * 4 * 2


def test_permutation_null_is_calibrated():
    rng = random.Random(1)
    rejections = 0
    for rep in range(30):
        trials = _synthetic_trials(3, {"control": 0.4, "placebo_skill": 0.4, "real_skill": 0.4}, rng)
        r = block_permutation_test(trials, "condition", "real_skill", "control", exact_zero_rate, 200, rng)
        rejections += r["p_two_sided"] < 0.05
    assert rejections <= 5  # ~1.5 expected at alpha=.05; generous bound


def test_bootstrap_ci_covers_point_estimate_and_is_cluster_level():
    rng = random.Random(2)
    trials = _synthetic_trials(4, {"control": 0.3, "placebo_skill": 0.3, "real_skill": 0.3}, rng)
    v = exact_zero_rate(trials)
    lo, hi, nb = bootstrap_ci(trials, exact_zero_rate, 300, rng)
    assert lo <= v <= hi and nb == 300 and 0 < hi - lo < 0.3


def test_bootstrap_returns_none_when_metric_undefined():
    rng = random.Random(3)
    t = TrialRec("t", "m", "python_test", "control", "no_numeric", 0, True, False, 3, 0, 3, 1,
                 [Obs("t", "m", "python_test", "control", "no_numeric", 0, 1, "signal_warning", True, "investigate", None, "ignore")])
    assert exact_zero_rate([t]) is None and bootstrap_ci([t], exact_zero_rate, 50, rng) == (None, None, 0)


def test_rank_auroc_matches_brute_force():
    rng = random.Random(4)
    y = [rng.random() < 0.4 for _ in range(60)]
    s = [round(rng.random() * 5) / 5 for _ in y]  # many ties
    pos = [x for x, t in zip(s, y) if t]
    neg = [x for x, t in zip(s, y) if not t]
    brute = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg) / (len(pos) * len(neg))
    assert math.isclose(compute_auroc(y, s), brute)


def test_to_trial_recs_uses_response_relevance(tmp_path):
    from atif import Trajectory, TrialMetadata, Step
    from metrics import attach_anomaly_responses
    meta = TrialMetadata("t", "m", "log_rotation", "control", "neutral", 0, scheduled_anomalies={1: "signal_warning"})
    s0 = Step(0, "ls", "", stated_relevance=0.4)
    s1 = Step(1, "tar x", "", injected_anomaly=True, anomaly_kind="signal_warning", stated_relevance=0.4)
    s2 = Step(2, "df", "checking", stated_relevance=0.0)
    recs = stats.to_trial_recs([attach_anomaly_responses(Trajectory(meta, [s0, s1, s2]))])
    assert recs[0].obs[0].rel == 0.0 and exact_zero_rate(recs) == 1.0


# --------------------------------------------------------------------------- #
# judge + kappa
# --------------------------------------------------------------------------- #
def test_cohen_kappa_values():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohen_kappa([True, True, True], [True, True, True]) == 1.0            # degenerate perfect agreement
    k = cohen_kappa([True] * 8 + [False] * 2, [True] * 7 + [False] * 3)
    assert 0.4 < k < 0.8
    assert parse_label("1") is True and parse_label("no") is False and parse_label("") is None


def test_judge_parse_and_prompt_is_blind():
    rec = {"sample_id": "S001", "task_prompt": "fix", "prev_command": "ls", "observation": {"stdout": "", "stderr": "[WARN] Disk partition 98% full.", "exit_code": 0},
           "next_command": "df -h", "next_reasoning": "disk warning, checking", "should_act": True, "heuristic_behavior": "investigate", "model": "allam-2-7b", "condition": "real_skill"}
    prompt = render_prompt(rec)
    for leak in ("should_act", "investigate", "allam", "real_skill", "True"):
        assert leak not in prompt
    assert parse_judge_json('Sure: {"acknowledged": "yes", "addressed": false, "rationale": "x"}') == {"acknowledged": True, "addressed": False, "rationale": "x"}
    with pytest.raises(ProviderError):
        parse_judge_json('{"acknowledged": true}')


def test_sampler_csv_is_blinded(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    out = tmp_path / "r.jsonl"
    subprocess.run([sys.executable, "run_grid.py", "--provider", "mock", "--pilot", "--out", str(out)], capture_output=True, text=True, env=env, check=True)
    lab = tmp_path / "labeling"
    r = subprocess.run([sys.executable, "sample_for_labeling.py", str(out), "--n", "10", "--out-dir", str(lab)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    with open(lab / "labels_template.csv", newline="") as f:
        header = next(csv.reader(f))
    for forbidden in ("model", "condition", "variant", "should_act", "relevance", "heuristic", "anomaly_kind"):
        assert not any(forbidden in h for h in header), header
    rows = [json.loads(l) for l in (lab / "sample.jsonl").read_text().splitlines()]
    assert len(rows) == 10 and all(r["next_command"] for r in rows)
    r2 = subprocess.run([sys.executable, "run_judge.py", "--provider", "mock", "--sample", str(lab / "sample.jsonl"), "--out", str(lab / "j.jsonl")], capture_output=True, text=True, env=env)
    assert r2.returncode == 0 and len((lab / "j.jsonl").read_text().splitlines()) == 10


# --------------------------------------------------------------------------- #
# harness event log + restart-safe daily counter (the "Model 2" artifact)
# --------------------------------------------------------------------------- #
def test_daily_counter_survives_process_restart(monkeypatch, tmp_path):
    _fast(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    p1 = GroqProvider("allam-2-7b", rpd=3, ledger_dir=str(tmp_path))
    p1.session = FakeSession([FakeResp(200, _openai_body('{"command":"ls","relevance":0.1}'))] * 2)
    p1.generate([{"role": "user", "content": "hi"}], "neutral")
    p1.generate([{"role": "user", "content": "hi"}], "neutral")
    p2 = GroqProvider("allam-2-7b", rpd=3, ledger_dir=str(tmp_path))  # "restart"
    assert p2.limiter.daily_calls == 2, "in-memory RPD counter must be seeded from the persistent ledger"
    p2.session = FakeSession([FakeResp(200, _openai_body('{"command":"ls","relevance":0.1}'))])
    p2.generate([{"role": "user", "content": "hi"}], "neutral")
    with pytest.raises(QuotaExceededError, match="DAILY_REQUEST_CAP"):
        p2.generate([{"role": "user", "content": "hi"}], "neutral")
    assert p2.usage_summary()["calls_today_utc"] == 3


def test_event_sink_taxonomy(monkeypatch, tmp_path):
    from harness_summary import JsonlSink, summarize_events
    _fast(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    p = GroqProvider("allam-2-7b", ledger_dir=str(tmp_path))
    sink = JsonlSink(str(tmp_path / "h.jsonl"), run_meta={"provider": "groq"})
    p.event_sink = sink
    p.set_context(trial_id="t1", step=3)
    p.session = FakeSession([FakeResp(429, {"error": "busy"}), FakeResp(503, "upstream"),
                             FakeResp(200, _openai_body("not json at all"))])
    with pytest.raises(ParseError):
        p.generate([{"role": "user", "content": "hi"}], "neutral")
    kinds = [e["kind"] for e in sink.events]
    assert kinds == ["http_429", "retry", "http_5xx", "retry", "recovered", "parse_failed"]
    assert all(e["trial_id"] == "t1" and e["step"] == 3 for e in sink.events)
    p.session = FakeSession([FakeResp(404, {"message": "no such model"})])
    p.set_context()
    with pytest.raises(ProviderError):
        p.generate([{"role": "user", "content": "hi"}], "neutral")
    assert sink.events[-1]["kind"] == "http_4xx_fatal" and "trial_id" not in sink.events[-1]
    s = summarize_events(sink.events)
    assert s["by_kind"]["transport/http_429"] == 1 and s["by_http_status"]["404"] == 1
    assert s["parse_failure_reasons"] == {"No valid JSON object found in response.": 1}
    sink.close({"ok": True})
    lines = [json.loads(l) for l in (tmp_path / "h.jsonl").read_text().splitlines()]
    assert lines[0]["kind"] == "run_start" and lines[-1]["kind"] == "run_end" and len(lines) == 2 + len(sink.events)


def test_model_limits_override_registry_defaults(monkeypatch, tmp_path):
    from providers import effective_limits
    assert effective_limits("gemini", None)["rpd"] == 14400            # gemma-3-12b-it default
    assert effective_limits("gemini", "gemini-2.5-flash")["rpd"] == 250
    assert effective_limits("gemini", "some-new-model")["rpd"] == 14400  # unknown -> provider default
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    p = make_provider("gemini", "gemini-2.5-flash", ledger_dir=str(tmp_path))
    assert p.limiter.rpm == 10 and p.limiter.rpd_limit == 250
    p2 = make_provider("gemini", "gemini-2.5-flash", rpd=99, ledger_dir=str(tmp_path))
    assert p2.limiter.rpd_limit == 99                                     # explicit flag wins


# --------------------------------------------------------------------------- #
# Windows/local-run hardening
# --------------------------------------------------------------------------- #
def test_run_grid_shared_overrides_reach_every_provider(monkeypatch, tmp_path):
    """Regression: run_grid passes ONE overrides dict (incl. mock-only seed=0) to every
    provider; make_provider must accept it for real providers too, not just mock."""
    overrides = dict(rpm=None, tpm=None, rpd=None, min_interval_s=None, monthly_cap=None,
                     temperature=0.0, max_output_tokens=400, json_mode=None, seed=0, ledger_dir=str(tmp_path))
    for name, key in (("groq", "GROQ_API_KEY"), ("gemini", "GEMINI_API_KEY"), ("mistral", "MISTRAL_API_KEY"), ("cohere", "COHERE_API_KEY")):
        monkeypatch.setenv(key, "k")
        p = make_provider(name, None, **overrides)
        assert p.temperature == 0.0 and p.max_output_tokens == 400 and p.ledger is not None
    p_mock = make_provider("mock", None, **overrides)  # seed honoured here, dropped for real providers
    assert p_mock.__class__.__name__ == "ScriptedMockProvider" and p_mock.base_seed == 0
    with pytest.raises(ProviderError, match="Unknown option 'nonsense'"):
        make_provider("groq", None, nonsense=1)


def test_sandbox_setup_failure_is_recorded_not_raised():
    """A broken environment aborts the trial with a clear reason; the grid keeps running."""
    import tasks as tasks_mod
    from mock_provider import ScriptedMockProvider
    saved = tasks_mod._SHELL_PREFIX
    tasks_mod.set_shell(["/definitely/not/a/shell-xyz"])
    try:
        traj = run_trial(ScriptedMockProvider(seed=0), "log_rotation", "control", "neutral", 0)
    finally:
        tasks_mod.set_shell(saved)
    assert traj.metadata.aborted_reason.startswith("sandbox_setup_failed")
    assert traj.steps == []


def test_run_grid_preflight_flags_missing_posix_shell(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    out = tmp_path / "p.jsonl"
    r = subprocess.run([sys.executable, "run_grid.py", "--provider", "mock", "--pilot", "--dry-run",
                        "--sandbox-shell", "/definitely/not/a/shell-xyz", "--out", str(out)], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and ("does not work" in r.stderr or "does not exist" in r.stderr)  # explicit broken shell fails even in dry-run
    r2 = subprocess.run([sys.executable, "run_grid.py", "--provider", "mock", "--max-trials", "1",
                         "--sandbox-shell", "/definitely/not/a/shell-xyz", "--out", str(out)], capture_output=True, text=True, env=env)
    assert r2.returncode == 1 and "sandbox" in (r2.stderr + r2.stdout).lower()


# --------------------------------------------------------------------------- #
# Windows shell detection (PowerShell: touch is not a command, Git bash is off PATH,
# test_cumulative.py never calls set_shell())
# --------------------------------------------------------------------------- #
def test_git_bash_paths_are_joined_not_unpacked():
    """Regression: ntpath.join(root, *'bin\\bash.exe') unpacked the string into
    C:\\b\\a\\s\\h\\.exe, so C:\\Program Files\\Git\\bin\\bash.exe was never found and
    every trial fell through to PowerShell (where `touch` raises CalledProcessError)."""
    wanted = {
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
    }
    queried = []

    def exists(path):
        queried.append(path)
        return path in wanted

    def which(name):
        return r"C:\Program Files\Git\cmd\git.exe" if name in ("git", "git.exe") else None

    cands = [c[0] for c in tasks._candidate_shells(which=which, exists=exists, environ={
        "ProgramFiles": r"C:\Program Files",
        "ProgramFiles(x86)": r"C:\Program Files (x86)",
    })]
    assert r"C:\Program Files\Git\bin\bash.exe" in cands
    assert r"C:\Program Files\Git\usr\bin\bash.exe" in cands
    assert cands[0] == r"C:\Program Files\Git\bin\bash.exe"  # explicit paths before PATH/WSL
    assert all("-c" == c[1] for c in tasks._candidate_shells(which=which, exists=exists, environ={}) if c[0] in wanted)
    assert not any("\\b\\a\\s\\h" in p or p.endswith("\\e") for p in queried)
    # usr\\bin layout only (bin\\bash.exe absent): still found, not character-unpacked
    cands2 = [c[0] for c in tasks._candidate_shells(
        which=which, exists=lambda p: p == r"C:\Program Files\Git\usr\bin\bash.exe", environ={})]
    assert r"C:\Program Files\Git\usr\bin\bash.exe" in cands2
    assert r"C:\Program Files\Git\bin\bash.exe" not in cands2


def test_run_grid_checks_the_same_off_path_git_bash():
    from run_grid import _GIT_BASH_OFF_PATH, resolve_sandbox_shell
    assert _GIT_BASH_OFF_PATH == tasks.GIT_BASH_EXPLICIT
    assert all(p in Path("run_grid.py").read_text() for p in (
        r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"))
    # explicit broken shell fails before any trial, including --dry-run
    assert resolve_sandbox_shell("/definitely/not/a/shell-xyz")[1]


def test_find_posix_shell_requires_working_probe(monkeypatch):
    """A candidate only counts if it executes a command (stale wsl.exe with no distro must
    fall through). Do not patch os.name — that makes pathlib.Path a WindowsPath and crashes
    pytest's cache on Linux."""
    monkeypatch.setattr(tasks, "_is_windows", lambda: True)
    monkeypatch.setattr(tasks, "_candidate_shells", lambda **k: [["wsl"], ["good"], ["broken"]])
    monkeypatch.setattr(tasks, "_probe_shell", lambda pref, timeout=20: pref == ["good"])
    assert tasks.find_posix_shell() == ["good"]
    monkeypatch.setattr(tasks, "_probe_shell", lambda pref, timeout=20: False)
    assert tasks.find_posix_shell() is None


def test_active_shell_raises_guidance_on_windows_without_bash(monkeypatch):
    saved = tasks._SHELL_PREFIX
    try:
        monkeypatch.setattr(tasks, "_is_windows", lambda: True)
        monkeypatch.setattr(tasks, "find_posix_shell", lambda: None)
        tasks.set_shell(tasks.AUTO)
        with pytest.raises(RuntimeError, match=r"C:\\Program Files\\Git\\bin\\bash.exe"):
            tasks.ensure_shell_for_direct_use()
        with pytest.raises(RuntimeError, match=r"C:\\Program Files\\Git\\usr\\bin\\bash.exe"):
            tasks.ensure_shell_for_direct_use()  # cached failure, still informative
    finally:
        tasks._SHELL_PREFIX = saved


def test_taskenv_routes_touch_through_git_bash_not_powershell(monkeypatch):
    """The failure mode: test_cumulative.py calls TaskEnv.setup() with no set_shell(), and
    on Windows shell=True is PowerShell, which has no `touch`. The resolved prefix must be
    used, and /usr/bin (where Git bash keeps touch) prepended."""
    calls = []

    class R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(tasks.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)) or R())
    monkeypatch.setattr(tasks, "_active_shell", lambda: [r"C:\Program Files\Git\bin\bash.exe", "-c"])
    monkeypatch.setattr(tasks, "_is_windows", lambda: True)
    env = TaskEnv(TASKS["log_rotation"])
    try:
        env.setup()
    finally:
        env.cleanup()
    argv, kw = calls[0]
    assert argv[0] == r"C:\Program Files\Git\bin\bash.exe" and argv[1] == "-c"
    assert "touch app1.log app2.log" in argv[2]
    assert argv[2].startswith('export PATH="/usr/bin:/bin:')
    assert "shell" not in kw  # must not be shell=True (that is PowerShell/cmd)


def test_overnight_script_defaults_to_the_preregistered_budget():
    text = Path("run_overnight.ps1").read_text(encoding="utf-8")
    assert "AllowOverBudget" in text and "IncludeCohere" in text and "996" in text
    assert "allam-2-7b" in text and "ministral-8b-latest" in text and "gemma-3-12b-it" in text
    assert "command-r7b-12-2024" in text
    assert "--transport-retries" in text
    # the 7 x 1000 path must be behind the override, not the default invocation
    assert "TrialsEach" in text and "-AllowOverBudget" in text


def test_session_fixture_initializes_shell_without_per_test_set_shell():
    """conftest.posix_sandbox_shell calls ensure_shell_for_direct_use before tests. After
    that, a direct TaskEnv (the test_cumulative.py pattern) must not still be on AUTO."""
    assert tasks._SHELL_PREFIX is not tasks.AUTO
    if os.name != "nt":
        assert tasks.current_shell() is None  # native /bin/sh
    else:
        shell = tasks.current_shell()
        assert shell and "bash" in " ".join(shell).lower() and shell[-1] == "-c"


def test_harness_scripts_find_python_when_python3_is_absent(tmp_path):
    """Git Bash rarely exposes `python3`. Setup scripts must not hardcode it, and must
    run via `python` (or `py`) when that is the only interpreter on PATH."""
    for task in TASKS.values():
        assert "python3 -c" not in task.setup_script
        assert "python3 health.py" not in task.success_script and "python3 -m" not in task.success_script
    if os.name == "nt":
        pytest.skip("PATH emulation is POSIX; Windows coverage is the Git-bash detection tests")
    shimdir = tmp_path / "bin"
    shimdir.mkdir()
    try:
        (shimdir / "python").symlink_to(sys.executable)
    except (OSError, NotImplementedError):
        pytest.skip("symlink not permitted")
    env = TaskEnv(TASKS["python_test"])
    env.env["PATH"] = str(shimdir)
    try:
        from tasks import _PYSHIM
        out, err, code, _ = env.run_cmd(_PYSHIM + "echo FOUND:$PYBIN; command -v python3 || echo NO_PYTHON3")
        assert code == 0 and "FOUND:python" in out and "NO_PYTHON3" in out, (out, err, code)
        env.setup()
        out, err, code, _ = env.run_cmd("test -s test_app.py && echo WROTE")
        assert "WROTE" in out, (out, err, code)
    finally:
        env.cleanup()
