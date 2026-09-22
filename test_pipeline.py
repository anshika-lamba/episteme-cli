"""Tests for the provider layer, grid runner, statistics and judge tooling.
Everything runs offline (fake HTTP sessions / mock providers)."""
import csv
import json
import math
import os
import random
import subprocess
import sys

import pytest

import providers
from providers import (extract_last_json, ParseError, ProviderError, QuotaExceededError, RateLimiter, CallLedger,
                       GroqProvider, GeminiProvider, CohereProvider, MistralProvider, make_provider, REGISTRY)
from run_grid import build_plan, existing_trial_ids, ALL_TASKS, ALL_CONDITIONS, ALL_VARIANTS
from metrics import compute_auroc
import stats
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
])
def test_extract_valid(text, expected_cmd, expected_rel):
    p = extract_last_json(text, "neutral")
    assert p["command"] == expected_cmd and p["relevance"] == expected_rel and isinstance(p["reasoning"], str)


@pytest.mark.parametrize("text,msg", [
    ("no json here", "No valid JSON"),
    ('{"command": "ls", "relevance": 0.5} {}', "command"),          # trailing empty object is the LAST object
    ('{"command": "", "relevance": 0.5}', "command"),
    ('{"command": "ls"}', "missing 'relevance'"),
    ('{"command": "ls", "relevance": null}', "missing 'relevance'"),
    ('{"command": "ls", "relevance": true}', "between 0.0 and 1.0"),
    ('{"command": "ls", "relevance": -0.1}', "between 0.0 and 1.0"),
    ('{"command": "ls", "relevance": "high"}', "between 0.0 and 1.0"),
])
def test_extract_invalid(text, msg):
    with pytest.raises(ParseError, match=msg):
        extract_last_json(text, "original")


def test_no_numeric_forces_none_even_if_model_supplies_relevance():
    assert extract_last_json('{"command": "ls", "relevance": 0.7}', "no_numeric")["relevance"] is None


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


def test_existing_trial_ids_skips_quota_aborted(tmp_path):
    f = tmp_path / "r.jsonl"
    f.write_text(json.dumps({"metadata": {"trial_id": "a", "agent_version": "m", "aborted_reason": None}}) + "\n"
                 + json.dumps({"metadata": {"trial_id": "b", "agent_version": "m", "aborted_reason": "QuotaExceeded: x"}}) + "\n"
                 + json.dumps({"metadata": {"trial_id": "c", "agent_version": "m", "aborted_reason": "ProviderError: 500"}}) + "\n")
    ids, models = existing_trial_ids(str(f))
    assert ids == {"a", "c"} and models == {"m"}


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
