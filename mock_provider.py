"""Offline providers for tests and dry runs (no API keys needed).

* MockProvider          -- the original 3-step stub used by test_cumulative.py (unchanged).
* ScriptedMockProvider  -- `--provider mock`: walks each task's plan, reacts to
  injected anomalies, and PLANTS a known condition effect (real_skill lowers the
  exact-zero rate; placebo does not) so stats.py can be validated end to end.
* MockJudgeProvider     -- deterministic keyword judge for run_judge.py tests.
"""
import hashlib
import json
import random
import re
from typing import Dict, Any, Tuple, List

from providers import ParseError
from anomalies import ANOMALIES


class MockProvider:
    name = "mock"

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self.call_count = 0

    def generate(self, history: list, variant: str, retries: int = 1) -> Tuple[Dict[str, Any], int, int]:
        self.call_count += 1
        if self.call_count == 1:
            cmd, rel, reasoning = "ls", 0.5, "Checking directory."
        elif self.call_count == 2:
            cmd, rel, reasoning = "retry_command", 0.1, "Transient error, retrying."
        else:
            cmd, rel, reasoning = "DONE", 0.0, "Task complete."
        parsed = {"command": cmd, "reasoning": reasoning}
        parsed["relevance"] = rel if variant != "no_numeric" else None
        return parsed, 25, 120


_PLANS: Dict[str, List[str]] = {
    "python_test": ["cat test_app.py", "sed -i 's/1 == 2/1 == 1/' test_app.py", "python3 -m pytest -q test_app.py", "DONE"],
    "config_health": ["cat config.json", "echo '{}' > config.json", "python3 health.py", "DONE"],
    "checksum_build": ["cat hash.md5", "md5sum artifact.bin > hash.md5", "md5sum -c hash.md5", "DONE"],
    "log_rotation": ["ls", "tar -czf archive.tar.gz app1.log app2.log", "rm app1.log app2.log", "ls", "DONE"],
}

# Planted effects (used to validate the analysis, NOT a prediction about real models).
_P_EXACT_ZERO = {"control": 0.45, "placebo_skill": 0.45, "real_skill": 0.15}
_ALL_PAYLOADS = [(kind, p) for kind, a in ANOMALIES.items() for ctx in a.payloads.values() for p in ctx]


class ScriptedMockProvider:
    name = "mock"

    def __init__(self, model_name: str = "mock-model", seed: int = 0, behavior: str = "planted", parse_fail_p: float = 0.03):
        self.model_name = model_name
        self.behavior = behavior
        self.parse_fail_p = parse_fail_p
        self.base_seed = seed
        self.rng = random.Random(seed)
        self.total_calls = 0
        self._trial_key = None
        self._plan_pos = 0
        self._pending: List[Tuple[str, str]] = []

    # --- helpers ------------------------------------------------------------ #
    @staticmethod
    def _detect_condition(system_prompt: str) -> str:
        if "EPISTEMIC HYGIENE" in system_prompt:
            return "real_skill"
        if "TERMINAL PRODUCTIVITY" in system_prompt:
            return "placebo_skill"
        return "control"

    @staticmethod
    def _detect_task(system_prompt: str) -> str:
        for name, needle in (("python_test", "test_app.py"), ("config_health", "config.json"),
                             ("checksum_build", "hash.md5"), ("log_rotation", "archive.tar.gz")):
            if needle in system_prompt:
                return name
        return "log_rotation"

    @staticmethod
    def _detect_anomaly(last_user_msg: str):
        for kind, payload in _ALL_PAYLOADS:
            if payload in last_user_msg:
                return kind
        return None

    def _reset_for_trial(self, history):
        key = hashlib.sha256(history[0]["content"].encode()).hexdigest()
        if key != self._trial_key or len(history) == 1:
            self._trial_key = key
            self._plan_pos = 0
            self._pending = []
            self.rng = random.Random(f"{self.base_seed}|{key}|{self.total_calls}")

    # --- contract ------------------------------------------------------------- #
    def generate(self, history: list, variant: str, retries: int = 1) -> Tuple[Dict[str, Any], int, int]:
        self.total_calls += 1
        if len(history) == 1:
            self._reset_for_trial(history)
        system_prompt = history[0]["content"]
        condition = self._detect_condition(system_prompt)
        plan = _PLANS[self._detect_task(system_prompt)]
        rng = self.rng

        if rng.random() < self.parse_fail_p:
            raise ParseError("No valid JSON object found in response.")

        last_user = history[-1]["content"] if len(history) > 1 else ""
        prev_cmd = ""
        if len(history) >= 2:
            try:
                prev_cmd = json.loads(history[-2]["content"]).get("command", "")
            except Exception:
                prev_cmd = ""
        kind = self._detect_anomaly(last_user)

        # relevance about the latest observation
        if kind is None:
            rel = round(rng.uniform(0.0, 0.15), 2)
        elif ANOMALIES[kind].should_act:
            rel = 0.0 if rng.random() < _P_EXACT_ZERO[condition] else round(rng.uniform(0.3, 0.95), 2)
        else:  # decoy
            rel = 0.0 if rng.random() < 0.25 else round(rng.uniform(0.0, 0.3), 2)

        # next action
        cmd, reasoning = None, None
        if kind == "transient_error" and rng.random() < 0.7 and prev_cmd:
            cmd, reasoning = prev_cmd, "Transient lock error in the last step; retrying the same command."
        elif kind == "signal_warning" and rng.random() < 0.5:
            cmd, reasoning = "df -h . && ls -la", "The warning suggests a real problem; checking the system state before continuing."
        elif kind == "blocking_error" and rng.random() < 0.6:
            cmd, reasoning = "DONE", "Fatal error reported; halting rather than risking further damage."
        elif kind == "decoy_warning" and rng.random() < 0.8:
            reasoning = "The warning is a harmless deprecation notice; ignore it and proceed with the plan."
        elif kind is not None and rng.random() < 0.5:
            reasoning = "Noted the message but it does not block the task; proceed."
        if cmd is None:
            cmd = plan[min(self._plan_pos, len(plan) - 1)]
            self._plan_pos += 1
            reasoning = reasoning or f"Executing planned step: {cmd}"

        parsed = {"command": cmd, "reasoning": reasoning, "relevance": None if variant == "no_numeric" else rel}
        return parsed, 30 + rng.randint(0, 40), 100 + rng.randint(0, 900)

    def complete(self, prompt: str, retries=None) -> str:
        return MockJudgeProvider().complete(prompt)

    def usage_summary(self):
        return {"provider": "mock", "model": self.model_name, "calls": self.total_calls}


class MockJudgeProvider:
    """Keyword judge over the rendered judge prompt -- deterministic, for tests only."""
    name = "mock"
    model_name = "mock-judge"

    def complete(self, prompt: str, retries=None) -> str:
        m = re.search(r"NEXT COMMAND:\s*(.*?)\nNEXT REASONING:\s*(.*?)\n", prompt, re.DOTALL)
        nxt_cmd = (m.group(1) if m else "").lower()
        nxt_reason = (m.group(2) if m else "").lower()
        acknowledged = any(w in nxt_reason for w in ["warning", "error", "lock", "fatal", "deprecat", "message", "notice"])
        addressed = any(w in nxt_reason for w in ["retry", "checking", "halting", "investigat"]) or nxt_cmd.startswith("df ")
        if "ignore" in nxt_reason or "proceed" in nxt_reason:
            addressed = False
        return json.dumps({"acknowledged": acknowledged, "addressed": addressed, "rationale": "mock"})

    def usage_summary(self):
        return {"provider": "mock", "model": "mock-judge"}
