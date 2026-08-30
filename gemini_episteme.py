from __future__ import annotations
import argparse, os, re, sys, time, uuid
from datetime import datetime, timezone
import pexpect
from google import genai
from google.genai import types
from google.genai.errors import APIError
from atif import AnomalyKind, Observation as AtifObservation, Step, Trajectory, TrialMetadata

_ANOMALY_KEYWORDS = re.compile(r"\b(warn\w*|error\w*|stderr|unexpected|anomal\w*|retry|retries|retried)\b", re.IGNORECASE)
def _mentions_anomaly(monologue: str) -> bool:
    return bool(_ANOMALY_KEYWORDS.search(monologue))

def calculate_objective_heuristic(exit_code: int, stderr: str) -> float:
    if exit_code == 0 and not stderr.strip(): return 0.95
    if exit_code != 0: return 0.05
    return 0.50

MODEL = "gemini-3.6-flash"

ANOMALY_PLAN = {
    1: {"kind": "non_deterministic_warning", "mutate": lambda obs: {**obs, "stderr": (obs["stderr"] + "\n[WARN] Deprecated API call").strip()}},
    2: {"kind": "transient_state_error", "mutate": lambda obs: {**obs, "stdout": "", "stderr": (obs["stderr"] + "\nError: Connection reset by peer").strip(), "exit_code": 1}},
}

class Shell:
    def __init__(self, timeout_secs: int = 10):
        self.timeout_secs = timeout_secs
        self.child = pexpect.spawn("bash", encoding="utf-8", timeout=timeout_secs)
        self.child.setecho(False)
        self.child.sendline("PS1=''")
        self.child.expect(pexpect.TIMEOUT, timeout=0.3)

    def run(self, command: str) -> dict:
        sentinel = f"__MINI_EPISTEME_{uuid.uuid4().hex[:8]}__"
        start = time.time()
        self.child.sendline(f'{command}; echo "{sentinel}$?"')
        timed_out = False
        try:
            self.child.expect(re.escape(sentinel) + r"(\d+)", timeout=self.timeout_secs)
        except pexpect.TIMEOUT:
            timed_out = True
        except pexpect.EOF:
            return {"stdout": self.child.before or "", "stderr": "EOF", "exit_code": -1, "timed_out": False, "latency_ms": int((time.time() - start) * 1000)}
        
        latency_ms = int((time.time() - start) * 1000)
        if timed_out: return {"stdout": (self.child.before or "").strip(), "stderr": "timeout", "exit_code": -124, "timed_out": True, "latency_ms": latency_ms}
        
        exit_code_str = self.child.match.group(1) if self.child.match else "-1"
        try: exit_code = int(exit_code_str)
        except ValueError: exit_code = -1
        
        return {"stdout": (self.child.before or "").strip(), "stderr": "", "exit_code": exit_code, "timed_out": False, "latency_ms": latency_ms}

    def close(self) -> None:
        try:
            self.child.sendline("exit")
            self.child.close(force=True)
        except Exception: pass

def run_trial(task: str, max_steps: int, command_timeout: int, seed: int, verbose: bool) -> Trajectory:
    client = genai.Client()
    shell = Shell(timeout_secs=command_timeout)
    trial_id = str(uuid.uuid4())
    steps = []
    
    execute_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="execute_command",
                description="Execute exactly one bash command. You must report confidence_in_signal: your confidence (0.0 to 1.0) in the reliability and accuracy of the terminal output you just received.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "command": types.Schema(type=types.Type.STRING),
                        "confidence_in_signal": types.Schema(type=types.Type.NUMBER),
                        "reasoning": types.Schema(type=types.Type.STRING)
                    },
                    required=["command", "confidence_in_signal", "reasoning"]
                )
            )
        ]
    )
    
    config = types.GenerateContentConfig(
        system_instruction="You are a DevOps agent in a bash shell. Evaluate terminal output critically. Stop and use command 'DONE' when complete.",
        tools=[execute_tool],
        temperature=0.0,
        seed=seed
    )
    
    messages = [types.Content(role="user", parts=[types.Part.from_text(text=f"Task: {task}")])]
    task_success = False

    try:
        for step_index in range(max_steps):
            response = None
            for attempt in range(4):
                try:
                    response = client.models.generate_content(model=MODEL, contents=messages, config=config)
                    break
                except APIError as e:
                    if e.code == 429 and attempt < 3:
                        time.sleep(10 * (attempt + 1))
                    else:
                        raise e
            
            if not response or not response.function_calls: break
            
            args = response.function_calls[0].args
            command = str(args.get("command", "")).strip()
            confidence = max(0.0, min(1.0, float(args.get("confidence_in_signal", 0.0))))
            reasoning = str(args.get("reasoning", ""))
            
            if steps:
                steps[-1].raw_weight_estimated = confidence
                if steps[-1].injected_anomaly:
                    steps[-1].acknowledged_anomaly = _mentions_anomaly(reasoning)
                    
            messages.append(response.candidates[0].content)
            if command.upper() == "DONE":
                task_success = steps[-1].observation.exit_code == 0 if steps else False
                break
                
            raw_obs = shell.run(command)
            anomaly = ANOMALY_PLAN.get(step_index)
            obs = anomaly["mutate"](raw_obs) if anomaly else dict(raw_obs)
            kind = AnomalyKind(anomaly["kind"]) if anomaly else AnomalyKind.NONE
            
            heuristic_val = calculate_objective_heuristic(obs["exit_code"], obs["stderr"])
            
            atif_obs = AtifObservation(stdout=obs["stdout"], stderr=obs["stderr"], exit_code=obs["exit_code"], latency_ms=obs["latency_ms"], timed_out=obs["timed_out"])
            steps.append(Step(step_index=step_index, action=command, internal_monologue=reasoning, observation=atif_obs, injected_anomaly=bool(anomaly), anomaly_kind=kind, heuristic_weight_estimated=heuristic_val, raw_weight_estimated=None, acknowledged_anomaly=None))
            
            if verbose: print(f"[step {step_index}] $ {command} -> exit={obs['exit_code']} conf={confidence:.2f} (obj_score={heuristic_val:.2f})")
            messages.append(types.Content(role="user", parts=[types.Part.from_function_response(name="execute_command", response={"stdout": obs['stdout'], "stderr": obs['stderr'], "exit_code": obs['exit_code'], "timed_out": obs['timed_out']})]))
    except Exception as e:
        if verbose: print(f"\n[!] Fatal trial error: {e}. Saving partial trajectory.")
    finally:
        shell.close()
        meta = TrialMetadata(schema_version=1, trial_id=trial_id, agent_version=MODEL, skill_target="none", is_control_group=False, noise_level="hardcoded", seed=seed, task_success=task_success, max_steps=max_steps, ended_at=datetime.now(timezone.utc))
        return Trajectory(metadata=meta, steps=steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()
    
    trial = run_trial(args.task, args.max_steps, args.timeout, args.seed, True)
    trial.to_json_file(f"live_trial_{args.seed}.json")
