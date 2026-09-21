import json
from atif import Trajectory, TrialMetadata, Step, Observation
from schedule import generate_schedule
from anomalies import ANOMALIES, inject_anomaly
from tasks import TASKS, TaskEnv
from prompts import build_system_prompt
from providers import ParseError
from behavior import classify_behavior

def run_trial(provider, task_name: str, condition: str, variant: str, seed: int, expected_steps: int = 6, max_steps: int = 10) -> Trajectory:
    task = TASKS[task_name]
    env = TaskEnv(task)
    env.setup()
    schedule = generate_schedule(seed, task_name, expected_steps, max_steps)
    meta = TrialMetadata(f"{task_name}_{condition}_{variant}_{seed}", provider.model_name, task_name, condition, variant, seed, False, schedule)
    traj = Trajectory(meta)
    history = [{"role": "user", "content": build_system_prompt(task.prompt, condition, variant)}]
    consecutive_failures = 0
    
    try:
        for step_idx in range(max_steps):
            try:
                parsed, out_tok, latency = provider.generate(history, variant)
                consecutive_failures = 0
            except ParseError as e:
                consecutive_failures += 1
                obs = Observation(stdout="", stderr=f"ParseError: {e}", exit_code=-1, timed_out=False)
                step = Step(step_idx, "PARSE_FAILED", str(e), obs, False, "none", None)
                traj.steps.append(step)
                if consecutive_failures >= 2: break
                continue
                
            cmd = parsed.get("command", "echo no-op")
            rel = parsed.get("relevance")
            reasoning = parsed.get("reasoning", "")
            curr_is_done = cmd.strip().upper() == "DONE"
            
            prev_valid = next((s for s in reversed(traj.steps) if s.action != "PARSE_FAILED"), None)
            if prev_valid and prev_valid.injected_anomaly:
                prev_valid.next_action_behavior = classify_behavior(prev_valid.action, cmd, reasoning, curr_is_done)
                    
            if curr_is_done:
                traj.steps.append(Step(step_idx, cmd, reasoning, Observation("", "", 0, False), False, "none", rel, None, None, out_tok, latency))
                break
                
            out, err, code, to = env.run_cmd(cmd)
            injected, kind = False, "none"
            if step_idx in schedule:
                kind = schedule[step_idx]
                out, err, code = inject_anomaly(out, err, code, ANOMALIES[kind], cmd, seed, step_idx)
                injected = True
                
            obs = Observation(out[:1000], err[:1000], code, to)
            traj.steps.append(Step(step_idx, cmd, reasoning, obs, injected, kind, rel, None, None, out_tok, latency))
            history.extend([{"role": "assistant", "content": json.dumps(parsed)}, {"role": "user", "content": f"stdout: {obs.stdout}\nstderr: {obs.stderr}\nexit: {obs.exit_code}"}])
            
        traj.metadata.task_success = env.check_success()
    finally:
        env.cleanup()
    return traj
