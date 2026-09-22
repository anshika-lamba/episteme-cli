import datetime as _dt
import json
from atif import Trajectory, TrialMetadata, Step, Observation
from schedule import generate_schedule
from anomalies import ANOMALIES, inject_anomaly
from tasks import TASKS, TaskEnv
from prompts import build_system_prompt
from providers import ParseError, ProviderError, QuotaExceededError
from metrics import attach_anomaly_responses


def run_trial(provider, task_name: str, condition: str, variant: str, seed: int,
              expected_steps: int = 6, max_steps: int = 10) -> Trajectory:
    """Run one sandboxed trial. Never raises for provider trouble: partial trajectories
    are returned with `metadata.aborted_reason` set (QuotaExceeded is flagged so the
    grid runner can stop the whole run)."""
    task = TASKS[task_name]
    schedule = generate_schedule(seed, task_name, expected_steps, max_steps)
    meta = TrialMetadata(
        f"{task_name}_{condition}_{variant}_{seed}", provider.model_name, task_name, condition, variant, seed,
        False, schedule, provider_name=getattr(provider, "name", ""),
        started_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )
    traj = Trajectory(meta)
    env = TaskEnv(task)
    try:
        env.setup()
    except Exception as e:
        # Broken sandbox (e.g. no POSIX shell on Windows) must not look like model behaviour
        # and must not kill the grid: record it and move on.
        traj.metadata.aborted_reason = f"sandbox_setup_failed: {str(e)[:200]}"
        traj.metadata.task_success = False
        env.cleanup()
        return traj
    history = [{"role": "user", "content": build_system_prompt(task.prompt, condition, variant)}]
    consecutive_failures = 0

    set_ctx = getattr(provider, "set_context", None)
    try:
        for step_idx in range(max_steps):
            if set_ctx:
                set_ctx(trial_id=meta.trial_id, step=step_idx, condition=condition, variant=variant, task=task_name)
            try:
                parsed, out_tok, latency = provider.generate(history, variant)
                consecutive_failures = 0
            except ParseError as e:
                consecutive_failures += 1
                obs = Observation(stdout="", stderr=f"ParseError: {e}", exit_code=-1, timed_out=False)
                traj.steps.append(Step(step_idx, "PARSE_FAILED", str(e), obs, False, "none", None, parse_fail=True))
                if consecutive_failures >= 2:
                    break
                continue
            except QuotaExceededError as e:
                traj.steps.append(Step(step_idx, "PROVIDER_ERROR", str(e), Observation("", str(e), -1, False)))
                traj.metadata.aborted_reason = f"QuotaExceeded: {e}"
                break
            except ProviderError as e:
                traj.steps.append(Step(step_idx, "PROVIDER_ERROR", str(e), Observation("", str(e), -1, False)))
                traj.metadata.aborted_reason = f"ProviderError: {e}"
                break

            cmd = parsed.get("command", "echo no-op")
            rel = parsed.get("relevance")
            reasoning = parsed.get("reasoning", "")
            curr_is_done = cmd.strip().upper() == "DONE"

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
            history.extend([
                {"role": "assistant", "content": json.dumps({k: v for k, v in parsed.items() if not k.startswith("_")})},
                {"role": "user", "content": f"stdout: {obs.stdout}\nstderr: {obs.stderr}\nexit: {obs.exit_code}"},
            ])

        traj.metadata.task_success = env.check_success()
    finally:
        env.cleanup()

    if set_ctx:
        set_ctx()
    # Attribute each anomaly's *response* (next valid action + its stated relevance) to the anomaly step.
    attach_anomaly_responses(traj)
    return traj
