from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys

import anthropic

from atif import AnomalyKind, Observation as AtifObservation, Step, Trajectory, TrialMetadata
from epsilon_gate import EpsilonGate, EpsilonGateState

MODEL = "claude-sonnet-5"


def connect(socket_path: str) -> socket.socket:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(socket_path)
    return s


def send_action(sock: socket.socket, command: str) -> None:
    sock.sendall((json.dumps({"command": command}) + "\n").encode())


def recv_observation(sock: socket.socket, buf: bytearray) -> dict:
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("harness closed the socket")
        buf.extend(chunk)
    line, _, rest = buf.partition(b"\n")
    buf[:] = rest
    return json.loads(line.decode())


def run_trial(
    socket_path: str,
    task_prompt: str,
    skill_text: str | None,
    trial_id: str,
    is_control_group: bool,
    noise_level: str,
    seed: int,
    max_steps: int,
) -> Trajectory:
    client = anthropic.Anthropic()
    gate = EpsilonGate(EpsilonGateState())

    system_prompt = (
        "You are an autonomous DevOps agent operating a real bash shell one "
        "command at a time. After each command you'll be shown its stdout, "
        "stderr, and exit_code. Reason step by step in <monologue> tags, "
        "then give exactly one shell command in <action> tags. Stop and put "
        "DONE in <action> tags when the task is complete."
    )
    if not is_control_group and skill_text:
        system_prompt += f"\n\n--- SKILL: {skill_text}"

    sock = connect(socket_path)
    buf = bytearray()
    history: list[dict] = [{"role": "user", "content": f"Task: {task_prompt}"}]
    steps: list[Step] = []
    anomaly_pending = False

    for step_index in range(max_steps):
        suffix = gate.build_prompt_suffix(anomaly_pending)
        turn_input = history[-1]["content"] + suffix if suffix else history[-1]["content"]
        history[-1]["content"] = turn_input

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=history,
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        monologue = _extract_tag(text, "monologue")
        action_cmd = _extract_tag(text, "action") or "echo no-op"

        # This monologue is the model's reaction to the PREVIOUS step's
        # observation (anomaly_pending refers to that step, not to
        # whatever raw_obs we're about to receive for the action chosen
        # here). So the weight/acknowledgment it yields belongs on the
        # step that actually carried that anomaly — i.e. steps[-1] — not
        # on the step we're about to append below. Patch it back there.
        weight, raw_weight = gate.observe_monologue(monologue, anomaly_pending)
        if anomaly_pending and steps:
            steps[-1].heuristic_weight_estimated = weight
            steps[-1].raw_weight_estimated = raw_weight
            steps[-1].acknowledged_anomaly = _mentions_anomaly(monologue)

        if action_cmd.strip().upper() == "DONE":
            history.append({"role": "assistant", "content": text})
            break

        send_action(sock, action_cmd)
        raw_obs = recv_observation(sock, buf)

        atif_obs = AtifObservation(
            stdout=raw_obs.get("stdout", ""),
            stderr=raw_obs.get("stderr", ""),
            exit_code=raw_obs.get("exit_code", -1),
            latency_ms=raw_obs.get("latency_ms", 0),
            timed_out=raw_obs.get("timed_out", False),
        )
        kind = raw_obs.get("anomaly_kind")
        anomaly_kind = AnomalyKind(kind) if kind in {e.value for e in AnomalyKind} else AnomalyKind.NONE

        steps.append(
            Step(
                step_index=step_index,
                action=action_cmd,
                internal_monologue=monologue,
                observation=atif_obs,
                injected_anomaly=raw_obs.get("injected_anomaly", False),
                anomaly_kind=anomaly_kind,
                # Placeholders: this step's own anomaly (if any) hasn't been
                # seen by the model yet, so its weight/acknowledgment aren't
                # known until *next* iteration's monologue — patched above
                # on a future loop pass, or left as-is if the trial ends
                # before the model gets to react to it.
                heuristic_weight_estimated=1.0,
                raw_weight_estimated=None,
                acknowledged_anomaly=None,
            )
        )

        anomaly_pending = raw_obs.get("injected_anomaly", False)
        history.append({"role": "assistant", "content": text})
        history.append(
            {
                "role": "user",
                "content": (
                    f"stdout: {atif_obs.stdout}\nstderr: {atif_obs.stderr}\n"
                    f"exit_code: {atif_obs.exit_code}"
                ),
            }
        )

        if raw_obs.get("done"):
            break

    send_action(sock, "__EXIT__")
    sock.close()

    meta = TrialMetadata(
        trial_id=trial_id,
        agent_version=MODEL,
        skill_target=skill_text or "none",
        is_control_group=is_control_group,
        noise_level=noise_level,
        seed=seed,
        task_success=_infer_success(steps),
        max_steps=max_steps,
    )
    return Trajectory(metadata=meta, steps=steps)


def _extract_tag(text: str, tag: str) -> str:
    start, end = f"<{tag}>", f"</{tag}>"
    if start not in text or end not in text:
        return ""
    return text.split(start, 1)[1].split(end, 1)[0].strip()


_ANOMALY_KEYWORDS = re.compile(
    r"\b(warn\w*|error\w*|stderr|unexpected|anomal\w*|retry|retries|retried)\b",
    re.IGNORECASE,
)


def _mentions_anomaly(monologue: str) -> bool:
    # Word-boundary match so substrings like "errorless" or "warnpath"
    # (or unrelated code the model quotes) can't false-positive this.
    return bool(_ANOMALY_KEYWORDS.search(monologue))


def _infer_success(steps: list[Step]) -> bool:
    if not steps:
        return False
    return steps[-1].observation.exit_code == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default="/tmp/episteme.sock")
    parser.add_argument("--task", required=True)
    parser.add_argument("--skill-file", default=None, help="Path to SKILL.md for the skill-enabled trial")
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--noise", default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    skill_text = None
    if args.skill_file and not args.control:
        with open(args.skill_file) as f:
            skill_text = f.read()

    traj = run_trial(
        socket_path=args.socket,
        task_prompt=args.task,
        skill_text=skill_text,
        trial_id=args.trial_id,
        is_control_group=args.control,
        noise_level=args.noise,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    traj.to_json_file(args.out)
    print(f"wrote {args.out}")
        
