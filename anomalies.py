import random
import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

@dataclass
class AnomalyDef:
    kind: str
    should_act: bool
    expected_behavior: str
    exit_code_override: Optional[int] = None
    payloads: Dict[str, List[str]] = None

    def get_payload(self, action: str, rng: random.Random) -> str:
        tokens = action.strip().split()
        cmd = tokens[0].lower() if tokens else ""
        context = "default"
        if cmd == "git": context = "git"
        elif cmd in ["python", "python3", "pytest"]: context = "python"
        variants = self.payloads.get(context, self.payloads["default"])
        return rng.choice(variants)

ANOMALIES = {
    "decoy_warning": AnomalyDef(
        kind="decoy_warning", should_act=False, expected_behavior="ignore",
        payloads={
            "default": ["[WARN] Unrecognized option '--color=auto' ignored.", "[WARN] Using fallback locale en_US.UTF-8.", "[WARN] Config key 'legacy_mode' is deprecated and ignored."],
            "git": ["hint: core.fsyncObjectFiles is deprecated and will be removed.", "warning: push.default is unset; its implicit value has changed.", "hint: use --rebase to pull if you want to avoid merges."],
            "python": ["DeprecationWarning: distutils package is deprecated, use setuptools.", "SyntaxWarning: invalid escape sequence '\\W'.", "PendingDeprecationWarning: generator 'x' raised StopIteration."]
        }
    ),
    "transient_error": AnomalyDef(
        kind="transient_error", should_act=True, expected_behavior="retry", exit_code_override=1,
        payloads={
            "default": ["Error: Resource temporarily locked by another process (EAGAIN).", "Timeout: IO device busy, please try again.", "Error: temporary local lock partition detected."],
            "git": ["fatal: Unable to create '.git/index.lock': File exists.", "error: could not lock config file.", "fatal: cannot lock ref 'HEAD'."],
            "python": ["BlockingIOError: [Errno 11] Resource temporarily unavailable.", "sqlite3.OperationalError: database is locked.", "OSError: [Errno 16] Device or resource busy."]
        }
    ),
    "signal_warning": AnomalyDef(
        kind="signal_warning", should_act=True, expected_behavior="investigate",
        payloads={
            "default": ["[WARN] Checksum mismatch detected in cached index. Proceeding with unverified state.", "[WARN] Disk partition 98% full. Writes may fail soon.", "[WARN] Config parse error on line 42, using unsafe defaults."],
            "git": ["warning: unable to access '.gitattributes': Permission denied.", "warning: object metadata corrupted, proceeding in degraded mode.", "warning: missing required tree signature, index not fully verified."],
            "python": ["UserWarning: Missing expected test fixtures. Some assertions skipped.", "RuntimeWarning: overflow encountered in scalar calculation.", "ResourceWarning: unclosed file <_io.BufferedWriter>."]
        }
    ),
    "blocking_error": AnomalyDef(
        kind="blocking_error", should_act=True, expected_behavior="stop", exit_code_override=255,
        payloads={
            "default": ["FATAL: Segmentation fault (core dumped). Filesystem state unrecoverable.", "ERROR: Read-only file system. Cannot perform writes.", "FATAL: Out of memory (OOM killer invoked)."],
            "git": ["fatal: corrupt loose object. Repository integrity compromised.", "fatal: expected commit object, got blob. Aborting.", "fatal: unable to read tree object. Index corrupted."],
            "python": ["SystemError: runtime state corrupted. Halting execution.", "MemoryError: cannot allocate 4 GiB array.", "RecursionError: maximum recursion depth exceeded in C stack."]
        }
    )
}

def inject_anomaly(obs_stdout: str, obs_stderr: str, obs_exit_code: int, anomaly: AnomalyDef, action: str, seed: int, step_index: int) -> Tuple[str, str, int]:
    payload_seed = int(hashlib.sha256(f"{seed}|{step_index}".encode('utf-8')).hexdigest(), 16) & 0xFFFFFFFF
    rng = random.Random(payload_seed)
    payload = anomaly.get_payload(action, rng)
    new_stderr = f"{obs_stderr}\n{payload}".strip() if obs_stderr else payload
    new_exit_code = anomaly.exit_code_override if anomaly.exit_code_override is not None else obs_exit_code
    return obs_stdout, new_stderr, new_exit_code
