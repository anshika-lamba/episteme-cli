from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AnomalyKind(str, Enum):
    NONE = "none"
    NON_DETERMINISTIC_WARNING = "non_deterministic_warning"
    TRANSIENT_STATE_ERROR = "transient_state_error"
    BLACK_SWAN = "black_swan"


class Observation(BaseModel):
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    latency_ms: int = Field(default=0, ge=0)


class Step(BaseModel):
    step_index: int = Field(ge=0)
    action: str
    internal_monologue: str = ""
    observation: Observation
    injected_anomaly: bool = False
    anomaly_kind: AnomalyKind = AnomalyKind.NONE

    # The agent's own stated (or ε-gate-clamped) weight on this observation.
    # Must stay strictly inside (0, 1) once the ε-gate has processed it —
    # see epsilon_gate.py for how this floor/ceiling is enforced upstream.
    heuristic_weight_estimated: float = Field(ge=0.0, le=1.0)

    # Did the agent's monologue explicitly acknowledge an injected anomaly
    # before proceeding? Filled in by the audit engine during scoring, not
    # by the agent itself, to avoid the agent gaming its own label.
    acknowledged_anomaly: Optional[bool] = None

    @field_validator("heuristic_weight_estimated")
    @classmethod
    def warn_on_hard_zero_or_one(cls, v: float) -> float:
        # We don't hard-reject here (a raw, pre-gate agent trace SHOULD be
        # able to record a true zero — that's the failure mode we're
        # trying to detect). The audit engine treats exact 0.0/1.0 values
        # as evidence of heuristic collapse rather than rejecting them.
        return v


class TrialMetadata(BaseModel):
    trial_id: str
    agent_version: str
    skill_target: str
    is_control_group: bool
    noise_level: str = "none"
    seed: int = 0
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    task_success: Optional[bool] = None
    max_steps: int = 50


class Trajectory(BaseModel):
    """Top-level ATIF document — one per trial."""
    metadata: TrialMetadata
    steps: list[Step] = Field(default_factory=list)

    def to_json_file(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_json_file(cls, path: str) -> "Trajectory":
        with open(path, "r") as f:
            return cls.model_validate_json(f.read())
