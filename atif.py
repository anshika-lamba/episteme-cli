"""ATIF -- Agent Trajectory Interchange Format (schema 2.2).

One JSONL line per trial: {"metadata": TrialMetadata, "steps": [Step, ...]}.

Relevance bookkeeping (important -- this was the source of an off-by-one):
  * `Step.stated_relevance`  -- the number the model emitted *alongside this
    step's command*. By the prompt's definition it rates the LATEST observation,
    i.e. the PREVIOUS step's stdout/stderr.
  * `Step.anomaly_response_relevance` -- set only on steps where an anomaly was
    injected: the relevance the model stated in its NEXT valid response, i.e.
    its rating of THIS anomaly. All relevance metrics use this field.
  * `Step.next_action_behavior` -- heuristic label of the next valid action
    (retry / investigate / ignore / stop), same alignment as above.
Both derived fields are (re)computable from raw steps: metrics.attach_anomaly_responses().
"""
import json
from dataclasses import dataclass, field, asdict, fields as _dc_fields
from typing import List, Optional, Dict, Any

SCHEMA_VERSION = "2.2"


@dataclass
class Observation:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False


@dataclass
class Step:
    step_index: int
    action: str
    internal_monologue: str = ""
    observation: Observation = field(default_factory=Observation)
    injected_anomaly: bool = False
    anomaly_kind: str = "none"
    stated_relevance: Optional[float] = None
    next_action_behavior: Optional[str] = None
    acknowledged_anomaly: Optional[bool] = None
    output_tokens: int = 0
    latency_ms: int = 0
    anomaly_response_relevance: Optional[float] = None  # schema 2.2 (see module docstring)

    @property
    def is_valid_action(self) -> bool:
        return self.action not in ("PARSE_FAILED", "PROVIDER_ERROR")


@dataclass
class TrialMetadata:
    trial_id: str
    agent_version: str
    task_name: str
    condition: str
    prompt_variant: str
    seed: int
    task_success: bool = False
    scheduled_anomalies: Dict[int, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    provider_name: str = ""            # schema 2.2
    aborted_reason: Optional[str] = None  # schema 2.2: set when a ProviderError/quota ended the trial early
    started_at: str = ""               # schema 2.2: ISO-8601 UTC


@dataclass
class Trajectory:
    metadata: TrialMetadata
    steps: List[Step] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        step_names = {f.name for f in _dc_fields(Step)}
        obs_names = {f.name for f in _dc_fields(Observation)}
        meta_names = {f.name for f in _dc_fields(TrialMetadata)}
        steps = []
        for s in data.get("steps", []):
            obs_raw = s.get("observation", {}) or {}
            obs = Observation(**{k: v for k, v in obs_raw.items() if k in obs_names})
            step = Step(**{k: v for k, v in s.items() if k in step_names and k != "observation"})
            step.observation = obs
            steps.append(step)
        meta_data = {k: v for k, v in (data.get("metadata", {}) or {}).items() if k in meta_names}
        if "scheduled_anomalies" in meta_data and meta_data["scheduled_anomalies"] is not None:
            meta_data["scheduled_anomalies"] = {int(k): v for k, v in meta_data["scheduled_anomalies"].items()}
        meta = TrialMetadata(**meta_data)
        return cls(metadata=meta, steps=steps)
