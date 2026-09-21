import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

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
    schema_version: str = "2.1"

@dataclass
class Trajectory:
    metadata: TrialMetadata
    steps: List[Step] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        steps = []
        for s in data.get("steps", []):
            obs = Observation(**s.get("observation", {}))
            step_data = {k: v for k, v in s.items() if k != "observation"}
            step = Step(**step_data)
            step.observation = obs
            steps.append(step)
        meta_data = data.get("metadata", {})
        if "scheduled_anomalies" in meta_data:
            meta_data["scheduled_anomalies"] = {
                int(k): v for k, v in meta_data["scheduled_anomalies"].items()
            }
        meta = TrialMetadata(**meta_data)
        return cls(metadata=meta, steps=steps)
