from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:  # pragma: no cover - allows standalone use/testing
    class BaseCallbackHandler:  # type: ignore
        pass

EPSILON = 0.02  # optimal small positive weight floor
WEIGHT_RE = re.compile(r"w\s*\(\s*E_?t?\s*\)\s*[:=]\s*([01](?:\.\d+)?)", re.IGNORECASE)

CONFIDENCE_PROMPT_SUFFIX = (
    "\n\n[episteme-gate] Before proceeding, state your updated belief as "
    "w(E_t) = <number between 0 and 1> reflecting how much weight you assign "
    "to the most recent stderr/exit_code output. A value of exactly 0 means "
    "'this observation cannot possibly be relevant, ever' — only use it if "
    "you can justify why that's mathematically, not just practically, true."
)

OVERRIDE_SUFFIX_TEMPLATE = (
    "\n\n[episteme-gate: epsilon-floor enforced] Your stated w(E_t) was "
    "{raw:.4f}, which the harness has floored to {clamped:.4f}. You must "
    "acknowledge the prior stderr output explicitly and adjust your stated "
    "probability of task success before issuing your next action."
)


def clamp_weight(raw: float, epsilon: float = EPSILON) -> float:
    """Hard mathematical floor/ceiling — this is the actual epsilon-gate."""
    return max(epsilon, min(1.0 - epsilon, raw))


@dataclass
class EpsilonGateState:
    """Per-trial running state the callback needs across steps."""
    epsilon: float = EPSILON
    collapsed_events: int = 0          # times the agent tried to state exactly 0 or 1
    total_anomaly_events: int = 0
    last_raw_weight: Optional[float] = None
    last_clamped_weight: Optional[float] = None
    pending_override: Optional[str] = None
    step_log: list[dict[str, Any]] = field(default_factory=list)


class EpsilonGate(BaseCallbackHandler):
    """
    LangChain callback handler. Wire it in as one of the `callbacks=[...]`
    passed to your agent executor. It only acts on steps that the harness
    has flagged as `injected_anomaly=True` (that flag comes from the STH/EID
    over the socket, and should be pushed into `state` by your agent loop
    before each generation call — see run_paired_trial.py).
    """

    def __init__(self, state: Optional[EpsilonGateState] = None):
        self.state = state or EpsilonGateState()

    def build_prompt_suffix(self, anomaly_pending: bool) -> str:
        """Call this from your agent loop to build the next turn's prompt."""
        if self.state.pending_override:
            suffix = self.state.pending_override
            self.state.pending_override = None
            return suffix
        if anomaly_pending:
            return CONFIDENCE_PROMPT_SUFFIX
        return ""

    def observe_monologue(self, monologue: str, anomaly_pending: bool) -> float:
        """
        Parse the agent's stated w(E_t) out of its monologue, clamp it, and
        stage an override suffix for the *next* turn if clamping had to act.
        Returns the (possibly clamped) weight to record in the ATIF step.
        """
        if not anomaly_pending:
            return 1.0  # no anomaly this step: weight is not meaningfully constrained

        match = WEIGHT_RE.search(monologue)
        raw = float(match.group(1)) if match else 0.0  # silent zero if agent never states one
        self.state.total_anomaly_events += 1
        self.state.last_raw_weight = raw

        clamped = clamp_weight(raw, self.state.epsilon)
        self.state.last_clamped_weight = clamped

        if raw <= 0.0 or raw >= 1.0:
            self.state.collapsed_events += 1
            self.state.pending_override = OVERRIDE_SUFFIX_TEMPLATE.format(
                raw=raw, clamped=clamped
            )

        self.state.step_log.append(
            {"raw_weight": raw, "clamped_weight": clamped, "collapsed": raw in (0.0, 1.0)}
        )
        return clamped

    # --- LangChain hook points -------------------------------------------------
    def on_llm_end(self, response, **kwargs) -> None:  # type: ignore[override]
        """
        Best-effort auto-wire: if the agent framework surfaces raw text here,
        we still require the caller to tell us `anomaly_pending` via
        observe_monologue() directly, since BaseCallbackHandler has no notion
        of the STH's anomaly flag. This hook is a no-op placeholder to keep
        the class a valid drop-in callback; real scoring happens through
        observe_monologue(), called explicitly from the agent loop.
        """
        return None

    def collapse_rate(self) -> float:
        if self.state.total_anomaly_events == 0:
            return 0.0
        return self.state.collapsed_events / self.state.total_anomaly_events
