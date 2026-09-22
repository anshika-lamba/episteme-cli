"""LLM provider layer for the epistemic-rigidity benchmark.

Design goals
------------
* One `generate(history, variant)` contract for every backend, returning
  `(parsed_json, output_tokens, latency_ms)` and raising `ParseError` when the
  model did not produce a valid agent JSON (recorded, then excluded; PREREG §4).
* Plain HTTPS via `requests` -- no vendor SDKs (keeps Termux installs trivial and
  removes a whole class of "SDK changed its API" landmines).
* Rate limiting that is *real*: RPM sliding window, TPM budget (Groq's 6K TPM
  bites on long trajectories), RPD cap, optional minimum inter-call spacing
  (Mistral ~1 req/s) and a persistent monthly ledger (Cohere trial: 1000/month).
* Fail fast and loudly on non-retryable errors (bad model name, bad key) --
  every provider has its own landmines and silent retries hide them.

Public API (used by runner.py / run_grid.py / judge.py / tests)
---------------------------------------------------------------
    ProviderError, ParseError, QuotaExceededError
    RateLimiter(rpm, tpm_limit, rpd, min_interval_s=0.0)
    extract_last_json(text, variant) -> dict
    BaseProvider.generate(history, variant, retries=None) -> (dict, int, int)
    BaseProvider.complete(prompt) -> str          # raw text, used by the judge
    REGISTRY, make_provider(name, model=None, **overrides), list_models(name)
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

try:  # requests is the only third-party dependency of the whole pipeline
    import requests
except ImportError:  # pragma: no cover - allows offline unit tests of pure helpers
    requests = None  # type: ignore


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ProviderError(Exception):
    """Non-recoverable provider failure for this request (auth, bad model, budget)."""


class ParseError(ProviderError):
    """The model answered, but not with a valid agent JSON object."""


class QuotaExceededError(ProviderError):
    """A hard daily/monthly cap was reached; the whole run should stop cleanly."""


class _TransientError(Exception):
    """Internal: 429/5xx/timeouts that deserve a backoff-and-retry."""

    def __init__(self, msg: str, retry_after: Optional[float] = None):
        super().__init__(msg)
        self.retry_after = retry_after


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
def estimate_tokens(chars: int) -> int:
    """Conservative chars->tokens estimate (4 chars/token) used before we know real usage."""
    return max(1, int(chars) // 4)


class RateLimiter:
    """Sliding-window RPM + TPM limiter with a daily request cap.

    `wait_if_needed(estimated_input_chars)` blocks until the request fits in the
    per-minute budgets, raises `ProviderError` if a single request can never fit
    (est tokens > tpm_limit) and `QuotaExceededError` once `rpd` calls were made.
    `record_call(actual_input_tokens)` must be called after each request.
    """

    def __init__(self, rpm: int, tpm_limit: int, rpd: int, min_interval_s: float = 0.0):
        self.rpm = int(rpm)
        self.tpm_limit = int(tpm_limit)
        self.rpd_limit = int(rpd)
        self.min_interval_s = float(min_interval_s)
        self.window_calls: List[float] = []
        self.window_tokens: List[Tuple[float, int]] = []
        self.daily_calls = 0
        self.total_wait_s = 0.0
        self._last_call = 0.0
        self._day = _dt.date.today()

    def _roll_day(self) -> None:
        today = _dt.date.today()
        if today != self._day:  # long runs cross midnight; caps reset per day (provider clocks vary!)
            self._day = today
            self.daily_calls = 0

    def wait_if_needed(self, estimated_input_chars: int) -> int:
        est_tokens = estimate_tokens(estimated_input_chars)
        if est_tokens > self.tpm_limit:
            raise ProviderError(f"Request est_tokens ({est_tokens}) exceeds budget ({self.tpm_limit}).")
        self._roll_day()
        if self.daily_calls >= self.rpd_limit:
            raise QuotaExceededError("DAILY_REQUEST_CAP reached.")
        while True:
            now = time.time()
            self.window_calls = [t for t in self.window_calls if now - t < 60.0]
            self.window_tokens = [(t, n) for t, n in self.window_tokens if now - t < 60.0]
            current_tokens = sum(n for _, n in self.window_tokens)
            wait = 0.0
            if len(self.window_calls) >= self.rpm:
                wait = max(wait, 60.0 - (now - self.window_calls[0]) + 0.05)
            if self.window_tokens and current_tokens + est_tokens > self.tpm_limit:
                wait = max(wait, 60.0 - (now - self.window_tokens[0][0]) + 0.05)
            if self.min_interval_s and (now - self._last_call) < self.min_interval_s:
                wait = max(wait, self.min_interval_s - (now - self._last_call))
            if wait <= 0:
                return est_tokens
            wait = min(wait, 61.0)
            self.total_wait_s += wait
            time.sleep(wait)

    def record_call(self, actual_input_tokens: int) -> None:
        now = time.time()
        self.window_calls.append(now)
        self.window_tokens.append((now, int(actual_input_tokens or 0)))
        self.daily_calls += 1
        self._last_call = now


class CallLedger:
    """Persistent per-calendar-month call counter (Cohere trial keys: 1000 calls/month, all endpoints)."""

    def __init__(self, name: str, monthly_cap: int, ledger_dir: str = ".quota"):
        self.name = name
        self.monthly_cap = int(monthly_cap)
        self.path = os.path.join(ledger_dir, f"{name}.json")
        os.makedirs(ledger_dir, exist_ok=True)

    @property
    def month_key(self) -> str:
        return _dt.date.today().strftime("%Y-%m")

    def _load(self) -> Dict[str, int]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path) as f:
                return json.load(f)
        except Exception:
            return {}

    def used(self) -> int:
        return int(self._load().get(self.month_key, 0))

    def remaining(self) -> int:
        return max(0, self.monthly_cap - self.used())

    def check(self) -> None:
        if self.used() >= self.monthly_cap:
            raise QuotaExceededError(
                f"MONTHLY_CALL_CAP reached for {self.name}: {self.used()}/{self.monthly_cap} in {self.month_key}."
            )

    def record(self, n: int = 1) -> None:
        data = self._load()
        data[self.month_key] = int(data.get(self.month_key, 0)) + n
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------- #
# JSON extraction / validation (agent responses)
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _balanced_object_spans(text: str) -> List[str]:
    """Return every top-level {...} span, respecting string literals."""
    spans: List[str] = []
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start:i + 1])
                start = None
    return spans


def _last_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Last JSON object in `text` (prefers fenced blocks). None if nothing parses."""
    candidates = _FENCE_RE.findall(text) or _balanced_object_spans(text)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue  # e.g. prose in braces after the real answer
        if isinstance(obj, dict):
            return obj
    return None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Public, validation-free variant of the extractor (used by the judge)."""
    return _last_json_object(text)


def _coerce_relevance(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def extract_last_json(text: str, variant: str) -> Dict[str, Any]:
    """Parse + validate an agent response. Raises ParseError with a specific reason.

    Rules (PREREG §4 -- malformed output is recorded, then excluded):
      * take the LAST JSON object in the reply (a trailing `{}` therefore fails);
      * `command` must be a non-empty string;
      * `relevance` must be a number in [0, 1] unless variant == "no_numeric",
        in which case it is forced to None (that arm has no numeric channel).
    """
    parsed = _last_json_object(text or "")
    if parsed is None:
        raise ParseError("No valid JSON object found in response.")
    command = parsed.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ParseError("JSON missing or invalid 'command' string.")
    reasoning = parsed.get("reasoning", "")
    parsed["reasoning"] = reasoning if isinstance(reasoning, str) else json.dumps(reasoning)
    if variant == "no_numeric":
        parsed["relevance"] = None
        return parsed
    if "relevance" not in parsed or parsed.get("relevance") is None:
        raise ParseError("JSON missing 'relevance'.")
    rel = _coerce_relevance(parsed.get("relevance"))
    if rel is None or not (0.0 <= rel <= 1.0):
        raise ParseError("JSON 'relevance' must be a float between 0.0 and 1.0.")
    parsed["relevance"] = rel
    return parsed


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
_QUOTA_HINTS = ("per day", "daily", "per_day", "perday", "month", "quota exceeded", "monthly")


def _sleep_backoff(attempt: int, retry_after: Optional[float]) -> float:
    if retry_after and retry_after > 0:
        return min(float(retry_after) + 0.5, 120.0)
    return min(2.0 * (2 ** attempt), 60.0)


class BaseProvider:
    """Shared plumbing: limiter, ledger, retries, timing, parsing."""

    name = "base"
    env_key = ""
    default_model = ""

    def __init__(self, model_name: str, rpm: int = 15, tpm: int = 10000, rpd: int = 1000,
                 api_key: Optional[str] = None, max_output_tokens: int = 400, temperature: float = 0.0,
                 transport_retries: int = 4, min_interval_s: float = 0.0, monthly_cap: Optional[int] = None,
                 ledger_dir: str = ".quota", json_mode: bool = False, timeout_s: int = 60):
        if requests is None:
            raise ProviderError("The 'requests' package is required: pip install requests")
        self.model_name = model_name or self.default_model
        self.api_key = api_key or os.environ.get(self.env_key, "")
        if not self.api_key:
            raise ProviderError(f"{self.name}: missing API key (set {self.env_key}).")
        self.limiter = RateLimiter(rpm, tpm, rpd, min_interval_s)
        self.ledger = CallLedger(self.name, monthly_cap, ledger_dir) if monthly_cap else None
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        self.transport_retries = int(transport_retries)
        self.json_mode = bool(json_mode)
        self.timeout_s = int(timeout_s)
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_ratelimit_headers: Dict[str, str] = {}
        self.session = requests.Session()

    # -- subclass hooks ----------------------------------------------------- #
    def _call(self, history: List[Dict[str, str]]) -> Tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens). Raise _TransientError/ProviderError."""
        raise NotImplementedError

    # -- shared -------------------------------------------------------------- #
    def _classify_http(self, status: int, body_text: str, headers: Any) -> None:
        low = (body_text or "").lower()
        if status == 429:
            if any(h in low for h in _QUOTA_HINTS):
                raise QuotaExceededError(f"{self.name}: quota exhausted (HTTP 429): {body_text[:300]}")
            retry_after = None
            try:
                retry_after = float(headers.get("retry-after")) if headers and headers.get("retry-after") else None
            except (TypeError, ValueError):
                retry_after = None
            if retry_after is None:
                m = re.search(r'retrydelay"?\s*:\s*"?(\d+(?:\.\d+)?)s', low)  # Gemini puts retryDelay in the body
                retry_after = float(m.group(1)) if m else None
            raise _TransientError(f"HTTP 429: {body_text[:200]}", retry_after)
        if status >= 500:
            raise _TransientError(f"HTTP {status}: {body_text[:200]}")
        if status >= 400:
            raise ProviderError(f"{self.name}/{self.model_name}: HTTP {status}: {body_text[:500]}")

    def _post(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        try:
            resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout_s)
        except requests.RequestException as e:  # DNS, timeouts, resets
            raise _TransientError(f"network error: {e}")
        rl = {k.lower(): v for k, v in resp.headers.items() if k.lower().startswith("x-ratelimit")}
        if rl:
            self.last_ratelimit_headers = rl  # Groq/Mistral/Cohere expose the real per-minute/per-day caps here
        if resp.status_code != 200:
            self._classify_http(resp.status_code, resp.text, resp.headers)
        try:
            return resp.json()
        except ValueError:
            raise _TransientError(f"non-JSON body: {resp.text[:200]}")

    def _raw_generate(self, history: List[Dict[str, str]], retries: Optional[int]) -> Tuple[str, int, int, int]:
        chars = sum(len(m.get("content", "")) for m in history) + 256
        est = self.limiter.wait_if_needed(chars)
        if self.ledger:
            self.ledger.check()
        attempts = self.transport_retries if retries is None else int(retries)
        last_err: Optional[Exception] = None
        for attempt in range(attempts + 1):
            t0 = time.time()
            try:
                text, in_tok, out_tok = self._call(history)
            except _TransientError as e:
                last_err = e
                self.limiter.record_call(est)
                if self.ledger:
                    self.ledger.record()
                if attempt >= attempts:
                    break
                time.sleep(_sleep_backoff(attempt, e.retry_after))
                continue
            latency_ms = int((time.time() - t0) * 1000)
            self.limiter.record_call(in_tok or est)
            if self.ledger:
                self.ledger.record()
            self.total_calls += 1
            self.total_input_tokens += int(in_tok or est)
            self.total_output_tokens += int(out_tok or 0)
            return text, in_tok, out_tok, latency_ms
        raise ProviderError(f"{self.name}: gave up after {attempts + 1} attempts: {last_err}")

    def generate(self, history: List[Dict[str, str]], variant: str, retries: Optional[int] = None) -> Tuple[Dict[str, Any], int, int]:
        text, _in_tok, out_tok, latency_ms = self._raw_generate(history, retries)
        parsed = extract_last_json(text, variant)  # ParseError propagates to the runner
        parsed["_raw_text"] = text[:2000]
        return parsed, int(out_tok or 0), latency_ms

    def complete(self, prompt: str, retries: Optional[int] = None) -> str:
        """Single-turn raw completion (judge / diagnostics)."""
        text, _i, _o, _l = self._raw_generate([{"role": "user", "content": prompt}], retries)
        return text

    def usage_summary(self) -> Dict[str, Any]:
        return {
            "provider": self.name, "model": self.model_name, "calls": self.total_calls,
            "input_tokens": self.total_input_tokens, "output_tokens": self.total_output_tokens,
            "limiter_wait_s": round(self.limiter.total_wait_s, 1),
            "monthly_used": self.ledger.used() if self.ledger else None,
            "monthly_cap": self.ledger.monthly_cap if self.ledger else None,
        }


class _OpenAICompatProvider(BaseProvider):
    base_url = ""

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _call(self, history):
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": m["role"], "content": m["content"]} for m in history],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post(f"{self.base_url}/chat/completions", payload, self._headers())
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise _TransientError(f"unexpected response shape: {json.dumps(data)[:200]}")
        usage = data.get("usage") or {}
        return text, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)

    def list_models(self) -> List[str]:
        resp = self.session.get(f"{self.base_url}/models", headers=self._headers(), timeout=self.timeout_s)
        if resp.status_code != 200:
            raise ProviderError(f"{self.name}: list models failed HTTP {resp.status_code}: {resp.text[:300]}")
        return sorted(m.get("id", "") for m in resp.json().get("data", []))


class GroqProvider(_OpenAICompatProvider):
    name = "groq"
    env_key = "GROQ_API_KEY"
    default_model = "allam-2-7b"
    base_url = "https://api.groq.com/openai/v1"


class MistralProvider(_OpenAICompatProvider):
    name = "mistral"
    env_key = "MISTRAL_API_KEY"
    default_model = "ministral-8b-latest"
    base_url = "https://api.mistral.ai/v1"


class GeminiProvider(BaseProvider):
    name = "gemini"
    env_key = "GEMINI_API_KEY"
    default_model = "gemini-2.5-flash"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _call(self, history):
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in history]
        gen_cfg: Dict[str, Any] = {"temperature": self.temperature, "maxOutputTokens": self.max_output_tokens}
        if self.json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        payload = {"contents": contents, "generationConfig": gen_cfg}
        url = f"{self.base_url}/models/{self.model_name}:generateContent"
        data = self._post(url, payload, {"Content-Type": "application/json", "x-goog-api-key": self.api_key})
        cands = data.get("candidates") or []
        if not cands:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise ProviderError(f"gemini: empty response ({reason})")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata") or {}
        return text, int(usage.get("promptTokenCount") or 0), int(usage.get("candidatesTokenCount") or 0)

    def list_models(self) -> List[str]:
        resp = self.session.get(f"{self.base_url}/models", headers={"x-goog-api-key": self.api_key},
                                params={"pageSize": 200}, timeout=self.timeout_s)
        if resp.status_code != 200:
            raise ProviderError(f"gemini: list models failed HTTP {resp.status_code}: {resp.text[:300]}")
        return sorted(m.get("name", "").replace("models/", "") for m in resp.json().get("models", []))


class CohereProvider(BaseProvider):
    name = "cohere"
    env_key = "COHERE_API_KEY"
    default_model = "command-r7b-12-2024"
    base_url = "https://api.cohere.com"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _call(self, history):
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": m["role"], "content": m["content"]} for m in history],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post(f"{self.base_url}/v2/chat", payload, self._headers())
        content = (data.get("message") or {}).get("content") or []
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        tokens = ((data.get("usage") or {}).get("tokens")) or {}
        return text, int(tokens.get("input_tokens") or 0), int(tokens.get("output_tokens") or 0)

    def list_models(self) -> List[str]:
        resp = self.session.get(f"{self.base_url}/v1/models", headers=self._headers(),
                                params={"page_size": 100, "endpoint": "chat"}, timeout=self.timeout_s)
        if resp.status_code != 200:
            raise ProviderError(f"cohere: list models failed HTTP {resp.status_code}: {resp.text[:300]}")
        return sorted(m.get("name", "") for m in resp.json().get("models", []))


# --------------------------------------------------------------------------- #
# Registry -- free-tier defaults as of 2026-09; VERIFY against your console,
# they change without notice. Every value is overridable from run_grid.py.
# --------------------------------------------------------------------------- #
REGISTRY: Dict[str, Dict[str, Any]] = {
    "groq": dict(cls=GroqProvider, rpm=30, tpm=6000, rpd=1000, min_interval_s=0.0,
                 notes="allam-2-7b: 30 RPM / 6K TPM / 1K RPD. llama-3.1-8b-instant has ~14.4K RPD if you need volume."),
    "gemini": dict(cls=GeminiProvider, rpm=10, tpm=250000, rpd=250, min_interval_s=0.0,
                   notes="2.5-flash free tier ~10 RPM / 250 RPD (post Dec-2025 cuts); 2.5-flash-lite ~15-30 RPM / 1000 RPD."),
    "mistral": dict(cls=MistralProvider, rpm=30, tpm=500000, rpd=100000, min_interval_s=1.1,
                    notes="Experiment tier ~1 req/s (enforced via min_interval_s), 500K TPM, 1B tokens/month."),
    "cohere": dict(cls=CohereProvider, rpm=20, tpm=100000, rpd=1000, min_interval_s=0.0, monthly_cap=1000,
                   notes="Trial key: 20 RPM and a HARD 1000 calls/month across all endpoints (tracked in .quota/)."),
}


def make_provider(name: str, model: Optional[str] = None, **overrides: Any) -> BaseProvider:
    """Instantiate a provider by registry name. `mock` is handled in mock_provider.py."""
    if name == "mock":
        from mock_provider import ScriptedMockProvider  # local import: keeps providers.py dependency-free
        return ScriptedMockProvider(model_name=model or "mock-model", **{k: v for k, v in overrides.items() if k in ("seed", "behavior")})
    if name not in REGISTRY:
        raise ProviderError(f"Unknown provider '{name}'. Choose from: {', '.join(REGISTRY)} or mock.")
    spec = dict(REGISTRY[name])
    cls = spec.pop("cls")
    spec.pop("notes", None)
    spec.update({k: v for k, v in overrides.items() if v is not None})
    return cls(model_name=model or cls.default_model, **spec)


def list_models(name: str) -> List[str]:
    prov = make_provider(name)
    if not hasattr(prov, "list_models"):
        return []
    return prov.list_models()  # type: ignore[attr-defined]
