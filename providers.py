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
import inspect
import json
import os
import random
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
    """Persistent call counters per provider: per calendar month AND per UTC day.

    Why: an in-memory RPD counter resets when the process restarts while the
    server's window does not -> persistent 429s (this bit the pilot twice). Every
    provider seeds its RateLimiter.daily_calls from here at start-up. The monthly
    cap is only enforced when `monthly_cap` is set (Cohere trial: 1000/month, all
    endpoints). Files live in .quota/ (git-ignored, machine-local): run a capped
    provider from ONE machine.
    """

    def __init__(self, name: str, monthly_cap: Optional[int] = None, ledger_dir: str = ".quota"):
        self.name = name
        self.monthly_cap = int(monthly_cap) if monthly_cap else None
        self.path = os.path.join(ledger_dir, f"{name}.json")
        os.makedirs(ledger_dir, exist_ok=True)

    @staticmethod
    def _utc_today() -> _dt.date:
        return _dt.datetime.now(_dt.timezone.utc).date()

    @property
    def month_key(self) -> str:
        return self._utc_today().strftime("%Y-%m")

    @property
    def day_key(self) -> str:
        return self._utc_today().isoformat()

    def _load(self) -> Dict[str, Dict[str, int]]:
        if not os.path.exists(self.path):
            return {"months": {}, "days": {}}
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {"months": {}, "days": {}}
        if "months" not in data:  # legacy flat {"YYYY-MM": n}
            data = {"months": {k: v for k, v in data.items() if isinstance(v, int)}, "days": {}}
        data.setdefault("months", {})
        data.setdefault("days", {})
        return data

    def used(self) -> int:
        return int(self._load()["months"].get(self.month_key, 0))

    def daily_used(self) -> int:
        return int(self._load()["days"].get(self.day_key, 0))

    def remaining(self) -> Optional[int]:
        return None if self.monthly_cap is None else max(0, self.monthly_cap - self.used())

    def check(self) -> None:
        if self.monthly_cap is not None and self.used() >= self.monthly_cap:
            raise QuotaExceededError(
                f"MONTHLY_CALL_CAP reached for {self.name}: {self.used()}/{self.monthly_cap} in {self.month_key}."
            )

    def record(self, n: int = 1) -> None:
        data = self._load()
        data["months"][self.month_key] = int(data["months"].get(self.month_key, 0)) + n
        data["days"][self.day_key] = int(data["days"].get(self.day_key, 0)) + n
        # keep the file small
        data["days"] = {k: v for k, v in sorted(data["days"].items())[-45:]}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------- #
# JSON extraction / validation (agent responses)
# --------------------------------------------------------------------------- #
# Strict on purpose. Fuzzy repair (fences, regex extraction, string-to-float)
# masks structural collapse and would invent a relevance the model did not
# validly state. A json.loads failure is a formatting collapse: ParseError,
# parse_fail, rel=null. It does not abort the grid.
_RNG = random.Random()


def _json_number(value):
    """A JSON number. Booleans are ints in Python and are not relevance scores.
    Strings are not coerced: \"0.5\" is a formatting/schema failure, not 0.5."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def extract_last_json(text: str, variant: str) -> Dict[str, Any]:
    """Parse an agent reply with json.loads and nothing else.

    No markdown stripping, no regex extraction, no trailing-comma repair, no
    quote normalisation, no string-to-float coercion. If the reply is not a
    single JSON object, that is a formatting collapse (ParseError). The runner
    records parse_fail and a null relevance. It does not invent a number.
    """
    raw = text if isinstance(text, str) else ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise ParseError("Formatting collapse: response is not valid JSON.")
    if not isinstance(parsed, dict):
        raise ParseError("Formatting collapse: JSON value is not an object.")
    command = parsed.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ParseError("JSON missing or invalid 'command' string.")
    parsed["command"] = command.strip()
    reasoning = parsed.get("reasoning", "")
    parsed["reasoning"] = reasoning if isinstance(reasoning, str) else json.dumps(reasoning)
    if variant == "no_numeric":
        parsed["relevance"] = None
        return parsed
    if "relevance" not in parsed or parsed.get("relevance") is None:
        raise ParseError("JSON missing 'relevance'.")
    rel = _json_number(parsed.get("relevance"))
    if rel is None or not (0.0 <= rel <= 1.0):
        raise ParseError("JSON 'relevance' must be a JSON number between 0.0 and 1.0.")
    parsed["relevance"] = rel
    return parsed


def _balanced_object_spans(text: str) -> List[str]:
    """Top-level {...} spans, respecting strings. Judge only — never agent scoring."""
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


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Judge-only helper. Agent relevance scoring must use extract_last_json.

    The judge prompt asks for a bare object; a short preamble is tolerated here
    so a judge formatting slip does not silently drop a labeling batch. This
    function is not called on agent replies, and it does not repair commas,
    quotes, or numeric strings.
    """
    raw = text or ""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    for span in reversed(_balanced_object_spans(raw)):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
_QUOTA_HINTS = ("per day", "daily", "per_day", "perday", "month", "quota exceeded", "monthly")


# 8 retries after the first attempt (9 tries). Daily/monthly quota 429s are NOT
# retried — they cannot clear in 10–30s, and retrying them burns the night.
DEFAULT_TRANSPORT_RETRIES = 8


def _sleep_backoff(attempt: int, retry_after: Optional[float]) -> float:
    """Jittered exponential backoff for transient 429/503/network errors.

    With no Retry-After, attempt 0 sleeps uniform(10, 30) seconds (the floor that
    keeps a thundering herd off a just-reset window) and later attempts double
    that, capped at 180s. A server Retry-After is honored, plus a short jitter.
    """
    if retry_after and retry_after > 0:
        return min(max(float(retry_after), 1.0) + _RNG.uniform(0.25, 1.5), 180.0)
    return min(_RNG.uniform(10.0, 30.0) * (2 ** attempt), 180.0)


class BaseProvider:
    """Shared plumbing: limiter, ledger, retries, timing, parsing."""

    name = "base"
    env_key = ""
    default_model = ""

    def __init__(self, model_name: str, rpm: int = 15, tpm: int = 10000, rpd: int = 1000,
                 api_key: Optional[str] = None, max_output_tokens: int = 400, temperature: float = 0.0,
                 transport_retries: int = DEFAULT_TRANSPORT_RETRIES, min_interval_s: float = 0.0, monthly_cap: Optional[int] = None,
                 ledger_dir: str = ".quota", json_mode: bool = False, timeout_s: int = 60):
        if requests is None:
            raise ProviderError("The 'requests' package is required: pip install requests")
        self.model_name = model_name or self.default_model
        self.api_key = api_key or os.environ.get(self.env_key, "")
        if not self.api_key:
            raise ProviderError(f"{self.name}: missing API key (set {self.env_key}).")
        self.limiter = RateLimiter(rpm, tpm, rpd, min_interval_s)
        self.ledger = CallLedger(self.name, monthly_cap, ledger_dir)
        self.limiter.daily_calls = self.ledger.daily_used()  # survive process restarts (see CallLedger)
        self.event_sink: Optional[Any] = None  # callable(dict) -> None; run_grid installs a JSONL writer
        self.context: Dict[str, Any] = {}      # trial_id / step, set by the runner for event attribution
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

    # -- harness event log (taxonomy of everything that is NOT the model's decision) -- #
    def set_context(self, **kw: Any) -> None:
        self.context = {k: v for k, v in kw.items() if v is not None}

    def _emit(self, layer: str, kind: str, **fields: Any) -> None:
        if self.event_sink is None:
            return
        ev = {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), "provider": self.name,
              "model": self.model_name, "layer": layer, "kind": kind}
        ev.update(self.context)
        ev.update({k: v for k, v in fields.items() if v is not None})
        try:
            self.event_sink(ev)
        except Exception:  # logging must never take the run down
            pass

    # -- shared -------------------------------------------------------------- #
    def _classify_http(self, status: int, body_text: str, headers: Any) -> None:
        low = (body_text or "").lower()
        if status == 429:
            if any(h in low for h in _QUOTA_HINTS):
                self._emit("quota", "server_quota_exhausted", http_status=429, body=body_text[:300])
                raise QuotaExceededError(f"{self.name}: quota exhausted (HTTP 429): {body_text[:300]}")
            retry_after = None
            try:
                retry_after = float(headers.get("retry-after")) if headers and headers.get("retry-after") else None
            except (TypeError, ValueError):
                retry_after = None
            if retry_after is None:
                m = re.search(r'retrydelay"?\s*:\s*"?(\d+(?:\.\d+)?)s', low)  # Gemini puts retryDelay in the body
                retry_after = float(m.group(1)) if m else None
            self._emit("transport", "http_429", http_status=429, retry_after=retry_after, body=body_text[:300])
            raise _TransientError(f"HTTP 429: {body_text[:200]}", retry_after)
        if status >= 500:
            self._emit("transport", "http_5xx", http_status=status, body=body_text[:300])
            raise _TransientError(f"HTTP {status}: {body_text[:200]}")
        if status >= 400:
            self._emit("transport", "http_4xx_fatal", http_status=status, body=body_text[:500])
            raise ProviderError(f"{self.name}/{self.model_name}: HTTP {status}: {body_text[:500]}")

    def _post(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        try:
            resp = self.session.post(url, json=payload, headers=headers, timeout=self.timeout_s)
        except requests.RequestException as e:  # DNS, timeouts, resets
            self._emit("transport", "network_error", error=str(e)[:300])
            raise _TransientError(f"network error: {e}")
        rl = {k.lower(): v for k, v in resp.headers.items() if k.lower().startswith("x-ratelimit")}
        if rl:
            self.last_ratelimit_headers = rl  # Groq/Mistral/Cohere expose the real per-minute/per-day caps here
        if resp.status_code != 200:
            self._classify_http(resp.status_code, resp.text, resp.headers)
        try:
            return resp.json()
        except ValueError:
            self._emit("transport", "non_json_body", http_status=resp.status_code, body=resp.text[:300])
            raise _TransientError(f"non-JSON body: {resp.text[:200]}")

    def _raw_generate(self, history: List[Dict[str, str]], retries: Optional[int]) -> Tuple[str, int, int, int]:
        chars = sum(len(m.get("content", "")) for m in history) + 256
        wait_before = self.limiter.total_wait_s
        try:
            est = self.limiter.wait_if_needed(chars)
        except QuotaExceededError as e:
            self._emit("quota", "client_daily_cap", detail=str(e), daily_calls=self.limiter.daily_calls)
            raise
        except ProviderError as e:
            self._emit("client", "oversize_request", detail=str(e), est_tokens=estimate_tokens(chars))
            raise
        waited = self.limiter.total_wait_s - wait_before
        if waited >= 5:
            self._emit("client", "throttle_wait", wait_s=round(waited, 1), est_tokens=est)
        try:
            self.ledger.check()
        except QuotaExceededError as e:
            self._emit("quota", "client_monthly_cap", detail=str(e))
            raise
        attempts = self.transport_retries if retries is None else int(retries)
        last_err: Optional[Exception] = None
        for attempt in range(attempts + 1):
            t0 = time.time()
            try:
                text, in_tok, out_tok = self._call(history)
            except _TransientError as e:
                last_err = e
                self.limiter.record_call(est)
                self.ledger.record()
                if attempt >= attempts:
                    break
                backoff = _sleep_backoff(attempt, e.retry_after)
                self._emit("transport", "retry", attempt=attempt + 1, backoff_s=round(backoff, 1), error=str(e)[:200])
                time.sleep(backoff)
                continue
            latency_ms = int((time.time() - t0) * 1000)
            self.limiter.record_call(in_tok or est)
            self.ledger.record()
            self.total_calls += 1
            self.total_input_tokens += int(in_tok or est)
            self.total_output_tokens += int(out_tok or 0)
            if attempt:
                self._emit("transport", "recovered", attempts=attempt + 1)
            return text, in_tok, out_tok, latency_ms
        self._emit("transport", "gave_up", attempts=attempts + 1, error=str(last_err)[:300])
        raise ProviderError(f"{self.name}: gave up after {attempts + 1} attempts: {last_err}")

    def generate(self, history: List[Dict[str, str]], variant: str, retries: Optional[int] = None) -> Tuple[Dict[str, Any], int, int]:
        text, _in_tok, out_tok, latency_ms = self._raw_generate(history, retries)
        try:
            parsed = extract_last_json(text, variant)  # ParseError propagates to the runner
        except ParseError as e:
            self._emit("model_output", "parse_failed", reason=str(e), raw=(text or "")[:300], out_tokens=int(out_tok or 0))
            raise
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
            "calls_today_utc": self.ledger.daily_used(), "monthly_used": self.ledger.used(),
            "monthly_cap": self.ledger.monthly_cap,
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
    """Gemini API (also serves the open-weight Gemma 3 models, which have far higher free RPD)."""
    name = "gemini"
    env_key = "GEMINI_API_KEY"
    default_model = "gemma-3-12b-it"  # decision 2026-09-22: size-matched Google family member, ~14.4K RPD
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
            self._emit("transport", "empty_response", detail=str(reason))
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
    "gemini": dict(cls=GeminiProvider, rpm=30, tpm=15000, rpd=14400, min_interval_s=0.0,
                   notes="gemma-3-12b-it ~30 RPM / 15K TPM / 14.4K RPD. gemini-2.5-flash is only ~10 RPM / 250 RPD on the free tier."),
    "mistral": dict(cls=MistralProvider, rpm=30, tpm=500000, rpd=100000, min_interval_s=1.1,
                    notes="Experiment tier ~1 req/s (enforced via min_interval_s), 500K TPM, 1B tokens/month."),
    "cohere": dict(cls=CohereProvider, rpm=20, tpm=100000, rpd=1000, min_interval_s=0.0, monthly_cap=1000,
                   notes="Trial key: 20 RPM and a HARD 1000 calls/month across all endpoints (tracked in .quota/)."),
}

# Per-model overrides of the provider defaults (free tier, 2026-09; verify against the console/headers).
MODEL_LIMITS: Dict[str, Dict[str, int]] = {
    "gemini-2.5-flash": dict(rpm=10, tpm=250000, rpd=250),
    "gemini-2.5-flash-lite": dict(rpm=15, tpm=250000, rpd=1000),
    "gemini-2.0-flash": dict(rpm=10, tpm=250000, rpd=250),
    "gemma-3-12b-it": dict(rpm=30, tpm=15000, rpd=14400),
    "gemma-3-27b-it": dict(rpm=30, tpm=15000, rpd=14400),
    "allam-2-7b": dict(rpm=30, tpm=6000, rpd=1000),
    "llama-3.1-8b-instant": dict(rpm=30, tpm=6000, rpd=14400),
    "llama-3.3-70b-versatile": dict(rpm=30, tpm=12000, rpd=1000),
}


def effective_limits(name: str, model: Optional[str]) -> Dict[str, Any]:
    """Registry defaults, overridden by MODEL_LIMITS when the model is known."""
    spec = {k: v for k, v in REGISTRY[name].items() if k not in ("cls", "notes")}
    m = model or REGISTRY[name]["cls"].default_model
    spec.update(MODEL_LIMITS.get(m, {}))
    return spec


# Knobs that only the offline mock understands; run_grid passes one shared overrides dict.
MOCK_ONLY_OVERRIDES = ("seed", "behavior", "parse_fail_p")


def make_provider(name: str, model: Optional[str] = None, **overrides: Any) -> BaseProvider:
    """Instantiate a provider by registry name. `mock` is handled in mock_provider.py.

    Unknown options for a real provider raise ProviderError instead of TypeError,
    and mock-only knobs (e.g. --mock-seed) are dropped silently so `--provider groq`
    and `--provider mock` accept the same CLI surface."""
    if name == "mock":
        from mock_provider import ScriptedMockProvider  # local import: keeps providers.py dependency-free
        kw = {k: v for k, v in overrides.items() if k in MOCK_ONLY_OVERRIDES and v is not None}
        return ScriptedMockProvider(model_name=model or "mock-model", **kw)
    if name not in REGISTRY:
        raise ProviderError(f"Unknown provider '{name}'. Choose from: {', '.join(REGISTRY)} or mock.")
    cls = REGISTRY[name]["cls"]
    accepted = set(inspect.signature(BaseProvider.__init__).parameters) - {"self"}
    spec = effective_limits(name, model)
    for k, v in overrides.items():
        if v is None or k in MOCK_ONLY_OVERRIDES:
            continue
        if k not in accepted:
            raise ProviderError(f"Unknown option '{k}' for provider '{name}'. Accepted: {', '.join(sorted(accepted))}.")
        spec[k] = v
    return cls(model_name=model or cls.default_model, **spec)


def list_models(name: str) -> List[str]:
    if name == "mock":
        raise ProviderError("mock has no model list")
    prov = make_provider(name)
    if not hasattr(prov, "list_models"):
        return []
    return prov.list_models()  # type: ignore[attr-defined]
