import time
import json
import re
import requests
import os
from typing import Dict, Any, Tuple

class ProviderError(Exception): pass
class ParseError(ProviderError): pass
class QuotaExceededError(ProviderError): pass

class RateLimiter:
    def __init__(self, rpm: int, tpm_limit: int, rpd: int):
        self.rpm = rpm
        self.tpm_limit = tpm_limit
        self.rpd_limit = int(rpd * 0.9)
        self.window_calls, self.window_tokens, self.daily_calls = [], [], 0
        
    def wait_if_needed(self, estimated_input_chars: int):
        if self.daily_calls >= self.rpd_limit: raise QuotaExceededError("DAILY_REQUEST_CAP reached.")
        est_tokens = estimated_input_chars // 4
        if est_tokens >= self.tpm_limit: raise ProviderError(f"Request est_tokens ({est_tokens}) exceeds budget ({self.tpm_limit}).")
        now = time.time()
        self.window_calls = [t for t in self.window_calls if now - t < 60]
        self.window_tokens = [(t, tok) for (t, tok) in self.window_tokens if now - t < 60]
        current_tokens = sum(tok for _, tok in self.window_tokens)
        
        while len(self.window_calls) >= self.rpm or (current_tokens + est_tokens) >= self.tpm_limit:
            time.sleep(2.0)
            now = time.time()
            self.window_calls = [t for t in self.window_calls if now - t < 60]
            self.window_tokens = [(t, tok) for (t, tok) in self.window_tokens if now - t < 60]
            current_tokens = sum(tok for _, tok in self.window_tokens)
            
    def record_call(self, actual_input_tokens: int):
        now = time.time()
        self.window_calls.append(now)
        self.window_tokens.append((now, actual_input_tokens))
        self.daily_calls += 1

def extract_last_json(text: str, variant: str) -> dict:
    fences = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    parsed = None
    if fences:
        for fence in reversed(fences):
            try:
                parsed = json.loads(fence)
                break
            except json.JSONDecodeError: pass
    if parsed is None:
        idx = text.rfind('}')
        while idx != -1 and parsed is None:
            open_idx = text.rfind('{', 0, idx)
            while open_idx != -1:
                try:
                    parsed = json.loads(text[open_idx:idx+1])
                    break
                except json.JSONDecodeError: open_idx = text.rfind('{', 0, open_idx)
            idx = text.rfind('}', 0, idx)
    if parsed is None: raise ParseError("No valid JSON object found in response.")
    if not isinstance(parsed, dict) or "command" not in parsed or not isinstance(parsed["command"], str):
        raise ParseError("JSON missing or invalid 'command' string.")
    if variant != "no_numeric":
        if "relevance" not in parsed: raise ParseError("JSON missing 'relevance'.")
        rel = parsed["relevance"]
        if not isinstance(rel, (int, float)) or not (0.0 <= float(rel) <= 1.0):
            raise ParseError("JSON 'relevance' must be a float between 0.0 and 1.0.")
        parsed["relevance"] = float(rel)
    else: parsed["relevance"] = None
    return parsed
