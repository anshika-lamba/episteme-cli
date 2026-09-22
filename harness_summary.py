#!/usr/bin/env python3
"""Harness-failure taxonomy: everything that went wrong that was NOT the model's decision.

run_grid.py writes one JSON line per event to results/harness_{provider}_{model}.jsonl
(429s, 5xx, network errors, fatal 4xx, quota walls, oversize requests, throttle
waits, parse failures, retries/recoveries). This script aggregates them for the
research statement's "resource-constrained execution" section:

    python harness_summary.py "results/harness_*.jsonl"
    python harness_summary.py "results/harness_*.jsonl" --json results/harness_taxonomy.json

Layers: transport (HTTP/network), quota (daily/monthly caps, server or client
side), client (our own throttling / oversize requests), model_output (invalid
JSON), run (start/stop markers).
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional


class JsonlSink:
    """Appends events to a JSONL file and keeps them in memory for the end-of-run summary."""

    def __init__(self, path: str, run_meta: Optional[Dict[str, Any]] = None):
        self.path = path
        self.events: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "a")
        self._write({"layer": "run", "kind": "run_start", **(run_meta or {})})

    def _write(self, ev: Dict[str, Any]) -> None:
        import datetime as _dt
        ev.setdefault("ts", _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
        self._f.write(json.dumps(ev) + "\n")
        self._f.flush()

    def __call__(self, ev: Dict[str, Any]) -> None:
        self.events.append(ev)
        self._write(dict(ev))

    def close(self, summary: Optional[Dict[str, Any]] = None) -> None:
        try:
            self._write({"layer": "run", "kind": "run_end", **({"summary": summary} if summary else {})})
            self._f.close()
        except Exception:
            pass


def summarize_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_kind = Counter(f"{e.get('layer')}/{e.get('kind')}" for e in events if e.get("layer") != "run")
    by_status = Counter(str(e.get("http_status")) for e in events if e.get("http_status"))
    wait_s = sum(float(e.get("wait_s", 0) or 0) for e in events if e.get("kind") == "throttle_wait")
    backoff_s = sum(float(e.get("backoff_s", 0) or 0) for e in events if e.get("kind") == "retry")
    trials_hit = len({e.get("trial_id") for e in events if e.get("trial_id") and e.get("layer") in ("transport", "quota")})
    parse_reasons = Counter(e.get("reason") for e in events if e.get("kind") == "parse_failed")
    return {"n_events": sum(by_kind.values()), "by_kind": dict(by_kind), "by_http_status": dict(by_status),
            "throttle_wait_s": round(wait_s), "retry_backoff_s": round(backoff_s), "trials_with_transport_or_quota_events": trials_hit,
            "parse_failure_reasons": dict(parse_reasons.most_common(6))}


def load_events(patterns: List[str]) -> List[Dict[str, Any]]:
    events = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)) or [pat]:
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    events = load_events(args.inputs)
    if not events:
        print("no events", file=sys.stderr)
        return 1
    runs = sum(1 for e in events if e.get("kind") == "run_start")
    print(f"{len(events)} events from {runs} run(s)")
    per_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in events:
        if e.get("layer") != "run":
            per_model[f"{e.get('provider')}/{e.get('model')}"].append(e)
    out: Dict[str, Any] = {"per_model": {}, "all": summarize_events([e for e in events if e.get("layer") != "run"])}
    print(f"\n{'provider/model':34s} {'events':>6s} {'429':>5s} {'5xx':>5s} {'4xx!':>5s} {'quota':>5s} {'parse':>5s} {'gaveup':>6s} {'wait_s':>7s}")
    for key in sorted(per_model):
        evs = per_model[key]
        s = summarize_events(evs)
        k = s["by_kind"]
        quota = sum(v for kk, v in k.items() if kk.startswith("quota/"))
        print(f"{key:34s} {s['n_events']:6d} {k.get('transport/http_429', 0):5d} {k.get('transport/http_5xx', 0):5d} {k.get('transport/http_4xx_fatal', 0):5d} "
              f"{quota:5d} {k.get('model_output/parse_failed', 0):5d} {k.get('transport/gave_up', 0):6d} {s['throttle_wait_s'] + s['retry_backoff_s']:7d}")
        out["per_model"][key] = s
    print("\nby kind (all):")
    for kind, n in sorted(out["all"]["by_kind"].items(), key=lambda kv: -kv[1]):
        print(f"  {kind:36s} {n}")
    if out["all"]["parse_failure_reasons"]:
        print("parse-failure reasons:")
        for r, n in out["all"]["parse_failure_reasons"].items():
            print(f"  {n:5d}  {r}")
    fatal = [e for e in events if e.get("kind") in ("http_4xx_fatal", "server_quota_exhausted", "gave_up")]
    if fatal:
        print("\nfirst fatal/quota events (what the landmines looked like):")
        for e in fatal[:8]:
            print(f"  {e.get('ts')} {e.get('provider')}/{e.get('model')} {e.get('kind')} {e.get('http_status', '')} {str(e.get('body') or e.get('error') or e.get('detail'))[:140]!r}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
