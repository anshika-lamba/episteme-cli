#!/usr/bin/env python3
"""Phase 4 step 3: Cohen's kappa between your labels and the judge (and the heuristic).

    python kappa.py --labels labeling/labels.csv --judge labeling/judge_output.jsonl \
                    --sample labeling/sample.jsonl --json labeling/kappa.json

Primary: kappa(human_addressed, judge_addressed) -- must be >= 0.6 (PREREG) before
the judge is trusted. Also reported:
  * kappa(human_acknowledged, judge_acknowledged)
  * kappa(human_addressed, heuristic_addressed)  where heuristic_addressed =
    behavior.py label in {retry, investigate, stop}. If THIS already clears 0.6
    the heuristic alone is defensible and the LLM judge is optional.
  * percent agreement, confusion matrices, bootstrap 95% CI (item resampling),
    and per-anomaly-kind kappas (small n, informational only).
"""
import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any

from judge import load_judge_output, JUDGE_PROMPT_VERSION

TRUE_SET = {"1", "true", "yes", "y", "t"}
FALSE_SET = {"0", "false", "no", "n", "f"}


def parse_label(v: str) -> Optional[bool]:
    v = (v or "").strip().lower()
    if v in TRUE_SET:
        return True
    if v in FALSE_SET:
        return False
    return None


def cohen_kappa(a: List[bool], b: List[bool]) -> Optional[float]:
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1 - pe) < 1e-12:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def kappa_ci(a: List[bool], b: List[bool], B: int, rng: random.Random) -> Tuple[Optional[float], Optional[float]]:
    n = len(a)
    if n < 5:
        return None, None
    vals = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        k = cohen_kappa([a[i] for i in idx], [b[i] for i in idx])
        if k is not None:
            vals.append(k)
    vals.sort()
    return vals[int(0.025 * (len(vals) - 1))], vals[int(math.ceil(0.975 * (len(vals) - 1)))]


def confusion(a: List[bool], b: List[bool]) -> Dict[str, int]:
    c = Counter((x, y) for x, y in zip(a, b))
    return {"human1_other1": c[(True, True)], "human1_other0": c[(True, False)], "human0_other1": c[(False, True)], "human0_other0": c[(False, False)]}


def report(name: str, a: List[bool], b: List[bool], B: int, rng: random.Random, threshold: float) -> Dict[str, Any]:
    k = cohen_kappa(a, b)
    lo, hi = kappa_ci(a, b, B, rng)
    agree = sum(1 for x, y in zip(a, b) if x == y) / len(a) if a else None
    res = {"n": len(a), "kappa": k, "ci95": [lo, hi], "percent_agreement": agree, "confusion": confusion(a, b),
           "human_positive_rate": sum(a) / len(a) if a else None, "other_positive_rate": sum(b) / len(b) if b else None}
    verdict = "n/a" if k is None else ("PASS" if k >= threshold else "FAIL")
    ci = f"[{lo:.2f}, {hi:.2f}]" if lo is not None else "[--]"
    print(f"{name:42s} n={len(a):3d} kappa={k if k is None else round(k, 3)!s:6s} CI95={ci:14s} agree={agree if agree is None else round(agree, 3)} {verdict}")
    print(f"    confusion (human x other): {res['confusion']}  base rates human={res['human_positive_rate'] if a else None}, other={res['other_positive_rate'] if b else None}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="labeling/labels.csv")
    ap.add_argument("--judge", default="labeling/judge_output.jsonl")
    ap.add_argument("--sample", default="labeling/sample.jsonl")
    ap.add_argument("--json", default="labeling/kappa.json")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    sample = {json.loads(l)["sample_id"]: json.loads(l) for l in open(args.sample, encoding="utf-8") if l.strip()}
    judge = load_judge_output(args.judge)
    human: Dict[str, Dict[str, Optional[bool]]] = {}
    with open(args.labels, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            human[row["sample_id"]] = {"acknowledged": parse_label(row.get("human_acknowledged", "")),
                                       "addressed": parse_label(row.get("human_addressed", ""))}
    ids_all = sorted(sample)
    unlabeled = [i for i in ids_all if i not in human or human[i]["addressed"] is None]
    unjudged = [i for i in ids_all if i not in judge]
    if unlabeled:
        print(f"[warn] {len(unlabeled)} items without a human 'addressed' label (skipped): {unlabeled[:8]}{'...' if len(unlabeled) > 8 else ''}")
    if unjudged:
        print(f"[warn] {len(unjudged)} items without a judge output (skipped): {unjudged[:8]}{'...' if len(unjudged) > 8 else ''}")

    ids = [i for i in ids_all if i in human and human[i]["addressed"] is not None and i in judge]
    out: Dict[str, Any] = {"n_sample": len(ids_all), "n_scored": len(ids), "threshold": args.threshold,
                           "prompt_version": JUDGE_PROMPT_VERSION, "judge_model": next((judge[i].get("judge_model") for i in ids), None)}
    if not ids:
        print("nothing to score yet")
        return 1
    ha = [human[i]["addressed"] for i in ids]
    ja = [judge[i]["addressed"] for i in ids]
    print(f"\n== Cohen's kappa (threshold {args.threshold}) ==")
    out["addressed_human_vs_judge"] = report("addressed: human vs LLM judge (PRIMARY)", ha, ja, args.boot, rng, args.threshold)

    ids_ack = [i for i in ids if human[i]["acknowledged"] is not None]
    if ids_ack:
        out["acknowledged_human_vs_judge"] = report("acknowledged: human vs LLM judge", [human[i]["acknowledged"] for i in ids_ack],
                                                    [judge[i]["acknowledged"] for i in ids_ack], args.boot, rng, args.threshold)
    ids_h = [i for i in ids_all if i in human and human[i]["addressed"] is not None and sample[i].get("heuristic_behavior")]
    if ids_h:
        heur = [sample[i]["heuristic_behavior"] in ("retry", "investigate", "stop") for i in ids_h]
        out["addressed_human_vs_heuristic"] = report("addressed: human vs behavior.py heuristic", [human[i]["addressed"] for i in ids_h], heur, args.boot, rng, args.threshold)
    ids_jh = [i for i in ids if sample[i].get("heuristic_behavior")]
    if ids_jh:
        out["addressed_judge_vs_heuristic"] = report("addressed: LLM judge vs heuristic (info)", [judge[i]["addressed"] for i in ids_jh],
                                                     [sample[i]["heuristic_behavior"] in ("retry", "investigate", "stop") for i in ids_jh], args.boot, rng, args.threshold)

    print("\n== per anomaly kind (primary, informational) ==")
    by_kind: Dict[str, List[str]] = defaultdict(list)
    for i in ids:
        by_kind[sample[i]["anomaly_kind"]].append(i)
    out["per_kind"] = {}
    for kind in sorted(by_kind):
        ks = by_kind[kind]
        k = cohen_kappa([human[i]["addressed"] for i in ks], [judge[i]["addressed"] for i in ks])
        agree = sum(1 for i in ks if human[i]["addressed"] == judge[i]["addressed"]) / len(ks)
        out["per_kind"][kind] = {"n": len(ks), "kappa": k, "agreement": agree}
        print(f"{kind:16s} n={len(ks):3d} kappa={k if k is None else round(k, 3)} agree={agree:.2f}")

    disagreements = [{"sample_id": i, "human": human[i]["addressed"], "judge": judge[i]["addressed"], "kind": sample[i]["anomaly_kind"],
                      "next_command": sample[i]["next_command"], "rationale": judge[i].get("rationale", "")} for i in ids if human[i]["addressed"] != judge[i]["addressed"]]
    out["disagreements"] = disagreements
    if disagreements:
        print(f"\n== {len(disagreements)} disagreements (read these before revising the judge prompt) ==")
        for d in disagreements[:15]:
            print(f"{d['sample_id']} {d['kind']:16s} human={int(d['human'])} judge={int(d['judge'])} next={d['next_command'][:50]!r} :: {d['rationale'][:90]}")

    k = out["addressed_human_vs_judge"]["kappa"]
    out["primary_kappa"] = k
    out["passed"] = bool(k is not None and k >= args.threshold)
    print(f"\nPRIMARY kappa(addressed) = {k:.3f} -> {'PASS' if out['passed'] else 'FAIL'} (>= {args.threshold} required; n={len(ids)})")
    if not out["passed"]:
        print("If FAIL: revise judge.py prompt (bump JUDGE_PROMPT_VERSION), re-run run_judge.py --no-resume, re-run kappa.py.\n"
              "If it still fails: report it as a limitation; relevance-based metrics do not depend on the judge.")
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
