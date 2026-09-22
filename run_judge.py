#!/usr/bin/env python3
"""Phase 4 step 2: run the LLM judge over the labeling sample (resumable).

    python run_judge.py --sample labeling/sample.jsonl --provider gemini [--model gemini-2.5-flash]
    python run_judge.py --sample labeling/sample.jsonl --provider mock   # offline smoke test

Use a judge model from a DIFFERENT family than the agents in the sample when you
can; self-judging inflates agreement with the agent's own framing. The script
warns if the judge model also appears as an agent model in the sample.
Each judged item = 1 API call (60 items ~ 60 calls; mind Cohere's monthly cap).
"""
import argparse
import json
import os
import sys
from collections import Counter

from judge import judge_records, JUDGE_PROMPT_VERSION
from providers import make_provider, ProviderError, REGISTRY


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="labeling/sample.jsonl")
    ap.add_argument("--out", default="labeling/judge_output.jsonl")
    ap.add_argument("--provider", required=True, choices=list(REGISTRY) + ["mock"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--rpm", type=int, default=None)
    ap.add_argument("--rpd", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not os.path.exists(args.sample):
        print(f"missing {args.sample}: run sample_for_labeling.py first", file=sys.stderr)
        return 1
    records = [json.loads(l) for l in open(args.sample, encoding="utf-8") if l.strip()]
    if args.limit:
        records = records[: args.limit]

    if args.provider == "mock":
        from mock_provider import MockJudgeProvider
        provider = MockJudgeProvider()
    else:
        try:
            provider = make_provider(args.provider, args.model, rpm=args.rpm, rpd=args.rpd, max_output_tokens=200)
        except ProviderError as e:
            print(f"[fatal] {e}", file=sys.stderr)
            return 2
    agent_models = {r.get("model") for r in records}
    if provider.model_name in agent_models:
        print(f"[warn] judge model {provider.model_name} is also an agent model in this sample (self-judging)", file=sys.stderr)

    print(f"judge={provider.name}/{provider.model_name} prompt={JUDGE_PROMPT_VERSION} items={len(records)} out={args.out}", file=sys.stderr)
    if args.no_resume and os.path.exists(args.out):
        os.remove(args.out)
    results = judge_records(records, provider, args.out, resume=not args.no_resume)
    c = Counter("error" if r.get("judge_error") else f"ack={r['acknowledged']},addr={r['addressed']}" for r in results)
    print(json.dumps({"judged_now": len(results), "breakdown": dict(c),
                      "usage": provider.usage_summary() if hasattr(provider, "usage_summary") else None}, indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
