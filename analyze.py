import json
import argparse
from collections import Counter

def calculate_auroc(positives, negatives):
    if not positives or not negatives: return 0.0
    concordant = 0
    for p in positives:
        for n in negatives:
            if p > n: concordant += 1
            elif p == n: concordant += 0.5
    return concordant / (len(positives) * len(negatives))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to results.jsonl")
    args = parser.parse_args()

    behaviors = Counter()
    anomalies = []
    baselines = []
    parse_errors = 0

    with open(args.input) as f:
        for line in f:
            trial = json.loads(line)
            for step in trial.get("history", []):
                if step.get("raw", {}).get("error") == "parse_failure":
                    parse_errors += 1
                    continue
                
                behaviors[step.get("behavior", "unknown")] += 1
                
                score = step.get("relevance_score")
                if isinstance(score, (int, float)):
                    if step.get("is_anomaly"):
                        anomalies.append(score)
                    else:
                        baselines.append(score)

    print("=== EXPERIMENT RESULTS ===")
    print(f"Parse Failures Excluded: {parse_errors}")
    print(f"Behavior Distribution: {dict(behaviors)}")
    
    auroc = calculate_auroc(anomalies, baselines)
    print(f"Relevance AUROC: {auroc:.3f}")
    
    mean_anom = sum(anomalies)/len(anomalies) if anomalies else 0
    mean_base = sum(baselines)/len(baselines) if baselines else 0
    print(f"Exact-Zero Gap: Mean Anomaly={mean_anom:.3f}, Mean Baseline={mean_base:.3f}")

if __name__ == "__main__":
    main()
