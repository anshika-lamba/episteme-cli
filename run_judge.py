import argparse, json, random, os
from atif import Trajectory
from judge import evaluate_acknowledgment
from providers import GeminiProvider

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-n", type=int, default=50)
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--rpm", type=int, default=3)
    parser.add_argument("--tpm", type=int, default=8000)
    parser.add_argument("--rpd", type=int, default=18)
    args = parser.parse_args()

    all_anomalous_steps = []
    if os.path.exists(args.input):
        with open(args.input, "r") as f:
            for line in f:
                if not line.strip(): continue
                traj = Trajectory.from_dict(json.loads(line))
                for step in traj.steps:
                    if step.injected_anomaly and step.action != "PARSE_FAILED":
                        step._trial_id = traj.metadata.trial_id
                        all_anomalous_steps.append(step)
                        
    if not all_anomalous_steps: return
        
    random.seed(42)
    sample_size = min(args.sample_n, len(all_anomalous_steps))
    sampled_steps = random.sample(all_anomalous_steps, sample_size)
    
    provider = GeminiProvider(args.model, args.rpm, args.tpm, args.rpd)
    evaluate_acknowledgment(sampled_steps, provider)
    
    with open(args.output, "w") as f:
        for step in sampled_steps:
            f.write(json.dumps({
                "trial_id": step._trial_id, "step_index": step.step_index,
                "anomaly_kind": step.anomaly_kind, "reasoning": step.internal_monologue,
                "acknowledged_anomaly": step.acknowledged_anomaly
            }) + "\n")
            
if __name__ == "__main__":
    main()
