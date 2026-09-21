import os

os.makedirs(".github/workflows", exist_ok=True)

FILES = {
    "run_experiment.py": r'''
import argparse, json, os, sys
from itertools import product
from providers import GeminiProvider, QuotaExceededError
from runner import run_trial
from tasks import TASKS
from atif import Trajectory

def get_grids():
    tasks = list(TASKS.keys())
    main_grid = list(product(tasks, ["control", "real_skill", "placebo_skill"], ["neutral"], range(10)))
    abl_a = list(product(tasks, ["control"], ["original"], range(5)))
    abl_c = list(product(tasks, ["control"], ["no_numeric"], range(5)))
    return list(set(main_grid + abl_a + abl_c))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--rpm", type=int, required=True)
    parser.add_argument("--tpm", type=int, required=True)
    parser.add_argument("--rpd", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    completed = set()
    if os.path.exists(args.out):
        with open(args.out, "r") as f:
            for line in f:
                if not line.strip(): continue
                traj = Trajectory.from_dict(json.loads(line))
                meta = traj.metadata
                completed.add((meta.task_name, meta.condition, meta.prompt_variant, meta.seed))
                
    full_grid = get_grids()
    pending = [t for t in full_grid if t not in completed]
    
    total_trials = len(pending)
    est_calls = total_trials * 6
    print(f"--- Pre-flight for {args.model} ---")
    print(f"Pending Trials   : {total_trials}")
    print(f"Est. API Calls   : {est_calls}")
    print(f"------------------------------------")
    
    if total_trials == 0: return

    provider = GeminiProvider(args.model, args.rpm, args.tpm, args.rpd)
    with open(args.out, "a") as f:
        for task, cond, var, seed in pending:
            print(f"Running: Task={task} | Cond={cond} | Var={var} | Seed={seed}...")
            try:
                traj = run_trial(provider, task, cond, var, seed)
                f.write(traj.to_json() + "\n")
                f.flush()
            except QuotaExceededError as e:
                print(f"[!] Quota Reached: {e}. Exiting cleanly.")
                sys.exit(0)
            except Exception as e:
                print(f"[ERROR] Trial failed: {e}")

if __name__ == "__main__":
    main()
''',
    ".github/workflows/run_experiment.yml": r'''
name: AAAI-UC Experiment Runner
on:
  workflow_dispatch:
    inputs:
      model_a:
        description: 'Model A ID'
        default: 'gemma-4-26b-a4b-it'
        required: true
      model_b:
        description: 'Model B ID'
        default: 'gemma-4-31b-it'
        required: true
      rpm:
        description: 'RPM Limit'
        default: '21'
      tpm:
        description: 'TPM Limit'
        default: '11200'
      rpd:
        description: 'RPD Limit'
        default: '12960'

jobs:
  run-trials:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        model: [ "${{ github.event.inputs.model_a }}", "${{ github.event.inputs.model_b }}" ]
      max-parallel: 2
      fail-fast: false
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install pytest requests
      - uses: actions/download-artifact@v4
        continue-on-error: true
        with:
          name: results-${{ matrix.model }}
          path: ./
      - run: |
          python run_experiment.py \
            --model ${{ matrix.model }} \
            --rpm ${{ github.event.inputs.rpm }} \
            --tpm ${{ github.event.inputs.tpm }} \
            --rpd ${{ github.event.inputs.rpd }} \
            --out ${{ matrix.model }}_results.jsonl
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      - if: always()
        uses: actions/upload-artifact@v4
        with:
          name: results-${{ matrix.model }}
          path: ${{ matrix.model }}_results.jsonl
          retention-days: 14
''',
    "run_judge.py": r'''
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
''',
    "PREREG.md": r'''
# Preregistration: Epistemic Rigidity in LLM Agents

## 1. Hypotheses
*   **H1 (Confidence Collapse):** Under a neutral prompt, LLM agents in a shell environment will exhibit a non-zero "exact-zero collapse rate," assigning mathematically absolute 0.0 relevance to actionable anomaly signals.
*   **H2 (Verbal-Behavioral Gap):** There will be a measurable divergence between an agent's stated relevance and its behavioral execution.
*   **H3 (Intervention Efficacy):** Explicit methodological instructions (Skill) will significantly reduce the down-collapse rate compared to a control group and a placebo.

## 2. Experimental Design
*   **Models:** gemma-4-26b-a4b-it, gemma-4-31b-it.
*   **Conditions:** Control, Real Skill (epistemic hygiene), Placebo Skill.
*   **Prompt Variants:** Original (demand-heavy), Neutral, No-Numeric (behavior only).

## 3. Metrics & Analysis
*   **Primary Metric:** Down-Collapse Rate (stated relevance <= 0.02 on `should_act` anomalies).
*   **Judge Validation:** LLM Judge (gemini-3.6-flash) evaluated via Cohen's Kappa against a 50-sample human-labeled dataset, requiring kappa > 0.60 for validity.
'''
}

for filename, content in FILES.items():
    with open(filename, "w") as f:
        f.write(content.strip() + "\n")
print("Phase 3 files generated successfully.")

