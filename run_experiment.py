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
