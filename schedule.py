import random
import hashlib
from typing import Dict
from anomalies import ANOMALIES

def generate_schedule(seed: int, task_name: str, expected_steps: int = 6, max_steps: int = 10) -> Dict[int, str]:
    hash_input = f"{seed}|{task_name}".encode('utf-8')
    task_seed = int(hashlib.sha256(hash_input).hexdigest(), 16) & 0xFFFFFFFF
    rng = random.Random(task_seed)
    
    valid_steps = list(range(1, min(expected_steps + 1, max_steps)))
    if not valid_steps:
        return {}
        
    if len(valid_steps) < 3:
        num_anomalies = rng.randint(1, len(valid_steps))
    else:
        num_anomalies = 3
        
    steps_to_inject = sorted(rng.sample(valid_steps, k=num_anomalies))
    kinds = list(ANOMALIES.keys())
    
    if num_anomalies >= 3:
        while True:
            chosen_kinds = [rng.choice(kinds) for _ in range(num_anomalies)]
            has_decoy = any(k == "decoy_warning" for k in chosen_kinds)
            has_should_act = any(ANOMALIES[k].should_act for k in chosen_kinds)
            blocking_count = sum(1 for k in chosen_kinds if k == "blocking_error")
            if has_decoy and has_should_act and blocking_count <= 1:
                break
        if "blocking_error" in chosen_kinds:
            chosen_kinds.remove("blocking_error")
            chosen_kinds.append("blocking_error")
    else:
        chosen_kinds = [rng.choice(kinds) for _ in range(num_anomalies)]
        
    schedule = {}
    for step, kind in zip(steps_to_inject, chosen_kinds):
        schedule[step] = kind
        
    return schedule
