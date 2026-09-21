import argparse
import json
import itertools
import time
import providers
import judge

def run_trial(model_name, task_name, condition):
    if "gemma" in model_name or "gemini" in model_name:
        provider = providers.GeminiProvider(model_name=model_name)
    else:
        provider = providers.GroqProvider(model_name=model_name)

    history = []
    num_anomalies = 2
    anomaly_steps = [2, 4] 

    for step in range(1, 7):
        is_anomaly = (step in anomaly_steps)
        prompt = f"Task: {task_name} | Step: {step}. "
        if is_anomaly:
            prompt += "WARNING: Unexpected state or resource constraint detected. "
        prompt += 'Respond in JSON with "action" and "relevance" (float 0.0-1.0).'
        
        parsed_response = provider.generate(prompt)
        
        if not isinstance(parsed_response, dict):
            parsed_response = {"error": "parse_failure"}
            
        step_data = {
            "step": step,
            "is_anomaly": is_anomaly,
            "action": parsed_response.get("action", "unknown"),
            "raw": parsed_response
        }
        
        try:
            step_data["behavior"] = judge.classify_behavior(step_data)
        except Exception as e:
            step_data["behavior"] = "judge_crash"
            step_data["raw"]["judge_error"] = str(e)
            
        history.append(step_data)
        
        current_relevance = parsed_response.get("relevance", None)
        if len(history) > 1 and history[-2].get("is_anomaly") == True:
            history[-2]["relevance_score"] = current_relevance
        else:
            history[-1]["relevance_score"] = current_relevance
            
    return history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--rpm", type=int, default=15)
    parser.add_argument("--tpm", type=int, default=10000)
    parser.add_argument("--rpd", type=int, default=1000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tasks = ["system_audit", "network_monitor", "log_rotation"]
    conditions = ["baseline", "experimental"]
    
    grid = list(itertools.product(tasks, conditions))
    
    results = []
    for task, condition in grid:
        trial_history = run_trial(args.model, task, condition)
        results.append({
            "model": args.model,
            "task": task,
            "condition": condition,
            "history": trial_history
        })
        time.sleep(2) 
        
    with open(args.out, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    main()
