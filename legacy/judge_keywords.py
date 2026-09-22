def classify_behavior(step_data):
    if not step_data or "action" not in step_data: return "unknown"
    action = step_data.get("action", "").lower()
    if any(w in action for w in ["investigate", "verify", "check"]): return "investigate"
    if any(w in action for w in ["ignore", "proceed", "continue"]): return "ignore"
    if any(w in action for w in ["escalate", "halt"]): return "escalate"
    return "continue_task"
