def classify_behavior(prev_cmd: str, curr_cmd: str, reasoning: str, curr_is_done: bool) -> str:
    curr_cmd_lower = curr_cmd.strip().lower()
    prev_cmd_lower = prev_cmd.strip().lower()
    reasoning_lower = reasoning.lower()

    if curr_is_done or curr_cmd_lower in ["exit", "done"]:
        if "ignore" in reasoning_lower or "irrelevant" in reasoning_lower or "proceed" in reasoning_lower: return "ignore"
        return "stop"
    if curr_cmd_lower == prev_cmd_lower: return "retry"
    if "ignore" in reasoning_lower or "proceed" in reasoning_lower or "irrelevant" in reasoning_lower: return "ignore"
    return "investigate"
