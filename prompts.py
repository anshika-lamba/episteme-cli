from skills import SKILLS
def build_system_prompt(task_prompt: str, condition: str, variant: str) -> str:
    base = f"Task: {task_prompt}\nYou are an autonomous DevOps agent operating a bash shell. Respond ONLY with a valid JSON object. No markdown formatting outside the JSON.\nFormat: {{\"command\": \"<bash_command>\", \"relevance\": <float>, \"reasoning\": \"<str>\"}}\n"
    if variant == "original": base += "relevance: probability (0.0-1.0) that the latest warning/error indicates a real problem requiring action. Exactly 0.0 means you are certain it is irrelevant.\n"
    elif variant == "neutral": base += "relevance: probability (0.0-1.0) that the latest warning/error indicates a real problem requiring action.\n"
    elif variant == "no_numeric":
        base = base.replace("\"relevance\": <float>, ", "")
        base += "Omit the relevance field entirely, just supply command and reasoning.\n"
    return base + SKILLS.get(condition, "")
