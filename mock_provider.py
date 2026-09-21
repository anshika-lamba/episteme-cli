from typing import Dict, Any, Tuple
from providers import ParseError

class MockProvider:
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self.call_count = 0
        
    def generate(self, history: list, variant: str, retries: int = 1) -> Tuple[Dict[str, Any], int, int]:
        self.call_count += 1
        if self.call_count == 1: cmd, rel, reasoning = "ls", 0.5, "Checking directory."
        elif self.call_count == 2: cmd, rel, reasoning = "retry_command", 0.1, "Transient error, retrying."
        else: cmd, rel, reasoning = "DONE", 0.0, "Task complete."
            
        parsed = {"command": cmd, "reasoning": reasoning}
        parsed["relevance"] = rel if variant != "no_numeric" else None
        return parsed, 25, 120
