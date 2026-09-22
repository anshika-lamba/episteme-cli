# legacy/ — superseded, kept for the record

These files are the pre-ATIF pipeline (synthetic 6-step prompts with a literal
`WARNING:` string, no sandbox, keyword judge) and the SDK-based `providers.py`
from commit d5d3f22 ("Made pipeline bulletproof against Groq/judge crashes").

That commit replaced the real `providers.py` (RateLimiter / ParseError /
validating `extract_last_json`) with this simplified one, which broke
`runner.py`, `mock_provider.py`, `run_judge.py` and the whole test suite.
They are **not runnable** against the current `providers.py` and are not part
of the experiment. The live pipeline is documented in the top-level README.
