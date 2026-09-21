# AAAI-UC Epistemic Rigidity Benchmark

This repository contains an academic benchmarking pipeline designed to test LLM epistemic rigidity (exact-zero collapse) in automated agents executing resource-constrained tasks.

## Infrastructure
* **Workflow:** Automated execution via GitHub Actions (`.github/workflows/run_experiment.yml`).
* **Providers:** Supports Gemini and Groq (Llama-3) API backends with strict JSON parsing.
* **Evaluation:** Uses an independent `judge.py` to classify agent behavior based on action intent rather than reasoning keywords to prevent hallucinated false-positives.

## Analysis
To analyze the generated artifact from a GitHub Actions run:
```bash
python analyze.py --input results-<model>-<run_id>.jsonl


