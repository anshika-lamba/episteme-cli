# Preregistration: LLM Epistemic Rigidity

## 1. Hypothesis
Resource-constrained LLM agents will exhibit epistemic rigidity (exact-zero collapse) when encountering unexpected anomalies in standard IT operational tasks. Regardless of corporate alignment (Gemma vs Llama-3), agents will default to "ignore" or "continue_task" behaviors rather than halting or investigating.

## 2. Methodology
* **Task Structure:** 6-step sequential system operations (e.g., log rotation, system audit).
* **Anomalies:** 2 injected warnings at unexpected intervals (steps 2 and 4).
* **Models:** `gemini-1.5-flash` and `llama3-8b-8192`.
* **Prompting:** Zero-shot JSON generation requesting an `action` and `relevance` score (0.0 - 1.0).

## 3. Metrics
* **Behavior Distribution:** Categorical classification of the `action` output (investigate, ignore, escalate, continue_task) using a command-matching judge.
* **AUROC:** The area under the ROC curve comparing `relevance_score` outputs during anomaly steps (positives) vs standard steps (negatives).
* **Exact-Zero Gap:** Difference between mean anomaly relevance and mean baseline relevance.

## 4. Exclusions
Any trial step resulting in a malformed JSON output (`"error": "parse_failure"`) will be recorded for reliability tracking but strictly excluded from the AUROC and behavior distribution calculations.
