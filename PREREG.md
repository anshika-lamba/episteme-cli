# Preregistration: Epistemic Rigidity in LLM Agents

## 1. Hypotheses
*   **H1 (Confidence Collapse):** Under a neutral prompt, LLM agents in a shell environment will exhibit a non-zero "exact-zero collapse rate," assigning mathematically absolute 0.0 relevance to actionable anomaly signals.
*   **H2 (Verbal-Behavioral Gap):** There will be a measurable divergence between an agent's stated relevance and its behavioral execution.
*   **H3 (Intervention Efficacy):** Explicit methodological instructions (Skill) will significantly reduce the down-collapse rate compared to a control group and a placebo.

## 2. Experimental Design
*   **Models:** gemma-4-26b-a4b-it, gemma-4-31b-it.
*   **Conditions:** Control, Real Skill (epistemic hygiene), Placebo Skill.
*   **Prompt Variants:** Original (demand-heavy), Neutral, No-Numeric (behavior only).

## 3. Metrics & Analysis
*   **Primary Metric:** Down-Collapse Rate (stated relevance <= 0.02 on `should_act` anomalies).
*   **Judge Validation:** LLM Judge (gemini-3.6-flash) evaluated via Cohen's Kappa against a 50-sample human-labeled dataset, requiring kappa > 0.60 for validity.
