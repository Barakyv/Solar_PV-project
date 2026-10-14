"""Stage 5: explicit, logged cleaning rules (curtailment, inverter stops, faults, stuck values).

Spec: PROJECT_MASTER.md §7 stage 5. The night rule uses true-UTC solar geometry and only zeroes
small residual power (assumption A-10); real production is never zeroed (test T-16, finding F-01b).
Output: cleaning_policy_final.yaml, cleaning_logs/, cleaned data.
"""
