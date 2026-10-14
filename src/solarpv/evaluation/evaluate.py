"""Authoritative evaluation entry point (stage 8, PROJECT_MASTER.md §6.9-§6.10).

Reads predictions (all test hours, one file per model/seed), takes y_true from the feature file,
applies the daytime mask, and writes outputs/07_evaluation/benchmark_summary.csv plus DM, bootstrap,
route (true AC as truth on both routes, C-16), seasonal and common-mask tables.
"""
