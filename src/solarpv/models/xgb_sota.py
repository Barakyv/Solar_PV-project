"""XGB-Opt (PROJECT_MASTER.md §6.11). CLI: --n_trials N --seed S [--dev].

Optuna TPE on validation daytime nRMSE, explicit monotone constraints (D-15), early stopping
50 rounds, refit per D-10. Development: 20 trials, one seed (D-17); final: 60 trials x 3 seeds
on the frozen code (stage 10).
"""
