"""Tests of the neural worker. They run in `solar-gpu` only (torch, CUDA): in `solar` they are not
collected, and `tests/unit/test_neural_gpu_suite.py` runs them there through the GPU python."""

import importlib.util
import os

if importlib.util.find_spec("torch") is None:
    collect_ignore_glob = ["test_*.py"]
else:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before CUDA starts (X32)
