"""Runs tests/gpu in the GPU environment `solar-gpu` (D-29), so that the full test suite of `solar`
covers the training worker too. Skipped only where `solar-gpu` does not exist."""

import subprocess

import pytest

from solarpv.config.paths import REPO_ROOT
from solarpv.pipeline.neural import gpu_python, worker_env


@pytest.mark.slow
def test_the_gpu_tests_pass_in_solar_gpu():
    exe = gpu_python()
    if not exe.exists():
        pytest.skip(f"no GPU environment at {exe}")
    env = {**worker_env(), "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    proc = subprocess.run([str(exe), "-m", "pytest", "tests/gpu", "-q", "-p", "no:cacheprovider",
                           "-W", "ignore::UserWarning"], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=1800, encoding="utf-8",
                          errors="replace")
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    last = proc.stdout.strip().splitlines()[-1]
    assert " passed" in last and "skipped" not in last and "failed" not in last
