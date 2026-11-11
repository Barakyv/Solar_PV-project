"""The GPU environment of the neural training worker (D-29, docs/stage9_plan.md §18.5).

    <solar-gpu python> -m solarpv.models.neural_env [--no-probe]

Runs in `solar-gpu` only (it imports torch). Prints one JSON report: the pinned versions, the
CUDA runtime, the GPU, and whether the build ships kernels for the Tesla P40 (`sm_61`); with the
probe, an FP32 matrix product on the GPU compared with the CPU. The exit code is 1 when a check
fails. The runner's `check` and every neural step read this report (X25).
"""

import argparse
import importlib.metadata as md
import json
import platform
import sys

# distribution -> the version of D-29 (requirements-gpu.txt, environment-gpu.yml)
PINNED = {
    "torch": "2.5.1+cu121",
    "pytorch-forecasting": "1.7.0",
    "neuralforecast": "3.1.7",
    "lightning": "2.6.6",
    "pytorch-lightning": "2.6.6",
    "numpy": "2.4.6",
    "pandas": "3.0.6",
    "scipy": "1.17.1",
    "scikit-learn": "1.9.1",
    "optuna": "5.0.0",
}
CUDA_RUNTIME = "12.1"
REQUIRED_ARCH = "sm_61"  # Tesla P40, Pascal (docs/stage9_plan.md §1.1)
PROBE_SIZE = 512
PROBE_TOL = 1e-3  # FP32, relative to the largest |entry| of the CPU product


def versions() -> dict[str, str | None]:
    out = {}
    for dist in PINNED:
        try:
            out[dist] = md.version(dist)
        except md.PackageNotFoundError:
            out[dist] = None
    return out


def report(probe: bool = True) -> dict:
    """The environment and its checks; `errors` lists every failed check."""
    info: dict = {"python": sys.version.split()[0], "executable": sys.executable,
                  "platform": platform.platform(), "versions": versions(), "errors": []}
    for dist, want in PINNED.items():
        if info["versions"][dist] != want:
            info["errors"].append(f"{dist} {info['versions'][dist]} (pinned {want})")
    try:
        import torch
    except ImportError as exc:
        info["errors"].append(f"torch cannot be imported: {exc}")
        return info
    info["cuda_runtime"] = torch.version.cuda
    info["cudnn"] = torch.backends.cudnn.version()
    info["arch_list"] = torch.cuda.get_arch_list()
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_runtime"] != CUDA_RUNTIME:
        info["errors"].append(f"CUDA runtime {info['cuda_runtime']} (expected {CUDA_RUNTIME})")
    if REQUIRED_ARCH not in info["arch_list"]:
        info["errors"].append(f"no {REQUIRED_ARCH} kernels in {info['arch_list']}")
    if not info["cuda_available"]:
        info["errors"].append("CUDA is not available")
        return info
    props = torch.cuda.get_device_properties(0)
    info["gpu"] = {"name": props.name, "capability": f"{props.major}.{props.minor}",
                   "memory_gb": round(props.total_memory / 2**30, 2),
                   "device_count": torch.cuda.device_count()}
    if (props.major, props.minor) != (6, 1):
        info["errors"].append(f"GPU capability {props.major}.{props.minor}, expected 6.1")
    if probe:
        g = torch.Generator().manual_seed(0)
        a = torch.randn(PROBE_SIZE, PROBE_SIZE, generator=g, dtype=torch.float32)
        b = torch.randn(PROBE_SIZE, PROBE_SIZE, generator=g, dtype=torch.float32)
        cpu = a @ b
        gpu = (a.cuda() @ b.cuda()).cpu()
        rel = float((gpu - cpu).abs().max() / cpu.abs().max())
        info["probe"] = {"matmul_fp32_rel_diff": rel, "tol": PROBE_TOL}
        if not rel < PROBE_TOL:
            info["errors"].append(f"GPU FP32 product differs from the CPU by {rel:.2e}")
    for module in ("lightning", "pytorch_forecasting", "neuralforecast"):
        try:
            __import__(module)
        except Exception as exc:  # an import that fails here would fail in the worker
            info["errors"].append(f"{module} cannot be imported: {type(exc).__name__}: {exc}")
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-probe", action="store_true", help="skip the GPU matrix product")
    args = parser.parse_args(argv)
    info = report(probe=not args.no_probe)
    print(json.dumps(info, indent=2))
    return 1 if info["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
