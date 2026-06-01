#!/usr/bin/env python
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from typing import Any


def run_cmd(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"cmd": cmd, "error": str(exc)}


def pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> int:
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "env": {
            "CUDA_HOME": os.getenv("CUDA_HOME"),
            "CUDA_PATH": os.getenv("CUDA_PATH"),
            "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH"),
            "PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF"),
        },
        "binaries": {
            "nvidia-smi": shutil.which("nvidia-smi"),
            "nvcc": shutil.which("nvcc"),
        },
        "commands": {},
        "packages": {},
        "torch": {},
    }

    report["commands"]["nvidia_smi"] = run_cmd([
        "nvidia-smi",
        "--query-gpu=name,driver_version,cuda_version,memory.total,memory.used",
        "--format=csv,noheader",
    ])
    report["commands"]["nvcc"] = run_cmd(["nvcc", "--version"]) if shutil.which("nvcc") else None

    package_names = [
        "torch",
        "cuda-toolkit",
        "cuda-bindings",
        "nvidia-cuda-runtime",
        "nvidia-cudnn-cu13",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12",
        "transformers",
        "sentence-transformers",
        "FlagEmbedding",
        "mxbai-rerank",
        "flash-attn",
    ]
    for name in package_names:
        version = pkg_version(name)
        if version is not None:
            report["packages"][name] = version

    try:
        import torch

        report["torch"] = {
            "version": getattr(torch, "__version__", None),
            "version_cuda": getattr(torch.version, "cuda", None),
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            report["torch"]["device_name"] = torch.cuda.get_device_name(0)
            report["torch"]["device_capability"] = torch.cuda.get_device_capability(0)
            free, total = torch.cuda.mem_get_info(0)
            report["torch"]["mem_before_mb"] = {"free": free // 1024**2, "total": total // 1024**2}
            x = torch.empty((256, 1024, 1024), dtype=torch.float16, device=device)
            y = torch.mm(torch.randn((512, 512), device=device), torch.randn((512, 512), device=device))
            torch.cuda.synchronize()
            del x, y
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(0)
            report["torch"]["mem_after_mb"] = {"free": free // 1024**2, "total": total // 1024**2}
            report["torch"]["allocation_test"] = "ok"
    except Exception as exc:
        report["torch"]["error"] = repr(exc)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    torch_info = report.get("torch", {})
    if torch_info.get("error") or not torch_info.get("cuda_is_available"):
        print("\nCUDA check failed: PyTorch cannot use CUDA in this environment.", file=sys.stderr)
        return 2
    if any(name.endswith("cu12") for name in report["packages"]):
        print("\nWarning: cu12 NVIDIA packages are still installed next to a cu13 torch stack.", file=sys.stderr)
        return 3
    print("\nCUDA check ok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
