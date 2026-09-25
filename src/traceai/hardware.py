"""Low-cost, dependency-optional local hardware inspection."""

import importlib.util
import os
import platform
import subprocess
import sys


def inspect_hardware() -> dict:
    system = platform.system()
    machine = platform.machine()
    memory_bytes = None
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            memory_bytes = int(result.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    elif hasattr(os, "sysconf"):
        try:
            memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            pass
    cuda_available = False
    cuda_device = None
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_device = torch.cuda.get_device_name(0)
        except (RuntimeError, OSError, ImportError):
            pass
    return {
        "system": system,
        "machine": machine,
        "cpu": platform.processor(),
        "memory_bytes": memory_bytes,
        "apple_silicon": system == "Darwin" and machine == "arm64",
        "metal_expected": system == "Darwin" and machine == "arm64",
        "mlx_installed": importlib.util.find_spec("mlx_lm") is not None,
        "python": sys.version.split()[0],
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
    }
