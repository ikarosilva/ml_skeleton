"""
Environment verification utilities.

Run this module directly to check your environment:
    python -m explr.utils.verify

Or use programmatically:
    from ml_skeleton.utils.verify import verify_environment
    verify_environment()
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Tuple

# Minimum required versions
MIN_VERSIONS = {
    "python": (3, 10),
    "torch": (2, 5, 0),
    "tensorflow": (2, 18, 0),
    "mlflow": (2, 10, 0),
    "optuna": (3, 5, 0),
    "cuda": (12, 0),
}


def _parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string to tuple of ints."""
    # Handle versions like "2.5.0+cu124"
    version_str = version_str.split("+")[0].split("-")[0]
    parts = []
    for part in version_str.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


def _version_at_least(current: Tuple[int, ...], minimum: Tuple[int, ...]) -> bool:
    """Check if current version meets minimum requirement."""
    for c, m in zip(current, minimum):
        if c > m:
            return True
        if c < m:
            return False
    return len(current) >= len(minimum)


def check_python() -> Dict[str, Any]:
    """Check Python version."""
    current = sys.version_info[:3]
    minimum = MIN_VERSIONS["python"]
    ok = _version_at_least(current[:2], minimum)

    return {
        "name": "Python",
        "installed": True,
        "version": f"{current[0]}.{current[1]}.{current[2]}",
        "minimum": f"{minimum[0]}.{minimum[1]}",
        "ok": ok,
    }


def check_pytorch() -> Dict[str, Any]:
    """Check PyTorch installation and CUDA support."""
    result = {
        "name": "PyTorch",
        "installed": False,
        "version": None,
        "minimum": ".".join(map(str, MIN_VERSIONS["torch"])),
        "ok": False,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "compute_capability": None,
    }

    try:
        import torch

        result["installed"] = True
        result["version"] = torch.__version__
        current = _parse_version(torch.__version__)
        result["ok"] = _version_at_least(current, MIN_VERSIONS["torch"])

        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["cuda_version"] = torch.version.cuda
            props = torch.cuda.get_device_properties(0)
            result["gpu_name"] = props.name
            result["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
            result["compute_capability"] = f"{props.major}.{props.minor}"

            # Check CUDA version
            cuda_ver = _parse_version(torch.version.cuda)
            if not _version_at_least(cuda_ver, MIN_VERSIONS["cuda"]):
                result["cuda_warning"] = (
                    f"CUDA {torch.version.cuda} detected, "
                    f"but {'.'.join(map(str, MIN_VERSIONS['cuda']))}+ recommended for Blackwell"
                )

    except ImportError:
        pass

    return result


def check_tensorflow() -> Dict[str, Any]:
    """Check TensorFlow installation and GPU support."""
    result = {
        "name": "TensorFlow",
        "installed": False,
        "version": None,
        "minimum": ".".join(map(str, MIN_VERSIONS["tensorflow"])),
        "ok": False,
        "gpu_available": False,
    }

    try:
        import tensorflow as tf

        result["installed"] = True
        result["version"] = tf.__version__
        current = _parse_version(tf.__version__)
        result["ok"] = _version_at_least(current, MIN_VERSIONS["tensorflow"])

        gpus = tf.config.list_physical_devices("GPU")
        result["gpu_available"] = len(gpus) > 0
        result["gpu_count"] = len(gpus)

    except ImportError:
        pass

    return result


def check_mlflow() -> Dict[str, Any]:
    """Check MLflow installation."""
    result = {
        "name": "MLflow",
        "installed": False,
        "version": None,
        "minimum": ".".join(map(str, MIN_VERSIONS["mlflow"])),
        "ok": False,
    }

    try:
        import mlflow

        result["installed"] = True
        result["version"] = mlflow.__version__
        current = _parse_version(mlflow.__version__)
        result["ok"] = _version_at_least(current, MIN_VERSIONS["mlflow"])

    except ImportError:
        pass

    return result


def check_optuna() -> Dict[str, Any]:
    """Check Optuna installation."""
    result = {
        "name": "Optuna",
        "installed": False,
        "version": None,
        "minimum": ".".join(map(str, MIN_VERSIONS["optuna"])),
        "ok": False,
    }

    try:
        import optuna

        result["installed"] = True
        result["version"] = optuna.__version__
        current = _parse_version(optuna.__version__)
        result["ok"] = _version_at_least(current, MIN_VERSIONS["optuna"])

    except ImportError:
        pass

    return result


def check_ray() -> Dict[str, Any]:
    """Check Ray installation."""
    result = {
        "name": "Ray Tune",
        "installed": False,
        "version": None,
        "ok": False,
    }

    try:
        import ray

        result["installed"] = True
        result["version"] = ray.__version__
        result["ok"] = True  # Any version is acceptable

    except ImportError:
        pass

    return result


def verify_environment(verbose: bool = True) -> Dict[str, Any]:
    """
    Verify the environment meets all requirements.

    Args:
        verbose: If True, print results to stdout

    Returns:
        Dictionary with all check results and overall status
    """
    checks = [
        check_python(),
        check_pytorch(),
        check_tensorflow(),
        check_mlflow(),
        check_optuna(),
        check_ray(),
    ]

    # Determine overall status
    # Required: Python, MLflow, at least one of PyTorch/TensorFlow
    python_ok = checks[0]["ok"]
    pytorch_ok = checks[1]["installed"] and checks[1]["ok"]
    tensorflow_ok = checks[2]["installed"] and checks[2]["ok"]
    mlflow_ok = checks[3]["ok"]

    has_framework = pytorch_ok or tensorflow_ok
    all_ok = python_ok and mlflow_ok and has_framework

    result = {
        "checks": checks,
        "all_ok": all_ok,
        "has_gpu": checks[1].get("cuda_available", False) or checks[2].get("gpu_available", False),
        "warnings": [],
    }

    # Collect warnings
    if checks[1].get("cuda_warning"):
        result["warnings"].append(checks[1]["cuda_warning"])

    if not has_framework:
        result["warnings"].append("Neither PyTorch nor TensorFlow is properly installed")

    if verbose:
        _print_results(result)

    return result


def _print_results(result: Dict[str, Any]) -> None:
    """Print verification results."""
    print("\n" + "=" * 60)
    print("  explr Environment Verification")
    print("=" * 60 + "\n")

    for check in result["checks"]:
        status = "✓" if check["ok"] else "✗" if check["installed"] else "-"
        name = check["name"]
        version = check.get("version", "not installed")
        minimum = check.get("minimum", "")

        print(f"  [{status}] {name:<15} {version:<15} (min: {minimum})")

        # Extra info for PyTorch
        if check["name"] == "PyTorch" and check.get("cuda_available"):
            print(f"      └─ CUDA: {check['cuda_version']}")
            print(f"      └─ GPU: {check['gpu_name']} ({check['gpu_memory_gb']} GB)")
            print(f"      └─ Compute Capability: {check['compute_capability']}")

        # Extra info for TensorFlow
        if check["name"] == "TensorFlow" and check.get("gpu_available"):
            print(f"      └─ GPUs detected: {check['gpu_count']}")

    print("\n" + "-" * 60)

    if result["all_ok"]:
        print("  ✓ Environment is ready for explr!")
    else:
        print("  ✗ Environment has issues that need to be resolved")

    if result["warnings"]:
        print("\n  Warnings:")
        for warning in result["warnings"]:
            print(f"    - {warning}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    result = verify_environment(verbose=True)
    sys.exit(0 if result["all_ok"] else 1)
