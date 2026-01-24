"""
GPU Memory Management Utilities.

This module provides functions to limit and manage GPU memory usage.
This is important for:
- Sharing GPU with other processes (desktop, other training jobs)
- Preventing OOM errors
- Reproducible memory behavior

USAGE:
======

1. Set memory limit at the start of your script (BEFORE creating models):

    from ml_skeleton.utils.memory import limit_gpu_memory

    # Limit to 24GB (leave 8GB for system/other apps on a 32GB GPU)
    limit_gpu_memory(max_memory_gb=24)

2. Or use environment variable (set before running):

    export EXPLR_GPU_MEMORY_GB=24
    python your_script.py

3. Or in your train_model() function:

    def train_model(ctx: TrainingContext) -> TrainingResult:
        from ml_skeleton.utils.memory import limit_gpu_memory

        # Use memory limit from hyperparameters or default
        max_mem = ctx.hyperparameters.get("max_gpu_memory_gb", 24)
        limit_gpu_memory(max_memory_gb=max_mem)

        # ... rest of training code ...

IMPORTANT NOTES:
================
- Must be called BEFORE any GPU tensors are created
- For PyTorch: Uses CUDA memory fraction
- For TensorFlow: Uses memory growth + virtual device limit
- Default: 24GB (leaves 8GB free on a 32GB RTX 5090 for system/desktop)
- Set EXPLR_GPU_MEMORY_GB=0 to disable the limit and use all GPU memory
"""

from __future__ import annotations

import gc
import os
from typing import Optional

# Environment variable for setting memory limit
ENV_VAR_NAME = "EXPLR_GPU_MEMORY_GB"

# Default memory limit in GB (24GB leaves 8GB free on 32GB RTX 5090)
DEFAULT_MEMORY_LIMIT_GB = 24.0

# Track if memory has been configured (can only be done once)
_memory_configured = False


def limit_gpu_memory(
    max_memory_gb: Optional[float] = None,
    device_id: int = 0,
    framework: str = "auto",
) -> dict:
    """
    Limit GPU memory usage.

    MUST be called before creating any GPU tensors/models!

    Args:
        max_memory_gb: Maximum GPU memory in GB. If None, checks
                      EXPLR_GPU_MEMORY_GB environment variable, then
                      falls back to DEFAULT_MEMORY_LIMIT_GB (24GB).
                      Set to 0 to disable limit and use all GPU memory.
        device_id: GPU device ID to limit (default: 0)
        framework: "pytorch", "tensorflow", or "auto" (detect both)

    Returns:
        Dictionary with configuration results

    Example:
        # Use default 24GB limit
        limit_gpu_memory()

        # Custom limit
        limit_gpu_memory(max_memory_gb=16)

        # Disable limit (use all GPU memory)
        limit_gpu_memory(max_memory_gb=0)

        # Or via environment variable
        # export EXPLR_GPU_MEMORY_GB=24
        limit_gpu_memory()  # Will use env var
    """
    global _memory_configured

    # Check environment variable if no explicit limit
    if max_memory_gb is None:
        env_value = os.environ.get(ENV_VAR_NAME)
        if env_value is not None:
            try:
                max_memory_gb = float(env_value)
            except ValueError:
                print(f"Warning: Invalid {ENV_VAR_NAME}={env_value}, using default")
                max_memory_gb = DEFAULT_MEMORY_LIMIT_GB
        else:
            # Use default limit
            max_memory_gb = DEFAULT_MEMORY_LIMIT_GB

    # If explicitly set to 0, disable limit
    if max_memory_gb == 0:
        max_memory_gb = None

    result = {
        "max_memory_gb": max_memory_gb,
        "device_id": device_id,
        "pytorch_configured": False,
        "tensorflow_configured": False,
        "warnings": [],
    }

    if _memory_configured:
        result["warnings"].append(
            "GPU memory was already configured. "
            "Memory limits can only be set once before GPU initialization."
        )
        return result

    if max_memory_gb is None:
        result["warnings"].append(
            "No memory limit set. GPU will use all available memory. "
            f"Set {ENV_VAR_NAME} or pass max_memory_gb to limit."
        )
        return result

    # Configure PyTorch
    if framework in ("pytorch", "auto"):
        try:
            result["pytorch_configured"] = _limit_pytorch_memory(
                max_memory_gb, device_id
            )
        except Exception as e:
            result["warnings"].append(f"PyTorch memory config failed: {e}")

    # Configure TensorFlow
    if framework in ("tensorflow", "auto"):
        try:
            result["tensorflow_configured"] = _limit_tensorflow_memory(
                max_memory_gb, device_id
            )
        except Exception as e:
            result["warnings"].append(f"TensorFlow memory config failed: {e}")

    _memory_configured = True

    # Log the configuration
    if result["pytorch_configured"] or result["tensorflow_configured"]:
        print(f"[explr] GPU memory limited to {max_memory_gb:.1f} GB on device {device_id}")

    return result


def _limit_pytorch_memory(max_memory_gb: float, device_id: int) -> bool:
    """Configure PyTorch memory limit."""
    # Set allocation config BEFORE initializing CUDA context to avoid fragmentation
    # This must be done before any CUDA call
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "max_split_size_mb:512,expandable_segments:True"
        )

    try:
        import torch

        if not torch.cuda.is_available():
            return False

        # Get total memory
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        total_memory_gb = total_memory / (1024**3)

        if max_memory_gb >= total_memory_gb:
            # No need to limit
            return True

        # Calculate fraction
        fraction = max_memory_gb / total_memory_gb

        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(fraction, device_id)

        return True

    except ImportError:
        return False


def _limit_tensorflow_memory(max_memory_gb: float, device_id: int) -> bool:
    """Configure TensorFlow memory limit."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus or device_id >= len(gpus):
            return False

        gpu = gpus[device_id]

        # Enable memory growth first
        tf.config.experimental.set_memory_growth(gpu, True)

        # Set virtual device with memory limit
        tf.config.set_logical_device_configuration(
            gpu,
            [
                tf.config.LogicalDeviceConfiguration(
                    memory_limit=int(max_memory_gb * 1024)  # MB
                )
            ],
        )

        return True

    except ImportError:
        return False
    except RuntimeError as e:
        # TensorFlow has already initialized GPU
        if "already been initialized" in str(e):
            raise RuntimeError(
                "TensorFlow GPU already initialized. "
                "Call limit_gpu_memory() before importing TensorFlow models."
            ) from e
        raise


def get_gpu_memory_info(device_id: int = 0) -> dict:
    """
    Get current GPU memory usage information.

    Args:
        device_id: GPU device ID

    Returns:
        Dictionary with memory info (total, allocated, free, etc.)
    """
    result = {
        "device_id": device_id,
        "total_gb": None,
        "allocated_gb": None,
        "reserved_gb": None,
        "free_gb": None,
    }

    try:
        import torch

        if not torch.cuda.is_available():
            return result

        props = torch.cuda.get_device_properties(device_id)
        result["total_gb"] = round(props.total_memory / (1024**3), 2)

        # Current allocation stats
        result["allocated_gb"] = round(
            torch.cuda.memory_allocated(device_id) / (1024**3), 2
        )
        result["reserved_gb"] = round(
            torch.cuda.memory_reserved(device_id) / (1024**3), 2
        )
        result["free_gb"] = round(
            result["total_gb"] - result["reserved_gb"], 2
        )

    except ImportError:
        pass

    return result


def print_memory_summary(device_id: int = 0) -> None:
    """Print a summary of GPU memory usage."""
    info = get_gpu_memory_info(device_id)

    print("\n" + "-" * 40)
    print(f"  GPU {device_id} Memory Summary")
    print("-" * 40)

    if info["total_gb"] is not None:
        print(f"  Total:     {info['total_gb']:>8.2f} GB")
        print(f"  Allocated: {info['allocated_gb']:>8.2f} GB")
        print(f"  Reserved:  {info['reserved_gb']:>8.2f} GB")
        print(f"  Free:      {info['free_gb']:>8.2f} GB")
    else:
        print("  GPU not available or PyTorch not installed")

    print("-" * 40 + "\n")


# Convenience function to be called at module import
def auto_configure_memory() -> None:
    """
    Automatically configure memory if EXPLR_GPU_MEMORY_GB is set.

    This is called automatically when explr is imported if the
    environment variable is set.
    """
    if os.environ.get(ENV_VAR_NAME):
        limit_gpu_memory()


def cleanup_memory() -> None:
    """
    Force garbage collection and empty CUDA cache.

    Useful to reclaim memory between training stages or experiments
    without restarting the python process.
    """
    # Force Python garbage collection
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass
