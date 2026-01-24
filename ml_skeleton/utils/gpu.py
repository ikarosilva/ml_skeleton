"""
GPU detection and configuration utilities.

Provides functions for detecting available GPUs and getting
optimal training configurations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# NVIDIA GPU Architecture mapping by compute capability
_ARCHITECTURES = {
    (5, 0): "Maxwell",
    (5, 2): "Maxwell",
    (5, 3): "Maxwell",
    (6, 0): "Pascal",
    (6, 1): "Pascal",
    (6, 2): "Pascal",
    (7, 0): "Volta",
    (7, 2): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 7): "Ampere",
    (8, 9): "Ada Lovelace",
    (9, 0): "Hopper",
    (10, 0): "Blackwell",
    (10, 1): "Blackwell",
    (12, 0): "Blackwell",  # RTX 5090 reports as SM 12.0
}


def _get_architecture_name(major: int, minor: int) -> str:
    """Get architecture name from compute capability."""
    # Try exact match first
    if (major, minor) in _ARCHITECTURES:
        return _ARCHITECTURES[(major, minor)]

    # Try major version only
    for (maj, min_), arch in _ARCHITECTURES.items():
        if maj == major:
            return arch

    # Unknown
    if major >= 10:
        return "Blackwell or newer"
    return f"Unknown (SM {major}.{minor})"


def get_gpu_info() -> Dict[str, Any]:
    """
    Get comprehensive GPU information from both PyTorch and TensorFlow.

    Returns:
        Dictionary with GPU information from available frameworks
    """
    info = {
        "has_gpu": False,
        "pytorch": None,
        "tensorflow": None,
    }

    # PyTorch info
    try:
        import torch

        pytorch_info = {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            ),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [],
        }

        if torch.cuda.is_available():
            info["has_gpu"] = True
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                # Detect architecture generation
                arch = _get_architecture_name(props.major, props.minor)

                pytorch_info["devices"].append(
                    {
                        "index": i,
                        "name": props.name,
                        "memory_gb": round(props.total_memory / 1e9, 2),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "architecture": arch,
                        "multi_processor_count": props.multi_processor_count,
                    }
                )

        info["pytorch"] = pytorch_info

    except ImportError:
        pass

    # TensorFlow info
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")

        tensorflow_info = {
            "available": len(gpus) > 0,
            "version": tf.__version__,
            "device_count": len(gpus),
            "devices": [{"name": gpu.name, "type": gpu.device_type} for gpu in gpus],
        }

        if len(gpus) > 0:
            info["has_gpu"] = True

        info["tensorflow"] = tensorflow_info

    except ImportError:
        pass

    return info


def get_optimal_device(framework: str = "pytorch") -> str:
    """
    Get the optimal device string for training.

    Args:
        framework: "pytorch" or "tensorflow"

    Returns:
        Device string ("cuda", "gpu:0", "cpu", etc.)
    """
    if framework == "pytorch":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except ImportError:
            return "cpu"

    elif framework == "tensorflow":
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                return "/GPU:0"
            return "/CPU:0"
        except ImportError:
            return "/CPU:0"

    return "cpu"


def get_optimal_settings(framework: str = "pytorch") -> Dict[str, Any]:
    """
    Get optimal training settings based on detected hardware.

    Returns suggested settings for data loading, mixed precision, etc.

    Args:
        framework: "pytorch" or "tensorflow"

    Returns:
        Dictionary with recommended settings
    """
    settings = {
        "device": get_optimal_device(framework),
        "num_workers": 4,
        "pin_memory": False,
        "use_amp": False,
        "amp_dtype": None,
    }

    if framework == "pytorch":
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                settings["pin_memory"] = True
                settings["num_workers"] = 8

                # Enable AMP for compute capability >= 7.0 (Volta+)
                if props.major >= 7:
                    settings["use_amp"] = True
                    # BF16 for Ampere+ (compute 8.0+), FP16 otherwise
                    settings["amp_dtype"] = (
                        torch.bfloat16 if props.major >= 8 else torch.float16
                    )

                # Blackwell-specific optimizations (RTX 5090, compute 10.0+)
                is_blackwell = props.major >= 10
                if is_blackwell:
                    settings["use_amp"] = True
                    settings["amp_dtype"] = torch.bfloat16
                    settings["torch_compile"] = True  # torch.compile works well on Blackwell
                    settings["num_workers"] = 16

                # Adjust workers based on GPU memory
                elif props.total_memory > 20e9:  # > 20GB
                    settings["num_workers"] = 12

        except ImportError:
            pass

    elif framework == "tensorflow":
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                settings["use_amp"] = True
                settings["num_workers"] = 8

        except ImportError:
            pass

    return settings


def check_cuda_compatibility() -> Dict[str, Any]:
    """
    Check CUDA compatibility for the current environment.

    Returns:
        Dictionary with compatibility information and any warnings
    """
    result = {
        "compatible": True,
        "warnings": [],
        "cuda_version": None,
        "driver_version": None,
    }

    try:
        import torch

        if not torch.cuda.is_available():
            result["compatible"] = False
            result["warnings"].append("CUDA is not available")
            return result

        result["cuda_version"] = torch.version.cuda

        # Check compute capability
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cc = f"{props.major}.{props.minor}"

            if props.major < 6:
                result["warnings"].append(
                    f"GPU {i} ({props.name}) has compute capability {cc}, "
                    "which may not support all features"
                )

    except ImportError:
        result["warnings"].append("PyTorch not installed")

    return result


def configure_optimizations() -> None:
    """
    Apply global hardware-specific optimizations.

    Should be called at startup.
    """
    try:
        import torch

        if torch.cuda.is_available():
            # Enable TF32 for Ampere and newer (RTX 30xx, 40xx, 50xx)
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.set_float32_matmul_precision("high")
    except ImportError:
        pass
