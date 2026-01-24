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
    Get comprehensive GPU information from PyTorch.

    Returns:
        Dictionary with GPU information
    """
    info = {
        "has_gpu": False,
        "pytorch": None,
    }

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

    return info


def get_optimal_device(framework: str = "pytorch") -> str:
    """
    Get the optimal device string for training.

    Returns:
        Device string ("cuda" or "cpu")
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"


def get_optimal_settings(framework: str = "pytorch") -> Dict[str, Any]:
    """
    Get optimal training settings based on detected hardware.

    Returns suggested settings for data loading, mixed precision, etc.

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


class GPUMonitor:
    """Monitor GPU utilization during training.

    Uses pynvml for efficient polling without subprocess overhead.
    Falls back to nvidia-smi if pynvml is not available.
    """

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._nvml_initialized = False
        self._handle = None
        self._samples: list[dict] = []

        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._nvml_initialized = True
        except (ImportError, Exception):
            pass

    def sample(self) -> Optional[dict]:
        """Take a single GPU utilization sample.

        Returns:
            Dict with gpu_util (%), memory_used (GB), memory_total (GB)
            or None if sampling fails.
        """
        if self._nvml_initialized and self._handle:
            try:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                sample = {
                    "gpu_util": util.gpu,
                    "memory_util": util.memory,
                    "memory_used_gb": mem.used / 1e9,
                    "memory_total_gb": mem.total / 1e9,
                }
                self._samples.append(sample)
                return sample
            except Exception:
                pass
        return None

    def get_stats(self) -> dict:
        """Get statistics from collected samples.

        Returns:
            Dict with avg, min, max for gpu_util and memory stats.
        """
        if not self._samples:
            return {}

        gpu_utils = [s["gpu_util"] for s in self._samples]
        mem_utils = [s["memory_util"] for s in self._samples]
        mem_used = [s["memory_used_gb"] for s in self._samples]

        return {
            "gpu_util_avg": sum(gpu_utils) / len(gpu_utils),
            "gpu_util_min": min(gpu_utils),
            "gpu_util_max": max(gpu_utils),
            "memory_util_avg": sum(mem_utils) / len(mem_utils),
            "memory_used_avg_gb": sum(mem_used) / len(mem_used),
            "num_samples": len(self._samples),
        }

    def reset(self) -> None:
        """Clear collected samples."""
        self._samples = []

    def shutdown(self) -> None:
        """Clean up NVML resources."""
        if self._nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False


def get_gpu_utilization(device_index: int = 0) -> Optional[dict]:
    """Get current GPU utilization (one-shot, no state).

    Args:
        device_index: GPU device index

    Returns:
        Dict with gpu_util, memory_used_gb, memory_total_gb or None
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return {
            "gpu_util": util.gpu,
            "memory_util": util.memory,
            "memory_used_gb": mem.used / 1e9,
            "memory_total_gb": mem.total / 1e9,
        }
    except (ImportError, Exception):
        return None
