"""
Reproducibility utilities.

Provides functions for setting random seeds across different
frameworks to ensure reproducible experiments.
"""

from __future__ import annotations

import os
import random
from typing import Optional


def set_seed(seed: int, framework: str = "all") -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        framework: "pytorch", "tensorflow", or "all"
    """
    # Python random
    random.seed(seed)

    # NumPy
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    # Environment variable for some libraries
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    if framework in ("pytorch", "all"):
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    # TensorFlow
    if framework in ("tensorflow", "all"):
        try:
            import tensorflow as tf

            tf.random.set_seed(seed)
        except ImportError:
            pass


def set_deterministic(enabled: bool = True, framework: str = "all") -> None:
    """
    Enable or disable deterministic algorithms.

    Note: Deterministic mode may reduce performance.

    Args:
        enabled: Whether to enable deterministic mode
        framework: "pytorch", "tensorflow", or "all"
    """
    if framework in ("pytorch", "all"):
        try:
            import torch

            torch.backends.cudnn.deterministic = enabled
            torch.backends.cudnn.benchmark = not enabled

            # PyTorch 2.0+ deterministic flag
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(enabled)
                except Exception:
                    pass
        except ImportError:
            pass

    if framework in ("tensorflow", "all"):
        try:
            import tensorflow as tf

            if enabled:
                tf.config.experimental.enable_op_determinism()
        except (ImportError, AttributeError):
            pass


def setup_reproducibility(
    seed: Optional[int], deterministic: bool = True, framework: str = "all"
) -> None:
    """
    Complete reproducibility setup.

    Args:
        seed: Random seed (None to skip seeding)
        deterministic: Whether to enable deterministic mode
        framework: "pytorch", "tensorflow", or "all"
    """
    if seed is not None:
        set_seed(seed, framework)

    if deterministic:
        set_deterministic(True, framework)
