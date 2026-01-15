"""
PyTorch helper utilities for use within train_model().

Provides common functionality for PyTorch training including
device setup, reproducibility, and MLflow callbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from explr.core.protocols import TrainingContext


class PyTorchHelper:
    """
    Helper utilities for PyTorch training within train_model().

    Example:
        def train_model(ctx: TrainingContext) -> TrainingResult:
            from explr.frameworks.pytorch import PyTorchHelper

            # Set up device and reproducibility
            device = PyTorchHelper.setup_device(ctx)
            PyTorchHelper.setup_seed(ctx.seed)

            # Create callback for logging
            callback = PyTorchHelper.create_mlflow_callback(ctx)

            # Training loop
            for epoch in range(epochs):
                train_loss = train_epoch(...)
                val_loss, val_acc = validate(...)

                # Log and check for pruning
                should_continue = callback.on_epoch_end(
                    epoch, train_loss, val_loss, {"val_accuracy": val_acc}
                )
                if not should_continue:
                    break

            return TrainingResult(...)
    """

    @staticmethod
    def setup_device(ctx: "TrainingContext") -> Any:
        """
        Set up the appropriate PyTorch device.

        Args:
            ctx: Training context

        Returns:
            torch.device object
        """
        import torch

        if ctx.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        return device

    @staticmethod
    def setup_seed(seed: Optional[int]) -> None:
        """
        Set random seeds for reproducibility.

        Args:
            seed: Random seed (None to skip)
        """
        if seed is None:
            return

        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def create_mlflow_callback(ctx: "TrainingContext") -> "PyTorchMLflowCallback":
        """
        Create a callback for logging metrics during training.

        Args:
            ctx: Training context

        Returns:
            Callback instance
        """
        trial = getattr(ctx, "_optuna_trial", None)
        return PyTorchMLflowCallback(ctx.tracker, trial)

    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """
        Get information about available PyTorch devices.

        Returns:
            Dictionary with device information
        """
        import torch

        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            ),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info[f"device_{i}"] = {
                    "name": props.name,
                    "memory_gb": props.total_memory / 1e9,
                    "compute_capability": f"{props.major}.{props.minor}",
                }

        return info


class PyTorchMLflowCallback:
    """
    Callback for logging PyTorch training metrics to MLflow.

    Supports intermediate metric reporting for Optuna pruning.
    """

    def __init__(self, tracker: Any, trial: Optional[Any] = None):
        """
        Initialize the callback.

        Args:
            tracker: ExplrTracker instance
            trial: Optional Optuna trial for pruning
        """
        self.tracker = tracker
        self.trial = trial

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """
        Log metrics at end of epoch.

        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            metrics: Additional metrics to log

        Returns:
            True if training should continue, False if pruned
        """
        all_metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            **(metrics or {}),
        }
        self.tracker.log_metrics(all_metrics, step=epoch)

        # Check for pruning
        return self.tracker.report_intermediate(val_loss, epoch, self.trial)


def train_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    criterion: Callable,
    device: Any,
    scheduler: Optional[Any] = None,
) -> float:
    """
    Standard PyTorch training epoch.

    This is a utility function users can use or reference.

    Args:
        model: PyTorch model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        scheduler: Optional learning rate scheduler

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: Any,
    dataloader: Any,
    criterion: Callable,
    device: Any,
) -> Dict[str, float]:
    """
    Standard PyTorch validation.

    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Dictionary with val_loss and val_accuracy
    """
    import torch

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return {
        "val_loss": total_loss / len(dataloader),
        "val_accuracy": correct / total if total > 0 else 0.0,
    }
