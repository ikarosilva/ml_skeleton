"""
Core protocols and data structures for the training framework.

Defines the contract between the framework and user-provided train_model() functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Protocol

if TYPE_CHECKING:
    from explr.tracking.client import ExplrTracker


class Framework(Enum):
    """Supported deep learning frameworks."""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


@dataclass
class TrainingContext:
    """
    Context object passed to train_model() containing all necessary resources.

    This is the primary interface between the framework and user code.
    Users receive this object and use it to:
    - Access hyperparameters for the current trial
    - Log metrics, parameters, and artifacts via the tracker
    - Get trial/experiment metadata
    - Access device and path configurations

    Example:
        def train_model(ctx: TrainingContext) -> TrainingResult:
            lr = ctx.hyperparameters.get("learning_rate", 0.001)
            model = MyModel().to(ctx.device)

            for epoch in range(epochs):
                # ... training ...
                ctx.tracker.log_metric("loss", loss, step=epoch)

            return TrainingResult(primary_metric=val_loss)
    """

    # Hyperparameters for this trial (sampled by tuner or provided directly)
    hyperparameters: Dict[str, Any]

    # MLflow tracking client (pre-configured for this experiment/run)
    tracker: "ExplrTracker"

    # Trial information (populated during hyperparameter tuning)
    trial_id: Optional[str] = None
    trial_number: Optional[int] = None

    # Experiment metadata
    experiment_name: str = "default"
    run_name: Optional[str] = None

    # Device configuration
    device: str = "cuda"

    # Path configurations
    artifact_dir: str = "./artifacts"
    checkpoint_dir: str = "./checkpoints"

    # Framework hint (can be used by helpers)
    framework: Framework = Framework.PYTORCH

    # Random seed for reproducibility
    seed: Optional[int] = None


@dataclass
class TrainingResult:
    """
    Result object returned from train_model().

    The primary_metric is required and used by hyperparameter tuners
    to optimize the search. Additional metrics and artifacts can be
    provided for logging purposes.

    Example:
        return TrainingResult(
            primary_metric=best_val_loss,
            primary_metric_name="val_loss",
            minimize=True,
            metrics={
                "final_train_loss": train_loss,
                "final_val_accuracy": val_acc
            },
            best_model_path="./checkpoints/best_model.pt",
            epochs_completed=50
        )
    """

    # Primary metric for optimization (required)
    primary_metric: float

    # Name of the primary metric (for logging)
    primary_metric_name: str = "val_loss"

    # Whether to minimize or maximize the primary metric
    minimize: bool = True

    # Additional metrics (optional, logged to MLflow)
    metrics: Dict[str, float] = field(default_factory=dict)

    # Path to best model checkpoint (optional)
    best_model_path: Optional[str] = None

    # Any additional artifacts to log (name -> path mapping)
    artifacts: Dict[str, str] = field(default_factory=dict)

    # Training metadata
    epochs_completed: int = 0
    early_stopped: bool = False


class TrainFunction(Protocol):
    """
    Protocol defining the signature for user's train_model() function.

    Users must implement a function matching this signature. The function
    receives a TrainingContext with hyperparameters and tracking capabilities,
    and returns a TrainingResult with the optimization metric.

    Example:
        def train_model(ctx: TrainingContext) -> TrainingResult:
            # Extract hyperparameters
            lr = ctx.hyperparameters["learning_rate"]
            batch_size = ctx.hyperparameters["batch_size"]

            # Create model and train
            model = MyModel()
            # ... training loop ...

            # Log metrics
            ctx.tracker.log_metric("accuracy", accuracy)

            return TrainingResult(
                primary_metric=val_loss,
                metrics={"accuracy": accuracy}
            )
    """

    def __call__(self, ctx: TrainingContext) -> TrainingResult:
        """
        User-provided training function.

        Args:
            ctx: TrainingContext with hyperparameters and tracking client

        Returns:
            TrainingResult with primary metric and optional additional data
        """
        ...
