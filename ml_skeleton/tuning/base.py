"""
Base tuner abstract class.

Defines the interface that all hyperparameter tuners must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from ml_skeleton.core.protocols import TrainFunction, TrainingContext
from ml_skeleton.core.config import ExperimentConfig
from ml_skeleton.tracking.client import ExplrTracker


class BaseTuner(ABC):
    """
    Abstract base class for hyperparameter tuners.

    All tuner implementations (Optuna, Ray Tune) inherit from this class
    and implement the optimize() method.
    """

    def __init__(
        self,
        train_fn: TrainFunction,
        config: ExperimentConfig,
        mlflow_tracking_uri: str,
    ):
        """
        Initialize the tuner.

        Args:
            train_fn: User's train_model() function
            config: Experiment configuration
            mlflow_tracking_uri: URI of the MLflow tracking server
        """
        self.train_fn = train_fn
        self.config = config
        self.mlflow_tracking_uri = mlflow_tracking_uri

    @abstractmethod
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Returns:
            Dictionary containing:
            - best_params: Best hyperparameters found
            - best_value: Best metric value achieved
            - Additional tuner-specific results
        """
        pass

    @abstractmethod
    def _create_objective(self) -> Callable:
        """Create the objective function for the tuner."""
        pass

    def _build_context(
        self,
        hyperparameters: Dict[str, Any],
        trial_id: Optional[str] = None,
        trial_number: Optional[int] = None,
        parent_run_id: Optional[str] = None,
    ) -> TrainingContext:
        """
        Build TrainingContext for a trial.

        Args:
            hyperparameters: Sampled hyperparameters for this trial
            trial_id: Unique identifier for this trial
            trial_number: Sequential trial number
            parent_run_id: Optional MLflow parent run ID so the trial run is nested
                under the HPO parent (used by Optuna tuner).

        Returns:
            Configured TrainingContext
        """
        run_name = f"trial_{trial_number}" if trial_number is not None else None
        # Use resolved MLflow experiment name when tuner set it (e.g. after creating new experiment for deleted one)
        experiment_name = getattr(self, "_mlflow_experiment_name", None) or self.config.name
        tracker = ExplrTracker(
            tracking_uri=self.mlflow_tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
            nested=True,  # Required: trials must nest under HPO parent
            parent_run_id=parent_run_id,  # Required for nesting; Optuna sets this
        )

        return TrainingContext(
            hyperparameters=hyperparameters,
            tracker=tracker,
            trial_id=trial_id,
            trial_number=trial_number,
            experiment_name=experiment_name,
            seed=self.config.seed,
            checkpoint_dir=self.config.checkpoint_dir,
            artifact_dir=self.config.artifact_dir,
        )
