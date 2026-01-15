"""
Main experiment orchestrator.

Connects all framework components and provides the primary API
for running experiments and hyperparameter tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from explr.core.config import ExperimentConfig, TunerType
from explr.core.protocols import TrainFunction, TrainingContext, TrainingResult
from explr.tracking.client import ExplrTracker
from explr.tracking.server import MLflowServer


class Experiment:
    """
    Main experiment orchestrator.

    Provides the primary API for running experiments, either as single
    training runs or with hyperparameter tuning.

    Example:
        # Create experiment
        config = ExperimentConfig(name="my_experiment")
        exp = Experiment(config)

        # Single run
        result = exp.run(train_fn)

        # With hyperparameter tuning
        config.tuning.tuner_type = TunerType.OPTUNA
        config.tuning.n_trials = 50
        results = exp.tune(train_fn)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        mlflow_server: Optional[MLflowServer] = None,
    ):
        """
        Initialize the experiment.

        Args:
            config: Experiment configuration
            mlflow_server: Optional pre-configured MLflow server
        """
        self.config = config
        self._mlflow_server = mlflow_server
        self._tracking_uri = self._setup_tracking()

    def _setup_tracking(self) -> str:
        """Set up MLflow tracking, auto-starting server if needed."""
        if self._mlflow_server:
            return self._mlflow_server.tracking_uri

        if self.config.mlflow.auto_start:
            server = MLflowServer.ensure_running(
                port=int(self.config.mlflow.tracking_uri.split(":")[-1]),
                backend_store_uri=self.config.mlflow.backend_store_uri,
                artifact_root=self.config.mlflow.artifact_location,
            )
            return server.tracking_uri

        return self.config.mlflow.tracking_uri

    def run(
        self,
        train_fn: TrainFunction,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> TrainingResult:
        """
        Execute a single training run.

        Args:
            train_fn: User's train_model() function
            hyperparameters: Override default hyperparameters

        Returns:
            TrainingResult from the training run
        """
        # Ensure directories exist
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.artifact_dir).mkdir(parents=True, exist_ok=True)

        # Merge hyperparameters
        params = {**self.config.hyperparameters, **(hyperparameters or {})}

        # Create tracker
        tracker = ExplrTracker(
            tracking_uri=self._tracking_uri,
            experiment_name=self.config.name,
        )

        # Create context
        ctx = TrainingContext(
            hyperparameters=params,
            tracker=tracker,
            experiment_name=self.config.name,
            seed=self.config.seed,
            checkpoint_dir=self.config.checkpoint_dir,
            artifact_dir=self.config.artifact_dir,
        )

        # Run training
        with tracker:
            tracker.set_tags(self.config.tags)
            tracker.log_params(params)

            result = train_fn(ctx)

            # Log final metrics
            tracker.log_metrics(result.metrics)
            tracker.log_metric(result.primary_metric_name, result.primary_metric)

            # Log artifacts
            if result.best_model_path:
                tracker.log_artifact(result.best_model_path, "model")

            for name, path in result.artifacts.items():
                tracker.log_artifact(path, name)

        return result

    def tune(self, train_fn: TrainFunction) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.

        Args:
            train_fn: User's train_model() function

        Returns:
            Dictionary with best parameters and metrics
        """
        # Ensure directories exist
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.artifact_dir).mkdir(parents=True, exist_ok=True)

        tuner_type = self.config.tuning.tuner_type

        if tuner_type == TunerType.OPTUNA:
            from explr.tuning.optuna_tuner import OptunaTuner

            tuner = OptunaTuner(
                train_fn=train_fn,
                config=self.config,
                mlflow_tracking_uri=self._tracking_uri,
            )

        elif tuner_type == TunerType.RAY_TUNE:
            from explr.tuning.ray_tuner import RayTuneTuner

            tuner = RayTuneTuner(
                train_fn=train_fn,
                config=self.config,
                mlflow_tracking_uri=self._tracking_uri,
            )

        else:
            raise ValueError(
                f"Unknown tuner type: {tuner_type}. "
                "Use TunerType.OPTUNA or TunerType.RAY_TUNE"
            )

        return tuner.optimize()


def run_experiment(
    train_fn: TrainFunction,
    config: ExperimentConfig,
    tune: bool = False,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Convenience function to run an experiment.

    This is the primary entry point for most use cases.

    Args:
        train_fn: User's train_model() function
        config: Experiment configuration
        tune: Whether to run hyperparameter tuning
        hyperparameters: Override hyperparameters (for single run only)

    Returns:
        TrainingResult for single run, or tuning results dict

    Example:
        from explr import run_experiment, ExperimentConfig

        def train_model(ctx):
            # ... training code ...
            return TrainingResult(primary_metric=val_loss)

        # Single run
        result = run_experiment(train_model, ExperimentConfig(name="my_exp"))

        # With tuning
        config = ExperimentConfig(name="my_exp")
        config.tuning.tuner_type = TunerType.OPTUNA
        results = run_experiment(train_model, config, tune=True)
    """
    experiment = Experiment(config)

    if tune:
        return experiment.tune(train_fn)
    else:
        return experiment.run(train_fn, hyperparameters)
