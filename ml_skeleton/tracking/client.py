"""
MLflow tracking client wrapper for use inside train_model().

Provides a simplified interface for logging metrics, parameters,
and artifacts during training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import mlflow


class ExplrTracker:
    """
    Wrapped MLflow client providing a simplified interface for train_model().

    This class is passed to the user's train_model() function via the
    TrainingContext. It wraps MLflow's tracking API with a more convenient
    interface.

    Example usage inside train_model():
        def train_model(ctx: TrainingContext) -> TrainingResult:
            # Log hyperparameters
            ctx.tracker.log_params(ctx.hyperparameters)

            for epoch in range(epochs):
                # Training...
                ctx.tracker.log_metric("train_loss", train_loss, step=epoch)
                ctx.tracker.log_metric("val_loss", val_loss, step=epoch)

            # Log model artifact
            ctx.tracker.log_artifact("model.pt")

            return TrainingResult(...)
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
        nested: bool = False,
    ):
        """
        Initialize the tracker.

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
            run_id: Optional existing run ID to resume
            run_name: Optional name for the run
            nested: Whether this is a nested run (for tuning trials)
        """
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run_id = run_id
        self._run_name = run_name
        self._nested = nested
        self._active_run = None

    def __enter__(self) -> "ExplrTracker":
        """Start the MLflow run."""
        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)

        if self._run_id:
            self._active_run = mlflow.start_run(
                run_id=self._run_id, nested=self._nested
            )
        else:
            self._active_run = mlflow.start_run(
                run_name=self._run_name, nested=self._nested
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the MLflow run."""
        if self._active_run:
            mlflow.end_run()
        return False

    @property
    def run_id(self) -> str:
        """Get the current MLflow run ID."""
        active = mlflow.active_run()
        if active:
            return active.info.run_id
        raise RuntimeError("No active MLflow run")

    @property
    def artifact_uri(self) -> str:
        """Get the artifact URI for the current run."""
        active = mlflow.active_run()
        if active:
            return active.info.artifact_uri
        raise RuntimeError("No active MLflow run")

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        mlflow.log_params(params)

    def log_metric(
        self, key: str, value: float, step: Optional[int] = None
    ) -> None:
        """Log a single metric."""
        mlflow.log_metric(key, value, step=step)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(
        self, local_path: Union[str, Path], artifact_path: Optional[str] = None
    ) -> None:
        """Log a local file or directory as an artifact."""
        mlflow.log_artifact(str(local_path), artifact_path)

    def log_artifacts(
        self, local_dir: Union[str, Path], artifact_path: Optional[str] = None
    ) -> None:
        """Log all files in a directory as artifacts."""
        mlflow.log_artifacts(str(local_dir), artifact_path)

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """Log a matplotlib/plotly figure."""
        mlflow.log_figure(figure, artifact_file)

    def log_image(self, image: Any, artifact_file: str) -> None:
        """Log an image (numpy array or PIL Image)."""
        mlflow.log_image(image, artifact_file)

    def log_text(self, text: str, artifact_file: str) -> None:
        """Log a text string as an artifact."""
        mlflow.log_text(text, artifact_file)

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as a JSON/YAML artifact."""
        mlflow.log_dict(dictionary, artifact_file)

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        framework: str = "pytorch",
        **kwargs,
    ) -> None:
        """
        Log a model using the appropriate MLflow flavor.

        Args:
            model: The model to log
            artifact_path: Path within artifacts to store the model
            framework: "pytorch" or "tensorflow"
            **kwargs: Additional arguments passed to the MLflow logger
        """
        if framework == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path, **kwargs)
        elif framework == "tensorflow":
            mlflow.tensorflow.log_model(model, artifact_path, **kwargs)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run."""
        mlflow.set_tag(key, value)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set multiple tags."""
        mlflow.set_tags(tags)

    def report_intermediate(
        self, metric: float, step: int, trial: Optional[Any] = None
    ) -> bool:
        """
        Report intermediate metric for potential pruning.

        This is used during hyperparameter tuning to allow early stopping
        of unpromising trials.

        Args:
            metric: The metric value to report
            step: The current step/epoch
            trial: Optional Optuna trial object for pruning

        Returns:
            True if training should continue, False if pruned
        """
        self.log_metric("intermediate_metric", metric, step=step)

        if trial is not None:
            # Optuna pruning check
            trial.report(metric, step)
            if trial.should_prune():
                return False

        return True
