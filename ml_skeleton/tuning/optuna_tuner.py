"""
Optuna-based hyperparameter tuner.

Provides integration with Optuna for single-machine hyperparameter
optimization with advanced features like pruning and various samplers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import mlflow
import optuna

from ml_skeleton.core.config import ExperimentConfig
from ml_skeleton.core.protocols import TrainFunction
from ml_skeleton.tuning.base import BaseTuner


class OptunaTuner(BaseTuner):
    """
    Optuna-based hyperparameter tuner.

    Features:
    - TPE, CMA-ES, Random, and other samplers
    - Median, Hyperband, and other pruners for early stopping
    - MLflow integration for logging all trials
    - Optional persistent storage for resumable studies

    Example:
        tuner = OptunaTuner(
            train_fn=train_model,
            config=experiment_config,
            mlflow_tracking_uri="http://localhost:5000"
        )
        results = tuner.optimize()
        print(f"Best params: {results['best_params']}")
    """

    def __init__(
        self,
        train_fn: TrainFunction,
        config: ExperimentConfig,
        mlflow_tracking_uri: str,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ):
        """
        Initialize the Optuna tuner.

        Args:
            train_fn: User's train_model() function
            config: Experiment configuration
            mlflow_tracking_uri: MLflow tracking server URI
            study_name: Name for the Optuna study (defaults to experiment name)
            storage: Optional Optuna storage URI for persistence
        """
        super().__init__(train_fn, config, mlflow_tracking_uri)
        self.study_name = study_name or f"{config.name}_optuna"
        self.storage = storage or config.tuning.optuna_storage

    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample hyperparameters from the search space using Optuna.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of sampled hyperparameter values
        """
        params = {}
        search_space = self.config.tuning.search_space.parameters

        for name, space_def in search_space.items():
            space_type = space_def["type"]

            if space_type == "categorical":
                params[name] = trial.suggest_categorical(name, space_def["choices"])

            elif space_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    space_def["low"],
                    space_def["high"],
                    step=space_def.get("step", 1),
                    log=space_def.get("log", False),
                )

            elif space_type == "float":
                params[name] = trial.suggest_float(
                    name,
                    space_def["low"],
                    space_def["high"],
                    step=space_def.get("step"),
                    log=space_def.get("log", False),
                )

            elif space_type == "loguniform":
                params[name] = trial.suggest_float(
                    name, space_def["low"], space_def["high"], log=True
                )

        return params

    def _get_sampler(self) -> optuna.samplers.BaseSampler:
        """Get the configured sampler."""
        sampler_name = self.config.tuning.sampler

        samplers = {
            "TPESampler": optuna.samplers.TPESampler,
            "CmaEsSampler": optuna.samplers.CmaEsSampler,
            "RandomSampler": optuna.samplers.RandomSampler,
            "GridSampler": optuna.samplers.GridSampler,
        }

        sampler_class = samplers.get(sampler_name, optuna.samplers.TPESampler)
        return sampler_class(seed=self.config.seed)

    def _get_pruner(self) -> optuna.pruners.BasePruner:
        """Get the configured pruner."""
        pruner_name = self.config.tuning.pruner

        pruners = {
            "MedianPruner": optuna.pruners.MedianPruner,
            "SuccessiveHalvingPruner": optuna.pruners.SuccessiveHalvingPruner,
            "HyperbandPruner": optuna.pruners.HyperbandPruner,
            "NopPruner": optuna.pruners.NopPruner,
        }

        return pruners.get(pruner_name, optuna.pruners.MedianPruner)()

    def _create_objective(self) -> Callable[[optuna.Trial], float]:
        """Create Optuna objective function."""

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters from search space
            hyperparameters = self._sample_hyperparameters(trial)

            # Merge with default hyperparameters
            merged_params = {**self.config.hyperparameters, **hyperparameters}

            # Build training context
            ctx = self._build_context(
                hyperparameters=merged_params,
                trial_id=str(trial.number),
                trial_number=trial.number,
            )

            # Attach trial to context for pruning support
            ctx._optuna_trial = trial  # type: ignore

            # Run training with MLflow tracking
            with ctx.tracker:
                # Log trial info
                ctx.tracker.set_tags(
                    {
                        "tuner": "optuna",
                        "trial_number": str(trial.number),
                    }
                )
                ctx.tracker.log_params(merged_params)

                try:
                    # Execute user's train function
                    result = self.train_fn(ctx)

                    # Log final metrics
                    ctx.tracker.log_metrics(result.metrics)
                    ctx.tracker.log_metric(
                        result.primary_metric_name, result.primary_metric
                    )

                    return result.primary_metric

                except optuna.TrialPruned:
                    raise

        return objective

    def optimize(self) -> Dict[str, Any]:
        """
        Run Optuna optimization.

        Returns:
            Dictionary containing:
            - best_params: Best hyperparameters found
            - best_value: Best metric value
            - n_trials: Number of completed trials
            - study: The Optuna study object
        """
        # Set up MLflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.name)

        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self._get_sampler(),
            pruner=self._get_pruner(),
            direction="minimize",  # Assumes minimization; could be configurable
            load_if_exists=True,
        )

        # Create parent run for the entire study
        with mlflow.start_run(run_name=f"optuna_study_{self.study_name}"):
            mlflow.set_tag("tuner", "optuna")
            mlflow.set_tag("study_name", self.study_name)
            mlflow.log_params(
                {
                    "n_trials": self.config.tuning.n_trials,
                    "sampler": self.config.tuning.sampler,
                    "pruner": self.config.tuning.pruner,
                }
            )

            # Run optimization
            study.optimize(
                self._create_objective(),
                n_trials=self.config.tuning.n_trials,
                timeout=self.config.tuning.timeout,
                show_progress_bar=True,
            )

            # Log best results
            mlflow.log_params(
                {f"best_{k}": v for k, v in study.best_params.items()}
            )
            mlflow.log_metric("best_value", study.best_value)

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "study": study,
        }
