"""
Optuna-based hyperparameter tuner.

Provides integration with Optuna for single-machine hyperparameter
optimization with advanced features like pruning and various samplers.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import mlflow
import optuna
from mlflow.tracking import MlflowClient

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
                # Optuna SQL storage only supports categorical choices that are None, bool, int, float, str.
                # For list/tuple choices (e.g. hidden_dims), use string form only to avoid
                # "CategoricalDistribution does not support dynamic value space" and storage warnings.
                choices = space_def["choices"]
                has_complex = any(isinstance(c, (list, tuple)) for c in choices)
                if has_complex:
                    choices_for_storage = tuple(str(c) for c in choices)
                    params[name] = trial.suggest_categorical(name, choices_for_storage)
                    val = params[name]
                    params[name] = ast.literal_eval(val) if isinstance(val, str) and val.startswith("[") else list(val) if isinstance(val, tuple) else val
                else:
                    choices_for_storage = tuple(choices)
                    params[name] = trial.suggest_categorical(name, choices_for_storage)
                    val = params[name]
                    if isinstance(val, tuple):
                        params[name] = list(val)
                    elif isinstance(val, list) and val and isinstance(val[0], tuple):
                        params[name] = [list(x) for x in val]

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

            # Build training context (parent_run_id required so trials nest under HPO root)
            parent_run_id = getattr(self, "_parent_run_id", None)
            if parent_run_id is None:
                raise RuntimeError(
                    "Optuna HPO: _parent_run_id is not set. Trials must run inside "
                    "optimize()'s 'with mlflow.start_run(...)' block so trials nest under the parent."
                )
            ctx = self._build_context(
                hyperparameters=merged_params,
                trial_id=str(trial.number),
                trial_number=trial.number,
                parent_run_id=parent_run_id,
            )

            # Attach trial to context for pruning support
            ctx._optuna_trial = trial  # type: ignore

            # Run training with MLflow tracking
            with ctx.tracker:
                # Log trial info and config tags (e.g. version); run_type marks nested trial
                tags = {
                    "tuner": "optuna",
                    "run_type": "hpo_trial",
                    "trial_number": str(trial.number),
                }
                if self.config.tags:
                    tags.update(self.config.tags)
                ctx.tracker.set_tags(tags)
                # Only log the sampled hyperparameters (not the full merged config)
                # This avoids logging nested dicts which cause MLflow errors
                ctx.tracker.log_params(hyperparameters)

                try:
                    # Execute user's train function
                    result = self.train_fn(ctx)

                    # Log final metrics (MLflow requires real numbers; skip None/non-numeric)
                    safe_metrics = {
                        k: v for k, v in result.metrics.items()
                        if v is not None and isinstance(v, (int, float))
                    }
                    if safe_metrics:
                        ctx.tracker.log_metrics(safe_metrics)
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
        # Set up MLflow: ensure experiment exists and is active (create if missing or if deleted).
        # Store resolved name so trial trackers use the same experiment (not the deleted one).
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        client = MlflowClient()
        exp = client.get_experiment_by_name(self.config.name)
        if exp is None:
            mlflow.set_experiment(self.config.name)
            self._mlflow_experiment_name = self.config.name
        elif getattr(exp, "lifecycle_stage", None) == "deleted":
            new_name = f"{self.config.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            client.create_experiment(new_name)
            mlflow.set_experiment(new_name)
            self._mlflow_experiment_name = new_name
        else:
            mlflow.set_experiment(self.config.name)
            self._mlflow_experiment_name = self.config.name

        # Optional: delete existing study so this run starts fresh (e.g. --reset-study)
        reset_study = getattr(self.config.tuning, "reset_study", False)
        if reset_study and self.storage:
            try:
                optuna.delete_study(study_name=self.study_name, storage=self.storage)
            except KeyError:
                pass  # study did not exist

        # Create study (load_if_exists=False when we just reset, else True)
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self._get_sampler(),
            pruner=self._get_pruner(),
            direction="minimize",  # Assumes minimization; could be configurable
            load_if_exists=not reset_study,
        )

        # Create parent run for the entire study (all trial runs must be nested under this).
        # Do not run study.optimize() outside this block: trials need self._parent_run_id set.
        with mlflow.start_run(run_name=f"optuna_study_{self.study_name}") as parent_run:
            self._parent_run_id = parent_run.info.run_id
            mlflow.set_tag("tuner", "optuna")
            mlflow.set_tag("run_type", "hpo_parent")
            mlflow.set_tag("study_name", self.study_name)
            if self.config.tags:
                mlflow.set_tags(self.config.tags)
            mlflow.log_params(
                {
                    "n_trials": self.config.tuning.n_trials,
                    "sampler": self.config.tuning.sampler,
                    "pruner": self.config.tuning.pruner,
                }
            )

            # Run optimization (each trial creates a nested run under parent_run_id)
            n_trials_before = len(study.trials)
            study.optimize(
                self._create_objective(),
                n_trials=self.config.tuning.n_trials,
                timeout=self.config.tuning.timeout,
                show_progress_bar=True,
            )
            n_trials_after = len(study.trials)
            if n_trials_after == n_trials_before and n_trials_before >= self.config.tuning.n_trials:
                print(
                    f"\n[Optuna] Study '{self.study_name}' already has {n_trials_before} trials (requested {self.config.tuning.n_trials}). "
                    "No new trials run; best result above is from existing study. "
                    "Use --reset-study to start fresh, or increase -N to run more trials."
                )

            # At root (parent) run: log best results and parameters (MLflow params must be strings)
            n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            mlflow.log_param("n_completed_trials", n_completed)
            if n_completed > 0:
                mlflow.log_param("best_trial_number", study.best_trial.number)
                mlflow.log_metric("best_value", study.best_value)
                mlflow.set_tag("best_trial", str(study.best_trial.number))
                # Convert param values to strings (MLflow rejects list/dict; use json for complex types)
                best_params_str = {}
                for k, v in study.best_params.items():
                    if isinstance(v, (list, dict)):
                        best_params_str[f"best_{k}"] = json.dumps(v)
                    else:
                        best_params_str[f"best_{k}"] = str(v)
                mlflow.log_params(best_params_str)

                # Copy all metrics from the best-trial run to the parent (same names as in trials)
                client = MlflowClient()
                parent_run = mlflow.active_run()
                if parent_run:
                    children = client.search_runs(
                        experiment_ids=[parent_run.info.experiment_id],
                        filter_string=f'tags."mlflow.parentRunId" = "{self._parent_run_id}"',
                        max_results=500,
                    )
                    best_trial_run = None
                    for r in children:
                        if r.data.tags.get("trial_number") == str(study.best_trial.number):
                            best_trial_run = r
                            break
                    if best_trial_run is not None:
                        run = client.get_run(best_trial_run.info.run_id)
                        for key, value in run.data.metrics.items():
                            if value is not None and isinstance(value, (int, float)):
                                mlflow.log_metric(key, value)

            # Optional parent run display name: "prefix-{run_id}" (e.g. classifier-tune-hpo-abc123)
            prefix = getattr(self.config.tuning, "parent_run_name_prefix", None)
            if prefix:
                mlflow.set_tag("mlflow.runName", f"{prefix}-{self._parent_run_id}")

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "study": study,
            "mlflow_parent_run_id": getattr(self, "_parent_run_id", None),
        }
