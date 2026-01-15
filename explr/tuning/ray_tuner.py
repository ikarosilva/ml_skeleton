"""
Ray Tune-based hyperparameter tuner.

Provides distributed hyperparameter optimization with advanced
schedulers like ASHA and Population Based Training.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import mlflow

from explr.core.config import ExperimentConfig
from explr.core.protocols import TrainFunction
from explr.tuning.base import BaseTuner


class RayTuneTuner(BaseTuner):
    """
    Ray Tune-based hyperparameter tuner.

    Features:
    - Distributed training across multiple GPUs/nodes
    - ASHA scheduler for efficient early stopping
    - Population Based Training for adaptive hyperparameters
    - Integration with Optuna search algorithm
    - MLflow logging for all trials

    Example:
        tuner = RayTuneTuner(
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
        ray_address: Optional[str] = None,
    ):
        """
        Initialize the Ray Tune tuner.

        Args:
            train_fn: User's train_model() function
            config: Experiment configuration
            mlflow_tracking_uri: MLflow tracking server URI
            ray_address: Optional Ray cluster address (None for local)
        """
        super().__init__(train_fn, config, mlflow_tracking_uri)
        self.ray_address = ray_address

    def _build_search_space(self) -> Dict[str, Any]:
        """
        Convert config search space to Ray Tune format.

        Returns:
            Dictionary with Ray Tune search space definitions
        """
        # Import here to avoid requiring ray when using optuna
        from ray import tune

        ray_space = {}
        search_space = self.config.tuning.search_space.parameters

        for name, space_def in search_space.items():
            space_type = space_def["type"]

            if space_type == "categorical":
                ray_space[name] = tune.choice(space_def["choices"])

            elif space_type == "int":
                if space_def.get("log", False):
                    ray_space[name] = tune.lograndint(
                        space_def["low"], space_def["high"]
                    )
                else:
                    ray_space[name] = tune.randint(
                        space_def["low"], space_def["high"]
                    )

            elif space_type == "float":
                if space_def.get("log", False):
                    ray_space[name] = tune.loguniform(
                        space_def["low"], space_def["high"]
                    )
                else:
                    ray_space[name] = tune.uniform(
                        space_def["low"], space_def["high"]
                    )

            elif space_type == "loguniform":
                ray_space[name] = tune.loguniform(
                    space_def["low"], space_def["high"]
                )

        return ray_space

    def _get_scheduler(self) -> Any:
        """Get the configured scheduler."""
        from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

        scheduler_name = self.config.tuning.scheduler

        if scheduler_name == "ASHAScheduler":
            return ASHAScheduler(
                metric="primary_metric",
                mode="min",
                max_t=100,
                grace_period=10,
                reduction_factor=3,
            )
        elif scheduler_name == "PopulationBasedTraining":
            return PopulationBasedTraining(
                metric="primary_metric",
                mode="min",
                perturbation_interval=5,
            )
        else:
            return ASHAScheduler(
                metric="primary_metric",
                mode="min",
            )

    def _get_search_algorithm(self) -> Optional[Any]:
        """Get the configured search algorithm."""
        from ray.tune.search.optuna import OptunaSearch

        search_alg = self.config.tuning.search_alg

        if search_alg == "OptunaSearch":
            return OptunaSearch(
                metric="primary_metric",
                mode="min",
                seed=self.config.seed,
            )
        return None

    def _create_objective(self) -> Callable:
        """Create Ray Tune trainable function."""
        from ray import tune

        train_fn = self.train_fn
        config = self.config
        mlflow_uri = self.mlflow_tracking_uri

        def trainable(ray_config: Dict[str, Any]) -> None:
            """Ray Tune trainable that wraps user's train function."""
            from explr.tracking.client import ExplrTracker
            from explr.core.protocols import TrainingContext

            # Merge with default hyperparameters
            merged_params = {**config.hyperparameters, **ray_config}

            # Build context with sampled hyperparameters
            tracker = ExplrTracker(
                tracking_uri=mlflow_uri,
                experiment_name=config.name,
                nested=True,
            )

            ctx = TrainingContext(
                hyperparameters=merged_params,
                tracker=tracker,
                trial_id=tune.get_trial_id(),
                experiment_name=config.name,
                seed=config.seed,
                checkpoint_dir=config.checkpoint_dir,
                artifact_dir=config.artifact_dir,
            )

            # Run training with MLflow tracking
            with tracker:
                tracker.set_tags(
                    {
                        "tuner": "ray_tune",
                        "trial_id": tune.get_trial_id(),
                    }
                )
                tracker.log_params(merged_params)

                # Execute user's train function
                result = train_fn(ctx)

                # Report to Ray Tune
                tune.report(
                    primary_metric=result.primary_metric,
                    **result.metrics,
                )

        return trainable

    def optimize(self) -> Dict[str, Any]:
        """
        Run Ray Tune optimization.

        Returns:
            Dictionary containing:
            - best_params: Best hyperparameters found
            - best_value: Best metric value
            - results: Ray Tune ResultGrid object
        """
        import ray
        from ray import tune
        from ray.air import RunConfig
        from ray.air.integrations.mlflow import MLflowLoggerCallback

        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(address=self.ray_address)

        # Set up MLflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.name)

        # Build search space
        search_space = self._build_search_space()

        # Create MLflow callback
        mlflow_callback = MLflowLoggerCallback(
            tracking_uri=self.mlflow_tracking_uri,
            experiment_name=self.config.name,
            save_artifact=True,
        )

        # Run tuning
        tuner = tune.Tuner(
            self._create_objective(),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=self.config.tuning.num_samples,
                max_concurrent_trials=self.config.tuning.max_concurrent_trials,
                scheduler=self._get_scheduler(),
                search_alg=self._get_search_algorithm(),
            ),
            run_config=RunConfig(
                name=f"{self.config.name}_ray_tune",
                callbacks=[mlflow_callback],
                storage_path="./ray_results",
            ),
        )

        results = tuner.fit()
        best_result = results.get_best_result(metric="primary_metric", mode="min")

        return {
            "best_params": best_result.config,
            "best_value": best_result.metrics.get("primary_metric"),
            "results": results,
        }
