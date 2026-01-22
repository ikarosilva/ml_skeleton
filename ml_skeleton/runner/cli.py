"""
Command-line interface for the ml_skeleton framework.

Provides commands for running experiments, tuning, and managing MLflow.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Optional
import ast

import click


@click.group()
@click.version_option(version="0.1.0", prog_name="ml-skeleton")
def cli():
    """
    ml_skeleton - Deep Learning Training Framework

    A framework for training ML models with MLflow tracking
    and hyperparameter tuning (Optuna/Ray Tune).
    """
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--train-fn",
    required=True,
    help="Train function as 'module:function' (e.g., 'my_module:train_model')",
)
@click.option(
    "--override",
    "-o",
    multiple=True,
    help="Override config values as 'key=value'",
)
def run(config_path: str, train_fn: str, override: tuple):
    """
    Run a single training experiment.

    CONFIG_PATH: Path to YAML configuration file
    """
    from ml_skeleton.core.config import ExperimentConfig
    from ml_skeleton.runner.experiment import run_experiment

    # Load configuration
    config = ExperimentConfig.from_yaml(config_path)

    # Apply overrides
    for item in override:
        key, value = item.split("=", 1)
        _apply_override(config, key, value)

    # Load train function
    train_function = _load_function(train_fn)

    # Run experiment
    click.echo(f"Starting experiment: {config.name}")
    result = run_experiment(train_function, config, tune=False)

    click.echo(f"\nTraining completed!")
    click.echo(f"  {result.primary_metric_name}: {result.primary_metric:.6f}")
    click.echo(f"  Epochs completed: {result.epochs_completed}")
    if result.best_model_path:
        click.echo(f"  Best model saved: {result.best_model_path}")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--train-fn",
    required=True,
    help="Train function as 'module:function' (e.g., 'my_module:train_model')",
)
@click.option(
    "--n-trials",
    type=int,
    default=None,
    help="Number of trials (overrides config)",
)
@click.option(
    "--tuner",
    type=click.Choice(["optuna", "ray_tune"]),
    default=None,
    help="Tuner type (overrides config)",
)
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="Timeout in seconds",
)
def tune(
    config_path: str,
    train_fn: str,
    n_trials: Optional[int],
    tuner: Optional[str],
    timeout: Optional[int],
):
    """
    Run hyperparameter tuning.

    CONFIG_PATH: Path to YAML configuration file
    """
    from ml_skeleton.core.config import ExperimentConfig, TunerType
    from ml_skeleton.runner.experiment import run_experiment

    # Load configuration
    config = ExperimentConfig.from_yaml(config_path)

    # Apply CLI overrides
    if n_trials is not None:
        config.tuning.n_trials = n_trials
    if tuner is not None:
        config.tuning.tuner_type = TunerType(tuner)
    if timeout is not None:
        config.tuning.timeout = timeout

    # Ensure tuner is set
    if config.tuning.tuner_type == TunerType.NONE:
        config.tuning.tuner_type = TunerType.OPTUNA

    # Load train function
    train_function = _load_function(train_fn)

    # Run tuning
    click.echo(f"Starting hyperparameter tuning: {config.name}")
    click.echo(f"  Tuner: {config.tuning.tuner_type.value}")
    click.echo(f"  Trials: {config.tuning.n_trials}")

    results = run_experiment(train_function, config, tune=True)

    click.echo(f"\nTuning completed!")
    click.echo(f"  Best value: {results['best_value']:.6f}")
    click.echo(f"  Best parameters:")
    for key, value in results["best_params"].items():
        click.echo(f"    {key}: {value}")


@cli.command("mlflow-ui")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=5000, type=int, help="Port to run on")
@click.option("--backend", default="sqlite:///mlflow.db", help="Backend store URI")
@click.option("--artifacts", default="./mlruns", help="Artifact root directory")
def mlflow_ui(host: str, port: int, backend: str, artifacts: str):
    """
    Start the MLflow tracking UI.
    """
    from ml_skeleton.tracking.server import MLflowServer

    click.echo(f"Starting MLflow server at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop")

    server = MLflowServer(
        host=host,
        port=port,
        backend_store_uri=backend,
        artifact_root=artifacts,
    )

    try:
        server.start(wait=True)
        # Keep running until interrupted
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nStopping server...")
        server.stop()


@cli.command("gpu-info")
@click.option("--framework", type=click.Choice(["pytorch", "tensorflow", "both"]), default="both")
def gpu_info(framework: str):
    """
    Display GPU information.
    """
    click.echo("GPU Information")
    click.echo("=" * 40)

    if framework in ("pytorch", "both"):
        click.echo("\nPyTorch:")
        try:
            from ml_skeleton.frameworks.pytorch import PyTorchHelper

            info = PyTorchHelper.get_device_info()
            click.echo(f"  CUDA available: {info['cuda_available']}")
            click.echo(f"  CUDA version: {info['cuda_version']}")
            click.echo(f"  cuDNN version: {info['cudnn_version']}")
            click.echo(f"  Device count: {info['device_count']}")
            for key, value in info.items():
                if key.startswith("device_") and isinstance(value, dict):
                    click.echo(f"  {key}:")
                    click.echo(f"    Name: {value['name']}")
                    click.echo(f"    Memory: {value['memory_gb']:.1f} GB")
                    click.echo(f"    Compute: {value['compute_capability']}")
        except ImportError:
            click.echo("  PyTorch not installed")

    if framework in ("tensorflow", "both"):
        click.echo("\nTensorFlow:")
        try:
            from ml_skeleton.frameworks.tensorflow import TensorFlowHelper

            info = TensorFlowHelper.get_device_info()
            click.echo(f"  TensorFlow version: {info['tensorflow_version']}")
            click.echo(f"  GPU available: {info['gpu_available']}")
            click.echo(f"  Device count: {info['device_count']}")
            for device in info["devices"]:
                click.echo(f"    {device['name']}")
        except ImportError:
            click.echo("  TensorFlow not installed")


@cli.command("verify")
def verify():
    """
    Verify environment meets all requirements.
    """
    from ml_skeleton.utils.verify import verify_environment

    result = verify_environment(verbose=True)
    if not result["all_ok"]:
        raise SystemExit(1)


@cli.command("memory")
@click.option("--limit", type=float, default=None, help="Set GPU memory limit in GB")
@click.option("--show", is_flag=True, help="Show current memory usage")
def memory(limit: Optional[float], show: bool):
    """
    Manage GPU memory settings.

    Examples:
        explr memory --show              # Show current usage
        explr memory --limit 24          # Limit to 24GB
    """
    from ml_skeleton.utils.memory import limit_gpu_memory, print_memory_summary

    if limit is not None:
        result = limit_gpu_memory(max_memory_gb=limit)
        if result["warnings"]:
            for warning in result["warnings"]:
                click.echo(f"Warning: {warning}")

    if show or limit is None:
        print_memory_summary()


def _load_function(function_path: str):
    """
    Load a function from a module path.

    Args:
        function_path: Path as 'module:function'

    Returns:
        The loaded function
    """
    if ":" not in function_path:
        raise click.BadParameter(
            f"Invalid function path: {function_path}. "
            "Expected format: 'module:function'"
        )

    module_path, func_name = function_path.rsplit(":", 1)

    # Add current directory to path
    if "" not in sys.path:
        sys.path.insert(0, "")

    try:
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        return func
    except (ImportError, AttributeError) as e:
        raise click.BadParameter(f"Could not load function: {e}")


def _apply_override(config, key: str, value: str):
    """Apply a config override from CLI."""
    parts = key.split(".")
    obj = config

    for part in parts[:-1]:
        obj = getattr(obj, part)

    # Try to convert value to appropriate type
    try:
        converted = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Keep as string if it's not a literal
        converted = value

    setattr(obj, parts[-1], converted)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
