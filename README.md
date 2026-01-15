# explr

A deep learning training framework with MLflow experiment tracking and hyperparameter tuning (Optuna/Ray Tune).

## Features

- **Simple Interface**: Implement a single `train_model()` function with your training logic
- **MLflow Integration**: Automatic experiment tracking, metric logging, and artifact storage
- **Hyperparameter Tuning**: Support for Optuna and Ray Tune with pruning
- **Multi-Framework**: Works with both PyTorch and TensorFlow
- **Docker Ready**: Pre-configured for NVIDIA GPUs (including RTX 5090)

## Installation

```bash
# Basic installation
pip install -e .

# With PyTorch
pip install -e ".[pytorch]"

# With TensorFlow
pip install -e ".[tensorflow]"

# With Ray Tune
pip install -e ".[ray]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### 1. Define Your Training Function

```python
from explr import TrainingContext, TrainingResult

def train_model(ctx: TrainingContext) -> TrainingResult:
    """Your training function receives a context and returns results."""

    # Get hyperparameters
    lr = ctx.hyperparameters.get("learning_rate", 0.001)
    epochs = ctx.hyperparameters.get("epochs", 10)

    # Your training logic here...
    model = create_model()

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader)
        val_loss = validate(model, val_loader)

        # Log metrics using the provided tracker
        ctx.tracker.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)

    # Return results
    return TrainingResult(
        primary_metric=val_loss,
        primary_metric_name="val_loss",
        metrics={"final_accuracy": accuracy}
    )
```

### 2. Run a Single Experiment

```python
from explr import run_experiment, ExperimentConfig

config = ExperimentConfig(
    name="my_experiment",
    hyperparameters={"learning_rate": 0.001, "epochs": 50}
)

result = run_experiment(train_model, config)
print(f"Final loss: {result.primary_metric}")
```

### 3. Run Hyperparameter Tuning

```python
from explr import run_experiment, ExperimentConfig, TunerType
from explr.tuning import SearchSpaceBuilder

# Define search space
search_space = (
    SearchSpaceBuilder()
    .loguniform("learning_rate", 1e-5, 1e-1)
    .categorical("batch_size", [16, 32, 64])
    .uniform("dropout", 0.0, 0.5)
    .build()
)

config = ExperimentConfig(name="tuning_experiment")
config.tuning.tuner_type = TunerType.OPTUNA
config.tuning.n_trials = 50
config.tuning.search_space.parameters = search_space

results = run_experiment(train_model, config, tune=True)
print(f"Best params: {results['best_params']}")
```

## CLI Usage

```bash
# Run a single training
explr run config.yaml --train-fn my_module:train_model

# Run hyperparameter tuning
explr tune config.yaml --train-fn my_module:train_model --n-trials 50

# Start MLflow UI
explr mlflow-ui

# Show GPU info
explr gpu-info

# Verify environment
explr verify

# Show/set GPU memory limits
explr memory --show
explr memory --limit 24
```

## Docker Setup

### Using Docker Compose

```bash
# Start MLflow server
docker-compose up -d mlflow

# Run training environment
docker-compose run --rm training bash

# Inside container
cd /workspace
python examples/pytorch_example.py
```

### Using Existing kaggle:torch Image

```bash
# Start container with GPU support
docker run --shm-size 50G --runtime=nvidia --privileged -it \
  -p 8888:8888 -p 5000:5000 \
  --env TF_ENABLE_ONEDNN_OPTS=0 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v ~/git:/git \
  -v ~/PycharmProjects:/projects \
  --rm kaggle:torch bash

# Inside container
cd /git/explr
pip install -e .

# Start MLflow server (background)
mlflow server --host 0.0.0.0 --port 5000 &

# Run your experiment
python examples/pytorch_example.py
```

### Docker Commands via Makefile

```bash
make docker-up       # Start MLflow
make docker-shell    # Open training shell
make docker-down     # Stop services
make docker-kaggle   # Use kaggle:torch image
```

## Configuration

### YAML Configuration

```yaml
# configs/example.yaml
name: "my_experiment"
framework: "pytorch"

hyperparameters:
  epochs: 50
  batch_size: 32

mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "my_experiment"
  auto_start: true

tuning:
  tuner_type: "optuna"
  n_trials: 100
  sampler: "TPESampler"
  pruner: "MedianPruner"
  search_space:
    parameters:
      learning_rate:
        type: "loguniform"
        low: 0.00001
        high: 0.1
      batch_size:
        type: "categorical"
        choices: [16, 32, 64]

seed: 42
```

### Programmatic Configuration

```python
from explr import ExperimentConfig, TuningConfig, TunerType

config = ExperimentConfig(
    name="my_experiment",
    framework="pytorch",
    seed=42,
    tuning=TuningConfig(
        tuner_type=TunerType.OPTUNA,
        n_trials=100
    )
)
```

## Project Structure

```
explr/
├── core/           # Core protocols and configuration
├── tracking/       # MLflow integration
├── tuning/         # Optuna and Ray Tune
├── frameworks/     # PyTorch and TensorFlow helpers
├── runner/         # Experiment orchestration and CLI
└── utils/          # Reproducibility and GPU utilities
```

## API Reference

### TrainingContext

```python
@dataclass
class TrainingContext:
    hyperparameters: Dict[str, Any]  # Current hyperparameters
    tracker: ExplrTracker            # MLflow tracker
    trial_id: Optional[str]          # Trial ID (tuning)
    trial_number: Optional[int]      # Trial number (tuning)
    experiment_name: str             # Experiment name
    device: str                      # Device ("cuda" or "cpu")
    checkpoint_dir: str              # Checkpoint directory
    seed: Optional[int]              # Random seed
```

### TrainingResult

```python
@dataclass
class TrainingResult:
    primary_metric: float            # Metric for optimization
    primary_metric_name: str         # Metric name
    minimize: bool                   # Minimize or maximize
    metrics: Dict[str, float]        # Additional metrics
    best_model_path: Optional[str]   # Path to best model
    epochs_completed: int            # Epochs completed
```

### ExplrTracker Methods

```python
tracker.log_metric("loss", 0.5, step=epoch)
tracker.log_metrics({"loss": 0.5, "acc": 0.9}, step=epoch)
tracker.log_params({"lr": 0.001})
tracker.log_artifact("model.pt")
tracker.set_tag("version", "1.0")
```

## GPU Memory Management

Control how much GPU memory the framework uses. **Default: 24GB** (leaves 8GB free on a 32GB RTX 5090 for system/desktop).

### Default Behavior

```python
from explr.utils.memory import limit_gpu_memory

# Uses default 24GB limit
limit_gpu_memory()
```

### Custom Limit

```python
# Set a different limit
limit_gpu_memory(max_memory_gb=16)

# Disable limit entirely (use all GPU memory)
limit_gpu_memory(max_memory_gb=0)
```

### Environment Variable Override

```bash
# Override the default via environment variable
export EXPLR_GPU_MEMORY_GB=28
python my_training.py

# Disable limit via environment variable
export EXPLR_GPU_MEMORY_GB=0
```

### In train_model()

```python
def train_model(ctx: TrainingContext) -> TrainingResult:
    from explr.utils.memory import limit_gpu_memory

    # Get limit from hyperparameters or use default (24GB)
    max_mem = ctx.hyperparameters.get("max_gpu_memory_gb")
    limit_gpu_memory(max_memory_gb=max_mem)

    # ... training code ...
```

### CLI

```bash
explr memory --show        # Show current usage
explr memory --limit 16    # Set custom limit
```

## Environment Verification

Verify your environment has all required dependencies:

```bash
# Via CLI
explr verify

# Via Python
python -m explr.utils.verify
```

This checks Python version, PyTorch/TensorFlow installation, CUDA availability, and GPU detection.

## Requirements

- Python >= 3.10
- CUDA 12.8+ (for RTX 5090 Blackwell support)
- MLflow >= 2.10.0
- Optuna >= 3.5.0
- PyTorch >= 2.5.0 (optional, required for Blackwell GPUs)
- TensorFlow >= 2.18.0 (optional)
- Ray[tune] >= 2.9.0 (optional)

## License

MIT License
