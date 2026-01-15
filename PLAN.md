# Deep Learning Training Framework - Implementation Plan

## Overview

Create a skeleton framework called `explr` for training and deploying deep learning models with:
- User-provided `train_model()` entry point
- MLflow server integration for experiment tracking
- Hyperparameter tuning with Optuna and Ray Tune
- Support for PyTorch and TensorFlow
- Docker/CUDA environment for RTX 5090

---

## Target Environment

```
GPU: NVIDIA GeForce RTX 5090 (Blackwell architecture)
VRAM: 32GB
Driver: 570.195.03
CUDA: 12.8
```

---

## Design Choices (Based on User Preferences)

- **CLI + API**: Both CLI interface (`explr run config.yaml`) and programmatic API
- **MLflow Auto-start**: Framework automatically starts MLflow server if not running
- **Examples**: Minimal skeleton implementations showing the interface
- **Single GPU**: Start with single GPU support, distributed training can be added later

---

## Project Structure

```
explr/
├── pyproject.toml
├── requirements.txt
├── README.md
├── docker-compose.yml
├── Dockerfile
├── Makefile
│
├── explr/
│   ├── __init__.py
│   │
│   ├── core/                         # Core abstractions
│   │   ├── __init__.py
│   │   ├── protocols.py              # TrainFunction protocol, TrainingContext, TrainingResult
│   │   └── config.py                 # Configuration dataclasses
│   │
│   ├── tracking/                     # MLflow integration
│   │   ├── __init__.py
│   │   ├── server.py                 # MLflow server management (auto-start)
│   │   └── client.py                 # ExplrTracker wrapper for train_model()
│   │
│   ├── tuning/                       # Hyperparameter optimization
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseTuner abstract class
│   │   ├── optuna_tuner.py           # Optuna integration
│   │   ├── ray_tuner.py              # Ray Tune integration
│   │   └── search_space.py           # Unified search space builder
│   │
│   ├── frameworks/                   # Multi-framework support
│   │   ├── __init__.py
│   │   ├── pytorch.py                # PyTorch utilities
│   │   └── tensorflow.py             # TensorFlow utilities
│   │
│   ├── runner/                       # Experiment execution
│   │   ├── __init__.py
│   │   ├── experiment.py             # Main Experiment orchestrator
│   │   └── cli.py                    # CLI interface (explr command)
│   │
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── seed.py                   # Reproducibility
│       └── gpu.py                    # GPU detection
│
├── configs/                          # Example configurations
│   └── example.yaml
│
└── examples/                         # Example implementations
    ├── pytorch_example.py            # Minimal PyTorch skeleton
    └── tensorflow_example.py         # Minimal TensorFlow skeleton
```

---

## Core Interface Design

### 1. TrainingContext (what train_model() receives)

```python
@dataclass
class TrainingContext:
    hyperparameters: Dict[str, Any]      # Hyperparameters for this trial
    tracker: ExplrTracker                 # MLflow client for logging
    trial_id: Optional[str] = None        # Trial ID (for tuning)
    trial_number: Optional[int] = None    # Trial number (for tuning)
    experiment_name: str = "default"
    device: str = "cuda"
    checkpoint_dir: str = "./checkpoints"
    seed: Optional[int] = None
```

### 2. TrainingResult (what train_model() returns)

```python
@dataclass
class TrainingResult:
    primary_metric: float                 # Required: metric for optimization
    primary_metric_name: str = "val_loss"
    minimize: bool = True
    metrics: Dict[str, float] = None      # Additional metrics
    best_model_path: Optional[str] = None # Path to saved model
    epochs_completed: int = 0
```

### 3. User's train_model() signature

```python
def train_model(ctx: TrainingContext) -> TrainingResult:
    # User implements:
    # - Data loading
    # - Model creation
    # - Training loop
    # - Logging via ctx.tracker.log_metric()
    # - Model saving
    return TrainingResult(primary_metric=val_loss, ...)
```

---

## Implementation Steps

### Phase 1: Core Module
1. Create `explr/core/protocols.py` - Define TrainingContext, TrainingResult, TrainFunction protocol
2. Create `explr/core/config.py` - ExperimentConfig, TuningConfig, MLflowConfig dataclasses

### Phase 2: Tracking Module
3. Create `explr/tracking/client.py` - ExplrTracker class wrapping MLflow
4. Create `explr/tracking/server.py` - MLflowServer lifecycle management

### Phase 3: Tuning Module
5. Create `explr/tuning/search_space.py` - SearchSpaceBuilder for defining hyperparameter spaces
6. Create `explr/tuning/base.py` - BaseTuner abstract class
7. Create `explr/tuning/optuna_tuner.py` - Optuna integration
8. Create `explr/tuning/ray_tuner.py` - Ray Tune integration

### Phase 4: Framework Support
9. Create `explr/frameworks/pytorch.py` - PyTorch helper utilities
10. Create `explr/frameworks/tensorflow.py` - TensorFlow helper utilities

### Phase 5: Runner
11. Create `explr/runner/experiment.py` - Main Experiment class
12. Create `explr/runner/cli.py` - CLI interface with Click

### Phase 6: Utilities
13. Create `explr/utils/seed.py` - Reproducibility utilities
14. Create `explr/utils/gpu.py` - GPU detection and configuration

### Phase 7: Project Setup
15. Create `pyproject.toml` - Package configuration
16. Create `requirements.txt` - Dependencies
17. Create `Dockerfile` - CUDA-enabled training container
18. Create `docker-compose.yml` - MLflow + training orchestration
19. Create `Makefile` - Common commands
20. Create `README.md` - Documentation with Docker usage

### Phase 8: Examples
21. Create `examples/pytorch_example.py` - Minimal PyTorch skeleton
22. Create `examples/tensorflow_example.py` - Minimal TensorFlow skeleton
23. Create `configs/example.yaml` - Example YAML configuration

---

## Key Dependencies (requirements.txt)

```
# Core ML Frameworks
torch>=2.2.0
tensorflow>=2.15.0

# Experiment Tracking
mlflow>=2.10.0

# Hyperparameter Tuning
optuna>=3.5.0
ray[tune]>=2.9.0

# Configuration
pyyaml>=6.0
python-dotenv>=1.0.0

# Utilities
numpy>=1.26.0
tqdm>=4.66.0
click>=8.1.0
```

---

## Docker Configuration

### Dockerfile
```dockerfile
FROM nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.11 python3-pip

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 5000 8888
```

### docker-compose.yml
```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
    command: mlflow server --host 0.0.0.0 --port 5000

  training:
    build: .
    runtime: nvidia
    shm_size: 50G
    privileged: true
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - TF_ENABLE_ONEDNN_OPTS=0
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ~/git:/git
      - ~/PycharmProjects:/projects
    depends_on:
      - mlflow
```

### Usage Documentation (README.md)
```bash
# Start MLflow + training environment
docker-compose up -d mlflow
docker-compose run --rm training bash

# Or use existing kaggle:torch image
docker run --shm-size 50G --runtime=nvidia --privileged -it \
  -p 8888:8888 -p 5000:5000 \
  --env TF_ENABLE_ONEDNN_OPTS=0 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v ~/git:/git -v ~/PycharmProjects:/projects \
  --rm kaggle:torch bash

# Inside container
cd /git/explr
pip install -e .
mlflow server --host 0.0.0.0 --port 5000 &
python examples/pytorch_example.py
```

---

## How Tuning Works

### Optuna Flow
1. `OptunaTuner.optimize()` creates Optuna Study
2. For each trial, Optuna samples hyperparameters
3. `TrainingContext` built with sampled params + MLflow tracker
4. User's `train_model(ctx)` called
5. `TrainingResult.primary_metric` returned to Optuna
6. Best parameters logged to MLflow

### Ray Tune Flow
1. `RayTuneTuner.optimize()` creates Ray search space
2. Ray Tune scheduler (ASHA) manages parallel trials
3. Each trial receives sampled hyperparameters
4. `train_model(ctx)` called with `tune.report()` integration
5. Results aggregated by Ray Tune

---

## Verification Plan

1. **Unit Tests**: Run `pytest tests/` to verify each module
2. **Integration Test**: Run `examples/pytorch_example.py` end-to-end
3. **MLflow UI**: Verify experiments visible at http://localhost:5000
4. **Docker**: Test container startup with GPU access
5. **Tuning**: Run 10-trial Optuna optimization, verify pruning works

---

## Files to Create

| File | Purpose |
|------|---------|
| `explr/core/protocols.py` | TrainingContext, TrainingResult, TrainFunction |
| `explr/core/config.py` | Configuration dataclasses |
| `explr/tracking/client.py` | ExplrTracker MLflow wrapper |
| `explr/tracking/server.py` | MLflow server management |
| `explr/tuning/search_space.py` | Search space builder |
| `explr/tuning/base.py` | BaseTuner abstract class |
| `explr/tuning/optuna_tuner.py` | Optuna integration |
| `explr/tuning/ray_tuner.py` | Ray Tune integration |
| `explr/frameworks/pytorch.py` | PyTorch utilities |
| `explr/frameworks/tensorflow.py` | TensorFlow utilities |
| `explr/runner/experiment.py` | Experiment orchestrator |
| `explr/runner/cli.py` | CLI interface |
| `explr/utils/seed.py` | Reproducibility |
| `explr/utils/gpu.py` | GPU detection |
| `explr/utils/memory.py` | GPU memory management |
| `explr/utils/verify.py` | Environment verification |
| `pyproject.toml` | Package config |
| `requirements.txt` | Dependencies |
| `Dockerfile` | Training container |
| `docker-compose.yml` | Service orchestration |
| `Makefile` | Common commands |
| `README.md` | Documentation |
| `examples/pytorch_example.py` | Minimal PyTorch skeleton |
| `examples/tensorflow_example.py` | Minimal TensorFlow skeleton |
| `configs/example.yaml` | Example config |

---

## CLI Usage

```bash
# Run a single training
explr run config.yaml --train-fn my_module:train_model

# Run hyperparameter tuning
explr tune config.yaml --train-fn my_module:train_model --n-trials 50

# Start MLflow UI (auto-started, but can be manual)
explr mlflow-ui

# Show GPU info
explr gpu-info

# Verify environment
explr verify

# GPU memory management
explr memory --show
explr memory --limit 24
```

---

## GPU Memory Management

The framework limits GPU memory usage by default to share resources with other applications.

**Default: 24GB** (leaves 8GB free on a 32GB RTX 5090 for system/desktop)

### Methods:

1. **Default behavior** (24GB limit):
   ```python
   from explr.utils.memory import limit_gpu_memory
   limit_gpu_memory()  # Uses default 24GB
   ```

2. **Custom limit**:
   ```python
   limit_gpu_memory(max_memory_gb=16)
   ```

3. **Disable limit** (use all GPU memory):
   ```python
   limit_gpu_memory(max_memory_gb=0)
   # Or: export EXPLR_GPU_MEMORY_GB=0
   ```

4. **Environment Variable override**:
   ```bash
   export EXPLR_GPU_MEMORY_GB=28
   python my_script.py
   ```

---

## Future Enhancements

Potential features for future iterations:
- **Distributed Training**: Multi-GPU support with PyTorch DDP and TensorFlow MirroredStrategy
- **Model Registry**: MLflow model versioning and deployment
- **Experiment Comparison**: Built-in tools for comparing runs
- **Callbacks System**: Extensible callback hooks for training events
- **Data Versioning**: Integration with DVC or similar tools
- **Cloud Support**: AWS/GCP/Azure storage backends for artifacts
