# ml_skeleton

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
from ml_skeleton import TrainingContext, TrainingResult

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
from ml_skeleton import run_experiment, ExperimentConfig

config = ExperimentConfig(
    name="my_experiment",
    hyperparameters={"learning_rate": 0.001, "epochs": 50}
)

result = run_experiment(train_model, config)
print(f"Final loss: {result.primary_metric}")
```

### 3. Run Hyperparameter Tuning

```python
from ml_skeleton import run_experiment, ExperimentConfig, TunerType
from ml_skeleton.tuning import SearchSpaceBuilder

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
mlskel run config.yaml --train-fn my_module:train_model

# Run hyperparameter tuning
mlskel tune config.yaml --train-fn my_module:train_model --n-trials 50

# Start MLflow UI
ml_skeleton mlflow-ui

# Show GPU info
mlskel gpu-info

# Verify environment
mlskel verify

# Show/set GPU memory limits
mlskel memory --show
mlskel memory --limit 24
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
cd /git/ml_skeleton
pip install -e .

# Start MLflow server (background)
mlflow server --host 0.0.0.0 --port 5000 &

# Run your experiment
python examples/pytorch_example.py

# Stop MLflow server when done (optional - data persists regardless)
pkill -f "mlflow server"
```

### VS Code Dev Containers

You can attach VS Code directly to a running Docker container for full IDE support (debugging, IntelliSense, etc.):

1. Install the **Dev Containers** extension in VS Code
2. Start the Docker container (see above)
3. Open Command Palette (`Ctrl+Shift+P`)
4. Select **Dev Containers: Attach to Running Container...**
5. Select your `kaggle:torch` container
6. In the new VS Code window, click **Open Folder** and navigate to `/git/ml_skeleton`
7. Install the **Python** extension (`Ctrl+Shift+X`, search "Python" by Microsoft)
8. Select the Python interpreter (`Ctrl+Shift+P` → "Python: Select Interpreter" → choose `/opt/conda/bin/python` or similar)
9. Run scripts using the Play button (top right) or debug with F5

The project includes VS Code settings (`.vscode/settings.json`) that ensure scripts always run from the workspace root directory, regardless of which file you're editing. The attached VS Code window runs entirely inside the container, with access to the container's Python interpreter, GPU, and all installed packages.

### Git Authentication in Docker

To push code from inside the container, you need to authenticate.

**Option 1: HTTPS with Personal Access Token (No Restart)**
If you are already inside the container:
1. Run `git config --global credential.helper store`
2. Run `git push`
3. Enter your GitHub username
4. For the password, use a **Personal Access Token** (not your account password)

> **Note:** If `git push` fails immediately with a 403 error without prompting, you may have a conflicting environment variable. Run `unset GITHUB_TOKEN` and `rm -f ~/.git-credentials`, then try again.
> **Troubleshooting 403 Errors:**
> If `git push` fails immediately without prompting, an invalid token is likely active in the environment. Run:
> ```bash
> unset GITHUB_TOKEN
> git config --system --unset credential.helper
> git config --global --unset credential.helper
> git remote set-url origin https://<USERNAME>@github.com/<USERNAME>/ml_skeleton.git
> ```
>
> If you see "Permission denied", ensure your Personal Access Token has the **`repo`** scope selected.
>
> **Troubleshooting Hanging:**
> If `git push` hangs without output, the credential helper may be stuck. Run `git config --global --unset credential.helper` to disable it, then try again.
>
> **Last Resort (Token in URL):**
> If prompts are not appearing, embed your token directly in the remote URL (warning: this saves the token in plain text in `.git/config`):
> `git remote set-url origin https://<YOUR_TOKEN>@github.com/<USERNAME>/ml_skeleton.git`
>
> **Network Hanging (MTU):**
> If it hangs during connection or "Writing objects", it is likely a Docker network packet size issue. Run:
> `git config --global http.postBuffer 524288000`
> `ip link set eth0 mtu 1400` (requires root)
> `git config --global http.version HTTP/1.1`
> `ip link set eth0 mtu 1200` (requires root)

**Option 2: Mount Host Credentials (Requires Restart)**
Add these flags to your `docker run` command to share host credentials:
```bash
-v ~/.gitconfig:/root/.gitconfig -v ~/.git-credentials:/root/.git-credentials
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
from ml_skeleton import ExperimentConfig, TuningConfig, TunerType

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

## MLflow Server Management

The framework automatically manages the MLflow tracking server for you.

### Auto-Start Behavior (Default)

When `auto_start: true` (the default), the framework:

1. Checks if an MLflow server is already running on the configured port
2. If not running, automatically starts one as a subprocess
3. Automatically stops the server when your script exits (via `atexit` handler)

This means you can simply run your experiment without manually starting MLflow:

```bash
python examples/two_moons.py
# MLflow server starts automatically, runs experiment, then stops on exit
```

### Using an External Server

If you start MLflow manually before running your experiment, the framework detects it and reuses the existing server without stopping it when your script exits:

```bash
# Start server manually (stays running)
mlflow server --host 0.0.0.0 --port 5000 &

# Run experiments (server keeps running after script exits)
python examples/two_moons.py

# Stop when you're done
pkill -f "mlflow server"
```

### Disabling Auto-Start

To disable auto-start and require a pre-running server:

```yaml
mlflow:
  auto_start: false
  tracking_uri: "http://localhost:5000"
```

Or programmatically:

```python
config = ExperimentConfig(name="my_experiment")
config.mlflow.auto_start = False
```

## Project Structure

```
ml_skeleton/
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
from ml_skeleton.utils.memory import limit_gpu_memory

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
export ML_SKELETON_GPU_MEMORY_GB=28
python my_training.py

# Disable limit via environment variable
export ML_SKELETON_GPU_MEMORY_GB=0
```

### In train_model()

```python
def train_model(ctx: TrainingContext) -> TrainingResult:
    from ml_skeleton.utils.memory import limit_gpu_memory

    # Get limit from hyperparameters or use default (24GB)
    max_mem = ctx.hyperparameters.get("max_gpu_memory_gb")
    limit_gpu_memory(max_memory_gb=max_mem)

    # ... training code ...
```

### CLI

```bash
mlskel memory --show        # Show current usage
mlskel memory --limit 16    # Set custom limit
```

## Environment Verification

Verify your environment has all required dependencies:

```bash
# Via CLI
mlskel verify

# Via Python
python -m ml_skeleton.utils.verify
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

## Music Recommendation Use Case

The framework includes a complete music recommendation system implementation using Clementine database integration.

### Overview

Two-phase training pipeline:
1. **Stage 1: Encoder** - Train audio encoder (raw waveforms → embeddings)
2. **Stage 2: Classifier** - Train rating classifier (embeddings → ratings)
3. **Stage 3: Recommendations** - Generate ranked song recommendations

### Features

- **READ-ONLY Audio Loading** with multiprocessing (80% CPU cores default)
- **Multi-Album Support** - Songs can belong to multiple albums (original + compilations)
- **SQLite Embedding Storage** with multi-version support for A/B testing
- **Center-Crop Extraction** - Extract 30 seconds from center of songs
- **Baseline Models** - Simple 1D CNN encoder and MLP classifier (easily customizable)
- **Multiple Loss Functions** - MSE, contrastive learning, supervised contrastive
- **Speech Detection Filtering** - Optional filtering of spoken word content

### Quick Start

#### Prerequisites

1. **Install Dependencies**
   ```bash
   cd /git/ml_skeleton
   pip install -e ".[pytorch]"
   pip install torchaudio librosa tqdm pyyaml
   ```

2. **Configure Your Database Path**
   Edit [configs/music_recommendation.yaml](configs/music_recommendation.yaml) and set:
   ```yaml
   music:
     database_path: "/home/ikaro/Music/clementine.db"  # Your Clementine DB path
   ```

#### Full Training Pipeline

**Option 1: Using the convenience script (Recommended)**

```bash
# Make script executable (first time only)
chmod +x run_music_pipeline.sh

# Run entire pipeline (all 3 stages)
./run_music_pipeline.sh all

# Or run individual stages
./run_music_pipeline.sh encoder      # Stage 1 only
./run_music_pipeline.sh classifier   # Stage 2 only
./run_music_pipeline.sh recommend    # Stage 3 only

# Quick test with reduced epochs (for testing)
./run_music_pipeline.sh quick
```

**Option 2: Running stages manually**

Run all three stages in sequence:

```bash
# Stage 1: Train audio encoder (audio → embeddings)
# This will:
# - Load rated songs from Clementine DB
# - Train 1D CNN encoder on 30-second audio clips
# - Extract and store embeddings in embeddings.db
# - Save best model to checkpoints/encoder_best.pt
python examples/music_recommendation.py --stage encoder --config configs/music_recommendation.yaml

# Stage 2: Train rating classifier (embeddings → ratings)
# This will:
# - Load pre-extracted embeddings
# - Train MLP classifier to predict ratings
# - Save best model to checkpoints/classifier_best.pt
python examples/music_recommendation.py --stage classifier --config configs/music_recommendation.yaml

# Stage 3: Generate recommendations
# This will:
# - Load unrated songs
# - Predict ratings using trained models
# - Generate top-N recommendations
# - Save to recommendations.txt
python examples/music_recommendation.py --stage recommend --config configs/music_recommendation.yaml
```

#### Quick Test (Single Command)

For testing, you can reduce epochs in the config:

```bash
# Edit config first (reduce epochs for quick test)
# encoder.epochs: 5
# classifier.epochs: 5

# Then run all stages
python examples/music_recommendation.py --stage encoder --config configs/music_recommendation.yaml && \
python examples/music_recommendation.py --stage classifier --config configs/music_recommendation.yaml && \
python examples/music_recommendation.py --stage recommend --config configs/music_recommendation.yaml
```

#### Monitor Training

While training, you can monitor progress:

```bash
# In another terminal, start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Open browser: http://localhost:5000
# View metrics, compare runs, and download artifacts
```

#### Expected Output

```
Stage 1 (Encoder):
  ✓ Loads ~60K songs from Clementine DB
  ✓ Trains for 50 epochs (~1-2 hours on RTX 5090)
  ✓ Extracts embeddings for all songs
  ✓ Saves to: checkpoints/encoder_best.pt, embeddings.db

Stage 2 (Classifier):
  ✓ Loads pre-extracted embeddings
  ✓ Trains for 20 epochs (~10-15 minutes)
  ✓ Reports MAE, correlation metrics
  ✓ Saves to: checkpoints/classifier_best.pt

Stage 3 (Recommendations):
  ✓ Predicts ratings for unrated songs
  ✓ Generates top-100 recommendations
  ✓ Saves to: recommendations.txt
  ✓ Generates HITL playlists:
    - recommender_help.xspf (100 uncertain songs for learning)
    - recommender_best.xspf (50 best predictions for validation)
```

### Human-in-the-Loop Reinforcement Learning

Stage 3 automatically generates two XSPF playlists for continuous model improvement:

#### 1. High Uncertainty Playlist (`recommender_help.xspf`)
Contains 100 songs where the model is most uncertain (predictions near 0.5). Rating these songs provides maximum information gain for improving the model.

#### 2. Best Predictions Playlist (`recommender_best.xspf`)
Contains 50 songs with highest predicted ratings. Use this to validate that the model's top recommendations are actually good.

#### Workflow

```bash
# Step 1: Train and generate playlists
./run_music_pipeline.sh all

# Step 2: Open playlists in Clementine
# - File > Open Playlist > recommender_help.xspf
# - Listen and rate songs (right-click > Rate)
# - Clementine saves ratings to database automatically

# Step 3: Re-train with updated ratings
./run_music_pipeline.sh all

# Step 4: Repeat for continuous improvement
```

#### Configuration

```yaml
recommendations:
  human_feedback_uncertain: 100  # Songs for maximum learning
  human_feedback_best: 50        # Songs for validation
```

The playlists include annotations showing:
- Predicted rating (0-5 scale, compatible with Clementine)
- Uncertainty score (for help playlist)

This active learning approach focuses human effort where it matters most!

### Troubleshooting

**Problem: "No rated songs found"**
- Solution: Check your Clementine DB has rated songs (rating >= 0, not -1)
- Run: `sqlite3 /home/ikaro/Music/clementine.db "SELECT COUNT(*) FROM songs WHERE rating >= 0;"`

**Problem: "Audio files not found"**
- Solution: Verify audio files exist at paths stored in Clementine DB
- Clementine stores paths with `file://` prefix, the loader automatically handles this

**Problem: "Out of memory during training"**
- Solution: Reduce batch size in config:
  ```yaml
  encoder:
    batch_size: 16  # Reduce from 32
  classifier:
    batch_size: 128  # Reduce from 256
  ```

**Problem: "Training is slow"**
- Solution: Check multiprocessing settings:
  ```yaml
  music:
    num_workers: null  # Uses 80% CPU cores (default)
    dataloader_workers: 4  # Increase if CPU has many cores
  ```

**Problem: "No embeddings found for Stage 2"**
- Solution: Make sure Stage 1 (encoder) completed successfully
- Check: `ls -lh embeddings.db` (should be several MB)
- Verify: Use embedding store stats in Stage 1 output

**Problem: "Classifier not improving"**
- Solution: Check embedding quality from Stage 1
- Try different loss functions in Stage 1:
  ```yaml
  encoder:
    loss_type: "supervised_contrastive"  # or "mse", "contrastive"
  ```

**Problem: "All recommendations have similar ratings"**
- Solution: Train longer or tune hyperparameters
- Check correlation metric in Stage 2 (should be > 0.3)
- Consider using ensemble classifier

### Configuration

```yaml
# configs/music_recommendation.yaml
music:
  database_path: "/home/ikaro/Music/clementine.db"
  embedding_db_path: "./embeddings.db"
  sample_rate: 22050
  audio_duration: 30.0
  center_crop: true
  num_workers: null  # 80% CPU cores

encoder:
  embedding_dim: 512
  epochs: 50
  batch_size: 32
  learning_rate: 0.001

classifier:
  hidden_dims: [256, 128]
  epochs: 20
  batch_size: 256
  learning_rate: 0.0001
```

### Custom Models

Implement your own encoder and classifier by conforming to the protocols:

```python
# Custom encoder
class MyEncoder(nn.Module):
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (batch, num_samples) → embeddings: (batch, embedding_dim)"""
        # Your encoder logic
        return embeddings

    def get_embedding_dim(self) -> int:
        return 512

# Custom classifier
class MyClassifier(nn.Module):
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """embeddings: (batch, Z) → ratings: (batch, 1) in [0, 1]"""
        # Your classifier logic
        return ratings
```

### Architecture

```
Clementine DB → Audio Loader → Encoder → Embeddings DB → Classifier → Recommendations
```

Key components:
- [ml_skeleton/music/clementine_db.py](ml_skeleton/music/clementine_db.py) - READ-ONLY database interface
- [ml_skeleton/music/audio_loader.py](ml_skeleton/music/audio_loader.py) - Multiprocessing audio loading
- [ml_skeleton/music/dataset.py](ml_skeleton/music/dataset.py) - PyTorch datasets with multi-album support
- [ml_skeleton/music/embedding_store.py](ml_skeleton/music/embedding_store.py) - SQLite storage with versioning
- [ml_skeleton/music/baseline_encoder.py](ml_skeleton/music/baseline_encoder.py) - Simple CNN encoder
- [ml_skeleton/music/baseline_classifier.py](ml_skeleton/music/baseline_classifier.py) - MLP classifier
- [ml_skeleton/music/losses.py](ml_skeleton/music/losses.py) - Rating, contrastive, and multi-task losses
- [ml_skeleton/training/encoder_trainer.py](ml_skeleton/training/encoder_trainer.py) - Stage 1 orchestration
- [ml_skeleton/training/classifier_trainer.py](ml_skeleton/training/classifier_trainer.py) - Stage 2 orchestration

## License

MIT License
