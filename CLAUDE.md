# Project: ml_skeleton

## Overview
Deep learning training framework with MLflow experiment tracking and hyperparameter tuning (Optuna/Ray Tune). Python 3.10+ package. Designed for Docker environments with NVIDIA GPU support (RTX 5090/Blackwell).

## Architecture
- **[core/](ml_skeleton/core/)** - Protocol definitions, config dataclasses
  - `TrainingContext`: Input to user's `train_model()` (hyperparameters, tracker, device)
  - `TrainingResult`: Return value (primary_metric, artifacts, metadata)
  - `ExperimentConfig`: Main config (MLflow, tuning, hyperparams)
- **[tracking/](ml_skeleton/tracking/)** - MLflow integration
  - `ExplrTracker`: MLflow client wrapper (log_metric, log_artifact, etc.)
  - `MLflowServer`: Auto-start/manage MLflow server subprocess
- **[tuning/](ml_skeleton/tuning/)** - Hyperparameter optimization
  - `OptunaTuner`, `RayTuneTuner`: Backend implementations
  - `SearchSpaceBuilder`: DSL for search space definitions
- **[runner/](ml_skeleton/runner/)** - Orchestration and CLI
  - `Experiment`: Main orchestrator (run single trial or tune)
  - `cli.py`: CLI commands (run, tune, mlflow-ui, gpu-info, verify)
- **[frameworks/](ml_skeleton/frameworks/)** - PyTorch/TensorFlow helpers
- **[utils/](ml_skeleton/utils/)** - GPU memory limits, reproducibility (seeding), environment verification

## User Contract
User implements: `def train_model(ctx: TrainingContext) -> TrainingResult`
Framework provides: hyperparameters, MLflow tracker, device info, paths

## Entry Points
- **API**: `run_experiment(train_fn, config)` in [runner/experiment.py](ml_skeleton/runner/experiment.py)
- **CLI**: `ml_skeleton` command in [runner/cli.py](ml_skeleton/runner/cli.py)
- **Exports**: See [\_\_init\_\_.py](ml_skeleton/__init__.py)

## Key Files
- [pyproject.toml](pyproject.toml) - Package metadata, dependencies, CLI entrypoint
- [ml_skeleton/core/protocols.py](ml_skeleton/core/protocols.py) - Core data structures
- [ml_skeleton/runner/experiment.py](ml_skeleton/runner/experiment.py) - Main orchestrator
- [configs/example.yaml](configs/example.yaml) - Example configuration
- [examples/](examples/) - PyTorch/TensorFlow usage examples
- [Dockerfile](Dockerfile) - Container image definition
- [docker-compose.yml](docker-compose.yml) - MLflow + training services

## Docker Environment
- Base image: `kaggle:torch` (pre-built with PyTorch, CUDA 12.8, RTX 5090 support)
- MLflow runs on port 5000, Jupyter on port 8888
- Mount: `~/git:/git`, `~/PycharmProjects:/projects`
- VS Code Dev Containers supported (attach to running container)
- Git auth: credential helper or mount host credentials

## Music Recommendation System (Complete Implementation)
Two-phase training pipeline for music recommendations using Clementine database:

### Phase 1: Encoder Training (Audio → Embeddings)
- **[music/audio_loader.py](ml_skeleton/music/audio_loader.py)**: READ-ONLY audio loading with multiprocessing (80% CPU cores), center-crop extraction, torchaudio integration
- **[music/baseline_encoder.py](ml_skeleton/music/baseline_encoder.py)**: Simple 1D CNN encoder (SimpleAudioEncoder), mel-spectrogram encoder template, multi-task encoder wrapper
- **[music/dataset.py](ml_skeleton/music/dataset.py)**: MusicDataset for audio loading, EmbeddingDataset for classifier training, multi-album support
- **[music/losses.py](ml_skeleton/music/losses.py)**: RatingLoss (MSE), MultiTaskLoss (rating + album), NTXentLoss (SimCLR), SupervisedContrastiveLoss
- **[training/encoder_trainer.py](ml_skeleton/training/encoder_trainer.py)**: Encoder training orchestration, embedding extraction/storage, checkpoint management

### Phase 2: Classifier Training (Embeddings → Ratings)
- **[music/baseline_classifier.py](ml_skeleton/music/baseline_classifier.py)**: Simple MLP classifier, deep classifier with residual connections, ensemble classifier
- **[training/classifier_trainer.py](ml_skeleton/training/classifier_trainer.py)**: Classifier training orchestration, prediction generation, evaluation metrics (MSE, MAE, RMSE, correlation)

### Storage & Database
- **[music/clementine_db.py](ml_skeleton/music/clementine_db.py)**: READ-ONLY Clementine database interface, Song dataclass
- **[music/embedding_store.py](ml_skeleton/music/embedding_store.py)**: SQLite storage with multi-version support, batch operations, efficient retrieval
- **[music/speech_detector.py](ml_skeleton/music/speech_detector.py)**: Speech detection filtering using Silero VAD (optional)

### Protocols & Configuration
- **[protocols/encoder.py](ml_skeleton/protocols/encoder.py)**: AudioEncoder protocol for user-injectable encoders
- **[protocols/classifier.py](ml_skeleton/protocols/classifier.py)**: RatingClassifier protocol for user-injectable classifiers
- **[configs/music_recommendation.yaml](configs/music_recommendation.yaml)**: Complete configuration for encoder, classifier, and recommendation generation

### Examples
- **[examples/music_recommendation.py](examples/music_recommendation.py)**: Complete end-to-end example with 3 stages (encoder, classifier, recommend)

### Key Features
- Multi-album support (songs on multiple albums averaged in loss)
- Album key format: "artist|||album" for uniqueness
- Multiprocessing default: 80% CPU cores
- End-crop extraction: 60s from end of song (with z-normalization)
- Embedding versioning: A/B testing support
- READ-ONLY operations: All audio file access is read-only


## Conventions
- Config via YAML or dataclasses (`ExperimentConfig`)
- Auto-start MLflow server (port 5000) if not running
- Default GPU memory limit: 24GB (leaves 8GB for system on 32GB RTX 5090)
- Default seed: 42 (reproducibility)
- Checkpoint dir: `./checkpoints`, Artifact dir: `./artifacts`

## Update Instructions
**IMPORTANT**: When making significant changes or adding features:
1. Update this project summary section in CLAUDE.md if architecture changes
2. Update [README.md](README.md) if user-facing API/CLI changes

# Security Rules

NEVER display, cat, echo, read, or print the contents of files that may contain secrets, including:
- ~/.git-credentials
- .env files
- .env.local, .env.production, .env.* files
- **/credentials/**
- **/*token*
- **/*secret*
- *.pem, *.key files
- ~/.ssh/id_* (private keys)
- ~/.netrc
- ~/.aws/credentials
- **/*password*
- config files containing API keys

When checking these files:
- Only verify they exist using `test -f` or `ls`
- Check file format or line count without showing content
- Never use cat, head, tail, less, more, or Read tool on these files
- If you need to verify content, describe what to look for and let the user check manually

# Pre-Commit Safety Check

When the user asks to check code before committing (or uses phrases like "check before commit", "safe to commit", "review for commit"):

1. Run `git status` to see all staged and untracked files
2. Read each new or modified file that will be committed
3. Check for sensitive information:
   - API keys, tokens, secrets
   - Passwords or credentials
   - Private keys or certificates
   - Personal information (emails, names, addresses)
   - Hardcoded internal URLs or IPs
   - Database connection strings with credentials
4. Report findings in a table format:
   | File | Status | Safe? | Notes |
5. Give a clear ✅ Safe or ❌ Not Safe verdict
