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

## Music Recommendation Components
- **[music/clementine_db.py](ml_skeleton/music/clementine_db.py)**: Placeholder for loading song metadata from a Clementine music player database. Defines the `Song` data structure.
- **[music/speech_detector.py](ml_skeleton/music/speech_detector.py)**: Implements a pipeline to detect speech in audio files using a pre-trained VAD model (`silero-vad`). It runs in parallel and caches results in a SQLite database to avoid re-processing files.
- **[music/dataset.py](ml_skeleton/music/dataset.py)**: Contains the `MusicDataset` class, which handles loading audio data and can filter out files identified as speech based on a configurable threshold.
- **[examples/music_hello_world.py](examples/music_hello_world.py)**: A demonstration script that showcases how to use the speech detection and filtering pipeline.


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
