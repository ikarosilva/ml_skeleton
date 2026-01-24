# Hyperparameter Tuning Guide

This guide explains how to use hyperparameter optimization for the music recommendation system.

## Overview

The system now supports automated hyperparameter tuning using **Optuna** (default) or **Ray Tune**. The search spaces are defined in your configuration file and can be customized for both encoder and classifier stages.

## Quick Start

### 1. Tune the Encoder

```bash
python examples/music_recommendation.py \
  --stage tune-encoder \
  --config configs/music_recommendation.yaml \
  --n-trials 30
```

This will:
- Run 30 training trials with different hyperparameter combinations
- Use the search space defined in `encoder_search_space` (see config)
- Track all experiments in MLflow
- Report the best hyperparameters at the end

### 2. Tune the Classifier

```bash
python examples/music_recommendation.py \
  --stage tune-classifier \
  --config configs/music_recommendation.yaml \
  --n-trials 20
```

Note: You must run encoder training first to generate embeddings!

### 3. Apply Best Parameters

After tuning completes, update your config file with the best parameters:

```yaml
encoder:
  learning_rate: 0.000543  # From tuning results
  batch_size: 32           # From tuning results
  embedding_dim: 512       # From tuning results
  # ... other best parameters
```

Then run normal training:

```bash
python examples/music_recommendation.py --stage encoder --config configs/music_recommendation.yaml
```

## Command Line Options

```bash
--stage tune-encoder|tune-classifier    # Which stage to tune
--config PATH                           # Config file path
--n-trials N                            # Number of trials (default: from config)
--tuner optuna|ray_tune                 # Tuner backend (default: optuna)
--timeout SECONDS                       # Max time for tuning (optional)
```

## Customizing Search Spaces

Edit your config file to modify the search space:

### Encoder Search Space

```yaml
tuning:
  encoder_search_space:
    parameters:
      learning_rate:
        type: "loguniform"    # Log-scale sampling
        low: 0.00001
        high: 0.01

      batch_size:
        type: "categorical"   # Discrete choices
        choices: [16, 32, 64]

      embedding_dim:
        type: "categorical"
        choices: [256, 512, 1024]

      base_channels:
        type: "categorical"
        choices: [16, 32, 64]

      # Add more parameters:
      contrastive_temperature:
        type: "float"
        low: 0.1
        high: 1.0

      year_threshold:
        type: "int"
        low: 3
        high: 10

      weight_decay:
        type: "loguniform"
        low: 0.00001
        high: 0.001
```

### Classifier Search Space

```yaml
tuning:
  classifier_search_space:
    parameters:
      learning_rate:
        type: "loguniform"
        low: 0.00001
        high: 0.001

      batch_size:
        type: "categorical"
        choices: [128, 256, 512]

      dropout:
        type: "float"
        low: 0.1
        high: 0.5

      hidden_dims:
        type: "categorical"
        choices: [[256, 128], [512, 256], [512, 256, 128]]
```

## Parameter Types

| Type | Description | Example |
|------|-------------|---------|
| `loguniform` | Log-scale continuous (good for learning rates) | `low: 1e-5, high: 1e-1` |
| `float` | Linear continuous | `low: 0.0, high: 1.0` |
| `int` | Integer range | `low: 1, high: 10, step: 1` |
| `categorical` | Discrete choices | `choices: [16, 32, 64]` |

## MLflow Integration

All tuning experiments are tracked in MLflow:

```bash
# View results in MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Navigate to `http://localhost:5000` to see:
- All trial runs
- Hyperparameter comparisons
- Loss curves
- Best model selection

## Advanced: Optuna Features

### Pruning (Early Stopping)

Optuna automatically stops unpromising trials early using the `MedianPruner` (configured in your YAML):

```yaml
tuning:
  pruner: "MedianPruner"  # Stops trials worse than median
  sampler: "TPESampler"   # Tree-structured Parzen Estimator
```

### Visualization

After tuning, you can visualize the optimization:

```python
import optuna

# Load study from database
study = optuna.load_study(
    study_name="music_recommendation_encoder",
    storage="sqlite:///optuna.db"
)

# Plot optimization history
optuna.visualization.plot_optimization_history(study).show()

# Plot parameter importances
optuna.visualization.plot_param_importances(study).show()

# Plot parallel coordinates
optuna.visualization.plot_parallel_coordinate(study).show()
```

## Example Workflow

### Full Hyperparameter Optimization Pipeline

```bash
# 1. Tune encoder (30 trials, ~1-2 hours depending on data)
python examples/music_recommendation.py \
  --stage tune-encoder \
  --config configs/music_recommendation.yaml \
  --n-trials 30

# Output:
# Best value: 0.234567
# Best parameters:
#   learning_rate: 0.000543
#   batch_size: 32
#   embedding_dim: 512
#   base_channels: 32

# 2. Update config with best encoder parameters
vim configs/music_recommendation.yaml

# 3. Train encoder with best parameters
python examples/music_recommendation.py \
  --stage encoder \
  --config configs/music_recommendation.yaml

# 4. Tune classifier (20 trials, faster than encoder)
python examples/music_recommendation.py \
  --stage tune-classifier \
  --config configs/music_recommendation.yaml \
  --n-trials 20

# Output:
# Best value: 0.123456 (MAE)
# Best parameters:
#   learning_rate: 0.000089
#   batch_size: 256
#   dropout: 0.35
#   hidden_dims: [512, 256]

# 5. Update config with best classifier parameters
vim configs/music_recommendation.yaml

# 6. Train classifier with best parameters
python examples/music_recommendation.py \
  --stage classifier \
  --config configs/music_recommendation.yaml

# 7. Generate recommendations
python examples/music_recommendation.py \
  --stage recommend \
  --config configs/music_recommendation.yaml
```

## Tips for Effective Tuning

1. **Start Small**: Begin with 10-20 trials to get a sense of the landscape
2. **Narrow the Range**: Use initial results to refine search bounds
3. **Focus on Important Params**: Learning rate and batch size usually have the biggest impact
4. **Use Log-Scale**: Always use `loguniform` for learning rates
5. **Consider Dependencies**: Some parameters interact (e.g., batch_size and learning_rate)
6. **Monitor Progress**: Watch MLflow UI during tuning to spot trends
7. **Validate Results**: Always retrain with best parameters to confirm improvement

## Troubleshooting

### "AttributeError: 'ExperimentConfig' object has no attribute..."

Make sure your config YAML has all required fields. The tuning code will use defaults for missing fields.

### "OSError: [Errno 24] Too many open files"

Reduce `num_workers` in DataLoader or increase system file descriptor limit:

```bash
ulimit -n 4096
```

### Trials failing with OOM errors

Reduce batch size in search space or limit GPU memory in config:

```yaml
gpu_memory_limit_gb: 20  # Leave headroom for system
```

### Tuning is too slow

- Reduce number of epochs during tuning (e.g., 10 epochs instead of 50)
- Use smaller datasets for initial tuning
- Reduce batch size in search space
- Enable pruning to stop bad trials early

## Performance Expectations

| Stage | Trials | Time per Trial | Total Time |
|-------|--------|----------------|------------|
| Encoder | 30 | 10-20 min | 5-10 hours |
| Classifier | 20 | 2-5 min | 40-100 min |

*Times vary based on dataset size, hardware, and number of epochs*

## Next Steps

After finding optimal hyperparameters:

1. Document your best parameters in a separate config file
2. Run full training with best parameters
3. Compare performance against baseline
4. Consider A/B testing different hyperparameter sets
5. Retune periodically as your dataset grows
