# Hyperparameter Optimization Workflow

## Quick Start

```bash
# Run 50 trials (recommended starting point)
HPO_ENCODER_TRIALS=50 ./run_music_pipeline.sh hpo-encoder

# After completion, train with best params
./run_music_pipeline.sh encoder --best-params checkpoints/best_encoder_params.json
./run_music_pipeline.sh classifier
```

## Incremental Workflow (Multiple Rounds)

The `HPO_ENCODER_TRIALS` parameter specifies the **total target number of trials**, not additional trials. Optuna automatically resumes from the last completed trial.

### Round 1: Initial Exploration (1 day)
```bash
HPO_ENCODER_TRIALS=20 ./run_music_pipeline.sh hpo-encoder
# Runs trials 0-19 (20 trials total)
```

**After Round 1:**
- Check best params so far
- Optionally test the current best configuration
- Decide if more trials are needed

### Round 2: Expand Search (1-2 days)
```bash
HPO_ENCODER_TRIALS=50 ./run_music_pipeline.sh hpo-encoder
# Runs trials 20-49 (30 MORE trials, 50 total)
```

This continues from where Round 1 left off. Optuna will:
1. Load existing study from `optuna_study.db`
2. See that 20 trials already exist
3. Run 30 more trials to reach the target of 50

### Round 3: Final Refinement (1-2 days)
```bash
HPO_ENCODER_TRIALS=100 ./run_music_pipeline.sh hpo-encoder
# Runs trials 50-99 (50 MORE trials, 100 total)
```

## Monitoring Progress

### Check Current Status
```bash
python -c "
import optuna
study = optuna.load_study(study_name='music_moco_optuna', storage='sqlite:///optuna_study.db')
print(f'Completed trials: {len(study.trials)}')
print(f'Best trial: {study.best_trial.number}')
print(f'Best value: {study.best_value:.4f}')
print(f'Best params:')
for key, value in study.best_params.items():
    print(f'  {key}: {value}')
"
```

### View Trial History
```bash
python -c "
import optuna
study = optuna.load_study(study_name='music_moco_optuna', storage='sqlite:///optuna_study.db')
for trial in study.trials[-10:]:  # Last 10 trials
    status = 'COMPLETE' if trial.state == optuna.trial.TrialState.COMPLETE else trial.state.name
    value = f'{trial.value:.4f}' if trial.value else 'N/A'
    print(f'Trial {trial.number}: {status} - Value: {value}')
"
```

### Launch Optuna Dashboard (Optional)
```bash
# Install dashboard
pip install optuna-dashboard

# Launch (in separate terminal)
optuna-dashboard sqlite:///optuna_study.db
# Open http://localhost:8080
```

## Extracting Best Params Mid-Run

If you interrupt HPO (CTRL+C) before completion, extract best params manually:

```bash
python -c "
import optuna, json
from pathlib import Path

study = optuna.load_study(study_name='music_moco_optuna', storage='sqlite:///optuna_study.db')

# Save best params
Path('checkpoints').mkdir(exist_ok=True)
with open('checkpoints/best_encoder_params_partial.json', 'w') as f:
    json.dump(study.best_params, f, indent=2)

print(f'Completed {len(study.trials)} trials')
print(f'Best params saved to: checkpoints/best_encoder_params_partial.json')
print(f'Best validation loss: {study.best_value:.4f}')
"
```

Then use the partial results:
```bash
./run_music_pipeline.sh encoder --best-params checkpoints/best_encoder_params_partial.json
```

## Configuration

Current HPO settings (see `configs/music_moco.yaml`):
- **Epochs per trial**: 25 (faster trials, ~50-60 min each)
- **Final training epochs**: 70 (full convergence with best params)
- **Batch size**: 96 (optimized for RTX 5090)
- **Early stopping patience**: 7 epochs
- **Storage**: `sqlite:///optuna_study.db` (persistent, resumable)

## Search Space

The encoder HPO tunes 9 hyperparameters:
- `learning_rate`: 0.0001 - 0.001 (log scale)
- `adam_weight_decay`: 0.0001 - 0.01 (log scale)
- `adam_eps`: 1e-10 - 1e-6 (log scale)
- `adam_beta1`: 0.85 - 0.95 (log scale)
- `adam_beta2`: 0.99 - 0.9999 (log scale)
- `loss_weights_moco`: 0.4 - 0.8 (linear, step 0.05)
- `gain_db_max`: 2.0 - 6.0 (linear, step 0.5)
- `noise_prob`: 0.3 - 0.8 (linear, step 0.1)
- `mixup_alpha`: 0.1 - 0.4 (linear, step 0.05)

## Timeline Estimates

With current settings (25 epochs, batch size 96):

| Trials | Time (with early stopping) | Time (worst case) |
|--------|---------------------------|-------------------|
| 20     | ~15-20 hours             | ~24 hours         |
| 50     | ~35-45 hours             | ~60 hours         |
| 100    | ~70-90 hours             | ~120 hours        |

Early stopping typically triggers around epoch 12-18, saving significant time.

## After HPO Completes

1. **Review best parameters**: `cat checkpoints/best_encoder_params.json`

2. **Train encoder with best params** (70 epochs for full convergence):
   ```bash
   ./run_music_pipeline.sh encoder --best-params checkpoints/best_encoder_params.json
   ```

3. **Train classifier**:
   ```bash
   ./run_music_pipeline.sh classifier
   ```

4. **Generate recommendations**:
   ```bash
   ./run_music_pipeline.sh recommend
   ```

## Troubleshooting

### Reset HPO Study
If you want to start fresh (deletes all trial history):
```bash
rm optuna_study.db
rm checkpoints/best_encoder_params*.json
```

### Resume After Interruption
Just run the same command again. Optuna automatically continues from the last completed trial:
```bash
HPO_ENCODER_TRIALS=50 ./run_music_pipeline.sh hpo-encoder
```

### Check GPU Utilization
```bash
watch -n 1 nvidia-smi
```

Expect:
- GPU utilization: 95-100%
- Memory usage: ~20-24GB (out of 32GB available)
- Power: 400-450W

## Best Practices

1. **Start with 20-30 trials** to get initial insights
2. **Check convergence** - if the best trial keeps changing after 50+ trials, the search space might be too large
3. **Monitor early stopping** - if most trials stop before epoch 15, consider reducing epochs further
4. **Save checkpoints frequently** - the study is auto-saved after each trial completes
5. **Use tmux/screen** for long runs to prevent disconnection issues:
   ```bash
   tmux new -s hpo
   HPO_ENCODER_TRIALS=100 ./run_music_pipeline.sh hpo-encoder
   # CTRL+B, then D to detach
   # tmux attach -t hpo to reattach
   ```
