# Hyperparameter Optimization Quick Start

## One Command HPO Pipeline

Run the complete hyperparameter optimization workflow:

```bash
./run_music_pipeline.sh hpo
```

This will:
1. **Tune encoder** (30 trials by default)
2. **PAUSE** - Review best parameters and update config
3. **Train encoder** with optimized parameters
4. **Tune classifier** (20 trials by default)
5. **PAUSE** - Review best parameters and update config
6. **Train classifier** with optimized parameters
7. **Display model card** (with all stats)
8. **Generate recommendations**

## Customize Number of Trials

```bash
# More thorough tuning (recommended for production)
HPO_ENCODER_TRIALS=50 HPO_CLASSIFIER_TRIALS=30 ./run_music_pipeline.sh hpo

# Quick tuning (for testing the pipeline)
HPO_ENCODER_TRIALS=10 HPO_CLASSIFIER_TRIALS=10 ./run_music_pipeline.sh hpo
```

## What Happens During HPO

### Step 1: Encoder Tuning
The script will run 30 trials (default), testing different combinations of:
- Learning rate: 0.00001 → 0.01
- Batch size: [16, 32, 64]
- Embedding dim: [256, 512, 1024]
- Base channels: [16, 32, 64]

After completion, you'll see output like:
```
Best value: 0.234567
Best parameters:
  learning_rate: 0.000543
  batch_size: 32
  embedding_dim: 512
  base_channels: 32

PAUSE: Update encoder parameters in configs/music_recommendation.yaml
Press Enter when ready to continue...
```

### Step 2: Update Config

Edit `configs/music_recommendation.yaml` and update the encoder section:

```yaml
encoder:
  learning_rate: 0.000543  # From HPO
  batch_size: 32           # From HPO
  embedding_dim: 512       # From HPO
  base_channels: 32        # From HPO
```

Press Enter to continue.

### Step 3: Train Encoder
The script trains the encoder with your optimized parameters.

### Step 4: Classifier Tuning
Same process for classifier:
- Learning rate: 0.00001 → 0.001
- Batch size: [128, 256, 512]
- Dropout: 0.1 → 0.5
- Hidden dims: [[256, 128], [512, 256], [512, 256, 128]]

### Step 5: Update Config Again
Update classifier section in config file.

### Step 6-8: Train & Generate
Trains classifier and generates recommendations with optimized models.

## Time Estimates

| Stage | Trials | Estimated Time |
|-------|--------|----------------|
| Encoder HPO | 30 | 5-10 hours |
| Encoder Train | 1 | 20-40 minutes |
| Classifier HPO | 20 | 40-100 minutes |
| Classifier Train | 1 | 5-10 minutes |
| **Total** | | **7-12 hours** |

*Times vary based on dataset size and hardware*

## Manual HPO (Individual Stages)

If you prefer more control:

```bash
# 1. Tune encoder only
python examples/music_recommendation.py \
  --stage tune-encoder \
  --config configs/music_recommendation.yaml \
  --n-trials 30

# 2. Update config manually (see best parameters in output)

# 3. Train encoder
./run_music_pipeline.sh encoder

# 4. Tune classifier
python examples/music_recommendation.py \
  --stage tune-classifier \
  --config configs/music_recommendation.yaml \
  --n-trials 20

# 5. Update config manually

# 6. Train classifier
./run_music_pipeline.sh classifier

# 7. Generate recommendations
./run_music_pipeline.sh recommend
```

## View Results in MLflow

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Open browser
open http://localhost:5000
```

You'll see:
- All 30 encoder trials with hyperparameters
- All 20 classifier trials with hyperparameters
- Loss curves for each trial
- Parameter importance charts
- Easy comparison between trials

## Config Backup

The HPO pipeline automatically creates a backup:
```bash
configs/music_recommendation.yaml.hpo_backup
```

To restore original config:
```bash
cp configs/music_recommendation.yaml.hpo_backup configs/music_recommendation.yaml
```

## Tips

1. **Start Small**: Use 10 trials first to test the pipeline
2. **Monitor Progress**: Watch MLflow UI during tuning
3. **Review Logs**: Check `/tmp/encoder_hpo.log` and `/tmp/classifier_hpo.log`
4. **Save Best Configs**: Create separate config files for different HPO runs
5. **Document Results**: Note best parameters and final metrics

## Troubleshooting

### "Config not found"
Make sure you're in the ml_skeleton root directory:
```bash
cd /git/ml_skeleton
./run_music_pipeline.sh hpo
```

### "Embeddings not found" during classifier tuning
Run encoder training first:
```bash
./run_music_pipeline.sh encoder
```

### Out of memory during HPO
Reduce batch sizes in search space (edit config):
```yaml
encoder_search_space:
  parameters:
    batch_size:
      choices: [8, 16, 32]  # Smaller batch sizes
```

### HPO takes too long
Use fewer trials:
```bash
HPO_ENCODER_TRIALS=10 HPO_CLASSIFIER_TRIALS=10 ./run_music_pipeline.sh hpo
```

## Next Steps

After HPO completes:

1. **Review model card**: `checkpoints/MODEL_CARD.md`
2. **Check recommendations**: `recommendations.txt`
3. **Open playlists** in Clementine:
   - `recommender_help.xspf` - Uncertain predictions (for learning)
   - `recommender_best.xspf` - Top predictions (for validation)
4. **Rate songs** and re-run training for continuous improvement
5. **Save optimized config** for future use

## Advanced: Resume Interrupted HPO

If HPO is interrupted, Optuna saves progress to SQLite:

```python
import optuna

# Resume encoder study
study = optuna.load_study(
    study_name="music_recommendation_encoder",
    storage="sqlite:///optuna.db"
)

# Continue tuning
study.optimize(objective, n_trials=10)  # Add 10 more trials
```

See [HYPERPARAMETER_TUNING_GUIDE.md](HYPERPARAMETER_TUNING_GUIDE.md) for full documentation.
