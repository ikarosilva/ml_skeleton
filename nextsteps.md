# Recommended Next Steps

## Phase 1: Full Encoder Training with Best Parameters

### Step 1: Train Encoder (70 epochs, ~3 hours)

Use the best parameters from HPO Trial 13 to train the encoder to full convergence:

```bash
./run_music_pipeline.sh encoder --best-params checkpoints/best_encoder_params.json
```

**What this does:**
- Loads best hyperparameters (lr=1.5e-04, loss_weights_moco=0.4, etc.)
- Trains for 70 epochs instead of 25
- Saves final encoder to `checkpoints/encoder_best.pt`
- Extracts embeddings for all songs automatically
- Expected final validation loss: ~1.4-1.5 (may improve slightly from 25-epoch result)

**Monitor progress:**
```bash
# In another terminal, watch training progress
tail -f encoder_training.log

# Or check GPU utilization
watch -n 1 nvidia-smi
```

---

## Phase 2: Classifier Training

### Step 2: Train Rating Classifier (~1-2 hours)

After encoder completes, train the classifier using the extracted embeddings:

```bash
./run_music_pipeline.sh classifier
```

**What this does:**
- Loads embeddings from `music_embeddings.db`
- Trains MLP classifier to predict ratings (1-5 stars)
- Uses default classifier config from `configs/music_moco.yaml`
- Saves classifier to `checkpoints/classifier_best.pt`
- Expected performance: MAE ~0.6-0.8 (lower is better)

---

## Phase 3: Generate Recommendations

### Step 3: Generate Recommendations (~5-10 minutes)

```bash
./run_music_pipeline.sh recommend
```

**What this does:**
- Loads encoder + classifier
- Generates personalized recommendations for rated songs
- Creates XSPF playlists in `playlists/` directory
- Exports recommendations to `recommendations.json`

---

## Optional: HPO for Classifier

If you want to optimize classifier performance:

### Step 4: Classifier HPO (Optional, ~8-12 hours)

```bash
# Run 20 trials for classifier hyperparameter optimization
HPO_CLASSIFIER_TRIALS=20 ./run_music_pipeline.sh hpo-classifier
```

**What this tunes:**
- Learning rate
- Dropout rate
- Hidden layer sizes
- Weight decay
- Batch size

Then train with best classifier params:
```bash
./run_music_pipeline.sh classifier --best-params checkpoints/best_classifier_params.json
```

---

## Monitoring & Validation

### Check Training Progress

**MLflow UI** (view all experiments):
```bash
# Open browser to http://localhost:5000
# View:
# - Training curves (loss, learning rate)
# - Hyperparameter comparisons
# - Model artifacts
```

**Check Embedding Store Stats:**
```bash
python -c "
from ml_skeleton.music.embedding_store import EmbeddingStore
store = EmbeddingStore('music_embeddings.db')
stats = store.get_stats()
print(f'Embeddings: {stats[\"total_embeddings\"]}')
print(f'Versions: {stats[\"num_versions\"]}')
for v in stats['versions']:
    print(f'  {v[\"version\"]}: {v[\"count\"]} embeddings')
"
```

**Check Cache Stats:**
```bash
./run_music_pipeline.sh cache-stats
```

---

## Expected Timeline

| Step | Duration | Command |
|------|----------|---------|
| Encoder training (70 epochs) | ~3 hours | `./run_music_pipeline.sh encoder --best-params checkpoints/best_encoder_params.json` |
| Embedding extraction | Auto (included) | - |
| Classifier training | ~1-2 hours | `./run_music_pipeline.sh classifier` |
| Generate recommendations | ~5-10 min | `./run_music_pipeline.sh recommend` |
| **Total** | **~4-5 hours** | - |

---

## Validation Steps

### After Encoder Training:

```bash
# Verify encoder checkpoint exists
ls -lh checkpoints/encoder_best.pt

# Check embeddings were extracted
python -c "
from ml_skeleton.music.embedding_store import EmbeddingStore
store = EmbeddingStore('music_embeddings.db')
print(f'Total embeddings: {store.get_stats()[\"total_embeddings\"]}')
"
```

### After Classifier Training:

```bash
# Verify classifier checkpoint
ls -lh checkpoints/classifier_best.pt

# Check version compatibility
python -c "
from ml_skeleton.training.classifier_trainer import validate_model_compatibility
from pathlib import Path
validate_model_compatibility(
    encoder_checkpoint=Path('checkpoints/encoder_best.pt'),
    classifier_checkpoint=Path('checkpoints/classifier_best.pt')
)
print('✓ Models are compatible')
"
```

### After Recommendations:

```bash
# Check output files
ls -lh recommendations.json
ls -lh playlists/*.xspf

# View sample recommendations
head -50 recommendations.json
```

---

## Troubleshooting

### If Training Fails:

**Out of Memory:**
```bash
# Reduce batch size in config
# Edit configs/music_moco.yaml:
#   encoder:
#     batch_size: 64  # down from 96
```

**Resume from Checkpoint:**
```bash
./run_music_pipeline.sh encoder --resume-checkpoint checkpoints/encoder_best.pt
```

**Clear Cache and Restart:**
```bash
./run_music_pipeline.sh clear-cache
./run_music_pipeline.sh build-cache
```

---

## Quick Reference

**View Model Card:**
```bash
./run_music_pipeline.sh model-card
```

**Full Pipeline (All Stages):**
```bash
./run_music_pipeline.sh all
```

**Quick Test (5 epochs, 500 songs):**
```bash
./run_music_pipeline.sh quick
```

---

## Recommended Execution Order

**Execute these commands in sequence:**

```bash
# 1. Train encoder with best params (START HERE)
./run_music_pipeline.sh encoder --best-params checkpoints/best_encoder_params.json

# 2. After encoder completes, train classifier
./run_music_pipeline.sh classifier

# 3. Generate recommendations
./run_music_pipeline.sh recommend

# 4. View results in MLflow
# Open browser: http://localhost:5000
```

---

## HPO Results Summary (Trial 13 - Best)

**Best Hyperparameters:**
```yaml
learning_rate: 1.50e-04
adam_weight_decay: 7.58e-04
adam_eps: 3.52e-09
adam_beta1: 0.9499
adam_beta2: 0.9987
loss_weights_moco: 0.4      # 40% MoCo, 60% Genre
gain_db_max: 3.0 dB
noise_prob: 0.5
mixup_alpha: 0.1            # Minimal mixup
```

**Performance:**
- Validation loss: 1.5687
- 77% better than chance level (~6.9)
- 31% improvement over worst trial

**Key Insights:**
- Genre BCE task is more important than MoCo (60/40 split optimal)
- Low learning rate (1.5e-04) works best for multi-task learning
- Minimal mixup (0.1) provides sufficient regularization
- Moderate augmentation (gain 3dB, noise 50%) is effective
