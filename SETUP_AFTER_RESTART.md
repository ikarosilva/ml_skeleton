# Setup After Container Restart

## Prerequisites

You'll need to restart the Docker container with the music directory mounted.

### Docker Run Command

Add this volume mount to your docker run command:
```bash
-v ~/Music:/Music:ro  # Read-only mount for safety
```

Or update your existing mount if Music is under a different path.

Example full command:
```bash
docker run --shm-size 50G --runtime=nvidia --privileged -it \
  -p 8888:8888 -p 5000:5000 \
  --env TF_ENABLE_ONEDNN_OPTS=0 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v ~/git:/git \
  -v ~/PycharmProjects:/projects \
  -v ~/Music:/Music:ro \
  --rm kaggle:torch bash
```

## Quick Setup Inside Container

After restarting the container, run these commands:

```bash
# 1. Navigate to project
cd /git/ml_skeleton

# 2. Install with music dependencies
pip install -e ".[music]"

# 3. Verify Clementine DB path (default path in config)
ls -lh /Music/database/clementine_backup_2026-01.db
# Or set your own path using environment variable (see "Update Configuration" below)

# 4. Test the setup (optional - quick check)
python -c "import torch; import torchaudio; import librosa; print('✓ All imports work!')"

# 5. Run quick test (5 epochs)
./run_music_pipeline.sh quick
```

## Alternative: Full Installation

If you want all dependencies:

```bash
cd /git/ml_skeleton
pip install -e ".[all]"
```

This includes: PyTorch, TorchAudio, Librosa, TensorFlow, Ray Tune, and all extras.

## Update Configuration

You have **three options** to set the database path (in order of preference):

### Option 1: Environment Variable (Easiest)

Set the `CLEMENTINE_DB_PATH` environment variable:

```bash
export CLEMENTINE_DB_PATH="/Music/database/clementine_backup_2026-01.db"
./run_music_pipeline.sh all
```

Or inline:

```bash
CLEMENTINE_DB_PATH="/Music/database/clementine_backup_2026-01.db" ./run_music_pipeline.sh all
```

### Option 2: Edit Config File

Edit `configs/music_recommendation.yaml`:

```yaml
music:
  database_path: "/Music/database/clementine_backup_2026-01.db"
```

### Option 3: Create Custom Config

Copy and customize:

```bash
cp configs/music_recommendation.yaml configs/my_music.yaml
# Edit configs/my_music.yaml with your paths
CONFIG=configs/my_music.yaml ./run_music_pipeline.sh all
```

## Verify Setup

Check that everything is ready:

```bash
# Check rated songs count
sqlite3 /home/ikaro/Music/clementine.db "SELECT COUNT(*) FROM songs WHERE rating >= 0;"

# Should show number of rated songs (>0 for training to work)
```

## Run Training

```bash
# Full pipeline
./run_music_pipeline.sh all

# Or individual stages
./run_music_pipeline.sh encoder
./run_music_pipeline.sh classifier
./run_music_pipeline.sh recommend
```

## What You'll Get

After Stage 3 completes:
- `checkpoints/encoder_best.pt` - Trained audio encoder
- `checkpoints/classifier_best.pt` - Trained rating classifier
- `embeddings.db` - SQLite database with all embeddings
- `recommendations.txt` - Text file with top recommendations
- `recommender_help.xspf` - 100 uncertain songs for human rating (HITL)
- `recommender_best.xspf` - 50 best predictions for validation

## Troubleshooting

**If you get "No module named 'torchaudio'":**
```bash
pip install torchaudio librosa
```

**If Clementine DB not found:**
```bash
# Find it
find /home -name "clementine.db" 2>/dev/null
find /Music -name "clementine.db" 2>/dev/null

# Update config with correct path
nano configs/music_recommendation.yaml
```

**If "No rated songs found":**
```bash
# Check your database has ratings
sqlite3 /path/to/clementine.db "SELECT COUNT(*) FROM songs WHERE rating >= 0;"

# If 0, you need to rate some songs in Clementine first
```

## Git Status

All changes are committed. Current branch status:
```bash
git log --oneline -5
# Should show:
# b426a5a Document human-in-the-loop reinforcement learning feature
# d3c3fc9 Add human-in-the-loop reinforcement learning with XSPF playlists
# a594b60 Add comprehensive Quick Start guide and pipeline runner script
# c83b3ba Checkpoint 8: Update documentation for music recommendation system
# 25c49a2 Checkpoint 7: Add configuration and end-to-end example
```

You have 12 total commits with full checkpoint history!
