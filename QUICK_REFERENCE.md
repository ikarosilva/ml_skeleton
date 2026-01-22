# Quick Reference - Music Recommendation System

## Database Path Configuration

The database path is now: `/Music/database/clementine_backup_2026-01.db`

### Three Ways to Change It

**1. Environment Variable (Easiest - No file editing)**
```bash
export CLEMENTINE_DB_PATH="/path/to/your/clementine.db"
./run_music_pipeline.sh all
```

**2. Edit Config File**
```bash
nano configs/music_recommendation.yaml
# Change: database_path: "/your/path/here"
```

**3. Custom Config File**
```bash
cp configs/music_recommendation.yaml configs/my_config.yaml
nano configs/my_config.yaml  # Edit your paths
CONFIG=configs/my_config.yaml ./run_music_pipeline.sh all
```

## Quick Start Commands

### After Container Restart

```bash
cd /git/ml_skeleton

# Install dependencies
pip install -e ".[music]"

# Verify database
ls -lh /Music/database/clementine_backup_2026-01.db

# Quick test (5 epochs, ~10 min)
./run_music_pipeline.sh quick

# Full training (~2 hours)
./run_music_pipeline.sh all
```

### With Custom Database Path

```bash
# One-time use
CLEMENTINE_DB_PATH="/your/path.db" ./run_music_pipeline.sh all

# Set for session
export CLEMENTINE_DB_PATH="/your/path.db"
./run_music_pipeline.sh encoder
./run_music_pipeline.sh classifier
./run_music_pipeline.sh recommend
```

## Output Files

After training:
- `checkpoints/encoder_best.pt` - Audio encoder model
- `checkpoints/classifier_best.pt` - Rating classifier model
- `embeddings.db` - SQLite database with embeddings
- `recommendations.txt` - Top recommendations (text)
- `recommender_help.xspf` - 100 uncertain songs (HITL)
- `recommender_best.xspf` - 50 best predictions (validation)

## Human-in-the-Loop Workflow

```bash
# 1. Train models
./run_music_pipeline.sh all

# 2. Open playlists in Clementine
# File > Open Playlist > recommender_help.xspf

# 3. Listen and rate songs
# Right-click > Rate (Clementine saves to database)

# 4. Re-train with new ratings
./run_music_pipeline.sh all

# 5. Repeat for continuous improvement!
```

## Troubleshooting

**Database not found:**
```bash
find /Music -name "*.db" 2>/dev/null
export CLEMENTINE_DB_PATH="/path/you/found.db"
```

**Check rated songs:**
```bash
sqlite3 /Music/database/clementine_backup_2026-01.db \
  "SELECT COUNT(*) FROM songs WHERE rating >= 0;"
```

**Module not found:**
```bash
pip install torchaudio librosa
```

## Git Status

All changes committed (15 commits total).
Ready to push or restart container safely.

```bash
git log --oneline -5
# 71e2f30 Update database path and environment variable override
# ee71d32 Add pre-restart checklist
# b51e495 Add music dependencies to pyproject.toml and setup guide
# b426a5a Document human-in-the-loop reinforcement learning feature
# d3c3fc9 Add human-in-the-loop reinforcement learning with XSPF playlists
```
