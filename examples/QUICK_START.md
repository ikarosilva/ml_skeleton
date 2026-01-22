# Quick Start Guide - Music Recommendation System

## Your Files Are Safe ✅

**All operations on your music files and Clementine database are READ-ONLY.**

- Your 60,000 MP3/FLAC files in `/home/ikaro/Music/` will NOT be modified
- Your Clementine database will NOT be changed
- New files (embeddings, models, playlists) are created in separate locations

---

## File Locations Summary

| What | Where | Size | Read/Write |
|------|-------|------|------------|
| **Your Music Files** | `/home/ikaro/Music/` | ~300 GB | **READ-ONLY** |
| **Clementine Database** | `/home/ikaro/Music/clementine.db` | ~100 MB | **READ-ONLY** |
| **Embeddings Database** | `./embeddings.db` | ~30-240 MB | **NEW FILE** (write) |
| **Model Checkpoints** | `./checkpoints/` | ~500 MB | **NEW FILES** (write) |
| **Recommendations** | `~/Music/recommendations/test.xspf` | <1 MB | **NEW FILE** (write) |

**Note:** Songs longer than 15 minutes are skipped by default (configurable). This filters out live albums, DJ sets, and podcasts.

---

## Before You Start

### 1. Test the Hello World Stubs

```bash
cd /git/ml_skeleton
python examples/music_hello_world.py
```

Expected output: `✓ All tests passed!`

### 2. Verify Audio Files Are Accessible

```python
import sqlite3
from pathlib import Path
from urllib.parse import unquote

# Connect to Clementine DB (read-only)
conn = sqlite3.connect("file:///home/ikaro/Music/clementine.db?mode=ro", uri=True)

# Check first 10 files
cursor = conn.execute("SELECT filename FROM songs LIMIT 10")
for row in cursor:
    filename = unquote(row[0].replace("file://", ""))
    exists = Path(filename).exists()
    print(f"{'✓' if exists else '✗'} {filename}")

conn.close()
```

If you see all `✓`, you're ready to proceed.

---

## Customization Workflow

### Step 1: Customize Your Models

Edit [music_hello_world.py](music_hello_world.py):

```python
class HelloWorldEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512):  # Change 128 -> 512
        super().__init__()
        self.embedding_dim = embedding_dim

        # REPLACE THIS with your architecture
        self.my_model = nn.Sequential(
            # Your layers here
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Your preprocessing here
        return embeddings  # Must be (batch, embedding_dim)

    def get_embedding_dim(self) -> int:
        return self.embedding_dim
```

See [CUSTOMIZATION_GUIDE.md](CUSTOMIZATION_GUIDE.md) for detailed examples.

### Step 2: Test Your Models

```bash
python examples/music_hello_world.py
```

Make sure you see:
- `✓ Encoder output shape is correct`
- `✓ Classifier output shape and range are correct`

### Step 3: Wait for Framework Implementation

The training orchestration code (`ml_skeleton/music/`, `ml_skeleton/training/`) is not yet implemented. Once it's ready, you can:

```bash
# Train encoder (Stage 1)
mlskel run configs/music_example.yaml --train-fn examples.music_hello_world:train_encoder_stage

# Train classifier (Stage 2)
mlskel run configs/music_example.yaml --train-fn examples.music_hello_world:train_classifier_stage

# Generate recommendations
mlskel recommend checkpoints/encoder_best.pt checkpoints/classifier_best.pt
```

---

## Expected Timeline

| Phase | Status | ETA |
|-------|--------|-----|
| **Planning** | ✅ Complete | Done |
| **Hello World Stubs** | ✅ Complete | Done |
| **Phase 1: Core Infrastructure** | ⏳ Pending | 3-5 days |
| **Phase 2: Audio Processing** | ⏳ Pending | 3-5 days |
| **Phase 3: Training Orchestration** | ⏳ Pending | 4-6 days |
| **Phase 4: Recommendations** | ⏳ Pending | 3-4 days |
| **Phase 5: Examples & Docs** | ⏳ Pending | 2-3 days |
| **Phase 6: Testing & Polish** | ⏳ Pending | 3-5 days |

**Total:** 3-4 weeks

---

## What You Can Do Now

1. ✅ **Customize encoder/classifier** in [music_hello_world.py](music_hello_world.py)
2. ✅ **Test your models** work with the stubs
3. ✅ **Experiment with architectures** (CNN, Transformer, pre-trained models)
4. ✅ **Read the documentation**:
   - [CUSTOMIZATION_GUIDE.md](CUSTOMIZATION_GUIDE.md) - How to modify models
   - [AUDIO_FILES_README.md](AUDIO_FILES_README.md) - File locations and safety
   - [LISTEN.MD](../LISTEN.MD) - Complete implementation plan

---

## Safety Reminders

- ✅ Your music files are **never modified**
- ✅ Clementine database is **read-only**
- ✅ All new data goes to **separate files**
- ✅ You can safely experiment without risk

---

## Questions?

Check the documentation:
- [LISTEN.MD](../LISTEN.MD) - Full implementation plan
- [AUDIO_FILES_README.md](AUDIO_FILES_README.md) - File locations and safety
- [CUSTOMIZATION_GUIDE.md](CUSTOMIZATION_GUIDE.md) - Model customization

**Happy music recommendation building! 🎵**
