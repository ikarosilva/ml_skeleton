# Implementation Updates

## 2026-01-24: MoCo v2 + Genre BCE Encoder Implementation

Replaced SimSiam encoder with MoCo v2 + Genre BCE multi-task learning architecture.

### Architecture
```
Audio → 16kHz .npy cache (4 chunks/song) → nnAudio CQT (84 bins) → ResNet-50 2D → 2048-dim
├── MoCo v2 head (queue=4096, τ=0.07, m=0.999)
└── Genre BCE head (7 categories)

Loss = 0.6×MoCo(NT-Xent) + 0.4×Genre_BCE
```

### Files Created
| File | Description |
|------|-------------|
| `ml_skeleton/music/genre_mapper.py` | 7-category genre mapping with "/" splitting |
| `ml_skeleton/music/chunk_cache.py` | 4-chunk cache builder (30s × 4 per song) |
| `ml_skeleton/music/moco_encoder.py` | MoCo v2 + nnAudio CQT + ResNet-50 + Genre BCE |
| `ml_skeleton/music/moco_dataset.py` | Training dataset with augmentations |
| `configs/music_moco.yaml` | Complete configuration |

### Files Modified
| File | Changes |
|------|---------|
| `pyproject.toml` | Added `nnAudio>=0.3.2` dependency |
| `ml_skeleton/music/clementine_db.py` | Added `genre` field to Song dataclass + SQL query |
| `run_music_pipeline.sh` | Removed SimSiam, updated for MoCo v2, added `cache-stats` command |

### Genre Categories (7)
1. rock (includes blues, metal, punk, alternative)
2. pop (includes dance, disco, synth)
3. electronic (techno, house, trance, ambient, edm)
4. hiphop (hip-hop, rap, r&b, urban)
5. jazz_classical (jazz, classical, instrumental, new age)
6. country (country, bluegrass, americana, folk)
7. latin_world (latin, reggae, world, soundtrack)

### Key Parameters
- **CQT**: 84 bins, fmin=32.7Hz (C1), hop_length=512
- **MoCo v2**: queue=4096, momentum=0.999, temperature=0.07
- **Cache**: 4 chunks/song, 30s each, ~30GB for 60K songs
- **Batch size**: 128 (RTX 5090 optimized)
- **Encoder epochs**: 100
- **Classifier epochs**: 20

### Usage
```bash
# 1. Install dependencies
pip install -e ".[music]"

# 2. Build cache (~30GB, run once)
# Use OMP_NUM_THREADS to control ffmpeg CPU threading
OMP_NUM_THREADS=2 ./run_music_pipeline.sh build-cache

# 3. Train encoder (100 epochs)
./run_music_pipeline.sh encoder

# 4. Train classifier (20 epochs)
./run_music_pipeline.sh classifier

# 5. Generate recommendations
./run_music_pipeline.sh recommend

# Quick test (5 epochs)
./run_music_pipeline.sh quick

# Check cache stats
./run_music_pipeline.sh cache-stats
```

### CPU Resource Control
The cache building process uses multiprocessing for parallel audio loading. To prevent excessive CPU usage:

1. **Worker count**: Set `chunk_cache.num_workers` in config (default: 2)
   ```yaml
   chunk_cache:
     num_workers: 2  # Limit parallel workers
   ```

2. **FFmpeg threading**: Each worker spawns ffmpeg which uses multiple threads internally. Control with `OMP_NUM_THREADS`:
   ```bash
   OMP_NUM_THREADS=2 ./run_music_pipeline.sh build-cache
   ```

Without these controls, cache building can use 400-500% CPU per worker due to ffmpeg's internal threading.

### Dependencies Added
- `nnAudio>=0.3.2` - GPU-accelerated CQT transform
