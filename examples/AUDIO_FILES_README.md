# Audio File Locations and Safety

## Overview

This music recommendation system works with your existing music library. **All audio file operations are READ-ONLY** - your music files will never be modified, moved, or deleted.

---

## Audio File Location

### Primary Source: Clementine Database

Your audio files are located via the Clementine music player database:

```
Database Path: /home/ikaro/Music/clementine.db
Table: songs
Column: filename (stores file paths as file:// URIs)
```

### How File Paths Are Stored

The Clementine database stores file paths in the `filename` column using `file://` URI format with URL encoding:

**Example stored value:**
```
file:///home/ikaro/Music/Artist/Album/01%20-%20Song%20Title.mp3
```

**Actual file path:**
```
/home/ikaro/Music/Artist/Album/01 - Song Title.mp3
```

### Typical Music Directory Structure

Based on the Clementine database path, your music files are expected to be in:

```
/home/ikaro/Music/
├── Artist 1/
│   ├── Album 1/
│   │   ├── 01 - Song Title.mp3
│   │   ├── 02 - Song Title.mp3
│   │   └── ...
│   └── Album 2/
│       └── ...
├── Artist 2/
│   └── ...
└── ... (approximately 60,000 songs total)
```

### Supported Audio Formats

The system supports all formats that `torchaudio` and `librosa` can read:

- **MP3** (.mp3) - Most common
- **FLAC** (.flac) - Lossless
- **OGG Vorbis** (.ogg)
- **WAV** (.wav)
- **M4A/AAC** (.m4a)
- **OPUS** (.opus)

---

## READ-ONLY Guarantee

### What the System Does

✅ **READ audio files** to extract acoustic features
✅ **LOAD file metadata** from the Clementine database
✅ **CREATE embeddings** stored in a separate database (`embeddings.db`)
✅ **GENERATE playlists** in XSPF format (separate files)

### What the System NEVER Does

❌ **NEVER modifies** your audio files (MP3, FLAC, etc.)
❌ **NEVER moves** or renames your files
❌ **NEVER deletes** any files
❌ **NEVER writes** to the Clementine database
❌ **NEVER changes** file permissions
❌ **NEVER modifies** ID3 tags or metadata in audio files

### Implementation Safety

All audio loading operations use read-only mode:

```python
# torchaudio loads files in read-only mode by default
waveform, sample_rate = torchaudio.load(filepath)

# librosa also loads in read-only mode
waveform, sample_rate = librosa.load(filepath, sr=16000)
```

**No write operations exist in the codebase for audio files.**

---

## File Access Patterns

### During Encoder Training (Stage 1)

1. **Read** Clementine database to get file paths
2. **Check** song duration (skip if > 15 minutes - configurable)
3. **Load** each audio file (read-only)
4. **Extract** 30 seconds from center of each song
5. **Process** through encoder neural network
6. **Store** embeddings in `embeddings.db` (separate file)

**Duration Filtering:**
- Songs longer than 15 minutes (900 seconds) are skipped by default
- This excludes live albums, DJ sets, audiobooks, podcasts
- Configurable via `max_audio_duration` parameter
- Skipped songs are logged for review

**Estimated time:** ~1-2 hours for 60,000 songs on RTX 5090

### During Classifier Training (Stage 2)

1. **Read** Clementine database to get rated songs
2. **Load** pre-computed embeddings from `embeddings.db`
3. **Train** classifier (no audio file access needed)

**No audio files are accessed during this stage.**

### During Recommendation Generation

1. **Load** pre-computed embeddings from `embeddings.db`
2. **Predict** ratings using classifier
3. **Write** recommendations to XSPF playlist

**No audio files are accessed during this stage.**

---

## Verifying File Locations

### Check if Files Exist

You can verify your audio files are accessible:

```python
from pathlib import Path
import sqlite3
from urllib.parse import unquote

# Connect to Clementine DB
conn = sqlite3.connect("/home/ikaro/Music/clementine.db")
cursor = conn.execute("SELECT filename FROM songs LIMIT 10")

# Check if files exist
for row in cursor:
    filename = row[0]

    # Convert file:// URI to path
    if filename.startswith("file://"):
        filename = filename[7:]
    filename = unquote(filename)

    filepath = Path(filename)
    exists = filepath.exists()

    print(f"{'✓' if exists else '✗'} {filepath}")

conn.close()
```

### Expected Output

```
✓ /home/ikaro/Music/Artist/Album/Song.mp3
✓ /home/ikaro/Music/Artist/Album/Song.flac
✓ /home/ikaro/Music/Artist/Album/Song.ogg
...
```

If you see `✗` marks, those files may have been moved or deleted since Clementine last scanned your library.

---

## File Permissions

### Required Permissions

The system only needs **READ** permission on audio files:

```bash
# Check permissions (should show at least r-- for user)
ls -l /home/ikaro/Music/Artist/Album/*.mp3

# Example output:
-rw-r--r-- 1 ikaro ikaro 4567890 Jan 15 10:30 Song.mp3
  ^
  Read permission required (nothing else)
```

### No Write Permission Needed

Even if your files are read-only, the system will work:

```bash
# This is fine - system doesn't need write access
chmod 444 /home/ikaro/Music/**/*.mp3
```

---

## Network/Remote Files

### Local Files Only (Recommended)

The system is designed for local file access. If your music is on:

- **Local SSD/HDD:** ✅ Optimal performance
- **Network Share (NFS/SMB):** ⚠️ Will work but slower
- **Cloud Storage:** ❌ Not recommended (too slow)

### Network Share Performance

If using network shares mounted at `/home/ikaro/Music/`:

- **Read time:** ~5-10x slower than local files
- **Total processing time:** May take 10-20 hours instead of 1-2 hours
- **Recommendation:** Copy files to local disk for training, or run overnight

---

## Disk Space Requirements

### Your Music Library

```
60,000 songs × ~5 MB average = ~300 GB
```

Your existing music files remain unchanged.

### Additional Space Needed

| Component | Size | Location |
|-----------|------|----------|
| Embeddings DB | ~30-240 MB | `./embeddings.db` |
| Model Checkpoints | ~500 MB | `./checkpoints/` |
| MLflow Artifacts | ~100 MB | `./mlruns/` |
| **Total** | **~1 GB** | **Working directory** |

The embeddings size depends on embedding dimension:
- 128-dim: ~30 MB (60K × 128 × 4 bytes)
- 512-dim: ~120 MB
- 1024-dim: ~240 MB

---

## Troubleshooting

### Problem: "File not found" errors

**Cause:** Audio files moved since Clementine last scanned

**Solution:**
1. Open Clementine
2. Rescan your music library
3. Check database is updated: `~/.config/Clementine/clementine.db`
4. Copy to training location if needed

### Problem: "Permission denied" errors

**Cause:** No read permission on audio files

**Solution:**
```bash
# Add read permission
chmod u+r /home/ikaro/Music/**/*.mp3
```

### Problem: Slow audio loading

**Cause:** Files on network share or slow disk

**Solutions:**
1. Copy files to local SSD
2. Use faster network connection
3. Enable caching (optional feature)
4. Reduce `num_workers` if causing network congestion

### Problem: Very long songs being processed

**Cause:** Live albums, DJ sets, podcasts mixed with music

**Solution:**
```yaml
# In config file
music:
  max_audio_duration: 600.0  # 10 minutes instead of default 15
```

Songs longer than this limit will be skipped and logged.

### Problem: Corrupted audio files

**Cause:** Some audio files may be corrupted

**Solution:**
The system will skip corrupted files and log warnings. Check logs:
```python
# Files that failed to load will be logged
# Check: logs/audio_loading_errors.txt
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Your Music Library (READ-ONLY)                             │
│  /home/ikaro/Music/                                         │
│  ├── Artist/Album/*.mp3  (60,000 songs, ~300 GB)           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ (1) Read file paths
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Clementine Database (READ-ONLY)                            │
│  /home/ikaro/Music/clementine.db                            │
│  Table: songs (filename, title, artist, rating, etc.)       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ (2) Load audio (read-only)
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Audio Loader (Multi-process)                               │
│  - Loads 30 seconds from center                             │
│  - No modifications to source files                         │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ (3) Extract embeddings
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Embeddings Database (NEW FILE - WRITE)                     │
│  ./embeddings.db (~30-240 MB)                               │
│  Table: embeddings (filename, embedding, model_version)     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ (4) Train classifier
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Model Checkpoints (NEW FILES - WRITE)                      │
│  ./checkpoints/encoder_best.pt                              │
│  ./checkpoints/classifier_best.pt                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ (5) Generate recommendations
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  XSPF Playlist (NEW FILE - WRITE)                           │
│  ~/Music/recommendations/test.xspf                          │
│  (References original audio files, doesn't copy them)       │
└─────────────────────────────────────────────────────────────┘
```

**Key Point:** Your original audio files and Clementine database are NEVER modified. All outputs go to NEW files in separate locations.

---

## Summary

✅ **Your music files are safe** - All operations are read-only
✅ **Files stay in place** - No moving, renaming, or copying
✅ **Clementine database is safe** - Read-only access
✅ **New files are separate** - Embeddings and playlists in different locations
✅ **~1 GB additional disk space** needed for embeddings and models

**You can safely run this system on your music library without any risk to your files.**
