# Music Recommendation System - Implementation Plan

## Executive Summary

This plan extends the `ml_skeleton` framework to build a deep learning music recommendation system that:
- Analyzes acoustic signatures of 60,000 songs from Clementine music database
- Uses a two-stage PyTorch architecture: Encoder (audio → embedding) + Classifier (embedding → rating)
- Stores embeddings in SQLite database for fast retrieval
- Generates XSPF playlist recommendations based on predicted ratings
- Supports both two-phase training (encoder first, then classifier) and end-to-end joint training
- Maximizes multicore utilization with configurable worker count (default: 80% CPU cores)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MUSIC RECOMMENDATION SYSTEM                   │
└─────────────────────────────────────────────────────────────────┘

INPUT: Clementine Database (/home/ikaro/Music/clementine.db)
       └─> 60K songs with metadata (title, artist, filename, rating)

STAGE 1: AUDIO ENCODER
       Audio Files → [User's Encoder Model] → Embeddings (1×Z vector)
       └─> Store in embeddings.db (filename → embedding mapping)

STAGE 2: RATING CLASSIFIER
       Embeddings (Z×N batch) → [User's Classifier Model] → Ratings (1×N)
       └─> Train on rated songs, predict on unrated songs

OUTPUT: XSPF Playlist (~/Music/recommendations/test.xspf)
       └─> Top-N unrated songs with highest predicted ratings
```

---

## 1. New Directory Structure

Add the following modules to `ml_skeleton/`:

```
ml_skeleton/
├── music/                          # NEW: Music-specific functionality
│   ├── __init__.py                 # Exports: MusicDataset, EmbeddingStore, etc.
│   ├── audio_loader.py             # Audio file loading with multiprocessing
│   ├── dataset.py                  # PyTorch Dataset for songs
│   ├── embedding_store.py          # SQLite storage for embeddings (multi-version support)
│   ├── clementine_db.py            # Clementine database interface
│   ├── speech_detector.py          # (NEW) Speech detection from audio
│   ├── recommendation.py           # Recommendation generation & XSPF export
│   ├── losses.py                   # NTXentLoss (SimCLR contrastive loss)
│   ├── augmentation.py             # AudioAugmentor (pitch shift, time stretch, noise)
│   ├── multitask.py                # Multi-task encoder (artist/genre/period classification)
│   └── deduplication.py            # Audio fingerprinting & duplicate detection
├── training/                       # NEW: Training orchestration
│   ├── __init__.py
│   ├── encoder_trainer.py          # Stage 1: Encoder training
│   ├── classifier_trainer.py       # Stage 2: Classifier training
│   └── joint_trainer.py            # End-to-end training
├── protocols/                      # NEW: User-injectable interfaces
│   ├── __init__.py
│   ├── encoder.py                  # AudioEncoder protocol
│   └── classifier.py               # RatingClassifier protocol
└── cli_extensions/                 # NEW: Music-specific CLI commands
    └── music_commands.py           # Commands: train-encoder, train-classifier, recommend

configs/
└── music_example.yaml              # NEW: Example config for music recommendation

examples/
└── music_recommendation.py         # NEW: Complete example with mock encoder/classifier
```

---

## 2. Audio File Safety & Location

**CRITICAL: ALL AUDIO FILE OPERATIONS ARE READ-ONLY**

### Audio File Locations

- **Source Database:** `/home/ikaro/Music/clementine.db`
- **File Paths:** Stored in `songs.filename` column as `file://` URIs
- **Expected Location:** `/home/ikaro/Music/` (approximately 60,000 songs, ~300 GB)
- **Supported Formats:** MP3, FLAC, OGG, WAV, M4A, OPUS

### Read-Only Guarantees

The system will:
- ✅ **READ** audio files to extract acoustic features
- ✅ **LOAD** file metadata from Clementine database (read-only)
- ✅ **CREATE** embeddings in separate database (`./embeddings.db`)
- ✅ **GENERATE** playlists in separate XSPF files

The system will NEVER:
- ❌ **MODIFY** audio files (MP3, FLAC, etc.)
- ❌ **WRITE** to Clementine database
- ❌ **MOVE** or rename files
- ❌ **DELETE** any files
- ❌ **CHANGE** ID3 tags or metadata

### Implementation

All audio loading uses read-only operations:
```python
# torchaudio.load() - read-only by default
waveform, sr = torchaudio.load(filepath)

# librosa.load() - read-only by default
waveform, sr = librosa.load(filepath, sr=22050)
```

**See [examples/AUDIO_FILES_README.md](examples/AUDIO_FILES_README.md) for complete documentation.**

---

## 2.1 Duplicate Detection & Deduplication

**Problem**: The same audio file may exist at multiple filesystem locations, leading to:
- Wasted computation (processing same audio multiple times)
- Duplicate embeddings in database
- Training on duplicate samples

**Solution**: Audio fingerprinting with intelligent prioritization

### Deduplication Strategy

**Multi-Stage Approach** (optimized for speed):

**Stage 1: Fast Metadata Matching** (required before fingerprinting)
- Extract metadata: file size, duration, sample rate, bit rate, channels
- Group files by metadata signature (exact match)
- Only fingerprint files within same metadata group
- **Performance**: ~1000-5000 files/second (metadata extraction)
- **Caching**: Metadata cached in SQLite, invalidated on file modification

**Stage 2: Audio Fingerprinting** (only for metadata matches)
- Use chromaprint (AcoustID) for perceptual hash
- Only run on files with matching metadata from Stage 1
- Detects bit-identical files and transcoded versions
- **Performance**: ~5-10 files/second on single core, ~400-800 with 80 cores
- **Caching**: Fingerprints cached in SQLite

**Step 1: Metadata Extraction**
```python
@dataclass
class AudioMetadata:
    """Quick metadata for duplicate detection."""
    filename: str
    file_size: int       # Bytes (instant)
    duration: float      # Seconds (fast)
    sample_rate: int     # Hz (fast, e.g., 44100, 48000)
    bit_rate: int        # kbps (fast, e.g., 320, 256, 128)
    channels: int        # 1=mono, 2=stereo (fast)
    codec: str           # MP3, FLAC, OGG (fast)
    mtime: float         # File modification time (instant)

    def signature(self) -> tuple:
        """Create hashable signature for grouping."""
        return (
            self.file_size,
            round(self.duration, 1),  # Round to 0.1s tolerance
            self.sample_rate,
            self.bit_rate,
            self.channels
        )

def extract_metadata(filepath: Path) -> Optional[AudioMetadata]:
    """Extract metadata without loading full audio.

    Uses torchaudio.info() which is much faster than loading audio.
    """
    try:
        # Get file size (instant)
        file_size = filepath.stat().st_size
        mtime = filepath.stat().st_mtime

        # Get audio metadata (fast - no audio loading)
        info = torchaudio.info(str(filepath))

        return AudioMetadata(
            filename=str(filepath),
            file_size=file_size,
            duration=info.num_frames / info.sample_rate,
            sample_rate=info.sample_rate,
            bit_rate=info.bits_per_sample * info.sample_rate * info.num_channels // 1000,
            channels=info.num_channels,
            codec=filepath.suffix[1:].upper(),  # .mp3 -> MP3
            mtime=mtime
        )
    except Exception as e:
        print(f"Failed to extract metadata from {filepath}: {e}")
        return None
```

**Step 2: Metadata-Based Grouping**
```python
from ml_skeleton.music.deduplication import AudioDeduplicator

deduplicator = AudioDeduplicator()
songs = load_all_songs()  # From Clementine DB

# Stage 1: Group by metadata signature
metadata_groups = deduplicator.group_by_metadata(songs)

# Result: dict mapping metadata signature → list[Song]
# Example: {
#   (5242880, 180.5, 44100, 320, 2): [  # (size, duration, sr, bitrate, channels)
#       Song(filename="/Music/Album1/track.mp3", ...),
#       Song(filename="/Music/Compilation/track.mp3", ...)
#   ]
# }

# Stage 2: Only fingerprint groups with 2+ files
duplicates = deduplicator.find_duplicates(songs, use_metadata_filter=True)

# Result: dict mapping fingerprint → list[Song]
# Only groups that passed metadata check are fingerprinted
```

**Step 3: Selective Fingerprinting**

Only fingerprint files that share the same metadata signature:

```python
def should_fingerprint(metadata_group: list[AudioMetadata]) -> bool:
    """Decide if group needs fingerprinting.

    Returns:
        True if group has 2+ files (potential duplicates)
        False if group has only 1 file (unique)
    """
    return len(metadata_group) >= 2
```

**Metadata Fields Checked** (in priority order):

1. **File size** (bytes) - Instant, most discriminative
2. **Duration** (seconds, rounded to 0.1s) - Very fast
3. **Sample rate** (Hz) - Fast (44100, 48000, 22050, etc.)
4. **Bit rate** (kbps) - Fast (320, 256, 192, 128, etc.)
5. **Channels** (1=mono, 2=stereo) - Fast

**Optional additional checks** (can add if needed):
- **Codec/format** (MP3, FLAC, OGG) - Fast, but less useful (different formats can be same audio)
- **Bits per sample** (16, 24, 32) - Fast
- **File basename** (heuristic) - Instant, helps but not definitive

**Not recommended**:
- ❌ **ID3 tags** (title, artist, album) - Unreliable, can differ for same file
- ❌ **Full file hash** (MD5/SHA) - Defeats purpose (as expensive as loading audio)

**Benefits of Metadata Pre-Filtering**:
- **Speed**: Skip 90-95% of files (unique metadata → skip fingerprinting)
- **Cost**: Metadata extraction is ~100-500x faster than fingerprinting
- **Accuracy**: Bit-identical files MUST have same metadata
- **Caching**: Metadata cached, only compute once per file
- **Parallel**: Metadata extraction easily parallelized

**Example**: For 60,000 songs:
- **Without metadata filter**: Fingerprint all 60K songs (~2-5 minutes)
- **With metadata filter**:
  - Extract metadata: 60K songs (~10-15 seconds)
  - Fingerprint only ~3-6K candidates (~15-30 seconds)
  - **Total: ~25-45 seconds** (5-10x faster!)

**Step 4: Prioritization (when duplicates found)**

For each group of duplicates, keep ONE canonical version using this priority:

1. **Has rating** (rated > unrated)
   - Songs with `rating >= 0` are preferred
   - Helps maximize training data

2. **Latest modified time** (newer > older)
   - Use `os.path.getmtime()` for last modification timestamp
   - Assumes newer files are higher quality or more recent copies

3. **Shortest path** (tiebreaker)
   - Prefers `/Music/Artist/Album/song.mp3` over `/Music/Backups/2020/Artist/Album/song.mp3`
   - Helps avoid backup directories

**Example**:
```python
def choose_canonical_song(duplicates: list[Song]) -> Song:
    """Select best version from duplicate group."""

    # Sort by priority:
    # 1. Has rating (True > False)
    # 2. Latest modified time (descending)
    # 3. Shortest path (ascending)

    return sorted(
        duplicates,
        key=lambda s: (
            s.rating >= 0,           # Has rating (True first)
            s.mtime,                  # Latest modified (descending)
            -len(s.filename)          # Shortest path (ascending)
        ),
        reverse=True
    )[0]
```

### Rating Merging

When duplicates have different ratings, use **latest rating**:

```python
def merge_ratings(duplicates: list[Song]) -> float:
    """Merge ratings from duplicate files."""

    # Filter songs with valid ratings
    rated = [s for s in duplicates if s.rating >= 0]

    if not rated:
        return -1  # No ratings

    # Return rating from most recently modified file
    latest = max(rated, key=lambda s: s.mtime)
    return latest.rating
```

**Rationale**: Your most recent rating reflects current preference better than older ratings.

### Implementation

**New File**: `ml_skeleton/music/deduplication.py`

```python
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import multiprocessing
from multiprocessing import Pool
import chromaprint  # pip install pyacoustid
import torchaudio

def get_default_workers() -> int:
    """Get default number of workers (80% of CPU cores).

    Returns:
        Number of worker processes to use for parallel operations.
        Leaves 20% of cores free for system and other tasks.
    """
    cpu_count = multiprocessing.cpu_count()
    return max(1, int(cpu_count * 0.8))

@dataclass
class AudioMetadata:
    """Quick metadata for duplicate detection."""
    filename: str
    file_size: int       # Bytes (instant)
    duration: float      # Seconds (fast)
    sample_rate: int     # Hz (fast, e.g., 44100, 48000)
    bit_rate: int        # kbps (fast, e.g., 320, 256, 128)
    channels: int        # 1=mono, 2=stereo (fast)
    codec: str           # MP3, FLAC, OGG (fast)
    mtime: float         # File modification time (instant)

    def signature(self) -> tuple:
        """Create hashable signature for grouping."""
        return (
            self.file_size,
            round(self.duration, 1),  # Round to 0.1s tolerance
            self.sample_rate,
            self.bit_rate,
            self.channels
        )

@dataclass
class AudioFingerprint:
    """Audio fingerprint for duplicate detection."""
    filename: str
    fingerprint: str  # Chromaprint hash
    duration: float
    file_size: int
    mtime: float  # Last modification time

class AudioDeduplicator:
    """Detect and handle duplicate audio files with multi-stage optimization."""

    def __init__(self, cache_path: str = "./fingerprints.db"):
        self.cache_path = cache_path
        self._metadata_cache = {}  # filename -> AudioMetadata
        self._fingerprint_cache = {}  # filename -> fingerprint
        self._load_cache()

    def extract_metadata(self, filepath: Path) -> Optional[AudioMetadata]:
        """Extract quick metadata without loading audio.

        Uses torchaudio.info() - ~100-500x faster than fingerprinting.
        """
        # Check cache first
        cache_key = str(filepath)
        if cache_key in self._metadata_cache:
            cached_meta = self._metadata_cache[cache_key]
            # Validate cache: check if file modified
            if filepath.stat().st_mtime == cached_meta.mtime:
                return cached_meta

        # Extract fresh metadata
        try:
            file_size = filepath.stat().st_size
            mtime = filepath.stat().st_mtime

            # Fast: get audio info without loading
            info = torchaudio.info(str(filepath))

            metadata = AudioMetadata(
                filename=str(filepath),
                file_size=file_size,
                duration=info.num_frames / info.sample_rate,
                sample_rate=info.sample_rate,
                bit_rate=info.bits_per_sample * info.sample_rate * info.num_channels // 1000,
                channels=info.num_channels,
                codec=filepath.suffix[1:].upper(),
                mtime=mtime
            )

            # Cache for future
            self._metadata_cache[cache_key] = metadata
            return metadata

        except Exception as e:
            print(f"Failed to extract metadata from {filepath}: {e}")
            return None

    def group_by_metadata(
        self,
        songs: list[Song],
        num_workers: Optional[int] = None
    ) -> dict[tuple, list[Song]]:
        """Group songs by metadata signature (fast pre-filter).

        Args:
            songs: List of songs from Clementine DB
            num_workers: Parallel workers (default: 80% CPU cores)

        Returns:
            dict mapping metadata signature -> list of songs
            Only includes groups with 2+ songs (potential duplicates)
        """
        if num_workers is None:
            num_workers = get_default_workers()

        print(f"Using {num_workers} worker processes (out of {multiprocessing.cpu_count()} cores)")

        # Extract metadata in parallel
        with Pool(num_workers) as pool:
            metadata_list = pool.map(
                lambda s: self.extract_metadata(Path(s.filename)),
                songs
            )

        # Group by metadata signature
        groups = {}
        for song, metadata in zip(songs, metadata_list):
            if metadata is None:
                continue  # Skip files with extraction errors

            sig = metadata.signature()
            if sig not in groups:
                groups[sig] = []
            groups[sig].append(song)

        # Return only groups with 2+ files (potential duplicates)
        return {sig: group for sig, group in groups.items() if len(group) >= 2}

    def compute_fingerprint(self, filepath: Path) -> Optional[str]:
        """Compute chromaprint fingerprint for audio file.

        Returns:
            Fingerprint hash string, or None if file can't be processed.
        """
        try:
            # Load audio (first 30 seconds for speed)
            waveform, sr = torchaudio.load(
                str(filepath),
                frame_offset=0,
                num_frames=30 * sr  # 30 seconds
            )

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)

            # Generate chromaprint
            fingerprint = chromaprint.fingerprint(
                waveform.numpy(),
                sample_rate=sr
            )

            return fingerprint

        except Exception as e:
            print(f"Failed to fingerprint {filepath}: {e}")
            return None

    def find_duplicates(
        self,
        songs: list[Song],
        num_workers: Optional[int] = None,
        use_metadata_filter: bool = True
    ) -> dict[str, list[Song]]:
        """Find duplicate audio files with metadata pre-filtering.

        Args:
            songs: List of songs from Clementine DB
            num_workers: Parallel workers (default: 80% CPU cores)
            use_metadata_filter: If True, only fingerprint metadata-matched groups

        Returns:
            dict mapping fingerprint -> list of duplicate songs
            Only includes groups with 2+ songs (actual duplicates)
        """
        if num_workers is None:
            num_workers = get_default_workers()

        # Stage 1: Metadata filtering (optional but recommended)
        if use_metadata_filter:
            print("Stage 1: Grouping by metadata signature...")
            metadata_groups = self.group_by_metadata(songs, num_workers)

            # Flatten groups into candidate list
            candidates = []
            for group in metadata_groups.values():
                candidates.extend(group)

            print(f"  Metadata filter: {len(songs)} total → {len(candidates)} candidates")
            songs_to_fingerprint = candidates
        else:
            songs_to_fingerprint = songs

        # Stage 2: Fingerprint candidates only
        print(f"Stage 2: Fingerprinting {len(songs_to_fingerprint)} songs...")
        fingerprints = {}

        with Pool(num_workers) as pool:
            results = pool.map(
                self._fingerprint_song,
                [(s.filename, s) for s in songs_to_fingerprint]
            )

        # Group by fingerprint
        for fingerprint, song in results:
            if fingerprint is None:
                continue  # Skip files that couldn't be fingerprinted

            if fingerprint not in fingerprints:
                fingerprints[fingerprint] = []
            fingerprints[fingerprint].append(song)

        # Return only groups with duplicates (2+ songs)
        duplicates = {
            fp: group
            for fp, group in fingerprints.items()
            if len(group) > 1
        }

        print(f"  Found {len(duplicates)} duplicate groups")
        return duplicates

    def deduplicate(
        self,
        songs: list[Song],
        merge_ratings: bool = True
    ) -> list[Song]:
        """Remove duplicates and return canonical song list.

        Args:
            songs: All songs from Clementine DB
            merge_ratings: If True, merge ratings from duplicates

        Returns:
            Deduplicated song list with merged ratings
        """
        duplicates = self.find_duplicates(songs)

        # Build set of canonical filenames
        canonical = {}

        for fingerprint, dup_group in duplicates.items():
            # Choose best version
            best = self._choose_canonical(dup_group)

            # Merge ratings if requested
            if merge_ratings:
                merged_rating = self._merge_ratings(dup_group)
                best.rating = merged_rating

            canonical[best.filename] = best

        # Add non-duplicate songs
        all_dup_filenames = {
            s.filename
            for dup_group in duplicates.values()
            for s in dup_group
        }

        for song in songs:
            if song.filename not in all_dup_filenames:
                canonical[song.filename] = song

        return list(canonical.values())

    def _choose_canonical(self, duplicates: list[Song]) -> Song:
        """Select best version from duplicate group."""
        return sorted(
            duplicates,
            key=lambda s: (
                s.rating >= 0,        # Has rating
                s.mtime,              # Latest modified
                -len(s.filename)      # Shortest path
            ),
            reverse=True
        )[0]

    def _merge_ratings(self, duplicates: list[Song]) -> float:
        """Merge ratings - use latest rating."""
        rated = [s for s in duplicates if s.rating >= 0]

        if not rated:
            return -1  # No ratings

        # Return rating from most recently modified file
        latest = max(rated, key=lambda s: s.mtime)
        return latest.rating

    def _fingerprint_song(self, args):
        """Helper for parallel fingerprinting."""
        filepath, song = args
        fp = self.compute_fingerprint(Path(filepath))
        return (fp, song)

    def _load_cache(self):
        """Load fingerprint cache from disk."""
        # TODO: Implement SQLite cache for fingerprints
        pass

    def _save_cache(self):
        """Save fingerprint cache to disk."""
        # TODO: Implement SQLite cache for fingerprints
        pass
```

### Configuration

Add to `configs/music_example.yaml`:

```yaml
music:
  # Duplicate detection
  deduplicate: true                    # Enable deduplication
  fingerprint_cache: ./fingerprints.db # Cache fingerprints for speed
  merge_duplicate_ratings: true        # Merge ratings from duplicates
```

### Workflow Integration

```python
# In encoder training workflow
def prepare_training_data(config):
    # Load songs from Clementine
    songs = load_all_songs(config.music.database_path)

    # Deduplicate if enabled
    if config.music.deduplicate:
        deduplicator = AudioDeduplicator(config.music.fingerprint_cache)
        songs = deduplicator.deduplicate(
            songs,
            merge_ratings=config.music.merge_duplicate_ratings
        )

        print(f"After deduplication: {len(songs)} unique songs")

    # Continue with training...
```

### Performance

**Multi-Stage Performance** (60,000 songs on RTX 5090 system with 80 cores):

| Stage | Operation | Speed | Time (60K songs) |
|-------|-----------|-------|------------------|
| **Stage 1** | Metadata extraction | ~1000-5000 files/sec | **10-15 seconds** |
| **Stage 2** | Fingerprinting candidates | ~400-800 files/sec | **15-30 seconds** |
| **Total** | **First run** | - | **25-45 seconds** |
| **Cached** | **Subsequent runs** | - | **~1 second** |

**Without metadata filter** (baseline):
- Fingerprint all 60K songs: ~2-5 minutes
- **Speedup with metadata filter: 5-10x faster!**

**Memory Usage**:
- Metadata per song: ~100 bytes
- Fingerprint per song: ~100 bytes
- Total for 60K songs: ~12 MB (negligible)

**Caching Strategy**:
```sql
-- Cache table schema
CREATE TABLE dedup_cache (
    filename TEXT PRIMARY KEY,
    file_size INTEGER,
    duration REAL,
    sample_rate INTEGER,
    bit_rate INTEGER,
    channels INTEGER,
    codec TEXT,
    mtime REAL,           -- File modification time
    fingerprint TEXT,     -- Chromaprint hash (NULL if not computed)
    updated_at REAL       -- Cache timestamp
);

CREATE INDEX idx_metadata_sig ON dedup_cache(file_size, duration, sample_rate, bit_rate, channels);
CREATE INDEX idx_mtime ON dedup_cache(mtime);
```

**Cache Invalidation**:
- Check `mtime` (file modification time) on every run
- If file modified → re-extract metadata and fingerprint
- If file unchanged → use cached values
- **Typical scenario**: 99% cache hit rate (files rarely change)

### Benefits

1. **Avoid duplicate computation**: Process each unique audio only once
2. **Better training**: No duplicate samples in training set
3. **Cleaner recommendations**: No duplicate songs in playlists
4. **Preserve user ratings**: Merge ratings from all copies
5. **Automatic**: Runs transparently during data loading

### Alternative: Simple Path-Based Deduplication

If chromaprint is unavailable, fall back to simpler approach:

```python
def simple_deduplicate(songs: list[Song]) -> list[Song]:
    """Deduplicate by exact filename match (less robust)."""
    seen = {}

    for song in songs:
        filename = Path(song.filename).name  # Just filename, no path

        if filename not in seen:
            seen[filename] = song
        else:
            # Keep version with rating or latest mtime
            if song.rating >= 0 and seen[filename].rating < 0:
                seen[filename] = song
            elif song.mtime > seen[filename].mtime:
                seen[filename] = song

    return list(seen.values())
```

**Note**: This is less robust (misses transcoded duplicates) but requires no additional dependencies.

---
## 2.2 Speech Detection & Filtering

**Problem**: The music library may contain significant amounts of non-music audio, such as podcasts, interviews, or audiobooks. Including these in the training data can degrade the quality of the learned acoustic representations and lead to a poor user experience.

**Solution**: An automated pipeline step to detect and filter out files containing a high probability of speech before they are fed into the training process.

### Speech Detection Strategy

The process is designed as an independent module that runs after deduplication.

**Workflow:**
1.  **Input**: The list of unique, canonical `Song` objects from the `AudioDeduplicator`.
2.  **Analysis**: For each song, a 30-second audio sample is extracted from the **center** of the file. Files longer than 15 minutes are skipped.
3.  **VAD Model**: The audio sample is passed to a pre-trained Voice Activity Detection (VAD) model (`silero-vad`) which calculates the total duration of speech within the sample.
4.  **Probability Calculation**: The speech probability is calculated as `total_speech_duration / sample_duration`. A value of `1.0` means the sample is entirely speech.
5.  **Caching**: The calculated probability and the file's modification time are cached in a dedicated SQLite database (`speech_cache.db`). On subsequent runs, the cache is checked first, and only new or modified files are re-analyzed.
6.  **Filtering**: The `MusicDataset` is initialized with the list of songs and their speech probabilities. It filters out any song where the probability exceeds a user-configurable `speech_threshold`.

### Caching Database Schema

A new, dedicated SQLite database will be created for caching speech detection results.

**File**: `speech_cache.db`

**Table**: `is_speech_cache`

```sql
CREATE TABLE IF NOT EXISTS is_speech_cache (
    filename TEXT PRIMARY KEY,          -- Absolute path to the audio file
    speech_probability REAL NOT NULL,   -- Probability of speech in the sample [0.0, 1.0]
    mtime REAL NOT NULL,                -- File modification time, for cache invalidation
    updated_at REAL NOT NULL            -- Timestamp of when the cache entry was last updated
);

CREATE INDEX IF NOT EXISTS idx_speech_probability ON is_speech_cache(speech_probability);
```

### Implementation

**New File**: `ml_skeleton/music/speech_detector.py`

```python
# ml_skeleton/music/speech_detector.py

import torch
import torchaudio
from pathlib import Path
from dataclasses import dataclass
import multiprocessing
from multiprocessing import Pool
from typing import List, Optional

@dataclass
class SpeechDetectionResult:
    filename: str
    speech_probability: float

class SpeechDetector:
    """
    Analyzes audio files to detect the probability of speech using
    a pre-trained VAD model. Results are cached in an SQLite database.
    """

    def __init__(self, cache_path: str = "./speech_cache.db"):
        # ... Connect to DB, load VAD model from torch.hub ...
        pass

    def detect_speech_in_songs(
        self,
        songs: List[Song],
        num_workers: Optional[int] = None
    ) -> List[SpeechDetectionResult]:
        """
        Processes a list of songs in parallel to detect speech.
        """
        if num_workers is None:
            num_workers = get_default_workers()

        with Pool(num_workers) as pool:
            results = pool.map(self._process_single_song, songs)
        
        return [res for res in results if res is not None]

    def _process_single_song(self, song: Song) -> Optional[SpeechDetectionResult]:
        """
        Checks cache or runs detection for a single song.
        This is the target function for the multiprocessing pool.
        """
        # 1. Check cache first, return if valid
        # 2. Skip files longer than 15 minutes
        # 3. Load center 30s of audio
        # 4. Run VAD model, calculate probability
        # 5. Update cache
        # 6. Return SpeechDetectionResult
        pass
```

### Configuration

Add to `configs/music_example.yaml`:

```yaml
music:
  # ... other settings

  # Speech detection and filtering
  speech_detection:
    enabled: true                          # Master switch for this feature
    cache_path: "./speech_cache.db"        # Path to the speech probability cache
    speech_threshold: 0.5                  # Filter songs with speech prob > 50%
```

### Workflow Integration

```python
# In encoder training workflow
def prepare_training_data(config):
    songs = load_all_songs(config.music.database_path)

    # Step 1: Deduplication (existing)
    if config.music.deduplicate:
        # ... deduplication logic ...

    # Step 2: Speech Detection (new)
    speech_results = {}
    if config.music.speech_detection.enabled:
        detector = SpeechDetector(config.music.speech_detection.cache_path)
        results = detector.detect_speech_in_songs(songs)
        speech_results = {res.filename: res.speech_probability for res in results}

    # Step 3: Create Dataset, which now filters based on speech
    dataset = MusicDataset(
        songs,
        speech_results=speech_results,
        speech_threshold=config.music.speech_detection.speech_threshold
    )
    # ... continue with training ...
```


---

## 3. Core Components Design

### 3.1 User-Injectable Protocols

Define clear interfaces for user code injection:

**`ml_skeleton/protocols/encoder.py`**:
```python
from typing import Protocol
import torch

class AudioEncoder(Protocol):
    """User implements this to encode audio to embeddings.

    The encoder should:
    - Accept raw audio waveform tensor (batch_size, num_samples)
    - Return embedding tensor (batch_size, embedding_dim)
    - Be a torch.nn.Module for training/optimization
    """

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Raw waveform tensor, shape (batch_size, num_samples)
                   Sample rate is configurable (default: 22050 Hz)

        Returns:
            embeddings: shape (batch_size, embedding_dim)
        """
        ...

    def get_embedding_dim(self) -> int:
        """Return the dimensionality Z of output embeddings."""
        ...
```

**`ml_skeleton/protocols/classifier.py`**:
```python
from typing import Protocol
import torch

class RatingClassifier(Protocol):
    """User implements this to predict ratings from embeddings.

    The classifier should:
    - Accept embedding tensor (batch_size, embedding_dim)
    - Return rating predictions (batch_size, 1) in range [0, 1]
    - Be a torch.nn.Module for training/optimization
    """

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: shape (batch_size, embedding_dim)

        Returns:
            ratings: shape (batch_size, 1), values in [0, 1]
        """
        ...
```

---

### 3.2 Clementine Database Interface (READ-ONLY)

**`ml_skeleton/music/clementine_db.py`**:
- Connect to `/home/ikaro/Music/clementine.db` in **READ-ONLY mode**
- Query songs table: `SELECT rowid, title, artist, album, year, rating, filename`
- Parse `file://` URIs to filesystem paths (URL decode, remove protocol)
- Filter valid songs (non-empty title/artist)
- Separate rated vs unrated songs
- Reuse code from `/git/clementine/src/database/connection.py` and `models.py`

**Database Connection Safety:**
```python
# Open in read-only mode to prevent accidental writes
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
```

**Key Functions**:
```python
@dataclass
class Song:
    rowid: int
    title: str
    artist: str
    album: str
    year: int
    rating: float  # -1 = unrated, 0.0-1.0 = rated
    filename: str  # file:// URI

    @property
    def filepath(self) -> Path:
        """Convert file:// URI to filesystem Path"""
        # Remove file:// prefix, URL decode, resolve path
        ...

def load_all_songs(db_path: str = "/home/ikaro/Music/clementine.db") -> list[Song]:
    """Load all valid songs from Clementine database."""
    ...

def get_rated_songs(songs: list[Song]) -> list[Song]:
    """Filter songs with rating >= 0."""
    ...

def get_unrated_songs(songs: list[Song]) -> list[Song]:
    """Filter songs with rating == -1."""
    ...
```

---

### 3.3 Audio Loading with Multiprocessing (READ-ONLY)

**`ml_skeleton/music/audio_loader.py`**:
- **Load audio files in READ-ONLY mode** using `torchaudio.load()` or `librosa.load()`
- Handle multiple formats: MP3, FLAC, OGG, WAV, M4A, OPUS
- Configurable sample rate (default: 22050 Hz)
- **Extract from CENTER of song** (default: 30 seconds)
- **Skip songs longer than max_duration** (default: 15 minutes = 900 seconds)
- Parallel loading with `multiprocessing.Pool`
- Chunk processing to avoid memory overflow
- Progress tracking with `tqdm`
- **No modifications to source files** - only reading audio data
- **Log skipped files** (too long, corrupted, missing) for user review

**Key Functions**:
```python
def load_audio_file(
    filepath: Path,
    sample_rate: int = 22050,
    mono: bool = True,
    duration: Optional[float] = 30.0,
    center_crop: bool = True,
    max_duration: float = 900.0  # 15 minutes
) -> torch.Tensor:
    """Load single audio file to tensor.

    Args:
        filepath: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono
        duration: Duration in seconds to extract (None = full song)
        center_crop: If True, extract from center; if False, from beginning
        max_duration: Skip files longer than this (default: 900s = 15 minutes)
                     Prevents processing very long files (live albums, DJ sets, etc.)

    Returns:
        waveform: shape (num_samples,) if mono, (channels, num_samples) otherwise

    Raises:
        ValueError: If file duration exceeds max_duration
    """
    ...

def load_audio_batch(
    filepaths: list[Path],
    sample_rate: int = 22050,
    num_workers: Optional[int] = None
) -> list[torch.Tensor]:
    """Load multiple audio files in parallel.

    Args:
        filepaths: List of audio file paths to load
        sample_rate: Target sample rate
        num_workers: Number of parallel workers (default: 80% of CPU cores)

    Returns:
        List of audio tensors in same order as filepaths
    """
    from multiprocessing import Pool

    if num_workers is None:
        num_workers = get_default_workers()

    print(f"Loading {len(filepaths)} audio files with {num_workers} workers...")

    with Pool(num_workers) as pool:
        results = pool.map(
            lambda fp: load_audio_file(fp, sample_rate),
            filepaths
        )

    return results
```

---

### 3.4 PyTorch Dataset

**`ml_skeleton/music/dataset.py`**:
- `MusicDataset`: Dataset for encoder training. Now includes logic to filter out songs based on speech probability.
- `EmbeddingDataset`: Dataset for classifier training (embeddings → ratings).
- Lazy audio loading (load on `__getitem__`, not `__init__`).
- Audio preprocessing: resample, normalize, padding/truncating to fixed length.

**Classes**:
```python
class MusicDataset(torch.utils.data.Dataset):
    """Dataset for encoder training.

    Loads audio files on-the-fly and returns (audio_tensor, song_metadata).
    Filters out songs likely to be speech based on pre-computed probabilities.
    """

    def __init__(
        self,
        songs: list[Song],
        speech_results: Optional[Dict[str, float]] = None,
        speech_threshold: float = 0.5,
        sample_rate: int = 22050,
        duration: float = 30.0,
        cache_dir: Optional[Path] = None
    ):
        # Internally filters songs based on speech_results and speech_threshold
        ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Song]:
        # Load audio, preprocess, return with song metadata
        ...

class EmbeddingDataset(torch.utils.data.Dataset):
    """Dataset for classifier training.

    Loads pre-computed embeddings from database and returns (embedding, rating).
    """

    def __init__(
        self,
        songs: list[Song],
        embedding_store: 'EmbeddingStore'
    ):
        ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        # Load embedding from DB, return with rating
        ...
```

---

### 2.5 Embedding Storage Database

**`ml_skeleton/music/embedding_store.py`**:
- SQLite database: `embeddings.db`
- Schema supports multiple model versions per filename (composite primary key)
- Store embeddings as numpy arrays serialized with `numpy.save()` to bytes
- Batch insert/update operations for efficiency
- Thread-safe (use connection pooling or locks)

**Schema** (multi-version support):
```sql
CREATE TABLE embeddings (
    filename TEXT NOT NULL,             -- Song filename (matches Clementine DB)
    model_version TEXT NOT NULL,        -- Encoder model version/hash
    embedding BLOB NOT NULL,            -- Serialized numpy array (float32)
    embedding_dim INTEGER NOT NULL,     -- Z dimension for validation
    created_at REAL NOT NULL,           -- Unix timestamp
    updated_at REAL NOT NULL,           -- Unix timestamp
    PRIMARY KEY (filename, model_version)  -- Composite key for multi-version support
);

CREATE INDEX idx_model_version ON embeddings(model_version);
CREATE INDEX idx_updated_at ON embeddings(updated_at);
```

**Key Class**:
```python
class EmbeddingStore:
    """SQLite-backed storage for song embeddings."""

    def __init__(self, db_path: str = "./embeddings.db"):
        ...

    def store_embedding(self, filename: str, embedding: np.ndarray, model_version: str):
        """Store single embedding."""
        ...

    def store_embeddings_batch(self, data: list[tuple[str, np.ndarray]], model_version: str):
        """Batch insert/update embeddings (faster)."""
        ...

    def get_embedding(self, filename: str, model_version: str) -> Optional[np.ndarray]:
        """Retrieve embedding by filename and model version."""
        ...

    def get_embeddings_batch(self, filenames: list[str], model_version: str) -> dict[str, np.ndarray]:
        """Retrieve multiple embeddings for specific model version."""
        ...

    def get_all_versions(self, filename: str) -> list[str]:
        """Get all model versions that have embeddings for this file."""
        ...

    def has_embedding(self, filename: str, model_version: str) -> bool:
        """Check if embedding exists for specific filename and model version."""
        ...

    def count(self) -> int:
        """Total number of embeddings."""
        ...

    def clear(self):
        """Delete all embeddings."""
        ...
```

---

### 2.6 Recommendation Generation

**`ml_skeleton/music/recommendation.py`**:
- Generate top-N recommendations for unrated songs
- Export to XSPF playlist format
- Reuse code from `/git/clementine/src/output/formatter.py` (lines 179-226)
- Support filtering by predicted rating threshold
- Sort by predicted rating (descending)

**Key Functions**:
```python
@dataclass
class Recommendation:
    song: Song
    predicted_rating: float
    rank: int

def generate_recommendations(
    unrated_songs: list[Song],
    classifier: RatingClassifier,
    embedding_store: EmbeddingStore,
    top_n: int = 100,
    min_rating: float = 0.7,
    batch_size: int = 256,
    device: str = "cuda"
) -> list[Recommendation]:
    """Generate recommendations for unrated songs.

    Args:
        unrated_songs: Songs without ratings
        classifier: Trained rating classifier
        embedding_store: Pre-computed embeddings
        top_n: Number of recommendations
        min_rating: Minimum predicted rating threshold
        batch_size: Inference batch size
        device: cuda or cpu

    Returns:
        Top-N recommendations sorted by predicted rating
    """
    ...

def export_to_xspf(
    recommendations: list[Recommendation],
    output_path: Path = Path.home() / "Music/recommendations/test.xspf"
):
    """Export recommendations to XSPF playlist.

    Format matches Clementine's XSPF structure:
    - <location>: absolute file path
    - <title>, <creator>, <album>: metadata
    - <annotation>: predicted rating
    """
    ...
```

---

## 3. Training Orchestration

### 3.1 Encoder Training (Stage 1)

**`ml_skeleton/training/encoder_trainer.py`**:
- Load songs from Clementine database
- Create `MusicDataset` with audio loading
- Train user's encoder model
- Extract embeddings for ALL songs (rated + unrated)
- Store embeddings in `EmbeddingStore`
- Complies with `ml_skeleton` framework (uses `TrainingContext` and `TrainingResult`)

**Training Modes**:
1. **Self-supervised**: Contrastive learning, autoencoder, etc. (user implements loss)
2. **Supervised**: Use ratings as weak labels (user implements loss)

**Integration with Framework**:
```python
def train_encoder(ctx: TrainingContext) -> TrainingResult:
    """User-facing training function for encoder.

    Expected hyperparameters in ctx.hyperparameters:
    - encoder_lr: Learning rate
    - encoder_epochs: Number of epochs
    - batch_size: Training batch size
    - sample_rate: Audio sample rate
    - audio_duration: Seconds of audio to use
    - embedding_dim: Z dimension
    - num_workers: Multiprocessing workers (default: 80% cores)
    """

    # 1. Load songs from Clementine DB
    songs = load_all_songs()
    rated_songs = get_rated_songs(songs)  # For supervised learning

    # 2. Create dataset and dataloader
    dataset = MusicDataset(songs, sample_rate=ctx.hyperparameters["sample_rate"])
    dataloader = DataLoader(dataset, batch_size=ctx.hyperparameters["batch_size"],
                           num_workers=ctx.hyperparameters.get("num_workers", get_default_workers()))

    # 3. Get user's encoder from hyperparameters
    encoder_class = ctx.hyperparameters["encoder_class"]
    encoder = encoder_class(embedding_dim=ctx.hyperparameters["embedding_dim"])
    encoder = encoder.to(ctx.device)

    # 4. Train encoder (user implements training loop or uses helper)
    # ... training code ...

    # 5. Extract embeddings for all songs
    embedding_store = EmbeddingStore()
    extract_all_embeddings(encoder, songs, embedding_store, ctx)

    # 6. Return result
    return TrainingResult(
        primary_metric=final_loss,
        primary_metric_name="encoder_loss",
        minimize=True,
        best_model_path=ctx.checkpoint_dir / "encoder_best.pt"
    )
```

**Embedding Extraction**:
```python
def extract_all_embeddings(
    encoder: AudioEncoder,
    songs: list[Song],
    embedding_store: EmbeddingStore,
    ctx: TrainingContext,
    batch_size: int = 64
):
    """Extract embeddings for all songs and store in database.

    Uses multiprocessing for audio loading and batched GPU inference.
    Shows progress bar with tqdm.
    """
    ...
```

---

### 3.2 Classifier Training (Stage 2)

**`ml_skeleton/training/classifier_trainer.py`**:
- Load rated songs from Clementine database
- Load pre-computed embeddings from `EmbeddingStore`
- Create `EmbeddingDataset`
- Train user's classifier model
- Complies with `ml_skeleton` framework

**Integration with Framework**:
```python
def train_classifier(ctx: TrainingContext) -> TrainingResult:
    """User-facing training function for classifier.

    Expected hyperparameters:
    - classifier_lr: Learning rate
    - classifier_epochs: Number of epochs
    - batch_size: Training batch size
    - classifier_hidden_dims: List of hidden layer sizes
    - dropout: Dropout rate
    """

    # 1. Load rated songs only
    all_songs = load_all_songs()
    rated_songs = get_rated_songs(all_songs)

    # 2. Load embeddings from database
    embedding_store = EmbeddingStore()
    dataset = EmbeddingDataset(rated_songs, embedding_store)

    # 3. Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=ctx.hyperparameters["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=ctx.hyperparameters["batch_size"])

    # 4. Get user's classifier
    classifier_class = ctx.hyperparameters["classifier_class"]
    classifier = classifier_class(
        embedding_dim=embedding_store.get_embedding_dim(),
        hidden_dims=ctx.hyperparameters["classifier_hidden_dims"]
    )
    classifier = classifier.to(ctx.device)

    # 5. Train classifier (regression task: MSE loss)
    # ... training code ...

    # 6. Return result
    return TrainingResult(
        primary_metric=best_val_loss,
        primary_metric_name="val_mse",
        minimize=True,
        best_model_path=ctx.checkpoint_dir / "classifier_best.pt"
    )
```

---

### 3.3 Joint End-to-End Training

**`ml_skeleton/training/joint_trainer.py`**:
- Train encoder and classifier together in single pass
- Backpropagate through both stages
- Store embeddings to database after training completes

**Integration with Framework**:
```python
def train_joint(ctx: TrainingContext) -> TrainingResult:
    """User-facing training function for joint encoder+classifier.

    Trains both models end-to-end on rated songs only.
    Extracts embeddings for all songs after training.
    """

    # 1. Load rated songs
    all_songs = load_all_songs()
    rated_songs = get_rated_songs(all_songs)

    # 2. Create MusicDataset (audio files)
    dataset = MusicDataset(rated_songs, ...)
    train_loader, val_loader = create_dataloaders(dataset, ...)

    # 3. Get user's encoder and classifier
    encoder = ctx.hyperparameters["encoder_class"](...)
    classifier = ctx.hyperparameters["classifier_class"](...)

    # 4. Train jointly
    for epoch in range(epochs):
        for audio_batch, ratings in train_loader:
            embeddings = encoder(audio_batch)
            predictions = classifier(embeddings)
            loss = mse_loss(predictions, ratings)
            # ... backprop ...

    # 5. Extract embeddings for ALL songs
    embedding_store = EmbeddingStore()
    extract_all_embeddings(encoder, all_songs, embedding_store, ctx)

    # 6. Return result
    return TrainingResult(...)
```

---

## 4. Configuration System

### 4.1 Multiprocessing Configuration

Add to `ml_skeleton/core/config.py`:

```python
@dataclass
class ComputeConfig:
    """Multiprocessing and device configuration."""

    num_workers: Optional[int] = None  # None = auto-detect (80% cores)
    dataloader_workers: int = 4        # PyTorch DataLoader workers
    device: str = "cuda"               # cuda or cpu
    gpu_memory_limit_gb: Optional[int] = 24

    def get_num_workers(self) -> int:
        """Get number of workers (default: 80% of CPU cores)."""
        if self.num_workers is not None:
            return self.num_workers
        import multiprocessing
        total_cores = multiprocessing.cpu_count()
        return max(1, int(total_cores * 0.8))
```

Add to `ExperimentConfig`:
```python
@dataclass
class ExperimentConfig:
    # ... existing fields ...
    compute: ComputeConfig = field(default_factory=ComputeConfig)
```

---

### 4.2 Music-Specific Configuration

**`configs/music_example.yaml`**:
```yaml
name: music_recommendation
framework: pytorch
seed: 42

# Compute configuration
compute:
  num_workers: null  # null = 80% of CPU cores
  dataloader_workers: 4
  device: cuda
  gpu_memory_limit_gb: 24

# Clementine database
music:
  database_path: /home/ikaro/Music/clementine.db
  embedding_db_path: ./embeddings.db
  audio_sample_rate: 22050
  audio_duration: 30.0      # seconds (extracted from CENTER of song)
  center_crop: true         # Extract from center, not beginning
  max_audio_duration: 900.0 # Skip files longer than 15 minutes (live albums, DJ sets, etc.)

# Training mode: "encoder", "classifier", or "joint"
training_mode: encoder

# Hyperparameters for encoder training
hyperparameters:
  # Audio preprocessing
  sample_rate: 22050
  audio_duration: 30.0       # Extract 30 seconds
  center_crop: true          # From center of song
  max_audio_duration: 900.0  # Skip songs longer than 15 minutes

  # Model architecture
  embedding_dim: 512  # Z dimension
  encoder_class: "examples.music_recommendation:MyEncoder"
  classifier_class: "examples.music_recommendation:MyClassifier"

  # Encoder training
  encoder_lr: 0.001
  encoder_epochs: 50
  encoder_batch_size: 32

  # Classifier training
  classifier_lr: 0.0001
  classifier_epochs: 20
  classifier_batch_size: 256
  classifier_hidden_dims: [256, 128]
  dropout: 0.3

  # Joint training
  joint_lr: 0.001
  joint_epochs: 50
  joint_batch_size: 32

# MLflow tracking
mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: music_recommendation
  auto_start: true

# Hyperparameter tuning (optional)
tuning:
  tuner_type: optuna
  n_trials: 50
  search_space:
    parameters:
      embedding_dim:
        type: categorical
        choices: [128, 256, 512, 1024]
      encoder_lr:
        type: loguniform
        low: 0.00001
        high: 0.01
      classifier_lr:
        type: loguniform
        low: 0.000001
        high: 0.001
```

---

## 5. CLI Extensions

### 5.1 New Commands

Add to `ml_skeleton/runner/cli.py`:

```python
@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--encoder-module", required=True, help="User's encoder module:class")
@click.option("--batch-size", type=int, default=64, help="Batch size for embedding extraction")
def train_encoder(config_path: str, encoder_module: str, batch_size: int):
    """Train audio encoder and extract embeddings for all songs."""
    # Load config, run encoder training, extract embeddings
    ...

@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--classifier-module", required=True, help="User's classifier module:class")
def train_classifier(config_path: str, classifier_module: str):
    """Train rating classifier on pre-computed embeddings."""
    # Load config, run classifier training
    ...

@cli.command()
@click.argument("encoder_path", type=click.Path(exists=True))
@click.argument("classifier_path", type=click.Path(exists=True))
@click.option("--output", default="~/Music/recommendations/test.xspf", help="Output XSPF path")
@click.option("--top-n", type=int, default=100, help="Number of recommendations")
@click.option("--min-rating", type=float, default=0.7, help="Minimum predicted rating")
@click.option("--batch-size", type=int, default=256, help="Inference batch size")
def recommend(encoder_path: str, classifier_path: str, output: str, top_n: int, min_rating: float, batch_size: int):
    """Generate music recommendations and save to XSPF playlist."""

    # 1. Load models
    # 2. Load unrated songs
    # 3. Load embeddings
    # 4. Predict ratings
    # 5. Generate top-N recommendations
    # 6. Export to XSPF
    ...

@cli.command()
def music_stats():
    """Show statistics about Clementine music library."""
    # Total songs, rated songs, unrated songs, embeddings count
    ...
```

---

## 6. Example Implementation

**`examples/music_recommendation.py`**:

Provide complete working example with mock encoder and classifier:

```python
"""
Music Recommendation Example

This example demonstrates:
1. How to implement AudioEncoder and RatingClassifier protocols
2. Two training modes: separate encoder/classifier or joint end-to-end
3. How to generate recommendations

Usage:
    # Train encoder (Stage 1)
    mlskel run configs/music_example.yaml --train-fn examples.music_recommendation:train_encoder_stage

    # Train classifier (Stage 2)
    mlskel run configs/music_example.yaml --train-fn examples.music_recommendation:train_classifier_stage

    # Generate recommendations
    mlskel recommend checkpoints/encoder_best.pt checkpoints/classifier_best.pt
"""

import torch
import torch.nn as nn
from ml_skeleton import TrainingContext, TrainingResult
from ml_skeleton.music import load_all_songs, get_rated_songs, MusicDataset, EmbeddingStore

class SimpleAudioEncoder(nn.Module):
    """Example encoder: 1D CNN on raw waveform.

    User should replace with their own architecture (ResNet, Transformer, etc.)
    """

    def __init__(self, embedding_dim: int = 512, sample_rate: int = 22050):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Simple 1D CNN stack
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1024, stride=256),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Projection to embedding
        self.projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (batch_size, num_samples)
        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        # Add channel dimension: (B, num_samples) -> (B, 1, num_samples)
        x = audio.unsqueeze(1)

        # Conv layers: (B, 1, num_samples) -> (B, 256, 1)
        x = self.conv_layers(x)

        # Flatten: (B, 256, 1) -> (B, 256)
        x = x.squeeze(-1)

        # Project to embedding
        embeddings = self.projection(x)

        return embeddings

    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class SimpleRatingClassifier(nn.Module):
    """Example classifier: MLP from embeddings to rating.

    User should replace with their own architecture.
    """

    def __init__(self, embedding_dim: int, hidden_dims: list[int] = [256, 128]):
        super().__init__()

        layers = []
        prev_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        # Output layer: single rating value in [0, 1]
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, embedding_dim)
        Returns:
            ratings: (batch_size, 1), values in [0, 1]
        """
        return self.network(embeddings)


def train_encoder_stage(ctx: TrainingContext) -> TrainingResult:
    """Stage 1: Train encoder and extract embeddings."""

    from ml_skeleton.training.encoder_trainer import train_encoder_impl

    # Inject user's encoder class
    ctx.hyperparameters["encoder_class"] = SimpleAudioEncoder

    # Run encoder training
    return train_encoder_impl(ctx)


def train_classifier_stage(ctx: TrainingContext) -> TrainingResult:
    """Stage 2: Train classifier on pre-computed embeddings."""

    from ml_skeleton.training.classifier_trainer import train_classifier_impl

    # Inject user's classifier class
    ctx.hyperparameters["classifier_class"] = SimpleRatingClassifier

    # Run classifier training
    return train_classifier_impl(ctx)


def train_joint(ctx: TrainingContext) -> TrainingResult:
    """Joint end-to-end training of encoder and classifier."""

    from ml_skeleton.training.joint_trainer import train_joint_impl

    # Inject user's models
    ctx.hyperparameters["encoder_class"] = SimpleAudioEncoder
    ctx.hyperparameters["classifier_class"] = SimpleRatingClassifier

    # Run joint training
    return train_joint_impl(ctx)
```

---

## 7. Code Reuse from Clementine Repository

### Files to Adapt:

1. **Database Connection** (`/git/clementine/src/database/connection.py`):
   - `get_connection()`: SQLite connection helper - **MODIFY to use read-only mode**
   - `sync_database()`: Copy DB from Clementine snap directory - **Keep as-is (reads from source)**

2. **Song Data Model** (`/git/clementine/src/database/models.py`):
   - `Song` dataclass (rowid, title, artist, album, year, rating, filename) - **Keep as-is**
   - `load_all_songs()`: Query songs table - **Ensure uses read-only connection**

3. **XSPF Export** (`/git/clementine/src/output/formatter.py`, lines 179-226):
   - `export_to_xspf()`: Write XSPF playlist
   - Handle `file://` URI conversion
   - XML escaping for special characters

4. **Multiprocessing Patterns** (`/git/clementine/main.py`):
   - Worker pool management (80% CPU cores default)
   - Progress bars with `tqdm`
   - Batch processing patterns

### Integration Strategy:

- Copy relevant code into `ml_skeleton/music/` modules
- Adapt to match `ml_skeleton` conventions (dataclasses, type hints)
- Add dependency on `tqdm`, `unidecode` if not already present
- Keep Clementine code in separate directory for reference

---

## 8. Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing dependencies ...

    # Audio processing
    "torchaudio>=2.0.0",
    "librosa>=0.10.0",

    # Progress tracking
    "tqdm>=4.66.0",

    # Text normalization
    "unidecode>=1.3.0",

    # Audio fingerprinting (for duplicate detection)
    "pyacoustid>=1.2.0",  # Chromaprint wrapper
]
```

**Optional Dependencies** (for advanced features):
```toml
[project.optional-dependencies]
audio = [
    "soundfile>=0.12.0",  # Additional audio format support
    "resampy>=0.4.0",     # High-quality resampling
]
```

---

## 9. Workflow & User Journey

### 9.1 Two-Phase Training Workflow

```bash
# Step 1: Train encoder (extracts embeddings for all songs)
mlskel run configs/music_example.yaml --train-fn examples.music_recommendation:train_encoder_stage

# Step 2: Train classifier (on pre-computed embeddings)
mlskel run configs/music_example.yaml --train-fn examples.music_recommendation:train_classifier_stage

# Step 3: Generate recommendations
mlskel recommend checkpoints/encoder_best.pt checkpoints/classifier_best.pt \
    --output ~/Music/recommendations/test.xspf \
    --top-n 100 \
    --min-rating 0.75
```

### 9.2 Joint Training Workflow

```bash
# Single step: Train encoder + classifier together
mlskel run configs/music_example.yaml --train-fn examples.music_recommendation:train_joint

# Generate recommendations
mlskel recommend checkpoints/best_model.pt checkpoints/best_model.pt \
    --output ~/Music/recommendations/test.xspf
```

### 9.3 Hyperparameter Tuning

```bash
# Tune encoder hyperparameters
mlskel tune configs/music_example.yaml \
    --train-fn examples.music_recommendation:train_encoder_stage \
    --n-trials 50 \
    --tuner optuna

# Tune classifier hyperparameters
mlskel tune configs/music_example.yaml \
    --train-fn examples.music_recommendation:train_classifier_stage \
    --n-trials 30 \
    --tuner optuna
```

---

## 10. Testing & Verification

### 10.1 Unit Tests

Create `tests/test_music.py`:

```python
def test_clementine_db_loading():
    """Test loading songs from Clementine database."""
    songs = load_all_songs()
    assert len(songs) > 0
    assert all(song.title for song in songs)

def test_embedding_store():
    """Test embedding storage and retrieval."""
    store = EmbeddingStore(":memory:")
    embedding = np.random.randn(512).astype(np.float32)
    store.store_embedding("test.mp3", embedding, "v1")
    retrieved = store.get_embedding("test.mp3")
    np.testing.assert_array_equal(embedding, retrieved)

def test_audio_loading():
    """Test audio file loading."""
    # Use test audio file
    audio = load_audio_file("tests/data/test_audio.mp3", sample_rate=22050)
    assert audio.shape[0] > 0

def test_xspf_export():
    """Test XSPF playlist generation."""
    recommendations = [...]
    output_path = Path("/tmp/test.xspf")
    export_to_xspf(recommendations, output_path)
    assert output_path.exists()
```

### 10.2 Integration Tests

```python
def test_encoder_training_pipeline():
    """Test complete encoder training pipeline."""
    # Create mock config
    # Run encoder training
    # Verify embeddings stored in DB

def test_classifier_training_pipeline():
    """Test complete classifier training pipeline."""
    # Create mock embeddings
    # Run classifier training
    # Verify model checkpoint saved

def test_recommendation_generation():
    """Test end-to-end recommendation generation."""
    # Load trained models
    # Generate recommendations
    # Verify XSPF output format
```

---

## 11. Performance Optimizations

### 11.1 Multiprocessing Strategy

**CRITICAL: All CPU-intensive operations use multicore processing by default**

**Default Worker Count**: 80% of CPU cores
```python
def get_default_workers() -> int:
    """Get default number of workers (80% of CPU cores)."""
    cpu_count = multiprocessing.cpu_count()
    return max(1, int(cpu_count * 0.8))
```

**Example**: On RTX 5090 system with 80 cores → 64 workers by default

**Operations Using Multicore Processing**:

1. **Metadata Extraction** (Stage 1 of deduplication):
   - ~1000-5000 files/second with 64 workers
   - Uses `multiprocessing.Pool`
   - Parallelized across all songs

2. **Audio Fingerprinting** (Stage 2 of deduplication):
   - ~400-800 files/second with 64 workers
   - Chromaprint computation parallelized
   - Only runs on metadata-matched candidates

3. **Audio Loading** (for training):
   - Parallel file loading with `multiprocessing.Pool`
   - Chunk songs into batches (1000 songs per chunk)
   - Load audio in parallel, move tensors to GPU in main process

4. **Embedding Extraction** (GPU-bound):
   - Batch size: 64 (balance GPU memory and throughput)
   - CPU workers: prefetch next batch while GPU processes current
   - Use `torch.cuda.amp` for mixed precision inference (2x speedup)

**Configuration**:
```yaml
compute:
  num_workers: null  # null = auto-detect (80% cores)
  # OR specify explicitly:
  # num_workers: 64  # Use 64 workers

music:
  deduplicate: true
  # Deduplication uses same num_workers setting
```

**Manual Override**:
```python
# Use all available cores (not recommended - may starve system)
deduplicator.find_duplicates(songs, num_workers=multiprocessing.cpu_count())

# Use specific number
deduplicator.find_duplicates(songs, num_workers=32)

# Use default (80%)
deduplicator.find_duplicates(songs, num_workers=None)  # or omit parameter
```

**Database Writes** (not parallelized, but optimized):
- Batch insert embeddings (100-1000 per transaction)
- Use write-ahead logging (WAL mode) for concurrent reads
- Single writer, multiple readers pattern

### 11.2 Caching

**Audio Files** (optional):
- Cache loaded audio tensors in `./cache/audio/`
- Save as `.pt` files for faster loading
- Invalidate cache if source file modified

**Embeddings**:
- Store in SQLite for fast random access
- Index by filename for O(1) lookup
- Keep in-memory cache (LRU) for frequently accessed embeddings

### 11.3 Memory Management

**Audio Batching**:
- Limit batch size to avoid OOM
- Use `pin_memory=True` for faster GPU transfer
- Clear cache periodically: `torch.cuda.empty_cache()`

**Embedding Database**:
- Use memory-mapped mode for large databases
- Fetch embeddings in batches (not one-by-one)

---

## 12. Future Enhancements (Out of Scope for Initial Implementation)

1. **Advanced Audio Preprocessing**:
   - Mel-spectrogram extraction
   - Data augmentation (pitch shift, time stretch, noise)
   - Multi-resolution spectrograms

2. **Multi-Task Learning**:
   - Predict genre, artist, year in addition to rating
   - Use auxiliary tasks to improve embedding quality

3. **Active Learning**:
   - Recommend songs for user to rate (uncertainty sampling)
   - Iterative model improvement

4. **Similarity Search**:
   - Use encoder embeddings for "songs similar to X"
   - FAISS/Annoy for fast nearest neighbor search

5. **Playlist Generation**:
   - Generate diverse playlists (not just top-N)
   - Use embedding clustering for variety

6. **Model Explainability**:
   - Attention visualization
   - Feature importance analysis

---

## 13. Safety Checklist for Implementation

When implementing the music modules, ensure these safety measures:

### Database Access
- [ ] Clementine DB connection uses `mode=ro` (read-only)
- [ ] No `INSERT`, `UPDATE`, or `DELETE` statements on Clementine DB
- [ ] Connection opened with: `sqlite3.connect(f"file:{path}?mode=ro", uri=True)`

### Audio File Access
- [ ] All audio loading uses `torchaudio.load()` or `librosa.load()` (read-only)
- [ ] No file write operations (`open(path, 'w')`, etc.) on audio files
- [ ] No file move/rename operations (`shutil.move()`, `os.rename()`, etc.)
- [ ] No file deletion operations (`os.remove()`, `Path.unlink()`, etc.)
- [ ] No metadata modifications (no ID3 tag writing, etc.)

### Error Handling
- [ ] Corrupted audio files are skipped with warning (not fatal error)
- [ ] Missing files are logged but don't crash the pipeline
- [ ] Permission errors are caught and reported clearly
- [ ] Songs exceeding max_duration (15 min default) are skipped and logged

### Testing
- [ ] Unit tests verify no write operations on audio files
- [ ] Unit tests verify Clementine DB opened read-only
- [ ] Integration tests use temporary test files (not real music library)

**See [examples/AUDIO_FILES_README.md](examples/AUDIO_FILES_README.md) for complete safety documentation.**

---

## 14. Critical Files to Modify/Create

### New Files (Create):

| File | Purpose | Lines (Est.) |
|------|---------|--------------|
| `ml_skeleton/protocols/encoder.py` | AudioEncoder protocol | 50 |
| `ml_skeleton/protocols/classifier.py` | RatingClassifier protocol | 40 |
| `ml_skeleton/music/clementine_db.py` | Clementine DB interface | 150 |
| `ml_skeleton/music/audio_loader.py` | Multiprocessing audio loading (center crop) | 250 |
| `ml_skeleton/music/dataset.py` | PyTorch datasets | 250 |
| `ml_skeleton/music/embedding_store.py` | SQLite multi-version embedding storage | 350 |
| `ml_skeleton/music/recommendation.py` | Recommendation & XSPF export | 200 |
| `ml_skeleton/music/losses.py` | NTXentLoss (SimCLR contrastive) | 150 |
| `ml_skeleton/music/augmentation.py` | AudioAugmentor (pitch, time, noise) | 200 |
| `ml_skeleton/music/multitask.py` | Multi-task encoder (artist/album/genre/period) | 350 |
| `ml_skeleton/music/speech_detector.py` | Speech detection and filtering | 200 |
| `ml_skeleton/music/deduplication.py` | Audio fingerprinting & duplicate detection | 350 |
| `ml_skeleton/training/encoder_trainer.py` | Encoder training orchestration | 350 |
| `ml_skeleton/training/classifier_trainer.py` | Classifier training orchestration | 250 |
| `ml_skeleton/training/joint_trainer.py` | Joint training orchestration | 300 |
| `ml_skeleton/cli_extensions/music_commands.py` | New CLI commands | 250 |
| `configs/music_example.yaml` | Example configuration | 120 |
| `examples/music_recommendation.py` | Complete working example | 450 |
| `tests/test_music.py` | Unit tests | 350 |

**Total New Code**: ~4200 lines

### Files to Modify:

| File | Changes | Lines (Est.) |
|------|---------|--------------|
| `ml_skeleton/__init__.py` | Export music modules | +10 |
| `ml_skeleton/core/config.py` | Add ComputeConfig | +50 |
| `ml_skeleton/runner/cli.py` | Add music commands | +150 |
| `pyproject.toml` | Add audio dependencies | +10 |
| `README.md` | Document music recommendation use case | +100 |
| `CLAUDE.md` | Update project summary | +50 |

**Total Modified Code**: ~370 lines

---

## 14. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Create protocols (encoder, classifier)
- [ ] Implement Clementine DB interface
- [ ] Implement EmbeddingStore (SQLite)
- [ ] Implement AudioDeduplicator (chromaprint fingerprinting)
- [ ] Add ComputeConfig to core config
- [ ] Unit tests for database and storage

### Phase 2: Audio Processing (Week 1-2)
- [ ] Implement audio_loader with multiprocessing
- [ ] Implement MusicDataset and EmbeddingDataset
- [ ] Audio preprocessing pipeline
- [ ] Unit tests for audio loading

### Phase 3: Training Orchestration (Week 2)
- [ ] Implement encoder_trainer
- [ ] Implement classifier_trainer
- [ ] Implement joint_trainer
- [ ] Integration tests for training pipelines

### Phase 4: Recommendation & Export (Week 2-3)
- [ ] Implement recommendation generation
- [ ] Implement XSPF export (reuse Clementine code)
- [ ] CLI commands: train-encoder, train-classifier, recommend
- [ ] End-to-end integration test

### Phase 5: Example & Documentation (Week 3)
- [ ] Complete example with SimpleAudioEncoder/Classifier
- [ ] Configuration file examples
- [ ] Update README.md with music recommendation section
- [ ] Update CLAUDE.md project summary

### Phase 6: Optimization & Polish (Week 3-4)
- [ ] Performance profiling and optimization
- [ ] Memory usage optimization
- [ ] Error handling and logging
- [ ] Code review and refactoring

---

## 15. Open Questions & Clarifications

### Answered:
✅ Audio format: Raw audio waveforms (user handles preprocessing)
✅ Training mode: Support both two-phase and joint training
✅ Recommendation type: Predict ratings for unrated songs
✅ Embedding dimension: Configurable hyperparameter

### Confirmed Design Decisions:

1. **Audio Duration**: ✅
   - **Configurable duration from CENTER of song** (default: 30 seconds)
   - Extract from center (not beginning) to capture main musical content
   - Formula: `start_sample = (total_samples - duration_samples) // 2`

2. **Encoder Loss Function**: ✅
   - **Provide SimCLR-style contrastive loss helper**
   - Create augmented pairs of same song
   - Maximize similarity between positive pairs, minimize for negative pairs

3. **Classifier Loss Function**: ✅
   - **MSE (Mean Squared Error)** for continuous rating prediction
   - Ratings in range [0, 1]

4. **Model Versioning**: ✅
   - **Support multiple encoder versions simultaneously in DB**
   - Schema includes `model_version` column
   - Each (filename, model_version) pair stores separate embedding
   - Allows A/B testing different encoders without re-extraction

5. **Recommendation Filtering**: ✅
   - Initial version: no filtering (simple top-N by predicted rating)
   - Future enhancement: genre, artist, year filters

6. **Playlist Size**: ✅
   - Default top-N: 100 songs

7. **Maximum Song Duration**: ✅
   - **Skip songs longer than 15 minutes** (default: 900 seconds)
   - Filters out live albums, DJ sets, podcasts, audiobooks
   - Configurable via `max_audio_duration` parameter
   - Skipped songs logged for review

8. **Multi-Task Encoder Training** (Optional Enhancement): ✅
   - **Train encoder with auxiliary tasks**: artist, genre, 5-year period
   - Improves embedding quality by learning multiple levels of similarity
   - Captures artist style evolution and industry progression over time
   - **5-year period**: Rounded release year (1990, 1995, 2000, 2005, etc.)
   - **Combined loss**: Contrastive + Artist classification + Genre classification + Period classification
   - See Section 16.4 for detailed implementation

---

## 16. Additional Suggestions

### 16.1 Data Preprocessing Helpers

Provide optional helper functions for common audio preprocessing:

```python
from ml_skeleton.music.preprocessing import (
    extract_mel_spectrogram,
    normalize_audio,
    pad_or_truncate,
    apply_augmentation
)
```

### 16.2 Contrastive Learning Helper (SimCLR-style)

For self-supervised encoder training:

```python
from ml_skeleton.music.losses import NTXentLoss  # Normalized Temperature-scaled Cross Entropy
from ml_skeleton.music.augmentation import AudioAugmentor

# In user's train_encoder implementation
augmentor = AudioAugmentor(
    pitch_shift_range=(-2, 2),  # semitones
    time_stretch_range=(0.9, 1.1),
    noise_level=0.005
)
loss_fn = NTXentLoss(temperature=0.5)

# Training loop
for audio_batch in dataloader:
    # Create two augmented views of same audio
    audio_aug1 = augmentor(audio_batch)
    audio_aug2 = augmentor(audio_batch)

    # Get embeddings
    z1 = encoder(audio_aug1)
    z2 = encoder(audio_aug2)

    # Contrastive loss (maximize similarity within pairs, minimize across pairs)
    loss = loss_fn(z1, z2)
```

**New File to Create**: `ml_skeleton/music/losses.py` (NTXentLoss implementation)
**New File to Create**: `ml_skeleton/music/augmentation.py` (AudioAugmentor)

### 16.3 Model Checkpointing

Automatically save encoder/classifier checkpoints during training:

```python
# In encoder_trainer.py
ctx.tracker.log_model(
    encoder,
    artifact_path="encoder",
    framework="pytorch",
    signature=infer_signature(sample_audio, sample_embedding)
)
```

### 16.4 Multi-Task Encoder Training (Optional Enhancement)

**Concept**: Train the encoder with multiple auxiliary tasks to improve embedding quality.

**Why This Works**:
- Embeddings learn to capture multiple levels of musical similarity
- **Artist classification**: Captures artist-specific style and sound
- **Album classification**: Captures cohesive themes within albums (production, era, mood)
- **Genre classification**: Captures broad musical characteristics
- **5-year period**: Captures temporal evolution and production trends
- **Combined**: Rich, multi-dimensional representations with hierarchical similarity

**Architecture**:
```python
class MultiTaskEncoder(nn.Module):
    """Encoder with auxiliary classification heads."""

    def __init__(
        self,
        embedding_dim: int = 512,
        num_artists: int = 5000,      # Top 5000 artists
        num_albums: int = 10000,      # Top 10000 albums
        num_genres: int = 13,          # From Clementine genre categories
        num_periods: int = 10          # 1970-2025 in 5-year bins
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Main encoder (your custom architecture)
        self.encoder = YourAudioEncoder(embedding_dim)

        # Auxiliary classification heads
        self.artist_classifier = nn.Linear(embedding_dim, num_artists)
        self.album_classifier = nn.Linear(embedding_dim, num_albums)
        self.genre_classifier = nn.Linear(embedding_dim, num_genres)
        self.period_classifier = nn.Linear(embedding_dim, num_periods)

    def forward(self, audio: torch.Tensor) -> dict:
        """
        Args:
            audio: (batch, num_samples)

        Returns:
            dict with keys:
            - 'embedding': (batch, embedding_dim) - main output
            - 'artist_logits': (batch, num_artists)
            - 'album_logits': (batch, num_albums)
            - 'genre_logits': (batch, num_genres)
            - 'period_logits': (batch, num_periods)
        """
        # Extract embedding
        embedding = self.encoder(audio)

        # Auxiliary predictions (used during training only)
        artist_logits = self.artist_classifier(embedding)
        album_logits = self.album_classifier(embedding)
        genre_logits = self.genre_classifier(embedding)
        period_logits = self.period_classifier(embedding)

        return {
            'embedding': embedding,
            'artist_logits': artist_logits,
            'album_logits': album_logits,
            'genre_logits': genre_logits,
            'period_logits': period_logits
        }

    def get_embedding_dim(self) -> int:
        return self.embedding_dim
```

**Combined Loss Function**:
```python
class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task encoder training."""

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        artist_weight: float = 0.3,
        album_weight: float = 0.3,
        genre_weight: float = 0.2,
        period_weight: float = 0.1,
        temperature: float = 0.5
    ):
        super().__init__()
        self.contrastive_loss = NTXentLoss(temperature)
        self.ce_loss = nn.CrossEntropyLoss()

        self.contrastive_weight = contrastive_weight
        self.artist_weight = artist_weight
        self.album_weight = album_weight
        self.genre_weight = genre_weight
        self.period_weight = period_weight

    def forward(
        self,
        outputs1: dict,  # First augmented view
        outputs2: dict,  # Second augmented view
        artist_labels: torch.Tensor,      # -1 = missing/unknown
        album_labels: list[list[int]],    # list of album indices per song (handles multiple albums)
        genre_labels: torch.Tensor,       # -1 = missing/unknown
        period_labels: torch.Tensor       # -1 = missing/unknown
    ) -> dict:
        """
        Args:
            outputs1: First augmented view (dict with 'embedding', 'artist_logits', etc.)
            outputs2: Second augmented view (dict with 'embedding', 'artist_logits', etc.)
            artist_labels: (batch_size,) tensor, -1 for missing
            album_labels: List of lists, where album_labels[i] is list of album indices for song i
                         Empty list [] means no valid albums for that song
            genre_labels: (batch_size,) tensor, -1 for missing
            period_labels: (batch_size,) tensor, -1 for missing

        Returns:
            dict with keys:
            - 'total_loss': Combined weighted loss
            - 'contrastive_loss': Self-supervised contrastive
            - 'artist_loss': Artist classification (masked for missing labels)
            - 'album_loss': Album classification (averaged over all albums per song)
            - 'genre_loss': Genre classification (masked for missing labels)
            - 'period_loss': Period classification (masked for missing labels)
            - 'album_valid_pct': Percentage of songs with at least one album
            - 'album_avg_count': Average number of albums per song (for songs with albums)

        Note:
            - Artist/genre/period labels with value -1 are considered missing and excluded from loss.
            - Album labels: each song can have 0+ albums. Loss computed as average over all albums.
        """
        # Contrastive loss (self-supervised - always computed)
        contrastive = self.contrastive_loss(
            outputs1['embedding'],
            outputs2['embedding']
        )

        # Classification losses with masking for missing labels
        # Artist loss (only for known artists)
        artist_mask = artist_labels >= 0
        if artist_mask.any():
            artist_loss = self.ce_loss(
                outputs1['artist_logits'][artist_mask],
                artist_labels[artist_mask]
            )
        else:
            artist_loss = torch.tensor(0.0, device=outputs1['artist_logits'].device)

        # Album loss (handles multiple albums per song)
        # album_labels: list[list[int]] - each song has list of album indices
        album_loss = torch.tensor(0.0, device=outputs1['album_logits'].device)
        album_samples_with_labels = 0

        for i, album_indices in enumerate(album_labels):
            if album_indices:  # Song has at least one valid album
                # Compute loss for each album this song appears on
                logits = outputs1['album_logits'][i].unsqueeze(0)  # (1, num_albums)

                sample_loss = 0.0
                for album_idx in album_indices:
                    target = torch.tensor([album_idx], device=logits.device)
                    sample_loss += self.ce_loss(logits, target)

                # Average loss over all albums for this song
                album_loss += sample_loss / len(album_indices)
                album_samples_with_labels += 1

        # Average over all samples with album labels
        if album_samples_with_labels > 0:
            album_loss = album_loss / album_samples_with_labels

        # Genre loss (only for known genres)
        genre_mask = genre_labels >= 0
        if genre_mask.any():
            genre_loss = self.ce_loss(
                outputs1['genre_logits'][genre_mask],
                genre_labels[genre_mask]
            )
        else:
            genre_loss = torch.tensor(0.0, device=outputs1['genre_logits'].device)

        # Period loss (only for known years)
        period_mask = period_labels >= 0
        if period_mask.any():
            period_loss = self.ce_loss(
                outputs1['period_logits'][period_mask],
                period_labels[period_mask]
            )
        else:
            period_loss = torch.tensor(0.0, device=outputs1['period_logits'].device)

        # Weighted combination
        total_loss = (
            self.contrastive_weight * contrastive +
            self.artist_weight * artist_loss +
            self.album_weight * album_loss +
            self.genre_weight * genre_loss +
            self.period_weight * period_loss
        )

        return {
            'total_loss': total_loss,
            'contrastive_loss': contrastive.item(),
            'artist_loss': artist_loss.item(),
            'album_loss': album_loss.item(),
            'genre_loss': genre_loss.item(),
            'period_loss': period_loss.item(),
            # Return mask statistics for monitoring
            'artist_valid_pct': artist_mask.float().mean().item() * 100,
            'album_valid_pct': (album_samples_with_labels / len(album_labels)) * 100 if album_labels else 0.0,
            'album_avg_count': sum(len(labels) for labels in album_labels) / max(album_samples_with_labels, 1),
            'genre_valid_pct': genre_mask.float().mean().item() * 100,
            'period_valid_pct': period_mask.float().mean().item() * 100
        }
```

**5-Year Period Mapping**:
```python
def year_to_period(year: int) -> int:
    """Convert year to 5-year period bin.

    Args:
        year: Release year (e.g., 1987, 2003)

    Returns:
        Period index (0-based)

    Example:
        1987 -> 1985 -> bin 3 (1970=0, 1975=1, 1980=2, 1985=3, ...)
        2003 -> 2000 -> bin 6
    """
    if year < 1970:
        return 0  # Pre-1970

    # Round down to nearest 5-year period
    period_year = (year // 5) * 5

    # Map to index: 1970=0, 1975=1, 1980=2, etc.
    period_idx = (period_year - 1970) // 5

    # Cap at max period (2025 = index 11)
    return min(period_idx, 11)

# Example mappings:
# 1970-1974 -> 0
# 1975-1979 -> 1
# 1980-1984 -> 2
# 1985-1989 -> 3
# 1990-1994 -> 4
# 1995-1999 -> 5
# 2000-2004 -> 6
# 2005-2009 -> 7
# 2010-2014 -> 8
# 2015-2019 -> 9
# 2020-2024 -> 10
# 2025+ -> 11
```

**Genre Mapping** (from Clementine - 13 categories):
```python
# Reuse exact mapping from /git/clementine/src/config.py
GENRE_CATEGORIES = {
    'rock': 0,        # rock, metal, punk, grunge, alternative, indie
    'pop': 1,         # pop, dance, disco, synth, new wave
    'electronic': 2,  # electronic, techno, house, trance, edm, dubstep
    'hiphop': 3,      # hip hop, rap, r&b, urban, trap
    'jazz': 4,        # jazz, swing, bebop, fusion
    'blues': 5,       # blues, soul, motown, funk
    'classical': 6,   # classical, orchestra, symphony, opera
    'country': 7,     # country, bluegrass, americana, folk
    'latin': 8,       # latin, salsa, reggaeton, bossa nova, samba
    'reggae': 9,      # reggae, ska, dub, dancehall
    'world': 10,      # world, african, celtic, indian, asian
    'soundtrack': 11, # soundtrack, film, movie, score, game
    'instrumental': 12 # instrumental, acoustic, new age
}

def map_genre_to_category(genre: str) -> int | None:
    """Map Clementine genre string to category index.

    Returns None if genre is missing or cannot be mapped.
    This allows loss masking for songs without genre info.
    """
    if not genre:
        return None  # Missing genre - will be masked in loss

    # Use exact mapping from Clementine (see config.py lines 72-146)
    from clementine.src.config import get_genre_category

    category = get_genre_category(genre)
    if category is None:
        return None  # Unknown genre - will be masked

    return GENRE_CATEGORIES[category]
```

**Album Mapping** (top K albums, handles multiple albums per song):
```python
def build_song_to_albums_mapping(songs: list[Song]) -> dict[str, list[str]]:
    """Build mapping from audio filename to all albums it appears on.

    Important: A song (unique audio file) may appear on multiple albums:
    - Original studio album
    - Compilation albums ("Best of Country 2020")
    - Artist's "Greatest Hits"
    - Soundtrack albums

    After deduplication, we have one canonical Song entry per unique audio file,
    but in the original Clementine DB, the same filename may appear in multiple
    rows with different album values.

    Args:
        songs: List of ALL songs from Clementine DB (before deduplication)

    Returns:
        dict mapping filename -> list of album keys ("Artist - Album")

    Example:
        {
            "/Music/song.mp3": [
                "John Smith - Studio Album 2020",
                "Various Artists - Country Hits 2020"
            ]
        }
    """
    from collections import defaultdict

    filename_to_albums = defaultdict(set)

    for song in songs:
        if song.artist and song.album and song.filename:
            album_key = f"{song.artist} - {song.album}"
            filename_to_albums[song.filename].add(album_key)

    # Convert sets to sorted lists for deterministic ordering
    return {
        filename: sorted(albums)
        for filename, albums in filename_to_albums.items()
    }

def build_album_mapping(
    songs: list[Song],
    top_k: int = 10000,
    min_songs_per_album: int = 2
) -> dict[str, int]:
    """Build album-to-index mapping for top K albums.

    Args:
        songs: List of songs from Clementine DB
        top_k: Number of most common albums to include
        min_songs_per_album: Minimum songs required per album (default: 2, filters singles)

    Returns:
        dict mapping album key ("Artist - Album") -> index (0-based)
        Albums not in top K are excluded (will use -1 in dataset)

    Notes:
        - Album key format: "Artist - Album" (handles multiple artists with same album name)
        - Only albums with min_songs_per_album+ songs included
        - Sorted by frequency (most common albums first)
    """
    from collections import Counter

    # Count album occurrences (use "Artist - Album" as key)
    album_counts = Counter()
    for song in songs:
        if song.artist and song.album:  # Only count if both fields present
            album_key = f"{song.artist} - {song.album}"
            album_counts[album_key] += 1

    # Filter: only albums with min_songs_per_album+ songs
    album_counts = {
        album: count
        for album, count in album_counts.items()
        if count >= min_songs_per_album
    }

    # Get top K most common albums
    top_albums = [album for album, _ in album_counts.most_common(top_k)]

    # Build mapping
    album_to_idx = {album: idx for idx, album in enumerate(top_albums)}

    print(f"Album mapping: {len(album_to_idx)} albums (top {top_k} with {min_songs_per_album}+ songs)")

    return album_to_idx

def get_album_labels(
    song: Song,
    album_to_idx: dict[str, int],
    filename_to_albums: dict[str, list[str]]
) -> list[int]:
    """Get all album labels for a song (handles multiple albums).

    Args:
        song: Song from Clementine DB (after deduplication)
        album_to_idx: Album-to-index mapping from build_album_mapping()
        filename_to_albums: Filename-to-albums mapping from build_song_to_albums_mapping()

    Returns:
        List of album indices (0-based) for all albums this song appears on.
        Empty list if no valid albums found.

    Examples:
        - Song on 1 album: [42]
        - Song on 3 albums: [42, 105, 201]
        - Song on no mapped albums: []
    """
    if song.filename not in filename_to_albums:
        return []  # No album info

    album_keys = filename_to_albums[song.filename]

    # Map to indices (only include albums in top K)
    album_indices = []
    for album_key in album_keys:
        idx = album_to_idx.get(album_key)
        if idx is not None:  # Album in top K
            album_indices.append(idx)

    return album_indices

def get_primary_album_label(
    song: Song,
    album_to_idx: dict[str, int],
    filename_to_albums: dict[str, list[str]],
    prefer_artist_albums: bool = True
) -> int:
    """Get single "primary" album label for a song (for single-label training).

    When a song appears on multiple albums, choose one using heuristics:
    1. Prefer albums where artist matches song artist (original album vs compilation)
    2. If multiple artist albums, choose the one with most songs (main album vs EP)
    3. If no artist albums, choose any album with most songs
    4. Fallback: first album alphabetically

    Args:
        song: Song from Clementine DB
        album_to_idx: Album-to-index mapping
        filename_to_albums: Filename-to-albums mapping
        prefer_artist_albums: If True, prioritize albums by same artist

    Returns:
        Album index (0-based), or -1 if no valid albums found
    """
    album_labels = get_album_labels(song, album_to_idx, filename_to_albums)

    if not album_labels:
        return -1  # No valid albums

    if len(album_labels) == 1:
        return album_labels[0]  # Only one album, use it

    # Multiple albums - apply heuristics
    if prefer_artist_albums and song.artist:
        # Get album keys for this song
        album_keys = filename_to_albums.get(song.filename, [])

        # Filter to albums by same artist
        artist_album_keys = [
            key for key in album_keys
            if key.startswith(f"{song.artist} - ")
        ]

        if artist_album_keys:
            # Choose artist's album (first one alphabetically)
            chosen_key = sorted(artist_album_keys)[0]
            return album_to_idx.get(chosen_key, -1)

    # Fallback: return first album (arbitrary but deterministic)
    return album_labels[0]
```

**Dataset Implementation** (handling multiple albums):
```python
class MultiTaskMusicDataset(Dataset):
    """Dataset for multi-task encoder training with multiple albums per song."""

    def __init__(
        self,
        songs: list[Song],
        artist_to_idx: dict[str, int],
        album_to_idx: dict[str, int],
        filename_to_albums: dict[str, list[str]],
        sample_rate: int = 22050,
        audio_duration: float = 30.0
    ):
        self.songs = songs
        self.artist_to_idx = artist_to_idx
        self.album_to_idx = album_to_idx
        self.filename_to_albums = filename_to_albums
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration

    def __len__(self) -> int:
        return len(self.songs)

    def __getitem__(self, idx: int) -> dict:
        song = self.songs[idx]

        # Load audio
        audio = load_audio_file(
            song.filename,
            sample_rate=self.sample_rate,
            duration=self.audio_duration,
            center_crop=True
        )

        # Get artist label
        artist_label = self.artist_to_idx.get(song.artist, -1)

        # Get album labels (multiple albums per song)
        album_labels = get_album_labels(
            song,
            self.album_to_idx,
            self.filename_to_albums
        )

        # Get genre label
        genre_label = map_genre_to_category(song.genre)
        if genre_label is None:
            genre_label = -1

        # Get period label
        period_label = year_to_period(song.year) if song.year > 0 else -1

        return {
            'audio': audio,
            'artist': artist_label,
            'albums': album_labels,  # list[int] - multiple albums
            'genre': genre_label,
            'period': period_label,
            'filename': song.filename
        }

def collate_multitask_batch(batch: list[dict]) -> dict:
    """Custom collate function for batching multi-task data.

    Handles variable-length album lists per song.
    """
    return {
        'audio': torch.stack([item['audio'] for item in batch]),
        'artist': torch.tensor([item['artist'] for item in batch], dtype=torch.long),
        'albums': [item['albums'] for item in batch],  # list[list[int]] - keep as list
        'genre': torch.tensor([item['genre'] for item in batch], dtype=torch.long),
        'period': torch.tensor([item['period'] for item in batch], dtype=torch.long),
        'filename': [item['filename'] for item in batch]
    }

# Usage in training
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_multitask_batch,  # Custom collate for album lists
    num_workers=4
)
```

**Training Loop Example**:
```python
def train_multitask_encoder(ctx: TrainingContext) -> TrainingResult:
    """Train encoder with multi-task learning."""

    # Load songs with metadata
    all_songs = load_all_songs()  # Before deduplication

    # Build song-to-albums mapping (handles multiple albums per song)
    filename_to_albums = build_song_to_albums_mapping(all_songs)

    # Deduplicate songs
    songs = deduplicate_songs(all_songs)  # After deduplication

    # Build label mappings
    artist_to_idx = build_artist_mapping(songs, top_k=5000)
    album_to_idx = build_album_mapping(all_songs, top_k=10000)  # Use all_songs to count properly

    # Create dataset with labels
    dataset = MultiTaskMusicDataset(
        songs,
        artist_to_idx=artist_to_idx,
        album_to_idx=album_to_idx,
        filename_to_albums=filename_to_albums,
        sample_rate=ctx.hyperparameters['sample_rate']
    )

    # Create model and loss
    encoder = MultiTaskEncoder(
        embedding_dim=ctx.hyperparameters['embedding_dim'],
        num_artists=len(artist_to_idx),
        num_albums=len(album_to_idx),
        num_genres=13,
        num_periods=12
    ).to(ctx.device)

    loss_fn = MultiTaskLoss(
        contrastive_weight=1.0,
        artist_weight=0.3,
        album_weight=0.3,
        genre_weight=0.2,
        period_weight=0.1
    )

    optimizer = torch.optim.Adam(encoder.parameters(), lr=ctx.hyperparameters['encoder_lr'])

    # Training loop
    for epoch in range(ctx.hyperparameters['encoder_epochs']):
        for batch in dataloader:
            audio = batch['audio'].to(ctx.device)
            artist_labels = batch['artist'].to(ctx.device)
            album_labels = batch['albums']  # list[list[int]] - NO .to(device), already indices
            genre_labels = batch['genre'].to(ctx.device)
            period_labels = batch['period'].to(ctx.device)

            # Create two augmented views
            audio_aug1 = augmentor(audio)
            audio_aug2 = augmentor(audio)

            # Forward pass
            outputs1 = encoder(audio_aug1)
            outputs2 = encoder(audio_aug2)

            # Compute loss
            losses = loss_fn(outputs1, outputs2, artist_labels, album_labels, genre_labels, period_labels)

            # Backward pass
            optimizer.zero_grad()
            losses['total_loss'].backward()
            optimizer.step()

            # Log metrics
            ctx.tracker.log_metrics({
                'train/total_loss': losses['total_loss'].item(),
                'train/contrastive_loss': losses['contrastive_loss'],
                'train/artist_loss': losses['artist_loss'],
                'train/album_loss': losses['album_loss'],
                'train/genre_loss': losses['genre_loss'],
                'train/period_loss': losses['period_loss']
            }, step=epoch)

    # After training, ONLY use the encoder part (discard classification heads)
    # The embedding space now captures artist, album, genre, and period similarities
    return TrainingResult(...)
```

**Benefits**:
1. **Richer Embeddings**: Captures multiple levels of musical similarity
2. **Better Generalization**: Auxiliary tasks act as regularization
3. **Album Cohesion**: Captures thematic similarity within albums (production era, mood, concept)
4. **Artist Evolution**: 5-year periods capture how artists' styles change over time
5. **Industry Trends**: Captures production quality improvements, genre evolution
6. **Better Recommendations**: More nuanced similarity beyond just acoustic features

**Configuration**:
```yaml
hyperparameters:
  # Multi-task training
  use_multitask: true
  contrastive_weight: 1.0
  artist_weight: 0.3
  album_weight: 0.3       # Album similarity loss
  genre_weight: 0.2
  period_weight: 0.1
  top_k_artists: 5000     # Only use top K artists (rest = "other")
  top_k_albums: 10000     # Only use top K albums (rest = "other")
```

**Implementation Notes**:
- **Loss Masking**: Songs with missing artist, album, genre, or year are excluded from that task's loss
  - Missing labels encoded as -1 (for artist/genre/period) or [] (for albums)
  - Only contrastive loss is computed for songs with all missing metadata
  - Mask statistics logged to MLflow (% of batch with valid labels)
- **Artist Mapping**:
  - Only top K artists (e.g., 5000) included to keep model size reasonable
  - Rare artists (< 5 songs in dataset) → label -1 (masked)
  - Unknown artists → label -1 (masked)
- **Album Mapping** (handles multiple albums per song):
  - **CRITICAL**: A song can appear on multiple albums (original album + compilations/greatest hits)
  - Album key format: "Artist - Album" (handles multiple artists with same album name)
  - **Build song-to-albums mapping** before deduplication to capture all album appearances
  - Loss computed as **average over all valid albums** for each song
  - Dataset returns `album_labels: list[list[int]]` where each song has list of album indices
  - Example: `album_labels = [[42, 105], [201], [], [7]]` means:
    - Song 0 appears on albums 42 and 105
    - Song 1 appears on album 201
    - Song 2 has no valid albums
    - Song 3 appears on album 7
  - Only albums in top K (default: 10000) included
  - Only albums with 2+ songs included (filters out singles/rarities)
  - Missing artist or album field → empty list [] (masked)
- **Genre Mapping**:
  - Uses Clementine's exact 13-category mapping (`/git/clementine/src/config.py`)
  - Unmapped genres → label -1 (masked)
  - Empty genre field → label -1 (masked)
- **Period Mapping**:
  - Invalid years (year <= 0) → label -1 (masked)
  - Years outside range → label -1 (masked)
- Classification heads discarded after training; only embeddings used for recommendations
- Can toggle multi-task on/off via config for comparison

**Expected Improvements**:
- Better clustering by artist, album, and genre in embedding space
- Album cohesion: songs from same album will have more similar embeddings
- Captures temporal progression (60s vs 90s vs 2010s sound)
- More meaningful nearest neighbors (similar albums, similar artists)
- Better cold-start (songs from known artists/albums/genres/periods)
- Album-aware recommendations: can recommend full albums if user likes one song

**New File to Create**: `ml_skeleton/music/multitask.py` (MultiTaskEncoder, MultiTaskLoss, label mapping utilities)

### 16.4 Embeddings Visualization

Generate t-SNE visualization of embeddings:

```bash
mlskel visualize-embeddings --embeddings-db embeddings.db --output embeddings_tsne.png
```

### 16.5 Cold Start Problem

For new users with no rated songs:
- Use pre-trained encoder (transfer learning)
- Popularity-based recommendations initially
- Interactive rating UI to bootstrap

---

## 17. Success Criteria

The implementation is successful if:

1. ✅ **Compliance**: Follows `ml_skeleton` design patterns (protocols, config, MLflow)
2. ✅ **Injection Points**: Clear interfaces for user's encoder and classifier code
3. ✅ **Database**: SQLite storage for embeddings with efficient batch operations
4. ✅ **Multiprocessing**: Configurable worker count (default 80% CPU cores)
5. ✅ **Two Training Modes**: Support both two-phase and joint training
6. ✅ **XSPF Export**: Generate valid Clementine-compatible playlists
7. ✅ **Performance**: Process 60K songs in reasonable time (<2 hours on RTX 5090)
8. ✅ **Documentation**: Complete example and clear user guide
9. ✅ **Testing**: Unit tests for all core components
10. ✅ **Reusability**: Code reused from Clementine repo where applicable

---

## 18. Estimated Timeline

**Total Duration**: 3-4 weeks (assuming 20-30 hours/week)

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Infrastructure | 3-5 days | Protocols, DB interface, embedding store |
| Phase 2: Audio Processing | 3-5 days | Audio loader, datasets |
| Phase 3: Training | 4-6 days | Three trainer implementations |
| Phase 4: Recommendations | 3-4 days | Recommendation engine, XSPF export |
| Phase 5: Example & Docs | 2-3 days | Complete example, documentation |
| Phase 6: Optimization | 3-5 days | Performance tuning, polish |

**Parallelization Opportunities**:
- Phases 1 & 2 can partially overlap (different modules)
- Phases 4 & 5 can partially overlap (documentation while coding)

---

## Next Steps

1. **Review this plan** and provide feedback
2. **Answer remaining open questions** (section 15)
3. **Approve architecture** (sections 2-3)
4. **Confirm file structure** (section 1)
5. **Start Phase 1 implementation** (core infrastructure)

---

## Appendix A: Example XSPF Output

```xml
<?xml version="1.0" encoding="UTF-8"?>
<playlist version="1" xmlns="http://xspf.org/ns/0/">
  <trackList>
    <track>
      <location>/home/ikaro/Music/Artist/Album/01 - Song Title.mp3</location>
      <title>Song Title</title>
      <creator>Artist Name</creator>
      <album>Album Name</album>
      <annotation>Predicted rating: 0.89</annotation>
    </track>
    <!-- ... more tracks ... -->
  </trackList>
</playlist>
```

---

## Appendix B: Embedding Database Schema SQL

```sql
-- Create embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    filename TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    model_version TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- Create indices
CREATE INDEX IF NOT EXISTS idx_updated_at ON embeddings(updated_at);
CREATE INDEX IF NOT EXISTS idx_model_version ON embeddings(model_version);

-- Create metadata table (optional, for model info)
CREATE TABLE IF NOT EXISTS model_metadata (
    model_version TEXT PRIMARY KEY,
    encoder_class TEXT,
    hyperparameters TEXT,  -- JSON
    created_at REAL NOT NULL
);
```

---

## Appendix C: Multiprocessing Worker Calculation

```python
import multiprocessing
import os

def get_default_workers(target_percentage: float = 0.8) -> int:
    """Get number of workers based on CPU cores.

    Args:
        target_percentage: Target CPU usage (default: 80%)

    Returns:
        Number of workers (at least 1)
    """
    total_cores = multiprocessing.cpu_count()

    # Check for CPU affinity restrictions (e.g., in containers)
    try:
        affinity_cores = len(os.sched_getaffinity(0))
        total_cores = min(total_cores, affinity_cores)
    except AttributeError:
        pass  # os.sched_getaffinity not available on all platforms

    workers = max(1, int(total_cores * target_percentage))
    return workers

# Example usage:
# num_workers = get_default_workers()  # Returns 80% of cores
# num_workers = get_default_workers(0.5)  # Returns 50% of cores
```

---

**END OF PLAN**
