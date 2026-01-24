"""Chunked waveform cache builder for MoCo training.

Caches 4 evenly-spaced 30s chunks per song as .npy files for fast loading.
This enables:
- 15x faster loading vs MP3 decoding
- Fresh augmentations each epoch (CQT computed on-the-fly)
- Capturing song structure diversity (intro, verse, chorus, outro)

Cache structure:
    ./cache/chunks/{song_id}_0.npy  # First chunk (0-30s region)
    ./cache/chunks/{song_id}_1.npy  # Second chunk
    ./cache/chunks/{song_id}_2.npy  # Third chunk
    ./cache/chunks/{song_id}_3.npy  # Fourth chunk (end region)

Each .npy file contains float32 PCM at 16kHz, shape (480000,) = 30s.
Total storage: ~30GB for 60K songs (4 chunks × 30s × 16kHz × 4 bytes).
"""

import os
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import multiprocessing
from functools import partial
from tqdm import tqdm

from ml_skeleton.music.clementine_db import Song


# Default cache parameters
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_DURATION = 30.0  # seconds per chunk
DEFAULT_NUM_CHUNKS = 4
DEFAULT_CACHE_DIR = "./cache/chunks"


@dataclass
class ChunkInfo:
    """Information about a cached chunk."""
    song_id: int
    chunk_idx: int
    cache_path: Path
    exists: bool


def get_chunk_cache_path(
    song_id: int,
    chunk_idx: int,
    cache_dir: str = DEFAULT_CACHE_DIR
) -> Path:
    """Get the cache path for a specific chunk.

    Args:
        song_id: Song row ID from database
        chunk_idx: Chunk index (0-3)
        cache_dir: Base cache directory

    Returns:
        Path to the .npy cache file
    """
    return Path(cache_dir) / f"{song_id}_{chunk_idx}.npy"


def compute_chunk_offsets(
    file_duration: float,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    chunk_duration: float = DEFAULT_CHUNK_DURATION
) -> List[float]:
    """Compute evenly-spaced chunk start offsets.

    For a 4-minute song with 4 chunks of 30s each:
    - Total extractable: 240s - 30s = 210s range
    - Spacing: 210s / 3 = 70s between chunk starts
    - Offsets: [0, 70, 140, 210] seconds

    For short songs (< chunk_duration * num_chunks), chunks will overlap.

    Args:
        file_duration: Total duration of audio file in seconds
        num_chunks: Number of chunks to extract
        chunk_duration: Duration of each chunk in seconds

    Returns:
        List of start offsets in seconds
    """
    if file_duration <= chunk_duration:
        # Very short file - all chunks start at 0
        return [0.0] * num_chunks

    # Calculate spacing between chunk start positions
    # Last chunk ends at (last_offset + chunk_duration) = file_duration
    # So last_offset = file_duration - chunk_duration
    max_offset = file_duration - chunk_duration

    if num_chunks == 1:
        return [max_offset / 2]  # Center for single chunk

    # Evenly space chunks
    spacing = max_offset / (num_chunks - 1)
    offsets = [i * spacing for i in range(num_chunks)]

    return offsets


def extract_chunk(
    filepath: str,
    offset: float,
    duration: float = DEFAULT_CHUNK_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    normalize: bool = True
) -> Optional[np.ndarray]:
    """Extract a single chunk from an audio file.

    Args:
        filepath: Path to audio file
        offset: Start offset in seconds
        duration: Duration to extract in seconds
        sample_rate: Target sample rate
        normalize: Apply z-normalization

    Returns:
        Numpy array of shape (num_samples,) or None if extraction fails
    """
    try:
        # Get file info
        info = torchaudio.info(filepath)
        file_sr = info.sample_rate
        file_frames = info.num_frames

        # Calculate frame positions
        start_frame = int(offset * file_sr)
        num_frames = int(duration * file_sr)

        # Ensure we don't exceed file bounds
        start_frame = max(0, min(start_frame, file_frames - num_frames))
        num_frames = min(num_frames, file_frames - start_frame)

        # Load chunk
        waveform, sr = torchaudio.load(
            filepath,
            frame_offset=start_frame,
            num_frames=num_frames
        )

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        # Pad to exact duration if needed
        target_length = int(duration * sample_rate)
        if waveform.shape[0] < target_length:
            padding = target_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[0] > target_length:
            waveform = waveform[:target_length]

        # Z-normalization
        if normalize:
            mean = waveform.mean()
            std = waveform.std()
            if std > 1e-8:
                waveform = (waveform - mean) / std
            else:
                waveform = waveform - mean

        return waveform.numpy().astype(np.float32)

    except Exception as e:
        return None


def cache_song_chunks(
    song: Song,
    cache_dir: str = DEFAULT_CACHE_DIR,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_duration: float = 900.0,
    overwrite: bool = False
) -> Tuple[int, int]:
    """Cache all chunks for a single song.

    Args:
        song: Song object with filepath
        cache_dir: Base cache directory
        num_chunks: Number of chunks per song
        chunk_duration: Duration of each chunk in seconds
        sample_rate: Target sample rate
        max_duration: Skip files longer than this
        overwrite: Overwrite existing cache files

    Returns:
        Tuple of (num_cached, num_skipped)
    """
    filepath = str(song.filepath)
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    # Check if all chunks already cached
    if not overwrite:
        all_cached = all(
            get_chunk_cache_path(song.rowid, i, cache_dir).exists()
            for i in range(num_chunks)
        )
        if all_cached:
            return (0, num_chunks)  # All skipped (already cached)

    # Get file duration
    try:
        info = torchaudio.info(filepath)
        file_duration = info.num_frames / info.sample_rate

        # Skip very long files
        if file_duration > max_duration:
            return (0, 0)

    except Exception:
        return (0, 0)

    # Compute chunk offsets
    offsets = compute_chunk_offsets(file_duration, num_chunks, chunk_duration)

    num_cached = 0
    num_skipped = 0

    for chunk_idx, offset in enumerate(offsets):
        cache_path = get_chunk_cache_path(song.rowid, chunk_idx, cache_dir)

        # Skip if exists and not overwriting
        if cache_path.exists() and not overwrite:
            num_skipped += 1
            continue

        # Extract and cache chunk
        chunk_data = extract_chunk(
            filepath, offset, chunk_duration, sample_rate
        )

        if chunk_data is not None:
            np.save(cache_path, chunk_data)
            num_cached += 1

    return (num_cached, num_skipped)


def _cache_song_worker(
    song: Song,
    cache_dir: str,
    num_chunks: int,
    chunk_duration: float,
    sample_rate: int,
    max_duration: float,
    overwrite: bool
) -> Tuple[int, int, int]:
    """Worker function for parallel caching.

    Returns:
        Tuple of (song_id, num_cached, num_skipped)
    """
    num_cached, num_skipped = cache_song_chunks(
        song, cache_dir, num_chunks, chunk_duration,
        sample_rate, max_duration, overwrite
    )
    return (song.rowid, num_cached, num_skipped)


def build_chunk_cache(
    songs: List[Song],
    cache_dir: str = DEFAULT_CACHE_DIR,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_duration: float = 900.0,
    num_workers: Optional[int] = None,
    overwrite: bool = False,
    show_progress: bool = True
) -> dict:
    """Build chunk cache for all songs in parallel.

    Args:
        songs: List of Song objects
        cache_dir: Base cache directory
        num_chunks: Number of chunks per song
        chunk_duration: Duration of each chunk
        sample_rate: Target sample rate
        max_duration: Skip files longer than this
        num_workers: Number of parallel workers (default: 80% CPU)
        overwrite: Overwrite existing cache files
        show_progress: Show progress bar

    Returns:
        Dictionary with cache statistics:
        - total_songs: Number of songs processed
        - total_chunks_cached: New chunks written
        - total_chunks_skipped: Existing chunks skipped
        - cache_dir: Cache directory path
        - estimated_size_gb: Estimated cache size in GB
    """
    if num_workers is None:
        num_workers = max(1, int(multiprocessing.cpu_count() * 0.8))

    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Create worker function with fixed parameters
    worker_fn = partial(
        _cache_song_worker,
        cache_dir=cache_dir,
        num_chunks=num_chunks,
        chunk_duration=chunk_duration,
        sample_rate=sample_rate,
        max_duration=max_duration,
        overwrite=overwrite
    )

    total_cached = 0
    total_skipped = 0

    # Process songs in parallel
    if show_progress:
        print(f"Building chunk cache: {len(songs)} songs, {num_chunks} chunks/song")
        print(f"Cache directory: {cache_dir}")

    with multiprocessing.Pool(num_workers) as pool:
        iterator = pool.imap_unordered(worker_fn, songs)
        if show_progress:
            iterator = tqdm(iterator, total=len(songs), desc="Caching")

        for song_id, num_cached, num_skipped in iterator:
            total_cached += num_cached
            total_skipped += num_skipped

    # Calculate estimated size
    chunk_size_bytes = int(chunk_duration * sample_rate * 4)  # float32
    estimated_size_gb = (total_cached * chunk_size_bytes) / (1024 ** 3)

    stats = {
        "total_songs": len(songs),
        "total_chunks_cached": total_cached,
        "total_chunks_skipped": total_skipped,
        "cache_dir": cache_dir,
        "estimated_size_gb": estimated_size_gb
    }

    if show_progress:
        print(f"\nCache build complete:")
        print(f"  New chunks: {total_cached}")
        print(f"  Skipped (existing): {total_skipped}")
        print(f"  Estimated size: {estimated_size_gb:.2f} GB")

    return stats


def load_cached_chunk(
    song_id: int,
    chunk_idx: int,
    cache_dir: str = DEFAULT_CACHE_DIR
) -> Optional[torch.Tensor]:
    """Load a cached chunk as a torch tensor.

    Args:
        song_id: Song row ID
        chunk_idx: Chunk index (0-3)
        cache_dir: Cache directory

    Returns:
        Torch tensor of shape (num_samples,) or None if not found
    """
    cache_path = get_chunk_cache_path(song_id, chunk_idx, cache_dir)

    if not cache_path.exists():
        return None

    try:
        data = np.load(cache_path)
        return torch.from_numpy(data)
    except Exception:
        return None


def get_cached_songs(
    songs: List[Song],
    cache_dir: str = DEFAULT_CACHE_DIR,
    num_chunks: int = DEFAULT_NUM_CHUNKS
) -> List[Song]:
    """Filter songs to only those with complete cache.

    Args:
        songs: List of Song objects
        cache_dir: Cache directory
        num_chunks: Expected number of chunks per song

    Returns:
        List of songs that have all chunks cached
    """
    cached_songs = []

    for song in songs:
        all_cached = all(
            get_chunk_cache_path(song.rowid, i, cache_dir).exists()
            for i in range(num_chunks)
        )
        if all_cached:
            cached_songs.append(song)

    return cached_songs


def clear_cache(cache_dir: str = DEFAULT_CACHE_DIR) -> int:
    """Clear all cached chunk files.

    Args:
        cache_dir: Cache directory to clear

    Returns:
        Number of files deleted
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return 0

    count = 0
    for npy_file in cache_path.glob("*.npy"):
        npy_file.unlink()
        count += 1

    return count


def get_cache_stats(cache_dir: str = DEFAULT_CACHE_DIR) -> dict:
    """Get statistics about the current cache.

    Args:
        cache_dir: Cache directory

    Returns:
        Dictionary with cache statistics
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        return {
            "exists": False,
            "num_files": 0,
            "size_gb": 0.0,
            "num_songs": 0
        }

    npy_files = list(cache_path.glob("*.npy"))
    total_size = sum(f.stat().st_size for f in npy_files)

    # Count unique songs (files are named {song_id}_{chunk_idx}.npy)
    song_ids = set()
    for f in npy_files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            song_ids.add(parts[0])

    return {
        "exists": True,
        "num_files": len(npy_files),
        "size_gb": total_size / (1024 ** 3),
        "num_songs": len(song_ids),
        "cache_dir": str(cache_path)
    }
