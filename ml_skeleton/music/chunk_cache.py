"""Chunked waveform cache builder for MoCo training.

Caches N evenly-spaced 30s chunks per song as .npy files (default 8).
This enables:
- 15x faster loading vs MP3 decoding
- Fresh augmentations each epoch (CQT computed on-the-fly)
- Capturing song structure diversity (intro, verse, chorus, outro)

Cache structure:
    ./cache/chunks/{song_id}_0.npy ... {song_id}_{N-1}.npy

Each .npy file contains PCM at 16kHz, shape (480000,) = 30s. With use_16bit (default): int16, ~0.96 MB/chunk; else float32, ~1.92 MB/chunk.
Load path supports both dtypes; NN always receives float32 in [-1, 1].
"""

import os
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import multiprocessing
from functools import partial
from tqdm import tqdm
import threading

from ml_skeleton.music.clementine_db import Song


# In-process LRU cache: only used when _chunk_cache_use_cache is True (train). Val never cached.
_chunk_cache_use_cache = threading.local()

# Songs with <= this many chunks are never evicted (prioritize short songs in RAM). Set by dataset.
_chunk_cache_short_song_ids: Optional[set] = None
_chunk_cache_short_song_ids_lock = threading.Lock()


def set_chunk_cache_short_song_ids(song_ids: Optional[set]) -> None:
    """Set song IDs that have ≤5 chunks on disk; their chunks are never evicted from the LRU.
    Call from dataset creation (e.g. MoCoDataset) so short songs stay fully in cache."""
    global _chunk_cache_short_song_ids
    with _chunk_cache_short_song_ids_lock:
        _chunk_cache_short_song_ids = set(song_ids) if song_ids is not None else None


def set_chunk_cache_use_cache(value: bool) -> None:
    """Set whether to use the in-process chunk cache (True=train, False=val)."""
    _chunk_cache_use_cache.value = value


def _get_chunk_cache_use_cache() -> bool:
    """Get current use_cache flag; default False so only explicit train path uses cache."""
    return getattr(_chunk_cache_use_cache, "value", False)


def _get_chunk_cache_max_bytes() -> int:
    """Per-process max bytes for the LRU cache. With num_workers>=1 only workers load data,
    so budget is split across workers: max_gb/num_workers. With num_workers=0 the main process
    loads, so it gets full max_gb."""
    default_gb = 100
    gb = os.environ.get("CHUNK_CACHE_MAX_GB")
    if gb is not None:
        try:
            default_gb = int(float(gb))
        except ValueError:
            pass
    num_workers = 0
    nw = os.environ.get("CHUNK_CACHE_NUM_WORKERS")
    if nw is not None:
        try:
            num_workers = max(0, int(nw))
        except ValueError:
            pass
    if num_workers >= 1:
        per_process_gb = default_gb / num_workers  # only workers load; main process doesn't cache
    else:
        per_process_gb = default_gb  # main process loads
    return int(per_process_gb * (1024 ** 3))


class _ChunkLRUCache:
    """LRU cache for loaded chunks. Evicts by oldest when over max_bytes.
    Prioritizes short songs: chunks from songs with ≤5 chunks (short_song_ids) are never evicted.
    Among evictable chunks, evicts from the song that has the most chunks in cache first."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._current_bytes = 0
        self._order: OrderedDict = OrderedDict()  # key -> (array, size); iteration order = LRU (oldest first)
        self._song_chunk_counts: dict = {}  # song_id -> number of chunks from this song in cache

    def get(self, key: Tuple[int, int]) -> Optional[np.ndarray]:
        if key not in self._order:
            return None
        arr, size = self._order[key]
        self._order.move_to_end(key)
        return arr.copy()

    def _pick_eviction_victim(self) -> Optional[Tuple[int, int]]:
        """Pick the oldest chunk that is evictable: not in short_song_ids, and from a song with max chunks in cache."""
        with _chunk_cache_short_song_ids_lock:
            short_ids = _chunk_cache_short_song_ids
        if not self._order:
            return None
        # Among songs that have at least one chunk in cache, find max chunk count
        max_count = max(self._song_chunk_counts.values(), default=0)
        if max_count <= 0:
            return next(iter(self._order))
        # Prefer evicting from a song with max_count chunks; never evict from short_song_ids
        for key in self._order:
            song_id = key[0]
            if short_ids is not None and song_id in short_ids:
                continue
            if self._song_chunk_counts.get(song_id, 0) == max_count:
                return key
        # Fallback: evict oldest evictable (not short)
        for key in self._order:
            song_id = key[0]
            if short_ids is not None and song_id in short_ids:
                continue
            return key
        # All are short songs; must evict something (shouldn't happen if budget > short-song total)
        return next(iter(self._order))

    def put(self, key: Tuple[int, int], arr: np.ndarray) -> None:
        size = arr.nbytes
        song_id = key[0]
        while self._current_bytes + size > self.max_bytes and self._order:
            evicted_key = self._pick_eviction_victim()
            if evicted_key is None:
                break
            _, evicted_size = self._order.pop(evicted_key)
            self._current_bytes -= evicted_size
            old_song = evicted_key[0]
            self._song_chunk_counts[old_song] = self._song_chunk_counts.get(old_song, 1) - 1
            if self._song_chunk_counts[old_song] <= 0:
                del self._song_chunk_counts[old_song]
        if size <= self.max_bytes:
            self._order[key] = (arr.copy(), size)
            self._current_bytes += size
            self._song_chunk_counts[song_id] = self._song_chunk_counts.get(song_id, 0) + 1


_chunk_lru_cache: Optional[_ChunkLRUCache] = None
_chunk_lru_lock = threading.Lock()


def _get_chunk_lru_cache() -> _ChunkLRUCache:
    global _chunk_lru_cache
    with _chunk_lru_lock:
        if _chunk_lru_cache is None:
            _chunk_lru_cache = _ChunkLRUCache(_get_chunk_cache_max_bytes())
        return _chunk_lru_cache


# Default cache parameters
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_DURATION = 30.0  # seconds per chunk
DEFAULT_NUM_CHUNKS = 8
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
        chunk_idx: Chunk index (0 to num_chunks-1)
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

    For a 4-minute song with 8 chunks of 30s each:
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
    normalize: bool = True,
    use_16bit: bool = False
) -> Optional[np.ndarray]:
    """Extract a single chunk from an audio file.

    Args:
        filepath: Path to audio file
        offset: Start offset in seconds
        duration: Duration to extract in seconds
        sample_rate: Target sample rate
        normalize: Apply z-normalization
        use_16bit: If True, return int16 PCM (clip to [-1,1] then * 32767); else float32

    Returns:
        Numpy array of shape (num_samples,) float32 or int16, or None if extraction fails
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

        w = waveform.numpy()
        if use_16bit:
            w = np.clip(w, -1.0, 1.0)
            return (w * 32767).astype(np.int16)
        return w.astype(np.float32)

    except Exception as e:
        return None


def cache_song_chunks(
    song: Song,
    cache_dir: str = DEFAULT_CACHE_DIR,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_duration: float = 900.0,
    overwrite: bool = False,
    use_16bit: bool = False
) -> Tuple[int, int]:
    """Cache chunks for a single song (only 0..N-1 where N = effective chunks from duration).

    Short songs get fewer chunk files; same logic as prune-chunk-cache. When overwrite is True,
    any existing chunks with index >= N are removed.

    Args:
        song: Song object with filepath
        cache_dir: Base cache directory
        num_chunks: Maximum chunks per song
        chunk_duration: Duration of each chunk in seconds
        sample_rate: Target sample rate
        max_duration: Skip files longer than this
        overwrite: Overwrite existing cache files
        use_16bit: If True, save as int16 (half size); else float32

    Returns:
        Tuple of (num_cached, num_skipped)
    """
    filepath = str(song.filepath)
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    # Get file duration first (needed to compute how many chunks we actually store)
    try:
        info = torchaudio.info(filepath)
        file_duration = info.num_frames / info.sample_rate

        # Skip very long files
        if file_duration > max_duration:
            return (0, 0)

    except Exception:
        return (0, 0)

    # Only store chunks 0..effective_n-1 (saves space for short songs; same as prune-chunk-cache logic)
    effective_n = min(num_chunks, max(1, int(file_duration / chunk_duration)))

    # Check if all needed chunks already cached
    if not overwrite:
        all_cached = all(
            get_chunk_cache_path(song.rowid, i, cache_dir).exists()
            for i in range(effective_n)
        )
        if all_cached:
            return (0, effective_n)  # All skipped (already cached)

    # Compute chunk offsets (only for indices we will cache)
    offsets = compute_chunk_offsets(file_duration, num_chunks, chunk_duration)[:effective_n]

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
            filepath, offset, chunk_duration, sample_rate, use_16bit=use_16bit
        )

        if chunk_data is not None:
            np.save(cache_path, chunk_data)
            num_cached += 1

    # When overwriting, remove any redundant chunks (index >= effective_n) from a previous full build
    if overwrite and effective_n < num_chunks:
        for chunk_idx in range(effective_n, num_chunks):
            redundant = get_chunk_cache_path(song.rowid, chunk_idx, cache_dir)
            if redundant.exists():
                try:
                    redundant.unlink()
                except OSError:
                    pass

    return (num_cached, num_skipped)


def _cache_song_worker(
    song: Song,
    cache_dir: str,
    num_chunks: int,
    chunk_duration: float,
    sample_rate: int,
    max_duration: float,
    overwrite: bool,
    use_16bit: bool = False
) -> Tuple[int, int, int]:
    """Worker function for parallel caching.

    Returns:
        Tuple of (song_id, num_cached, num_skipped)
    """
    num_cached, num_skipped = cache_song_chunks(
        song, cache_dir, num_chunks, chunk_duration,
        sample_rate, max_duration, overwrite, use_16bit=use_16bit
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
    show_progress: bool = True,
    use_16bit: bool = True
) -> dict:
    """Build chunk cache for all songs in parallel.

    Only stores chunks 0..N-1 per song where N = min(num_chunks, floor(duration/chunk_duration))
    (same as prune-chunk-cache logic), so short songs get fewer chunk files and use less space.

    Args:
        songs: List of Song objects
        cache_dir: Base cache directory
        num_chunks: Maximum chunks per song (short songs get fewer)
        chunk_duration: Duration of each chunk
        sample_rate: Target sample rate
        max_duration: Skip files longer than this
        num_workers: Number of parallel workers (default: 80% CPU)
        overwrite: Overwrite existing cache files
        show_progress: Show progress bar
        use_16bit: If True, save chunks as int16 (half size); else float32

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
        overwrite=overwrite,
        use_16bit=use_16bit
    )

    total_cached = 0
    total_skipped = 0

    # Process songs in parallel
    if show_progress:
        print(f"Building chunk cache: {len(songs)} songs, up to {num_chunks} chunks/song (fewer for short songs)")
        print(f"Cache directory: {cache_dir}")

    with multiprocessing.Pool(num_workers) as pool:
        iterator = pool.imap_unordered(worker_fn, songs)
        if show_progress:
            iterator = tqdm(iterator, total=len(songs), desc="Caching")

        for song_id, num_cached, num_skipped in iterator:
            total_cached += num_cached
            total_skipped += num_skipped

    # Calculate estimated size (int16 = 2 bytes, float32 = 4 bytes)
    bytes_per_sample = 2 if use_16bit else 4
    chunk_size_bytes = int(chunk_duration * sample_rate * bytes_per_sample)
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


def _chunk_array_to_float32(arr: np.ndarray) -> np.ndarray:
    """Convert chunk array to float32 in [-1, 1]. Supports int16 (from 16-bit cache) and float32."""
    if arr.dtype == np.int16:
        return arr.astype(np.float32) / 32768.0
    if arr.dtype == np.float32:
        return arr.copy()
    return arr.astype(np.float32)


def load_cached_chunk(
    song_id: int,
    chunk_idx: int,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Optional[torch.Tensor]:
    """Load a cached chunk as a torch tensor (float32 in [-1, 1]).

    Supports both int16 and float32 .npy files (backward compatible). Uses mmap for disk
    reads and an in-process LRU cache (train only; val never cached) when
    set_chunk_cache_use_cache(True). LRU stores native dtype (int16 or float32) to save RAM.

    If the requested chunk_idx is missing (e.g. after prune-chunk-cache), falls back to
    the highest existing index <= chunk_idx so pruned caches still work.

    Args:
        song_id: Song row ID
        chunk_idx: Chunk index (0 to num_chunks-1)
        cache_dir: Cache directory

    Returns:
        Torch tensor of shape (num_samples,) float32, or None if not found
    """
    cache_path = get_chunk_cache_path(song_id, chunk_idx, cache_dir)
    actual_chunk_idx = chunk_idx
    if not cache_path.exists():
        for fallback in range(chunk_idx - 1, -1, -1):
            fallback_path = get_chunk_cache_path(song_id, fallback, cache_dir)
            if fallback_path.exists():
                cache_path = fallback_path
                actual_chunk_idx = fallback
                break
        else:
            return None

    use_cache = _get_chunk_cache_use_cache()
    # Store under actual chunk index only to avoid duplicate entries (same file for chunk 5 and 7 fallback)
    cache_key = (song_id, actual_chunk_idx)

    if use_cache:
        cache = _get_chunk_lru_cache()
        # Requested chunk_idx may have been satisfied by a lower index (fallback); try chunk_idx down to 0
        for k in range(chunk_idx, -1, -1):
            hit = cache.get((song_id, k))
            if hit is not None:
                return torch.from_numpy(_chunk_array_to_float32(hit))
        # No hit; will load and cache under cache_key
    else:
        cache = None

    try:
        data = np.load(cache_path, mmap_mode="r")
        # Reject corrupted/malformed .npy (avoids malloc invalid size from bad shape)
        if data.ndim != 1 or data.size < 100_000 or data.size > 2_000_000:
            return None
        arr = np.array(data, dtype=data.dtype, copy=True)
        if use_cache and cache is not None:
            cache.put(cache_key, arr)
        return torch.from_numpy(_chunk_array_to_float32(arr))
    except Exception:
        return None


class ChunkCacheFlagWrapper:
    """Wraps a dataset and sets use_chunk_cache before each __getitem__ (train=True, val=False)."""

    def __init__(self, dataset, use_cache: bool):
        self._dataset = dataset
        self.use_cache = use_cache

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        set_chunk_cache_use_cache(self.use_cache)
        return self._dataset[idx]


def get_cached_songs(
    songs: List[Song],
    cache_dir: str = DEFAULT_CACHE_DIR,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    min_chunks: int = 1,
) -> List[Song]:
    """Filter songs to those with at least min_chunks chunk files cached.

    After prune-chunk-cache, a song may have 1..num_chunks files. Use min_chunks=2 to exclude
    single-chunk songs (e.g. for MoCo so query/key can come from different chunks).

    Args:
        songs: List of Song objects
        cache_dir: Cache directory
        num_chunks: Max chunk index to check (0..num_chunks-1)
        min_chunks: Minimum number of consecutive chunks (0, 1, ...) required; default 1

    Returns:
        List of songs that have at least min_chunks chunks cached
    """
    if min_chunks < 1:
        min_chunks = 1
    cached_songs = []
    for song in songs:
        count = 0
        for i in range(num_chunks):
            if get_chunk_cache_path(song.rowid, i, cache_dir).exists():
                count += 1
            else:
                break
        if count >= min_chunks:
            cached_songs.append(song)
    return cached_songs


def _convert_one_file_to_16bit(npy_path: Path) -> Tuple[str, int, Optional[dict]]:
    """Convert a single .npy from float32 to int16 in place.
    Returns ('converted'|'skipped'|'error', 0 or 1, failure_info or None).
    On error, failure_info is a dict (path, reason, ...) for logging.
    """
    path = Path(npy_path) if not isinstance(npy_path, Path) else npy_path
    try:
        data = np.load(path)
    except Exception as e:
        return ("error", 0, {"path": str(path), "reason": "exception", "message": str(e)})
    if data.ndim != 1 or data.size < 100_000 or data.size > 2_000_000:
        return ("error", 0, {
            "path": str(path),
            "reason": "validation",
            "shape": tuple(data.shape),
            "dtype": str(data.dtype),
            "size": int(data.size),
        })
    if data.dtype == np.int16:
        return ("skipped", 0, None)
    try:
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        arr = np.clip(data, -1.0, 1.0)
        int16_arr = (arr * 32767).astype(np.int16)
        # Temp path must end in .npy so np.save() does not add .npy (which would create .npy.tmp.npy)
        tmp_path = path.parent / (path.stem + ".tmp.npy")
        np.save(tmp_path, int16_arr)
        # Atomic replace: prefer Path.replace; if it fails (e.g. dest disappeared on NFS),
        # overwrite destination by writing to path and then remove tmp.
        try:
            tmp_path.replace(path)
        except OSError as e:
            if not tmp_path.exists():
                raise RuntimeError(f"Temp file missing after np.save: {tmp_path}") from e
            # Fallback: write directly over path (non-atomic but works if replace failed)
            np.save(path, int16_arr)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return ("converted", 1, None)
    except Exception as e:
        return ("error", 0, {"path": str(path), "reason": "exception", "message": str(e)})


def convert_cache_to_16bit(
    cache_dir: str = DEFAULT_CACHE_DIR,
    num_workers: Optional[int] = None,
    show_progress: bool = True,
    max_files: Optional[int] = None,
) -> dict:
    """Convert existing float32 chunk .npy files to int16 in place (faster than clear + rebuild).

    Skips files already int16. Each file is loaded, converted, and overwritten atomically.

    Args:
        cache_dir: Chunk cache directory
        num_workers: Parallel workers (default: 80% CPU); 0 = sequential
        show_progress: Show progress bar
        max_files: If set, only process this many files (for quick diagnostics).

    Returns:
        Dict with num_converted, num_skipped, num_error, cache_dir.
    """
    cache_path = Path(cache_dir).resolve()
    if not cache_path.exists():
        return {"num_converted": 0, "num_skipped": 0, "num_error": 0, "cache_dir": cache_dir}

    npy_files = [p.resolve() for p in cache_path.glob("*.npy")]
    if max_files is not None and max_files > 0:
        npy_files = npy_files[:max_files]
    if not npy_files:
        return {"num_converted": 0, "num_skipped": 0, "num_error": 0, "cache_dir": cache_dir}

    if num_workers is None:
        env_w = os.environ.get("CONVERT_CACHE_NUM_WORKERS")
        if env_w is not None:
            try:
                num_workers = int(env_w)
            except ValueError:
                num_workers = max(1, int(multiprocessing.cpu_count() * 0.8))
        else:
            num_workers = max(1, int(multiprocessing.cpu_count() * 0.8))

    num_converted = 0
    num_skipped = 0
    num_error = 0
    failed: List[dict] = []

    if num_workers <= 0:
        iterator = npy_files
        if show_progress:
            iterator = tqdm(iterator, desc="Convert to 16-bit", total=len(npy_files))
        for npy_path in iterator:
            status, n, failure_info = _convert_one_file_to_16bit(npy_path)
            if status == "converted":
                num_converted += n
            elif status == "skipped":
                num_skipped += 1
            else:
                num_error += 1
                if failure_info:
                    failed.append(failure_info)
    else:
        with multiprocessing.Pool(num_workers) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap(_convert_one_file_to_16bit, npy_files),
                    total=len(npy_files),
                    desc="Convert to 16-bit"
                ))
            else:
                results = pool.map(_convert_one_file_to_16bit, npy_files)
        for status, n, failure_info in results:
            if status == "converted":
                num_converted += n
            elif status == "skipped":
                num_skipped += 1
            else:
                num_error += 1
                if failure_info:
                    failed.append(failure_info)

    failures_file = None
    if failed:
        import json
        failures_file = cache_path / "convert_16bit_failures.json"
        with open(failures_file, "w") as f:
            json.dump(failed, f, indent=2)
        if show_progress:
            print(f"  Failures logged to {failures_file} ({len(failed)} files)")

    if show_progress:
        print(f"  Converted: {num_converted}, Skipped (already int16): {num_skipped}, Errors: {num_error}")

    return {
        "num_converted": num_converted,
        "num_skipped": num_skipped,
        "num_error": num_error,
        "cache_dir": cache_dir,
        "failures_file": str(failures_file) if failures_file else None,
    }


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


def chunks_per_song_distribution(
    cache_dir: str = DEFAULT_CACHE_DIR,
    return_per_song: bool = False
) -> dict:
    """Compute frequency counts of number of chunks per song (from cache files).

    Files are named {song_id}_{chunk_idx}.npy. Returns a dict mapping
    num_chunks -> count of songs that have exactly that many chunks.

    Args:
        cache_dir: Chunk cache directory
        return_per_song: If True, return dict song_id_str -> num_chunk_files instead of histogram

    Returns:
        Dict[int, int]: e.g. {1: 100, 2: 500, 8: 40000}; or if return_per_song, {song_id: count}
    """
    cache_path = Path(cache_dir).resolve()
    if not cache_path.exists():
        return {}

    # song_id -> count of chunk files
    chunks_per_song: dict = {}
    for p in cache_path.glob("*.npy"):
        stem = p.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        try:
            int(parts[1])  # chunk_idx
        except ValueError:
            continue
        song_id = parts[0]
        chunks_per_song[song_id] = chunks_per_song.get(song_id, 0) + 1

    if return_per_song:
        return chunks_per_song

    # histogram: num_chunks -> number of songs
    distribution: dict = {}
    for count in chunks_per_song.values():
        distribution[count] = distribution.get(count, 0) + 1

    return dict(sorted(distribution.items()))


def prune_cache_by_duration(
    cache_dir: str = DEFAULT_CACHE_DIR,
    song_id_to_duration_seconds: Optional[dict] = None,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    dry_run: bool = False,
) -> dict:
    """Delete chunk files with index >= N for each song, where N = effective chunks from duration.

    effective_n = min(num_chunks, max(1, floor(duration_seconds / chunk_duration))).
    Keeps chunks 0..effective_n-1; deletes effective_n..num_chunks-1. Saves disk and keeps cache consistent.

    Args:
        cache_dir: Chunk cache directory
        song_id_to_duration_seconds: Map song_id (int) -> duration in seconds. Songs not in map are left unchanged.
        num_chunks: Max chunk index (0..num_chunks-1)
        chunk_duration: Duration per chunk in seconds
        dry_run: If True, only report what would be deleted

    Returns:
        Dict with num_deleted, num_songs_trimmed, bytes_freed.
    """
    if not song_id_to_duration_seconds:
        return {"num_deleted": 0, "num_songs_trimmed": 0, "bytes_freed": 0}
    cache_path = Path(cache_dir).resolve()
    if not cache_path.exists():
        return {"num_deleted": 0, "num_songs_trimmed": 0, "bytes_freed": 0}

    num_deleted = 0
    num_songs_trimmed = 0
    bytes_freed = 0
    for song_id, duration_sec in song_id_to_duration_seconds.items():
        if duration_sec is None or duration_sec <= 0:
            continue
        effective_n = min(num_chunks, max(1, int(duration_sec / chunk_duration)))
        if effective_n >= num_chunks:
            continue
        trimmed = 0
        for chunk_idx in range(effective_n, num_chunks):
            path = get_chunk_cache_path(int(song_id) if isinstance(song_id, str) and song_id.isdigit() else song_id, chunk_idx, cache_dir)
            if path.exists():
                if not dry_run:
                    try:
                        bytes_freed += path.stat().st_size
                        path.unlink()
                    except OSError:
                        pass
                num_deleted += 1
                trimmed += 1
        if trimmed > 0:
            num_songs_trimmed += 1
    return {"num_deleted": num_deleted, "num_songs_trimmed": num_songs_trimmed, "bytes_freed": bytes_freed}
