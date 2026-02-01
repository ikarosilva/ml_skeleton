"""Chromaprint fingerprint extraction from cached audio chunks.

This module provides functions to extract acoustic fingerprints from pre-cached
30-second audio chunks (.npy files) using the chromaprint/AcoustID library.

Key features:
- Extracts fingerprints from cached chunks without re-processing source MP3s
- Converts float32 normalized waveforms to int16 PCM for chromaprint
- Supports batch processing with multiprocessing
- Integrates with FingerprintDB for storage
- Tracks progress and handles errors gracefully

Usage:
    from ml_skeleton.music.chunk_fingerprinter import fingerprint_songs
    from ml_skeleton.music.clementine_db import ClementineDB
    from ml_skeleton.music.fingerprint_db import FingerprintDB

    # Load songs and databases
    db = ClementineDB("/Music/database/clementine_backup_2026-01.db")
    songs = db.get_all_songs()
    fp_db = FingerprintDB("./cache/fingerprints.db")

    # Extract fingerprints
    stats = fingerprint_songs(
        songs=songs,
        cache_dir="./cache/chunks",
        fp_db=fp_db,
        chunk_idx=1,  # Use middle chunk
        num_workers=8
    )
    print(f"Fingerprinted {stats['fingerprinted']} songs")
"""

import acoustid
import chromaprint
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import multiprocessing as mp
from tqdm import tqdm

from ml_skeleton.music.clementine_db import Song
from ml_skeleton.music.fingerprint_db import FingerprintDB, Fingerprint
from ml_skeleton.music.chunk_cache import (
    get_chunk_cache_path,
    load_cached_chunk,
    DEFAULT_SAMPLE_RATE
)
from ml_skeleton.music.metadata_utils import (
    is_unknown_artist,
    is_unknown_album,
    is_unknown_title,
    count_valid_metadata_fields
)


@dataclass
class FingerprintResult:
    """Result of fingerprinting a single song."""
    song_id: int
    chunk_idx: int
    fingerprint: Optional[str]  # None if extraction failed
    duration: float
    mtime: float
    error: Optional[str] = None


def _waveform_to_pcm(waveform: torch.Tensor, sample_rate: int) -> bytes:
    """Convert float32 normalized waveform to int16 PCM bytes for chromaprint.

    Args:
        waveform: Float32 tensor with values in [-1.0, 1.0], shape (T,)
        sample_rate: Sample rate in Hz

    Returns:
        PCM audio bytes (int16, mono)
    """
    # Chromaprint expects int16 PCM in range [-32768, 32767]
    # Convert from normalized float32 [-1.0, 1.0]
    waveform_np = waveform.numpy()

    # Clip to valid range and scale to int16
    waveform_clipped = np.clip(waveform_np, -1.0, 1.0)
    pcm_int16 = (waveform_clipped * 32767).astype(np.int16)

    return pcm_int16.tobytes()


def extract_fingerprint(
    waveform: torch.Tensor,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> Optional[str]:
    """Extract chromaprint fingerprint from a waveform.

    Args:
        waveform: Audio waveform tensor (float32, mono), shape (T,)
        sample_rate: Sample rate in Hz

    Returns:
        Base64-encoded chromaprint fingerprint string, or None if extraction failed
    """
    try:
        # Convert to int16 PCM bytes
        pcm_bytes = _waveform_to_pcm(waveform, sample_rate)

        # Create chromaprint fingerprinter and calculate fingerprint
        fpcalc = chromaprint.Fingerprinter()  # FIXED: Use Fingerprinter, not Chromaprint
        fpcalc.start(sample_rate, 1)  # 1 channel (mono)

        # Feed audio data
        fpcalc.feed(pcm_bytes)

        # Get encoded fingerprint (finish() returns bytes directly)
        encoded_fp = fpcalc.finish()

        if not encoded_fp:
            return None

        # Convert bytes to string
        return encoded_fp.decode('utf-8') if isinstance(encoded_fp, bytes) else encoded_fp

    except Exception as e:
        # Silently fail - caller will check for None
        return None


def fingerprint_chunk(
    song_id: int,
    chunk_idx: int,
    cache_dir: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> FingerprintResult:
    """Extract fingerprint from a single cached chunk.

    Args:
        song_id: Song ID from Clementine database
        chunk_idx: Chunk index (0-3)
        cache_dir: Directory containing cached chunks
        sample_rate: Audio sample rate

    Returns:
        FingerprintResult with fingerprint or error message
    """
    cache_path = get_chunk_cache_path(song_id, chunk_idx, cache_dir)

    # Check if cache file exists
    if not cache_path.exists():
        return FingerprintResult(
            song_id=song_id,
            chunk_idx=chunk_idx,
            fingerprint=None,
            duration=0.0,
            mtime=0.0,
            error=f"Cache file not found: {cache_path}"
        )

    # Load cached chunk
    waveform = load_cached_chunk(song_id, chunk_idx, cache_dir)
    if waveform is None:
        return FingerprintResult(
            song_id=song_id,
            chunk_idx=chunk_idx,
            fingerprint=None,
            duration=0.0,
            mtime=0.0,
            error="Failed to load cached chunk"
        )

    # Extract fingerprint
    fingerprint = extract_fingerprint(waveform, sample_rate)

    # Get chunk duration and mtime
    duration = len(waveform) / sample_rate
    mtime = cache_path.stat().st_mtime

    if fingerprint is None:
        return FingerprintResult(
            song_id=song_id,
            chunk_idx=chunk_idx,
            fingerprint=None,
            duration=duration,
            mtime=mtime,
            error="Chromaprint extraction failed"
        )

    return FingerprintResult(
        song_id=song_id,
        chunk_idx=chunk_idx,
        fingerprint=fingerprint,
        duration=duration,
        mtime=mtime,
        error=None
    )


def _fingerprint_song_worker(args: Tuple[int, int, str, int]) -> FingerprintResult:
    """Worker function for multiprocessing fingerprint extraction.

    Args:
        args: Tuple of (song_id, chunk_idx, cache_dir, sample_rate)

    Returns:
        FingerprintResult
    """
    song_id, chunk_idx, cache_dir, sample_rate = args
    return fingerprint_chunk(song_id, chunk_idx, cache_dir, sample_rate)


def prioritize_songs_by_missing_metadata(songs: List[Song]) -> List[Song]:
    """Sort songs to prioritize those with missing metadata fields.

    Songs are sorted by:
    1. Number of valid metadata fields (ascending) - most missing first
    2. Rating (descending) - highest rated first, unrated last

    Args:
        songs: List of Song objects to sort

    Returns:
        Sorted list with missing-metadata songs first, then by rating
    """
    def metadata_score(song: Song) -> tuple:
        """Calculate sort key: (metadata_completeness, -rating).

        Lower values = higher priority.
        Unrated songs (rating < 0) are mapped to very low priority.
        """
        valid_count = count_valid_metadata_fields(song.artist, song.album, song.title)
        # For rating: negate to sort descending, but put unrated songs last
        rating_key = -song.rating if song.rating >= 0 else 999  # Unrated at end
        return (valid_count, rating_key)

    return sorted(songs, key=metadata_score)


def fingerprint_songs(
    songs: List[Song],
    cache_dir: str,
    fp_db: FingerprintDB,
    chunk_idx: int = 1,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_workers: int = 8,
    skip_existing: bool = True,
    prioritize_missing_metadata: bool = True,
    max_songs: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, int]:
    """Extract fingerprints for multiple songs using multiprocessing.

    Args:
        songs: List of Song objects to fingerprint
        cache_dir: Directory containing cached chunks
        fp_db: FingerprintDB instance for storage
        chunk_idx: Which chunk to use for fingerprinting (0-3, default: 1 = middle)
        sample_rate: Audio sample rate
        num_workers: Number of parallel workers (default: 8)
        skip_existing: Skip songs that already have fingerprints in DB
        prioritize_missing_metadata: Process songs with missing metadata first
        max_songs: Maximum number of songs to process (None = all, default: None)
        verbose: Show progress bar

    Returns:
        Statistics dictionary:
        - total_songs: Total songs in input
        - processed: Songs actually processed
        - fingerprinted: Successfully fingerprinted
        - skipped: Skipped (already in DB or exceeded max_songs)
        - failed: Failed to extract fingerprint
        - rated: Number of rated songs processed (rating >= 0)
        - unrated: Number of unrated songs processed (rating < 0)
        - errors: List of error messages
    """
    stats = {
        "total_songs": len(songs),
        "processed": 0,
        "fingerprinted": 0,
        "skipped": 0,
        "failed": 0,
        "rated": 0,
        "unrated": 0,
        "errors": []
    }

    # Prioritize songs with missing metadata if requested
    if prioritize_missing_metadata:
        songs = prioritize_songs_by_missing_metadata(songs)
        if verbose:
            # Count how many have missing metadata
            missing_count = sum(1 for s in songs if count_valid_metadata_fields(s.artist, s.album, s.title) < 3)
            if missing_count > 0:
                print(f"Prioritizing {missing_count} songs with incomplete metadata")

    # Filter out songs that already have fingerprints
    if skip_existing:
        songs_to_process = [
            song for song in songs
            if not fp_db.has_fingerprints(song.rowid, num_chunks=1)
        ]
        stats["skipped"] = len(songs) - len(songs_to_process)

        if verbose and stats["skipped"] > 0:
            print(f"Skipping {stats['skipped']} songs with existing fingerprints")
    else:
        songs_to_process = songs

    # Limit to max_songs if specified (for free tier API limits)
    if max_songs is not None and len(songs_to_process) > max_songs:
        songs_to_process = songs_to_process[:max_songs]
        stats["skipped"] += (len(songs) - stats["skipped"] - max_songs)
        if verbose:
            print(f"Limiting to {max_songs} songs (free tier limit)")

    if not songs_to_process:
        if verbose:
            print("No songs to fingerprint")
        return stats

    stats["processed"] = len(songs_to_process)

    # Track rated vs unrated songs
    stats["rated"] = sum(1 for song in songs_to_process if song.rating >= 0)
    stats["unrated"] = sum(1 for song in songs_to_process if song.rating < 0)

    # Prepare arguments for workers
    worker_args = [
        (song.rowid, chunk_idx, cache_dir, sample_rate)
        for song in songs_to_process
    ]

    # Process in parallel
    fingerprints_to_add = []

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            results = pool.imap_unordered(_fingerprint_song_worker, worker_args)

            if verbose:
                results = tqdm(
                    results,
                    total=len(worker_args),
                    desc="Fingerprinting chunks",
                    unit="songs"
                )

            for result in results:
                if result.error:
                    stats["failed"] += 1
                    stats["errors"].append(f"Song {result.song_id}: {result.error}")
                else:
                    stats["fingerprinted"] += 1
                    fingerprints_to_add.append(
                        Fingerprint(
                            song_id=result.song_id,
                            chunk_idx=result.chunk_idx,
                            fingerprint=result.fingerprint,
                            duration=result.duration,
                            mtime=result.mtime
                        )
                    )
    else:
        # Single-threaded (for debugging)
        for args in (tqdm(worker_args, desc="Fingerprinting chunks") if verbose else worker_args):
            result = _fingerprint_song_worker(args)
            if result.error:
                stats["failed"] += 1
                stats["errors"].append(f"Song {result.song_id}: {result.error}")
            else:
                stats["fingerprinted"] += 1
                fingerprints_to_add.append(
                    Fingerprint(
                        song_id=result.song_id,
                        chunk_idx=result.chunk_idx,
                        fingerprint=result.fingerprint,
                        duration=result.duration,
                        mtime=result.mtime
                    )
                )

    # Batch insert into database
    if fingerprints_to_add:
        if verbose:
            print(f"\nStoring {len(fingerprints_to_add)} fingerprints in database...")

        fp_db.add_fingerprints_batch(fingerprints_to_add)

    return stats


def fingerprint_all_chunks(
    songs: List[Song],
    cache_dir: str,
    fp_db: FingerprintDB,
    num_chunks: int = 4,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_workers: int = 8,
    skip_existing: bool = True,
    verbose: bool = True
) -> Dict[str, int]:
    """Extract fingerprints for all chunks of multiple songs.

    Args:
        songs: List of Song objects to fingerprint
        cache_dir: Directory containing cached chunks
        fp_db: FingerprintDB instance for storage
        num_chunks: Number of chunks per song (default: 4)
        sample_rate: Audio sample rate
        num_workers: Number of parallel workers
        skip_existing: Skip songs that already have complete fingerprints
        verbose: Show progress bar

    Returns:
        Statistics dictionary (same as fingerprint_songs)
    """
    total_stats = {
        "total_songs": len(songs),
        "fingerprinted": 0,
        "skipped": 0,
        "failed": 0,
        "errors": []
    }

    # Process each chunk index
    for chunk_idx in range(num_chunks):
        if verbose:
            print(f"\n=== Processing chunk {chunk_idx + 1}/{num_chunks} ===")

        chunk_stats = fingerprint_songs(
            songs=songs,
            cache_dir=cache_dir,
            fp_db=fp_db,
            chunk_idx=chunk_idx,
            sample_rate=sample_rate,
            num_workers=num_workers,
            skip_existing=skip_existing,
            verbose=verbose
        )

        # Aggregate statistics (per-song, not per-chunk)
        if chunk_idx == 0:
            total_stats["skipped"] = chunk_stats["skipped"]
            total_stats["failed"] = chunk_stats["failed"]
        total_stats["fingerprinted"] = max(
            total_stats["fingerprinted"],
            chunk_stats["fingerprinted"]
        )
        total_stats["errors"].extend(chunk_stats["errors"])

    return total_stats


def compare_fingerprints(fp1: str, fp2: str) -> Optional[float]:
    """Compare two chromaprint fingerprints and return similarity score.

    Args:
        fp1: First fingerprint (base64 encoded string)
        fp2: Second fingerprint (base64 encoded string)

    Returns:
        Similarity score (0.0-1.0), or None if comparison failed
        Higher score = more similar
    """
    try:
        # Decode fingerprints
        raw_fp1 = chromaprint.decode_fingerprint(fp1.encode('utf-8'))[0]
        raw_fp2 = chromaprint.decode_fingerprint(fp2.encode('utf-8'))[0]

        # Calculate bit error rate (BER) between fingerprints
        # Lower BER = more similar
        # Convert to similarity score: 1.0 - BER
        min_len = min(len(raw_fp1), len(raw_fp2))
        if min_len == 0:
            return 0.0

        # Count matching bits
        matches = sum(
            bin(raw_fp1[i] ^ raw_fp2[i]).count('0')
            for i in range(min_len)
        )
        total_bits = min_len * 32  # Each uint32 has 32 bits
        similarity = matches / total_bits

        return similarity

    except Exception:
        return None
