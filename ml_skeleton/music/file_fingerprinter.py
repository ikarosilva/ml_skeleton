"""Chromaprint fingerprint extraction from original audio files.

This module provides functions to extract acoustic fingerprints from original
audio files (MP3, FLAC, etc.) using the acoustid library. Unlike chunk_fingerprinter.py
which fingerprints 30-second cached chunks, this module fingerprints the FULL original
files, which is required for AcoustID API matching.

Key features:
- Uses acoustid.fingerprint_file() to process original audio files
- Handles path remapping via MUSIC_PATH_REMAP environment variable
- Supports batch processing with multiprocessing
- Integrates with FingerprintDB for storage
- Tracks progress and handles errors gracefully

Usage:
    from ml_skeleton.music.file_fingerprinter import fingerprint_songs_from_files
    from ml_skeleton.music.clementine_db import ClementineDB
    from ml_skeleton.music.fingerprint_db import FingerprintDB

    # Load songs and databases
    db = ClementineDB("/Music/database/clementine_backup_2026-01.db")
    songs = db.get_all_songs()
    fp_db = FingerprintDB("./cache/fingerprints.db")

    # Extract fingerprints from original files
    stats = fingerprint_songs_from_files(
        songs=songs,
        fp_db=fp_db,
        num_workers=8
    )
    print(f"Fingerprinted {stats['fingerprinted']} songs")
"""

import acoustid
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from ml_skeleton.music.clementine_db import Song
from ml_skeleton.music.fingerprint_db import FingerprintDB, Fingerprint
from ml_skeleton.music.metadata_utils import count_valid_metadata_fields


@dataclass
class FileFingerprintResult:
    """Result of fingerprinting a single audio file."""
    song_id: int
    fingerprint: Optional[str]  # None if extraction failed
    duration: float
    mtime: float
    error: Optional[str] = None


def fingerprint_audio_file(song: Song) -> FileFingerprintResult:
    """Extract chromaprint fingerprint from an original audio file.

    Args:
        song: Song object with filepath property

    Returns:
        FileFingerprintResult with fingerprint or error message
    """
    try:
        # Get file path with remapping
        filepath = song.filepath

        # Check if file exists
        if not filepath.exists():
            return FileFingerprintResult(
                song_id=song.rowid,
                fingerprint=None,
                duration=0.0,
                mtime=0.0,
                error=f"File not found: {filepath}"
            )

        # Extract fingerprint from full audio file using fpcalc directly
        # This avoids audioread multiprocessing issues
        # acoustid._fingerprint_file_fpcalc() returns (duration, fingerprint)
        duration, fingerprint = acoustid._fingerprint_file_fpcalc(str(filepath), None)
        # Note: None as second argument means no max length (process full file)

        if not fingerprint:
            return FileFingerprintResult(
                song_id=song.rowid,
                fingerprint=None,
                duration=duration,
                mtime=filepath.stat().st_mtime,
                error="Chromaprint extraction failed"
            )

        # Get file modification time
        mtime = filepath.stat().st_mtime

        return FileFingerprintResult(
            song_id=song.rowid,
            fingerprint=fingerprint,
            duration=duration,
            mtime=mtime,
            error=None
        )

    except Exception as e:
        # Capture full error details for debugging
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        return FileFingerprintResult(
            song_id=song.rowid,
            fingerprint=None,
            duration=0.0,
            mtime=0.0,
            error=error_msg
        )


def _fingerprint_worker(song: Song) -> FileFingerprintResult:
    """Worker function for multiprocessing fingerprint extraction.

    Args:
        song: Song object to fingerprint

    Returns:
        FileFingerprintResult
    """
    return fingerprint_audio_file(song)


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


def fingerprint_songs_from_files(
    songs: List[Song],
    fp_db: FingerprintDB,
    num_workers: int = 4,
    skip_existing: bool = True,
    prioritize_missing_metadata: bool = True,
    max_songs: Optional[int] = None,
    max_duration: Optional[float] = None,
    verbose: bool = True
) -> Dict[str, int]:
    """Extract fingerprints from original audio files using multiprocessing.

    This function fingerprints the FULL original audio files, not cached chunks.
    This is required for AcoustID API matching.

    Args:
        songs: List of Song objects to fingerprint
        fp_db: FingerprintDB instance for storage
        num_workers: Number of parallel workers (default: 4)
        skip_existing: Skip songs that already have fingerprints in DB
        prioritize_missing_metadata: Process songs with missing metadata first
        max_songs: Maximum number of songs to process (None = all, default: None)
        max_duration: Maximum song duration in seconds (None = no limit, default: None)
        verbose: Show progress bar

    Returns:
        Statistics dictionary:
        - total_songs: Total songs in input
        - processed: Songs actually processed
        - fingerprinted: Successfully fingerprinted
        - skipped: Skipped (already in DB or exceeded max_songs)
        - skipped_duration: Skipped due to exceeding max_duration
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
        "skipped_duration": 0,
        "failed": 0,
        "rated": 0,
        "unrated": 0,
        "errors": []
    }

    # Filter out songs exceeding max duration
    if max_duration is not None:
        songs_before_filter = len(songs)
        songs = [s for s in songs if s.duration_seconds <= max_duration or s.duration_seconds == 0]
        stats["skipped_duration"] = songs_before_filter - len(songs)
        if verbose and stats["skipped_duration"] > 0:
            print(f"Filtered out {stats['skipped_duration']} songs exceeding {max_duration}s duration")

    # Prioritize songs with missing metadata if requested
    if prioritize_missing_metadata:
        songs = prioritize_songs_by_missing_metadata(songs)
        if verbose:
            # Count how many have missing metadata
            missing_count = sum(1 for s in songs if count_valid_metadata_fields(s.artist, s.album, s.title) < 3)
            if missing_count > 0:
                print(f"Prioritizing {missing_count} songs with incomplete metadata")

    # Filter out songs that already have fingerprints
    # Note: For file fingerprints, we only store chunk_idx=0 as a marker
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

    # Process in parallel
    fingerprints_to_add = []

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            results = pool.imap_unordered(_fingerprint_worker, songs_to_process)

            if verbose:
                results = tqdm(
                    results,
                    total=len(songs_to_process),
                    desc="Fingerprinting original files",
                    unit="songs"
                )

            for result in results:
                if result.error:
                    stats["failed"] += 1
                    stats["errors"].append(f"Song {result.song_id}: {result.error}")
                else:
                    stats["fingerprinted"] += 1
                    # Store with chunk_idx=0 as a marker for full-file fingerprints
                    fingerprints_to_add.append(
                        Fingerprint(
                            song_id=result.song_id,
                            chunk_idx=0,  # Marker: 0 = full file fingerprint
                            fingerprint=result.fingerprint,
                            duration=result.duration,
                            mtime=result.mtime
                        )
                    )
    else:
        # Single-threaded (for debugging)
        for song in (tqdm(songs_to_process, desc="Fingerprinting original files") if verbose else songs_to_process):
            result = _fingerprint_worker(song)
            if result.error:
                stats["failed"] += 1
                stats["errors"].append(f"Song {result.song_id}: {result.error}")
            else:
                stats["fingerprinted"] += 1
                fingerprints_to_add.append(
                    Fingerprint(
                        song_id=result.song_id,
                        chunk_idx=0,
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
