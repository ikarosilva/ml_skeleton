"""Fingerprint database for storing chromaprint hashes and duplicate mappings.

This module provides SQLite database operations for acoustic fingerprints extracted
from audio chunks. It supports:
- Storing fingerprints per chunk (song_id, chunk_idx)
- Tracking duplicate songs via fingerprint similarity
- Managing canonical song selection for duplicates
- Querying fingerprint coverage statistics

Database Schema:
    fingerprints: Chromaprint hashes for 30s cached chunks
    duplicates: Mapping of duplicate songs to canonical versions
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
from contextlib import contextmanager


@dataclass
class Fingerprint:
    """Chromaprint fingerprint for a single audio chunk."""
    song_id: int
    chunk_idx: int
    fingerprint: str  # Chromaprint hash (base64 encoded)
    duration: float  # Chunk duration in seconds (typically 30.0)
    mtime: float  # Cache file modification time


@dataclass
class DuplicateGroup:
    """Group of duplicate songs with similarity scores."""
    canonical_id: int  # Selected canonical song ID
    duplicate_ids: List[int]  # Other song IDs in the duplicate group
    similarities: List[float]  # Similarity scores (0.0-1.0)


class FingerprintDB:
    """SQLite database for acoustic fingerprints and duplicate detection.

    Args:
        db_path: Path to SQLite database file
        auto_create: If True, create tables if they don't exist

    Usage:
        db = FingerprintDB("./cache/fingerprints.db")
        db.add_fingerprint(song_id=123, chunk_idx=1, fingerprint="AQAA...")
        count = db.get_fingerprint_count()
        print(f"Total fingerprints: {count}")
    """

    def __init__(self, db_path: str = "./cache/fingerprints.db", auto_create: bool = True):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if auto_create:
            self._create_tables()

    @contextmanager
    def _get_conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_tables(self):
        """Create fingerprints and duplicates tables if they don't exist."""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # Fingerprints table: stores chromaprint hashes for each chunk
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fingerprints (
                    song_id INTEGER NOT NULL,
                    chunk_idx INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL,
                    duration REAL NOT NULL,
                    mtime REAL NOT NULL,
                    PRIMARY KEY (song_id, chunk_idx)
                )
            """)

            # Index for faster lookups by song_id
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_fingerprints_song_id
                ON fingerprints(song_id)
            """)

            # Duplicates table: maps duplicate songs to canonical versions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS duplicates (
                    song_id INTEGER PRIMARY KEY,
                    canonical_id INTEGER NOT NULL,
                    similarity REAL NOT NULL CHECK(similarity >= 0.0 AND similarity <= 1.0)
                )
            """)

            # Index for faster lookups by canonical_id
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_duplicates_canonical_id
                ON duplicates(canonical_id)
            """)

    def add_fingerprint(self, song_id: int, chunk_idx: int, fingerprint: str,
                       duration: float, mtime: float) -> None:
        """Add or update a fingerprint for a chunk.

        Args:
            song_id: Song ID from Clementine database
            chunk_idx: Chunk index (0-3 for 4-chunk cache)
            fingerprint: Chromaprint hash (base64 encoded string)
            duration: Chunk duration in seconds
            mtime: Cache file modification time (Unix timestamp)
        """
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO fingerprints
                (song_id, chunk_idx, fingerprint, duration, mtime)
                VALUES (?, ?, ?, ?, ?)
            """, (song_id, chunk_idx, fingerprint, duration, mtime))

    def add_fingerprints_batch(self, fingerprints: List[Fingerprint]) -> int:
        """Add multiple fingerprints in a single transaction.

        Args:
            fingerprints: List of Fingerprint objects

        Returns:
            Number of fingerprints added
        """
        with self._get_conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO fingerprints
                (song_id, chunk_idx, fingerprint, duration, mtime)
                VALUES (?, ?, ?, ?, ?)
            """, [(f.song_id, f.chunk_idx, f.fingerprint, f.duration, f.mtime)
                  for f in fingerprints])
            return len(fingerprints)

    def get_fingerprint(self, song_id: int, chunk_idx: int) -> Optional[Fingerprint]:
        """Get fingerprint for a specific chunk.

        Args:
            song_id: Song ID
            chunk_idx: Chunk index

        Returns:
            Fingerprint object or None if not found
        """
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT song_id, chunk_idx, fingerprint, duration, mtime
                FROM fingerprints
                WHERE song_id = ? AND chunk_idx = ?
            """, (song_id, chunk_idx))

            row = cursor.fetchone()
            if row:
                return Fingerprint(**dict(row))
            return None

    def get_song_fingerprints(self, song_id: int) -> List[Fingerprint]:
        """Get all fingerprints for a song (all chunks).

        Args:
            song_id: Song ID

        Returns:
            List of Fingerprint objects, sorted by chunk_idx
        """
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT song_id, chunk_idx, fingerprint, duration, mtime
                FROM fingerprints
                WHERE song_id = ?
                ORDER BY chunk_idx
            """, (song_id,))

            return [Fingerprint(**dict(row)) for row in cursor.fetchall()]

    def has_fingerprints(self, song_id: int, num_chunks: int = 4) -> bool:
        """Check if song has complete fingerprints for all chunks.

        Args:
            song_id: Song ID
            num_chunks: Expected number of chunks per song

        Returns:
            True if song has all fingerprints, False otherwise
        """
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) as count
                FROM fingerprints
                WHERE song_id = ?
            """, (song_id,))

            count = cursor.fetchone()['count']
            return count == num_chunks

    def get_fingerprinted_song_ids(self, num_chunks: int = 4) -> Set[int]:
        """Get set of song IDs that have complete fingerprints.

        Args:
            num_chunks: Expected number of chunks per song

        Returns:
            Set of song IDs with complete fingerprints
        """
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT song_id, COUNT(*) as count
                FROM fingerprints
                GROUP BY song_id
                HAVING count = ?
            """, (num_chunks,))

            return {row['song_id'] for row in cursor.fetchall()}

    def get_fingerprint_count(self) -> int:
        """Get total number of fingerprints in database."""
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM fingerprints")
            return cursor.fetchone()['count']

    def get_song_count(self) -> int:
        """Get number of unique songs with at least one fingerprint."""
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT COUNT(DISTINCT song_id) as count FROM fingerprints
            """)
            return cursor.fetchone()['count']

    def add_duplicate(self, song_id: int, canonical_id: int, similarity: float) -> None:
        """Mark a song as a duplicate of a canonical song.

        Args:
            song_id: Duplicate song ID
            canonical_id: Canonical song ID (the one to keep)
            similarity: Fingerprint similarity score (0.0-1.0)
        """
        if not (0.0 <= similarity <= 1.0):
            raise ValueError(f"Similarity must be between 0.0 and 1.0, got {similarity}")

        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO duplicates
                (song_id, canonical_id, similarity)
                VALUES (?, ?, ?)
            """, (song_id, canonical_id, similarity))

    def add_duplicates_batch(self, duplicates: List[Tuple[int, int, float]]) -> int:
        """Add multiple duplicate mappings in a single transaction.

        Args:
            duplicates: List of (song_id, canonical_id, similarity) tuples

        Returns:
            Number of duplicate mappings added
        """
        with self._get_conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO duplicates
                (song_id, canonical_id, similarity)
                VALUES (?, ?, ?)
            """, duplicates)
            return len(duplicates)

    def get_canonical_id(self, song_id: int) -> Optional[int]:
        """Get canonical song ID for a potentially duplicate song.

        Args:
            song_id: Song ID to check

        Returns:
            Canonical song ID if it's a duplicate, None if it's canonical
        """
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT canonical_id FROM duplicates WHERE song_id = ?
            """, (song_id,))

            row = cursor.fetchone()
            if row:
                return row['canonical_id']
            return None

    def is_canonical(self, song_id: int) -> bool:
        """Check if a song is canonical (not a duplicate).

        Args:
            song_id: Song ID to check

        Returns:
            True if song is canonical, False if it's a duplicate
        """
        return self.get_canonical_id(song_id) is None

    def get_duplicate_group(self, canonical_id: int) -> DuplicateGroup:
        """Get all duplicates for a canonical song.

        Args:
            canonical_id: Canonical song ID

        Returns:
            DuplicateGroup with all duplicate IDs and similarities
        """
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT song_id, similarity
                FROM duplicates
                WHERE canonical_id = ?
                ORDER BY similarity DESC
            """, (canonical_id,))

            rows = cursor.fetchall()
            return DuplicateGroup(
                canonical_id=canonical_id,
                duplicate_ids=[row['song_id'] for row in rows],
                similarities=[row['similarity'] for row in rows]
            )

    def get_all_canonical_ids(self) -> Set[int]:
        """Get set of all canonical song IDs (songs that are not duplicates).

        Returns:
            Set of canonical song IDs
        """
        with self._get_conn() as conn:
            # Get all unique song IDs from fingerprints
            cursor = conn.execute("""
                SELECT DISTINCT song_id FROM fingerprints
            """)
            all_song_ids = {row['song_id'] for row in cursor.fetchall()}

            # Get all duplicate song IDs
            cursor = conn.execute("""
                SELECT song_id FROM duplicates
            """)
            duplicate_ids = {row['song_id'] for row in cursor.fetchall()}

            # Canonical songs = all songs - duplicates
            return all_song_ids - duplicate_ids

    def get_duplicate_count(self) -> int:
        """Get total number of duplicate songs."""
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM duplicates")
            return cursor.fetchone()['count']

    def clear_duplicates(self) -> int:
        """Clear all duplicate mappings (but keep fingerprints).

        Returns:
            Number of duplicates removed
        """
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM duplicates")
            count = cursor.fetchone()['count']

            conn.execute("DELETE FROM duplicates")
            return count

    def get_stats(self) -> Dict[str, any]:
        """Get comprehensive database statistics.

        Returns:
            Dictionary with statistics:
            - total_fingerprints: Total fingerprint count
            - unique_songs: Number of songs with fingerprints
            - songs_with_complete_fingerprints: Songs with all 4 chunks
            - canonical_songs: Number of canonical (non-duplicate) songs
            - duplicate_songs: Number of duplicate songs
            - duplicate_groups: Number of canonical songs with duplicates
            - db_size_mb: Database file size in MB
        """
        total_fingerprints = self.get_fingerprint_count()
        unique_songs = self.get_song_count()
        complete_songs = len(self.get_fingerprinted_song_ids(num_chunks=4))
        duplicate_count = self.get_duplicate_count()
        canonical_ids = self.get_all_canonical_ids()

        # Count canonical songs that have duplicates
        duplicate_groups = 0
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT COUNT(DISTINCT canonical_id) as count FROM duplicates
            """)
            duplicate_groups = cursor.fetchone()['count']

        # Get database file size
        db_size_mb = 0.0
        if self.db_path.exists():
            db_size_mb = self.db_path.stat().st_size / (1024 * 1024)

        return {
            "total_fingerprints": total_fingerprints,
            "unique_songs": unique_songs,
            "songs_with_complete_fingerprints": complete_songs,
            "canonical_songs": len(canonical_ids),
            "duplicate_songs": duplicate_count,
            "duplicate_groups": duplicate_groups,
            "db_size_mb": round(db_size_mb, 2)
        }
