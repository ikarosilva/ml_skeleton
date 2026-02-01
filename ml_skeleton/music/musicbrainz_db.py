"""MusicBrainz metadata database for storing enriched metadata from AcoustID/MusicBrainz APIs.

This module provides SQLite database operations for storing and retrieving enriched
metadata obtained from acoustic fingerprint lookups. It supports:
- Storing AcoustID lookup results with confidence scores
- Storing MusicBrainz metadata (artist, album, title, year, genres)
- 30-genre taxonomy with mapping to 7 categories
- Confidence scores per metadata field
- Original song filename tracking for potential file updates

Database Schema:
    mb_metadata: Enriched metadata from AcoustID/MusicBrainz APIs
    genre_mapping: 30-genre to 7-category mapping table

Usage:
    from ml_skeleton.music.musicbrainz_db import MusicBrainzDB

    # Initialize database
    mb_db = MusicBrainzDB("./musicbrainz_metadata.db")

    # Add enriched metadata
    mb_db.add_metadata(
        song_id=123,
        filename="/Music/Artist/Album/Song.mp3",
        acoustid_id="abc123",
        musicbrainz_recording_id="def456",
        artist_mb="Pink Floyd",
        artist_confidence=0.95,
        album_mb="The Dark Side of the Moon",
        album_confidence=0.92,
        genre_tags_mb=["progressive_rock", "art_rock"],
        api_response='{"score": 0.95, ...}'
    )

    # Retrieve metadata
    metadata = mb_db.get_metadata(123)
    print(f"Artist: {metadata.artist_mb} (confidence: {metadata.artist_confidence})")
"""

import sqlite3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Set
from contextlib import contextmanager


@dataclass
class MusicBrainzMetadata:
    """Enriched metadata from AcoustID/MusicBrainz for a single song."""
    song_id: int  # References Clementine songs.ROWID
    filename: str  # Original song filename from Clementine

    # AcoustID identification
    acoustid_id: Optional[str] = None
    musicbrainz_recording_id: Optional[str] = None

    # Enriched metadata fields
    artist_mb: Optional[str] = None
    album_mb: Optional[str] = None
    title_mb: Optional[str] = None
    year_mb: Optional[int] = None

    # Genre system (30 common genres from MB folksonomy)
    genre_tags_mb: Optional[List[str]] = None  # ["progressive_rock", "art_rock"]
    genre_7cat: Optional[List[str]] = None  # ["rock"] (mapped to 7 categories)

    # Confidence scores (0.0-1.0)
    artist_confidence: float = 0.0
    album_confidence: float = 0.0
    title_confidence: float = 0.0
    genre_confidence: float = 0.0

    # API metadata
    api_response: Optional[str] = None  # Full JSON response for debugging
    last_updated: Optional[float] = None  # Unix timestamp


@dataclass
class GenreMapping:
    """Mapping from 30-genre taxonomy to 7 categories."""
    mb_genre: str  # e.g., "progressive_rock"
    category_7: str  # Maps to "rock"
    weight: float = 1.0  # For multi-genre weighting


class MusicBrainzDB:
    """SQLite database for MusicBrainz enriched metadata.

    Stores metadata obtained from AcoustID/MusicBrainz API lookups in a separate
    database from Clementine (read-only separation). Includes original filenames
    for potential file updates and confidence scores for metadata quality assessment.

    Args:
        db_path: Path to SQLite database file
        auto_create: If True, create tables if they don't exist

    Usage:
        db = MusicBrainzDB("./musicbrainz_metadata.db")

        # Add metadata from API lookup
        db.add_metadata(
            song_id=123,
            filename="/Music/Artist/Song.mp3",
            artist_mb="The Beatles",
            artist_confidence=0.95
        )

        # Retrieve metadata
        metadata = db.get_metadata(123)
        if metadata and metadata.artist_confidence > 0.7:
            print(f"High-confidence artist: {metadata.artist_mb}")
    """

    def __init__(self, db_path: str = "./musicbrainz_metadata.db", auto_create: bool = True):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if auto_create:
            self._create_tables()
            self._populate_genre_mapping()

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
        """Create mb_metadata and genre_mapping tables if they don't exist."""
        with self._get_conn() as conn:
            cursor = conn.cursor()

            # Main MusicBrainz metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mb_metadata (
                    song_id INTEGER PRIMARY KEY,
                    filename TEXT NOT NULL,

                    acoustid_id TEXT,
                    musicbrainz_recording_id TEXT,

                    artist_mb TEXT,
                    album_mb TEXT,
                    title_mb TEXT,
                    year_mb INTEGER,

                    genre_tags_mb TEXT,
                    genre_7cat TEXT,

                    artist_confidence REAL DEFAULT 0.0,
                    album_confidence REAL DEFAULT 0.0,
                    title_confidence REAL DEFAULT 0.0,
                    genre_confidence REAL DEFAULT 0.0,

                    api_response TEXT,
                    last_updated REAL
                )
            """)

            # Index for faster filename lookups (for file updates)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mb_metadata_filename
                ON mb_metadata(filename)
            """)

            # Index for AcoustID lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mb_metadata_acoustid
                ON mb_metadata(acoustid_id)
            """)

            # 30-genre to 7-category mapping table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS genre_mapping (
                    mb_genre TEXT PRIMARY KEY,
                    category_7 TEXT NOT NULL,
                    weight REAL DEFAULT 1.0
                )
            """)

    def _populate_genre_mapping(self):
        """Populate genre_mapping table with default 30-genre to 7-category mappings."""
        # Check if already populated
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM genre_mapping")
            if cursor.fetchone()['count'] > 0:
                return  # Already populated

        # 30-genre taxonomy mapped to 7 categories
        mappings = [
            # Rock family (8 genres)
            ("rock", "rock", 1.0),
            ("alternative", "rock", 1.0),
            ("indie", "rock", 1.0),
            ("punk", "rock", 1.0),
            ("metal", "rock", 1.0),
            ("progressive_rock", "rock", 1.0),
            ("hard_rock", "rock", 1.0),
            ("grunge", "rock", 1.0),

            # Pop family (5 genres)
            ("pop", "pop", 1.0),
            ("dance", "pop", 1.0),
            ("synth_pop", "pop", 1.0),
            ("new_wave", "pop", 1.0),
            ("disco", "pop", 1.0),

            # Electronic family (6 genres)
            ("electronic", "electronic", 1.0),
            ("techno", "electronic", 1.0),
            ("house", "electronic", 1.0),
            ("trance", "electronic", 1.0),
            ("ambient", "electronic", 1.0),
            ("dubstep", "electronic", 1.0),

            # Hip-hop/R&B (4 genres)
            ("hip_hop", "hiphop", 1.0),
            ("rap", "hiphop", 1.0),
            ("r_and_b", "hiphop", 1.0),
            ("soul", "hiphop", 1.0),

            # Jazz/Classical (4 genres)
            ("jazz", "jazz", 1.0),
            ("classical", "classical", 1.0),
            ("blues", "jazz", 1.0),
            ("funk", "jazz", 1.0),

            # Country/Folk (3 genres)
            ("country", "country", 1.0),
            ("folk", "country", 1.0),
            ("bluegrass", "country", 1.0),
        ]

        with self._get_conn() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO genre_mapping (mb_genre, category_7, weight)
                VALUES (?, ?, ?)
            """, mappings)

    def add_metadata(
        self,
        song_id: int,
        filename: str,
        acoustid_id: Optional[str] = None,
        musicbrainz_recording_id: Optional[str] = None,
        artist_mb: Optional[str] = None,
        album_mb: Optional[str] = None,
        title_mb: Optional[str] = None,
        year_mb: Optional[int] = None,
        genre_tags_mb: Optional[List[str]] = None,
        genre_7cat: Optional[List[str]] = None,
        artist_confidence: float = 0.0,
        album_confidence: float = 0.0,
        title_confidence: float = 0.0,
        genre_confidence: float = 0.0,
        api_response: Optional[str] = None,
        last_updated: Optional[float] = None
    ) -> None:
        """Add or update MusicBrainz metadata for a song.

        Args:
            song_id: Song ID from Clementine database
            filename: Original song filename (for potential file updates)
            acoustid_id: AcoustID identifier
            musicbrainz_recording_id: MusicBrainz recording ID
            artist_mb: Artist name from MusicBrainz
            album_mb: Album title from MusicBrainz
            title_mb: Track title from MusicBrainz
            year_mb: Release year
            genre_tags_mb: List of 30-genre tags from MB folksonomy
            genre_7cat: List of 7-category genres (mapped)
            artist_confidence: Artist metadata confidence (0.0-1.0)
            album_confidence: Album metadata confidence (0.0-1.0)
            title_confidence: Title metadata confidence (0.0-1.0)
            genre_confidence: Genre metadata confidence (0.0-1.0)
            api_response: Full API response JSON (for debugging)
            last_updated: Unix timestamp of last update
        """
        # Serialize lists to JSON
        genre_tags_json = json.dumps(genre_tags_mb) if genre_tags_mb else None
        genre_7cat_json = json.dumps(genre_7cat) if genre_7cat else None

        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO mb_metadata (
                    song_id, filename,
                    acoustid_id, musicbrainz_recording_id,
                    artist_mb, album_mb, title_mb, year_mb,
                    genre_tags_mb, genre_7cat,
                    artist_confidence, album_confidence, title_confidence, genre_confidence,
                    api_response, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                song_id, filename,
                acoustid_id, musicbrainz_recording_id,
                artist_mb, album_mb, title_mb, year_mb,
                genre_tags_json, genre_7cat_json,
                artist_confidence, album_confidence, title_confidence, genre_confidence,
                api_response, last_updated
            ))

    def get_metadata(self, song_id: int) -> Optional[MusicBrainzMetadata]:
        """Get MusicBrainz metadata for a song.

        Args:
            song_id: Song ID from Clementine database

        Returns:
            MusicBrainzMetadata object or None if not found
        """
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT * FROM mb_metadata WHERE song_id = ?
            """, (song_id,))

            row = cursor.fetchone()
            if not row:
                return None

            # Parse JSON fields
            row_dict = dict(row)
            if row_dict['genre_tags_mb']:
                row_dict['genre_tags_mb'] = json.loads(row_dict['genre_tags_mb'])
            if row_dict['genre_7cat']:
                row_dict['genre_7cat'] = json.loads(row_dict['genre_7cat'])

            return MusicBrainzMetadata(**row_dict)

    def get_metadata_by_filename(self, filename: str) -> Optional[MusicBrainzMetadata]:
        """Get MusicBrainz metadata by original filename.

        Useful for tracking metadata when files are moved or renamed.

        Args:
            filename: Original song filename

        Returns:
            MusicBrainzMetadata object or None if not found
        """
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT * FROM mb_metadata WHERE filename = ?
            """, (filename,))

            row = cursor.fetchone()
            if not row:
                return None

            # Parse JSON fields
            row_dict = dict(row)
            if row_dict['genre_tags_mb']:
                row_dict['genre_tags_mb'] = json.loads(row_dict['genre_tags_mb'])
            if row_dict['genre_7cat']:
                row_dict['genre_7cat'] = json.loads(row_dict['genre_7cat'])

            return MusicBrainzMetadata(**row_dict)

    def has_metadata(self, song_id: int) -> bool:
        """Check if song has MusicBrainz metadata.

        Args:
            song_id: Song ID

        Returns:
            True if metadata exists, False otherwise
        """
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM mb_metadata WHERE song_id = ?
            """, (song_id,))
            return cursor.fetchone()['count'] > 0

    def get_enriched_song_ids(self) -> Set[int]:
        """Get set of song IDs that have MusicBrainz metadata.

        Returns:
            Set of song IDs with enriched metadata
        """
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT song_id FROM mb_metadata")
            return {row['song_id'] for row in cursor.fetchall()}

    def get_high_confidence_songs(
        self,
        min_artist_confidence: float = 0.7,
        min_album_confidence: float = 0.7
    ) -> List[int]:
        """Get song IDs with high-confidence metadata.

        Args:
            min_artist_confidence: Minimum artist confidence threshold
            min_album_confidence: Minimum album confidence threshold

        Returns:
            List of song IDs with confidence above thresholds
        """
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT song_id FROM mb_metadata
                WHERE artist_confidence >= ? OR album_confidence >= ?
                ORDER BY (artist_confidence + album_confidence) DESC
            """, (min_artist_confidence, min_album_confidence))
            return [row['song_id'] for row in cursor.fetchall()]

    def get_genre_mapping(self, mb_genre: str) -> Optional[GenreMapping]:
        """Get 7-category mapping for a 30-genre tag.

        Args:
            mb_genre: MusicBrainz genre tag (e.g., "progressive_rock")

        Returns:
            GenreMapping object or None if not found
        """
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT * FROM genre_mapping WHERE mb_genre = ?
            """, (mb_genre,))

            row = cursor.fetchone()
            if row:
                return GenreMapping(**dict(row))
            return None

    def map_genres_to_7cat(self, mb_genres: List[str]) -> List[str]:
        """Map list of 30-genre tags to 7-category genres.

        Args:
            mb_genres: List of MusicBrainz genre tags

        Returns:
            List of unique 7-category genres
        """
        categories = set()
        for genre in mb_genres:
            mapping = self.get_genre_mapping(genre)
            if mapping:
                categories.add(mapping.category_7)
        return sorted(list(categories))

    def get_count(self) -> int:
        """Get total number of enriched songs in database."""
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM mb_metadata")
            return cursor.fetchone()['count']

    def get_stats(self) -> Dict[str, any]:
        """Get comprehensive database statistics.

        Returns:
            Dictionary with statistics:
            - total_songs: Total enriched songs
            - with_acoustid: Songs with AcoustID match
            - with_musicbrainz: Songs with MusicBrainz recording ID
            - high_confidence_artist: Songs with artist confidence > 0.7
            - high_confidence_album: Songs with album confidence > 0.7
            - avg_artist_confidence: Average artist confidence
            - avg_album_confidence: Average album confidence
            - db_size_mb: Database file size in MB
        """
        with self._get_conn() as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN acoustid_id IS NOT NULL THEN 1 ELSE 0 END) as with_acoustid,
                    SUM(CASE WHEN musicbrainz_recording_id IS NOT NULL THEN 1 ELSE 0 END) as with_mb,
                    SUM(CASE WHEN artist_confidence > 0.7 THEN 1 ELSE 0 END) as high_conf_artist,
                    SUM(CASE WHEN album_confidence > 0.7 THEN 1 ELSE 0 END) as high_conf_album,
                    AVG(artist_confidence) as avg_artist_conf,
                    AVG(album_confidence) as avg_album_conf
                FROM mb_metadata
            """)

            row = cursor.fetchone()

            # Get database file size
            db_size_mb = 0.0
            if self.db_path.exists():
                db_size_mb = self.db_path.stat().st_size / (1024 * 1024)

            return {
                "total_songs": row['total'] or 0,
                "with_acoustid": row['with_acoustid'] or 0,
                "with_musicbrainz": row['with_mb'] or 0,
                "high_confidence_artist": row['high_conf_artist'] or 0,
                "high_confidence_album": row['high_conf_album'] or 0,
                "avg_artist_confidence": round(row['avg_artist_conf'] or 0.0, 3),
                "avg_album_confidence": round(row['avg_album_conf'] or 0.0, 3),
                "db_size_mb": round(db_size_mb, 2)
            }

    def clear_metadata(self) -> int:
        """Clear all MusicBrainz metadata (but keep genre mappings).

        Returns:
            Number of records removed
        """
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM mb_metadata")
            count = cursor.fetchone()['count']

            conn.execute("DELETE FROM mb_metadata")
            return count
