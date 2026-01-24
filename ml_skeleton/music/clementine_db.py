"""
Placeholder for Clementine DB interface.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.parse import unquote

@dataclass
class Song:
    """A song from the Clementine database."""
    rowid: int
    title: str
    artist: str
    album: str
    year: int
    rating: float  # -1 = unrated, 0.0-5.0 = rated (star rating scale)
    filename: str  # file:// URI
    genre: str = ""  # Genre string (may contain "/" for multi-genre, e.g. "Rock/Alternative")
    mtime: float = 0.0  # Last modification time, to be filled in later

    @property
    def filepath(self) -> Path:
        """Convert file:// URI to a filesystem Path object with path remapping.

        Environment variable MUSIC_PATH_REMAP can be used to remap paths:
        Format: "old_prefix:new_prefix"
        Example: "/home/ikaro/Music:/Music"
        """
        import os

        # Handle both bytes and str (SQLite can return either)
        filename_str = self.filename.decode('utf-8') if isinstance(self.filename, bytes) else self.filename

        if filename_str.startswith("file://"):
            path_str = unquote(filename_str[7:])
        else:
            path_str = filename_str

        # Apply path remapping if configured
        remap = os.environ.get('MUSIC_PATH_REMAP')
        if remap and ':' in remap:
            old_prefix, new_prefix = remap.split(':', 1)
            if path_str.startswith(old_prefix):
                path_str = new_prefix + path_str[len(old_prefix):]

        return Path(path_str)

    @property
    def is_rated(self) -> bool:
        """Check if the song has a rating (not -1)."""
        return self.rating >= 0.0

def load_all_songs(db_path: str = "/home/ikaro/Music/clementine.db", min_songs: int = 3) -> List[Song]:
    """
    Loads all songs from the Clementine database.

    If the database exists, reads from SQLite. Otherwise, falls back to
    placeholder mode for development and testing.

    Args:
        db_path: Path to the Clementine database
        min_songs: Minimum number of songs to generate (for placeholder mode)

    Returns:
        List of Song objects from the database

    Note:
        - Clementine rating scale: 0.0-1.0 per star (0-5 stars total)
        - Unrated songs have rating = -1
        - This function is READ-ONLY
    """
    import os
    import sqlite3
    from pathlib import Path

    db_file = Path(db_path)

    # Check if database exists
    if not db_file.exists():
        print(f"--- WARNING: Database not found at {db_path} ---")
        print(f"--- Using placeholder mode with dummy songs ---")

        # Check for environment variable override
        env_min_songs = os.environ.get('MIN_RATED_SONGS')
        if env_min_songs:
            min_songs = max(min_songs, int(env_min_songs))
            print(f"--- MIN_RATED_SONGS env var set: generating {min_songs} songs ---")
        else:
            print(f"--- Generating {min_songs} dummy songs for testing ---")

        # Return placeholder songs (see below)
        return _generate_placeholder_songs(min_songs)

    # Real database exists - read from SQLite
    print(f"Loading songs from Clementine database: {db_path}")

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Query the songs table
        # Clementine schema: ROWID, title, artist, album, year, rating, filename, genre, mtime
        # Rating: -1 = unrated, 0-1 = 0-5 stars (0.2 per star)
        query = """
            SELECT
                ROWID,
                title,
                artist,
                album,
                CAST(year AS INTEGER) as year,
                CAST(rating AS REAL) as rating,
                filename,
                genre,
                CAST(mtime AS REAL) as mtime
            FROM songs
            WHERE filename IS NOT NULL
            ORDER BY ROWID
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        songs = []
        for row in rows:
            rowid, title, artist, album, year, rating, filename, genre, mtime = row

            # Convert bytes to strings if needed (SQLite text_factory compatibility)
            def to_str(val):
                if isinstance(val, bytes):
                    return val.decode('utf-8')
                return val

            # Convert Clementine rating to 0-5 scale
            # Clementine: -1 = unrated, 0.0-1.0 = stars (0.2 per star)
            # Our scale: -1 = unrated, 0.0-5.0 = stars
            if rating is None or rating < 0:
                rating_converted = -1.0
            else:
                rating_converted = rating * 5.0  # 0.0-1.0 -> 0.0-5.0

            songs.append(Song(
                rowid=rowid,
                title=to_str(title) if title else "Unknown",
                artist=to_str(artist) if artist else "Unknown Artist",
                album=to_str(album) if album else "Unknown Album",
                year=year or 0,
                rating=rating_converted,
                filename=to_str(filename) if filename else "",
                genre=to_str(genre) if genre else "",
                mtime=mtime or 0.0
            ))

        conn.close()

        print(f"Loaded {len(songs)} songs from database")
        return songs

    except sqlite3.Error as e:
        print(f"--- ERROR: Failed to read database: {e} ---")
        print(f"--- Falling back to placeholder mode ---")
        return _generate_placeholder_songs(min_songs)


def _generate_placeholder_songs(min_songs: int) -> List[Song]:
    """Generate placeholder songs for testing when no real database exists.

    Args:
        min_songs: Minimum number of songs to generate

    Returns:
        List of dummy Song objects for testing
    """
    dummy_songs = []

    # Always include the 3 actual placeholder files
    base_songs = [
        Song(
            rowid=1,
            title="Test Song 1 (Music)",
            artist="Test Artist",
            album="Test Album",
            year=2024,
            rating=4.0,  # 4 stars (0-5 scale)
            filename=f"file:///git/ml_skeleton/examples/placeholder_music.mp3",
            genre="Rock/Alternative",
            mtime=1674259200.0
        ),
        Song(
            rowid=2,
            title="Test Song 2 (Speech)",
            artist="Test Speaker",
            album="Test Speech",
            year=2024,
            rating=-1,  # Unrated
            filename=f"file:///git/ml_skeleton/examples/placeholder_speech.mp3",
            genre="",
            mtime=1674259200.0
        ),
        Song(
            rowid=3,
            title="A Long Song",
            artist="Test Artist",
            album="Test Album",
            year=2024,
            rating=3.5,  # 3.5 stars (0-5 scale)
            filename=f"file:///git/ml_skeleton/examples/placeholder_long.mp3",
            genre="Electronic/Dance",
            mtime=1674259200.0
        ),
    ]

    # We need the placeholder files to exist
    for song in base_songs:
        if not song.filepath.exists():
            song.filepath.parent.mkdir(parents=True, exist_ok=True)
            song.filepath.touch()

    dummy_songs.extend(base_songs)

    # Generate additional rated songs if min_songs > 3
    # These will all use the same placeholder music file
    if min_songs > 3:
        import random
        random.seed(42)  # Reproducible dummy data

        artists = ["Artist A", "Artist B", "Artist C", "Artist D", "Artist E"]
        genres = ["Rock", "Pop", "Electronic", "Hip-Hop", "Jazz", "Rock/Alternative", "Pop/Dance"]
        albums_per_artist = 5

        for i in range(4, min_songs + 1):
            artist_idx = (i - 4) // (albums_per_artist * 10)
            album_idx = ((i - 4) // 10) % albums_per_artist

            artist = artists[artist_idx % len(artists)]
            album = f"Album {album_idx + 1}"
            genre = genres[i % len(genres)]

            # 80% rated, 20% unrated
            is_rated = random.random() < 0.8
            rating = random.uniform(0.0, 5.0) if is_rated else -1

            dummy_songs.append(Song(
                rowid=i,
                title=f"Song {i}",
                artist=artist,
                album=f"{artist}|||{album}",  # Use album key format
                year=2020 + (i % 6),
                rating=rating,
                filename=f"file:///git/ml_skeleton/examples/placeholder_music.mp3",  # Reuse same file
                genre=genre,
                mtime=1674259200.0 + i
            ))

    return dummy_songs

class ClementineDB:
    """READ-ONLY interface to Clementine database.

    This is a simple wrapper around the load_all_songs function
    that provides a class-based interface for accessing the database.

    Args:
        db_path: Path to the Clementine SQLite database file
        min_songs: Minimum number of songs to generate (for placeholder/testing)
    """

    def __init__(self, db_path: str = "/home/ikaro/Music/clementine.db", min_songs: int = 3):
        self.db_path = db_path
        self.min_songs = min_songs
        self._songs = None

    def get_all_songs(self) -> List[Song]:
        """Load all songs from the database.

        Results are cached after the first call.

        Returns:
            List of Song objects from the database
        """
        if self._songs is None:
            self._songs = load_all_songs(self.db_path, min_songs=self.min_songs)
        return self._songs


def get_default_workers() -> int:
    """
    Get default number of workers (80% of CPU cores).
    This is a helper function used across the music module.
    """
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    return max(1, int(cpu_count * 0.8))
