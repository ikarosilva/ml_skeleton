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
    mtime: float = 0.0 # Last modification time, to be filled in later

    @property
    def filepath(self) -> Path:
        """Convert file:// URI to a filesystem Path object."""
        if self.filename.startswith("file://"):
            return Path(unquote(self.filename[7:]))
        return Path(self.filename)

    @property
    def is_rated(self) -> bool:
        """Check if the song has a rating (not -1)."""
        return self.rating >= 0.0

def load_all_songs(db_path: str = "/home/ikaro/Music/clementine.db", min_songs: int = 3) -> List[Song]:
    """
    Loads all songs from the Clementine database.

    This is a placeholder implementation. It returns dummy Song objects
    for development and testing purposes.

    Args:
        db_path: Path to the Clementine database
        min_songs: Minimum number of songs to generate (for testing)
    """
    import os
    print(f"--- WARNING: Using placeholder `load_all_songs` from clementine_db.py ---")
    print(f"--- Would connect to: {db_path} ---")
    print(f"--- Generating {min_songs} dummy songs for testing ---")

    # In a real implementation, this would connect to SQLite and query the 'songs' table.
    # For now, return dummy data.
    # NOTE: Clementine uses 0-5 star rating scale (0.0-1.0 per star)

    # Check for environment variable override
    env_min_songs = os.environ.get('MIN_RATED_SONGS')
    if env_min_songs:
        min_songs = max(min_songs, int(env_min_songs))
        print(f"--- MIN_RATED_SONGS env var set: generating {min_songs} songs ---")

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
        albums_per_artist = 5

        for i in range(4, min_songs + 1):
            artist_idx = (i - 4) // (albums_per_artist * 10)
            album_idx = ((i - 4) // 10) % albums_per_artist

            artist = artists[artist_idx % len(artists)]
            album = f"Album {album_idx + 1}"

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
