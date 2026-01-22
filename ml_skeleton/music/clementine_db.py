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
    rating: float  # -1 = unrated, 0.0-1.0 = rated
    filename: str  # file:// URI
    mtime: float = 0.0 # Last modification time, to be filled in later

    @property
    def filepath(self) -> Path:
        """Convert file:// URI to a filesystem Path object."""
        if self.filename.startswith("file://"):
            return Path(unquote(self.filename[7:]))
        return Path(self.filename)

def load_all_songs(db_path: str = "/home/ikaro/Music/clementine.db") -> List[Song]:
    """
    Loads all songs from the Clementine database.
    
    This is a placeholder implementation. It returns a few dummy Song objects
    for development and testing purposes.
    """
    print(f"--- WARNING: Using placeholder `load_all_songs` from clementine_db.py ---")
    print(f"--- Would connect to: {db_path} ---")
    
    # In a real implementation, this would connect to SQLite and query the 'songs' table.
    # For now, return dummy data.
    dummy_songs = [
        Song(
            rowid=1,
            title="Test Song 1 (Music)",
            artist="Test Artist",
            album="Test Album",
            year=2024,
            rating=0.8,
            filename=f"file:///git/ml_skeleton/examples/placeholder_music.mp3",
            mtime=1674259200.0
        ),
        Song(
            rowid=2,
            title="Test Song 2 (Speech)",
            artist="Test Speaker",
            album="Test Speech",
            year=2024,
            rating=-1,
            filename=f"file:///git/ml_skeleton/examples/placeholder_speech.mp3",
            mtime=1674259200.0
        ),
        Song(
            rowid=3,
            title="A Long Song",
            artist="Test Artist",
            album="Test Album",
            year=2024,
            rating=0.5,
            filename=f"file:///git/ml_skeleton/examples/placeholder_long.mp3",
            mtime=1674259200.0
        ),
    ]
    # We need some placeholder files to exist for the demo to work
    for song in dummy_songs:
        if not song.filepath.exists():
            song.filepath.parent.mkdir(parents=True, exist_ok=True)
            song.filepath.touch()

    return dummy_songs

def get_default_workers() -> int:
    """
    Get default number of workers (80% of CPU cores).
    This is a helper function used across the music module.
    """
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    return max(1, int(cpu_count * 0.8))
