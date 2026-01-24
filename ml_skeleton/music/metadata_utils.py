"""Metadata validation and unknown detection utilities.

Detects unknown/placeholder metadata values in artist, album, and title fields.
Based on patterns from Clementine music player codebase.

Supports loading exclusion lists from CSV files for filtering during training.
"""

import csv
from pathlib import Path
from typing import Optional, Set

# Cache for loaded exclusion lists
_excluded_artists: Optional[Set[str]] = None
_excluded_albums: Optional[Set[str]] = None

# Unknown patterns compiled from Clementine codebase
# Reference: /git/clementine/main.py (stats command)
UNKNOWN_ARTIST_PATTERNS = {
    # Empty/null indicators
    "", "*",

    # English
    "unknown artist", "artist", "various artists", "various", "va",
    "track", "no artist", "unknown",

    # Portuguese
    "artista desconhecido", "artista", "intérprete desconhecido",
    "interprete desconhecido",

    # Spanish
    "varios", "varios artistas", "vários artistas",

    # French
    "artistes divers",
}

UNKNOWN_ALBUM_PATTERNS = {
    # Empty/null indicators
    "", "*",

    # English
    "unknown album", "album", "unknown", "no album",

    # Portuguese
    "álbum desconhecido", "album desconhecido", "desconhecido",

    # Spanish
    "álbum desconocido", "album desconocido", "desconocido",
}

UNKNOWN_TITLE_PATTERNS = {
    # Empty/null indicators
    "", "*",

    # English
    "unknown", "unknown title", "title", "track",

    # Portuguese
    "desconhecido", "faixa", "trilha",

    # Spanish
    "desconocido", "pista",
}


def is_unknown_artist(artist: Optional[str]) -> bool:
    """Check if artist is unknown/placeholder value.

    Args:
        artist: Artist name (or None)

    Returns:
        True if artist is unknown/placeholder, False otherwise
    """
    if artist is None:
        return True

    artist_clean = artist.strip().lower()

    # Empty string
    if not artist_clean:
        return True

    # Check against known patterns
    if artist_clean in UNKNOWN_ARTIST_PATTERNS:
        return True

    # Check for numeric-only with 2 or fewer characters (e.g., "1", "12")
    if len(artist_clean) <= 2 and artist_clean.isdigit():
        return True

    # Check for generic track patterns like "track 01", "track 1"
    if artist_clean.startswith("track") or artist_clean.startswith("faixa"):
        return True

    return False


def is_unknown_album(album: Optional[str]) -> bool:
    """Check if album is unknown/placeholder value.

    Args:
        album: Album name (or None)

    Returns:
        True if album is unknown/placeholder, False otherwise
    """
    if album is None:
        return True

    album_clean = album.strip().lower()

    # Empty string
    if not album_clean:
        return True

    # Check against known patterns
    if album_clean in UNKNOWN_ALBUM_PATTERNS:
        return True

    # Check for numeric-only with 2 or fewer characters
    if len(album_clean) <= 2 and album_clean.isdigit():
        return True

    return False


def is_unknown_title(title: Optional[str]) -> bool:
    """Check if title is unknown/placeholder value.

    Args:
        title: Title/track name (or None)

    Returns:
        True if title is unknown/placeholder, False otherwise
    """
    if title is None:
        return True

    title_clean = title.strip().lower()

    # Empty string
    if not title_clean:
        return True

    # Check against known patterns
    if title_clean in UNKNOWN_TITLE_PATTERNS:
        return True

    # Check for generic track patterns like "track 01", "faixa 5"
    # But allow if it has more context (e.g., "track 1 - introduction")
    words = title_clean.split()
    if len(words) >= 2:
        first_word = words[0]
        second_word = words[1]

        # "track 01", "faixa 3", "trilha 5"
        if first_word in ["track", "faixa", "trilha"]:
            # If second word is a number, likely placeholder
            if second_word.isdigit():
                return True

    return False


def has_valid_metadata(
    artist: Optional[str],
    album: Optional[str],
    title: Optional[str] = None
) -> bool:
    """Check if song has at least some valid metadata.

    For encoder training, we want to skip songs where EVERYTHING is unknown.
    But if even one field is valid, we can use it.

    Args:
        artist: Artist name
        album: Album name
        title: Optional title name

    Returns:
        True if at least one metadata field is valid, False if all are unknown
    """
    has_artist = not is_unknown_artist(artist)
    has_album = not is_unknown_album(album)

    if title is not None:
        has_title = not is_unknown_title(title)
        return has_artist or has_album or has_title

    return has_artist or has_album


def count_valid_metadata_fields(
    artist: Optional[str],
    album: Optional[str],
    title: Optional[str] = None
) -> int:
    """Count how many metadata fields are valid.

    Useful for filtering or statistics.

    Args:
        artist: Artist name
        album: Album name
        title: Optional title name

    Returns:
        Number of valid metadata fields (0-3)
    """
    count = 0

    if not is_unknown_artist(artist):
        count += 1

    if not is_unknown_album(album):
        count += 1

    if title is not None and not is_unknown_title(title):
        count += 1

    return count


def validate_metadata_for_contrastive_loss(
    artist: str,
    album: str
) -> tuple[bool, bool]:
    """Validate metadata fields for contrastive loss.

    Returns which fields are valid for creating positive pairs.

    Args:
        artist: Artist name
        album: Album name

    Returns:
        (artist_valid, album_valid): Boolean tuple indicating which fields are valid
    """
    artist_valid = not is_unknown_artist(artist)
    album_valid = not is_unknown_album(album)

    return artist_valid, album_valid


def load_exclusion_lists(
    artist_csv: Optional[str] = None,
    album_csv: Optional[str] = None
) -> tuple[Set[str], Set[str]]:
    """Load artist and album exclusion lists from CSV files.

    CSV files should have the first column as the name to exclude.
    Case-insensitive matching is used (all names stored lowercase).

    Args:
        artist_csv: Path to artist exclusion CSV file
        album_csv: Path to album exclusion CSV file

    Returns:
        (excluded_artists, excluded_albums): Sets of excluded names (lowercase)
    """
    global _excluded_artists, _excluded_albums

    # Default paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    default_artist_csv = project_root / "unknown_artists.csv"
    default_album_csv = project_root / "unknown_album.csv"

    # Use defaults if not specified
    artist_csv = artist_csv or str(default_artist_csv)
    album_csv = album_csv or str(default_album_csv)

    # Load artist exclusion list
    _excluded_artists = set()
    artist_path = Path(artist_csv)
    if artist_path.exists():
        try:
            with open(artist_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if row:  # Non-empty row
                        artist_name = row[0].strip().lower()
                        if artist_name:
                            _excluded_artists.add(artist_name)
            print(f"Loaded {len(_excluded_artists)} excluded artists from {artist_csv}")
        except Exception as e:
            print(f"Warning: Could not load artist exclusion list: {e}")
    else:
        print(f"Note: Artist exclusion file not found: {artist_csv}")

    # Load album exclusion list
    _excluded_albums = set()
    album_path = Path(album_csv)
    if album_path.exists():
        try:
            with open(album_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if row:  # Non-empty row
                        album_name = row[0].strip().lower()
                        if album_name:
                            _excluded_albums.add(album_name)
            print(f"Loaded {len(_excluded_albums)} excluded albums from {album_csv}")
        except Exception as e:
            print(f"Warning: Could not load album exclusion list: {e}")
    else:
        print(f"Note: Album exclusion file not found: {album_csv}")

    return _excluded_artists, _excluded_albums


def is_excluded_artist(artist: Optional[str]) -> bool:
    """Check if artist is in the exclusion list.

    Requires load_exclusion_lists() to be called first.

    Args:
        artist: Artist name

    Returns:
        True if artist is in exclusion list, False otherwise
    """
    global _excluded_artists

    if _excluded_artists is None:
        return False  # No exclusion list loaded

    if artist is None:
        return True  # Treat None as excluded

    artist_clean = artist.strip().lower()
    return artist_clean in _excluded_artists


def is_excluded_album(album: Optional[str]) -> bool:
    """Check if album is in the exclusion list.

    Requires load_exclusion_lists() to be called first.

    Args:
        album: Album name

    Returns:
        True if album is in exclusion list, False otherwise
    """
    global _excluded_albums

    if _excluded_albums is None:
        return False  # No exclusion list loaded

    if album is None:
        return True  # Treat None as excluded

    album_clean = album.strip().lower()
    return album_clean in _excluded_albums


def has_excluded_metadata(artist: Optional[str], album: Optional[str]) -> bool:
    """Check if song should be excluded based on artist OR album.

    Uses the CSV exclusion lists. A song is excluded if EITHER:
    - The artist is in the artist exclusion list, OR
    - The album is in the album exclusion list

    This is OR logic, not AND logic.

    Args:
        artist: Artist name
        album: Album name

    Returns:
        True if song should be excluded (artist OR album is excluded)
    """
    return is_excluded_artist(artist) or is_excluded_album(album)
