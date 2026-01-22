"""PyTorch Dataset classes for music data.

Supports:
- Audio loading with multiprocessing
- Multi-album labels per song
- Rating prediction datasets
- Embedding-based training
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional

from .clementine_db import Song
from .audio_loader import load_audio_file


class MusicDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for audio files with ratings and multi-album support.

    Loads raw audio waveforms for encoder training. Supports songs appearing
    on multiple albums (e.g., original album + compilation).

    Args:
        songs: List of Song objects from Clementine DB
        album_to_idx: Mapping from album key to integer index
        filename_to_albums: Mapping from filename to list of album keys
        sample_rate: Target sample rate (Hz)
        duration: Audio duration to extract (seconds)
        center_crop: Extract from center (True) or beginning (False)
        speech_results: Optional speech detection scores for filtering
        speech_threshold: Threshold for speech filtering
        only_rated: If True, only include rated songs
    """

    def __init__(
        self,
        songs: list[Song],
        album_to_idx: dict[str, int],
        filename_to_albums: dict[str, list[str]],
        sample_rate: int = 16000,
        duration: float = 30.0,
        center_crop: bool = True,
        speech_results: Optional[dict[str, float]] = None,
        speech_threshold: float = 0.5,
        only_rated: bool = True
    ):
        super().__init__()
        self.album_to_idx = album_to_idx
        self.filename_to_albums = filename_to_albums
        self.sample_rate = sample_rate
        self.duration = duration
        self.center_crop = center_crop

        # Filter songs
        self.songs = self._filter_songs(
            songs,
            speech_results,
            speech_threshold,
            only_rated
        )

    def _filter_songs(
        self,
        songs: list[Song],
        speech_results: Optional[dict[str, float]],
        threshold: float,
        only_rated: bool
    ) -> list[Song]:
        """Filter songs by speech detection and rating status."""
        filtered = []

        for song in songs:
            # Filter by rating if requested
            if only_rated and not song.is_rated:
                continue

            # Filter by speech detection
            if speech_results:
                prob = speech_results.get(song.filename, 0.0)
                if prob > threshold:
                    continue

            # Check file exists
            if not song.filepath.exists():
                continue

            filtered.append(song)

        if len(filtered) < len(songs):
            removed = len(songs) - len(filtered)
            print(f"Filtered {removed} songs (speech/rating/missing)")

        return filtered

    def __len__(self) -> int:
        return len(self.songs)

    def __getitem__(self, idx: int) -> dict:
        """Load audio and return with metadata.

        Returns:
            Dictionary with:
            - audio: Waveform tensor (num_samples,)
            - rating: Rating value in [0, 1] (or -1 if unrated)
            - albums: List of album indices this song belongs to
            - filename: Song filename
        """
        song = self.songs[idx]

        # Load audio (returns None if loading fails)
        audio = load_audio_file(
            song.filename,
            sample_rate=self.sample_rate,
            mono=True,
            duration=self.duration,
            center_crop=self.center_crop
        )

        # Fallback to zeros if loading fails
        if audio is None:
            audio = torch.zeros(int(self.sample_rate * self.duration))

        # Get album labels (may be multiple)
        albums = self._get_album_labels(song.filename)

        # Normalize rating to [0, 1] (Clementine uses 0-5 scale, -1 = unrated)
        rating = song.rating / 5.0 if song.is_rated else -1.0

        return {
            "audio": audio,
            "rating": rating,
            "albums": albums,
            "filename": song.filename
        }

    def _get_album_labels(self, filename: str) -> list[int]:
        """Get all album indices for a song (multi-album support).

        Args:
            filename: Song filename

        Returns:
            List of album indices (may be empty if song not in any album)
        """
        if filename not in self.filename_to_albums:
            return []

        album_keys = self.filename_to_albums[filename]
        album_indices = []

        for album_key in album_keys:
            idx = self.album_to_idx.get(album_key)
            if idx is not None:
                album_indices.append(idx)

        return album_indices


def music_collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for MusicDataset to handle variable-length album lists.

    Args:
        batch: List of dictionaries from MusicDataset.__getitem__

    Returns:
        Batched dictionary with:
        - audio: Stacked tensor (batch_size, num_samples)
        - rating: Tensor (batch_size,)
        - albums: List of lists (variable length per sample)
        - filename: List of strings
    """
    audio_list = []
    rating_list = []
    albums_list = []
    filename_list = []

    for item in batch:
        audio_list.append(item["audio"])
        rating_list.append(item["rating"])
        albums_list.append(item["albums"])  # Keep as list (variable length)
        filename_list.append(item["filename"])

    return {
        "audio": torch.stack(audio_list),
        "rating": torch.tensor(rating_list, dtype=torch.float32),
        "albums": albums_list,  # List of lists
        "filename": filename_list
    }


class EmbeddingDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for pre-extracted embeddings with ratings.

    Used for classifier training (Stage 2) when embeddings are already
    extracted and stored in embedding store.

    Args:
        embeddings: Dictionary mapping filename -> embedding array
        songs: List of Song objects
        only_rated: If True, only include rated songs
    """

    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        songs: list[Song],
        only_rated: bool = True
    ):
        super().__init__()

        # Filter songs that have embeddings and meet criteria
        self.data = []
        for song in songs:
            if only_rated and not song.is_rated:
                continue

            if song.filename not in embeddings:
                continue

            # Normalize rating to [0, 1]
            rating = song.rating / 5.0

            self.data.append({
                "embedding": embeddings[song.filename],
                "rating": rating,
                "filename": song.filename,
                "song": song
            })

        print(f"EmbeddingDataset: {len(self.data)} songs with embeddings")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Return embedding and rating.

        Returns:
            Dictionary with:
            - embedding: Embedding vector (embedding_dim,)
            - rating: Rating value in [0, 1]
            - filename: Song filename
        """
        item = self.data[idx]

        return {
            "embedding": torch.from_numpy(item["embedding"]).float(),
            "rating": torch.tensor(item["rating"], dtype=torch.float32),
            "filename": item["filename"]
        }

