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
from .genre_mapper import NUM_GENRES
from .audio_loader import load_audio_file, load_audio_file_with_jitter
from .metadata_utils import has_valid_metadata, has_excluded_metadata, load_exclusion_lists


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
        crop_position: Where to extract from - "start", "center", or "end"
        normalize: Apply z-normalization (zero mean, unit variance)
        speech_results: Optional speech detection scores for filtering
        speech_threshold: Threshold for speech filtering
        only_rated: If True, only include rated songs
        skip_unknown_metadata: If True, skip songs where all metadata (artist, album, title) is unknown
        use_augmentation: If True, return two different crops per song for contrastive learning
        crop_jitter: Random offset in seconds for second crop (used when use_augmentation=True)
        center_crop: DEPRECATED - use crop_position instead
    """

    def __init__(
        self,
        songs: list[Song],
        album_to_idx: dict[str, int],
        filename_to_albums: dict[str, list[str]],
        sample_rate: int = 16000,
        duration: float = 60.0,
        crop_position: str = "end",
        normalize: bool = True,
        speech_results: Optional[dict[str, float]] = None,
        speech_threshold: float = 0.5,
        only_rated: bool = True,
        skip_unknown_metadata: bool = False,
        use_augmentation: bool = False,
        crop_jitter: float = 5.0,
        noise_level: float = 0.0,
        center_crop: Optional[bool] = None
    ):
        super().__init__()
        self.album_to_idx = album_to_idx
        self.filename_to_albums = filename_to_albums
        self.sample_rate = sample_rate
        self.duration = duration

        # Handle deprecated center_crop parameter
        if center_crop is not None:
            self.crop_position = "center" if center_crop else "start"
        else:
            self.crop_position = crop_position

        self.normalize = normalize

        # Augmentation settings for contrastive learning
        self.use_augmentation = use_augmentation
        self.crop_jitter = crop_jitter
        self.noise_level = noise_level

        # Filter songs and store statistics
        self.songs, self.filter_counts = self._filter_songs(
            songs,
            speech_results,
            speech_threshold,
            only_rated,
            skip_unknown_metadata
        )

    def _filter_songs(
        self,
        songs: list[Song],
        speech_results: Optional[dict[str, float]],
        threshold: float,
        only_rated: bool,
        skip_unknown_metadata: bool
    ) -> tuple[list[Song], dict[str, int]]:
        """Filter songs by speech detection, rating status, and metadata validity.

        Returns:
            Tuple of (filtered_songs, filter_counts_dict)
        """
        filtered = []
        counts = {
            "rating": 0,
            "speech": 0,
            "missing_file": 0,
            "unknown_metadata": 0
        }

        # Load exclusion lists if filtering by metadata
        if skip_unknown_metadata:
            load_exclusion_lists()

        for song in songs:
            # Filter by rating if requested
            if only_rated and not song.is_rated:
                counts["rating"] += 1
                continue

            # Filter by speech detection
            if speech_results:
                prob = speech_results.get(song.filename, 0.0)
                if prob > threshold:
                    counts["speech"] += 1
                    continue

            # Check file exists
            if not song.filepath.exists():
                counts["missing_file"] += 1
                continue

            # Filter by metadata validity (for encoder training)
            # Uses OR logic: exclude if artist OR album is in exclusion lists
            if skip_unknown_metadata:
                # First check CSV exclusion lists (OR logic)
                if has_excluded_metadata(song.artist, song.album):
                    counts["unknown_metadata"] += 1
                    continue
                # Also check basic patterns (all fields unknown)
                if not has_valid_metadata(song.artist, song.album, song.title):
                    counts["unknown_metadata"] += 1
                    continue

            filtered.append(song)

        # Print filtering statistics
        if len(filtered) < len(songs):
            removed = len(songs) - len(filtered)
            print(f"Filtered {removed} songs:")
            if counts["rating"] > 0:
                print(f"  - {counts['rating']} unrated songs")
            if counts["speech"] > 0:
                print(f"  - {counts['speech']} speech-detected songs")
            if counts["missing_file"] > 0:
                print(f"  - {counts['missing_file']} missing files")
            if counts["unknown_metadata"] > 0:
                print(f"  - {counts['unknown_metadata']} songs with excluded metadata (artist OR album)")

        return filtered, counts

    def __len__(self) -> int:
        return len(self.songs)

    def __getitem__(self, idx: int) -> dict:
        """Load audio and return with metadata.

        Returns:
            Dictionary with:
            - audio: Waveform tensor (num_samples,) or None if augmentation enabled
            - audio_view1: First crop tensor (only when use_augmentation=True)
            - audio_view2: Second crop tensor with jitter (only when use_augmentation=True)
            - rating: Rating value in [0, 1] (or -1 if unrated)
            - albums: List of album indices this song belongs to
            - filename: Song filename
            - artist: Artist name (for metadata contrastive loss)
            - album: Album name (for metadata contrastive loss)
            - year: Release year (for metadata contrastive loss)
        """
        song = self.songs[idx]

        # Get album labels (may be multiple)
        albums = self._get_album_labels(song.filename)

        # Normalize rating to [0, 1] (Clementine uses 0-5 scale, -1 = unrated)
        rating = song.rating / 5.0 if song.is_rated else -1.0

        if self.use_augmentation:
            # Load two different crops for contrastive learning
            view1, view2 = load_audio_file_with_jitter(
                song.filename,
                sample_rate=self.sample_rate,
                mono=True,
                duration=self.duration,
                crop_position=self.crop_position,
                normalize=self.normalize,
                jitter_seconds=self.crop_jitter,
                noise_level=self.noise_level
            )

            # Fallback to zeros if loading fails
            zero_tensor = torch.zeros(int(self.sample_rate * self.duration))
            if view1 is None:
                view1 = zero_tensor
            if view2 is None:
                view2 = zero_tensor

            return {
                "audio_view1": view1,
                "audio_view2": view2,
                "rating": rating,
                "albums": albums,
                "filename": song.filename,
                "artist": song.artist,
                "album": song.album,
                "year": song.year
            }
        else:
            # Standard single crop
            audio = load_audio_file(
                song.filename,
                sample_rate=self.sample_rate,
                mono=True,
                duration=self.duration,
                crop_position=self.crop_position,
                normalize=self.normalize,
                noise_level=self.noise_level
            )

            # Fallback to zeros if loading fails
            if audio is None:
                audio = torch.zeros(int(self.sample_rate * self.duration))

            return {
                "audio": audio,
                "rating": rating,
                "albums": albums,
                "filename": song.filename,
                "artist": song.artist,
                "album": song.album,
                "year": song.year
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

    Supports both standard mode (single audio) and augmentation mode (dual views).

    Args:
        batch: List of dictionaries from MusicDataset.__getitem__

    Returns:
        Batched dictionary with:
        - audio: Stacked tensor (batch_size, num_samples) - standard mode only
        - audio_view1: First view tensor (batch_size, num_samples) - augmentation mode
        - audio_view2: Second view tensor (batch_size, num_samples) - augmentation mode
        - rating: Tensor (batch_size,)
        - albums: List of lists (variable length per sample)
        - filename: List of strings
        - artist: List of strings (for metadata contrastive loss)
        - album: List of strings (for metadata contrastive loss)
        - year: List of ints (for metadata contrastive loss)
    """
    # Check if using augmentation mode (look at first item)
    use_augmentation = "audio_view1" in batch[0]

    rating_list = []
    albums_list = []
    filename_list = []
    artist_list = []
    album_list = []
    year_list = []

    if use_augmentation:
        view1_list = []
        view2_list = []

        for item in batch:
            view1_list.append(item["audio_view1"])
            view2_list.append(item["audio_view2"])
            rating_list.append(item["rating"])
            albums_list.append(item["albums"])
            filename_list.append(item["filename"])
            artist_list.append(item["artist"])
            album_list.append(item["album"])
            year_list.append(item["year"])

        return {
            "audio_view1": torch.stack(view1_list),
            "audio_view2": torch.stack(view2_list),
            "rating": torch.tensor(rating_list, dtype=torch.float32),
            "albums": albums_list,
            "filename": filename_list,
            "artist": artist_list,
            "album": album_list,
            "year": year_list
        }
    else:
        audio_list = []

        for item in batch:
            audio_list.append(item["audio"])
            rating_list.append(item["rating"])
            albums_list.append(item["albums"])
            filename_list.append(item["filename"])
            artist_list.append(item["artist"])
            album_list.append(item["album"])
            year_list.append(item["year"])

        return {
            "audio": torch.stack(audio_list),
            "rating": torch.tensor(rating_list, dtype=torch.float32),
            "albums": albums_list,
            "filename": filename_list,
            "artist": artist_list,
            "album": album_list,
            "year": year_list
        }


class EmbeddingDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for pre-extracted embeddings with ratings.

    Used for classifier training (Stage 2) when embeddings are already
    extracted and stored in embedding store.

    Args:
        embeddings: Dictionary mapping filename -> embedding array
        songs: List of Song objects
        only_rated: If True, only include rated songs
        classification_mode: "regression" (0-1 continuous) or "binary" (0/1 labels)
        binary_positive_threshold: Rating >= this is positive class (default: 4)
        binary_negative_threshold: Rating <= this is negative class (default: 2)
        use_genre: If True, add 7-dim genre multi-hot per sample (from metadata or centroid imputation)
        genre_centroids: (NUM_GENRES, D) for imputing missing genre; required when use_genre and many songs lack genre
        genre_impute_top_k: Top-k closest centroids per chunk for imputation (default 2 for mix-ins)
        genre_impute_min_votes: Min votes to set a category in imputed multi-hot (default 1)
        binary_include_middle: If True, use three-way labels with middle band (default False)
        replace_embeddings_with_noise: If True, return N(0,1) noise instead of real embeddings (same shape); for debugging.
        noise_seed: RNG seed when replace_embeddings_with_noise is True (default 42).
    """

    def __init__(
        self,
        embeddings: dict[str, np.ndarray],
        songs: list[Song],
        only_rated: bool = True,
        classification_mode: str = "regression",
        binary_positive_threshold: float = 4.0,
        binary_negative_threshold: float = 2.0,
        use_genre: bool = False,
        genre_centroids: Optional[np.ndarray] = None,
        genre_impute_top_k: int = 2,
        genre_impute_min_votes: int = 1,
        binary_include_middle: bool = False,
        replace_embeddings_with_noise: bool = False,
        noise_seed: Optional[int] = None,
    ):
        super().__init__()

        self.classification_mode = classification_mode
        self._binary_include_middle = binary_include_middle
        self.use_genre = use_genre
        self.replace_embeddings_with_noise = replace_embeddings_with_noise
        self._noise_seed = noise_seed if noise_seed is not None else 42
        self._genre_centroids = genre_centroids
        self._genre_impute_top_k = genre_impute_top_k
        self._genre_impute_min_votes = genre_impute_min_votes

        # Filter songs that have embeddings and meet criteria
        self.data = []
        excluded_ambiguous = 0

        if use_genre:
            from .genre_centroids import get_genre_features

        for song in songs:
            if only_rated and not song.is_rated:
                continue

            if song.filename not in embeddings:
                continue

            # Handle classification mode
            if classification_mode == "binary":
                if not self._binary_include_middle:
                    # Simple binary: 0 = rating < positive_threshold, 1 = rating >= positive_threshold (no exclusions)
                    label = 1.0 if song.rating >= binary_positive_threshold else 0.0
                    is_middle = False
                else:
                    # With middle band: positive (>=4), negative (<=negative_threshold), middle with penalty toward 0.5
                    if song.rating >= binary_positive_threshold:
                        label = 1.0
                        is_middle = False
                    elif song.rating <= binary_negative_threshold:
                        label = 0.0
                        is_middle = False
                    else:
                        label = 0.5
                        is_middle = True
            else:
                # Regression: normalize rating to [0, 1]
                label = song.rating / 5.0
                is_middle = False  # unused in regression

            # Original 1-5 rating normalized to [0, 1] (for HPO metrics: MSE/correlation with predicted prob)
            rating_continuous = song.rating / 5.0

            item = {
                "embedding": embeddings[song.filename],
                "rating": label,
                "rating_continuous": rating_continuous,
                "filename": song.filename,
            }
            if classification_mode == "binary":
                item["is_middle"] = is_middle
            if use_genre:
                item["genre"] = get_genre_features(
                    song,
                    embeddings[song.filename],
                    genre_centroids,
                    top_k=genre_impute_top_k,
                    min_votes=genre_impute_min_votes,
                )
            self.data.append(item)

        if classification_mode == "binary":
            pos_count = sum(1 for d in self.data if d["rating"] == 1.0)
            neg_count = sum(1 for d in self.data if d["rating"] == 0.0)
            mid_count = sum(1 for d in self.data if d.get("is_middle"))
            print(f"EmbeddingDataset (binary): {len(self.data)} songs")
            if self._binary_include_middle:
                print(f"  Positive (rating >= {binary_positive_threshold}): {pos_count}")
                print(f"  Negative (rating <= {binary_negative_threshold}): {neg_count}")
                print(f"  Middle (|pred-0.5| penalty): {mid_count}")
            else:
                print(f"  Positive (rating >= {binary_positive_threshold}): {pos_count}")
                print(f"  Negative (rating < {binary_positive_threshold}): {neg_count}")
            if excluded_ambiguous:
                print(f"  Excluded (ambiguous): {excluded_ambiguous}")
        else:
            print(f"EmbeddingDataset: {len(self.data)} songs with embeddings")
        if self.replace_embeddings_with_noise:
            print("  Replace embeddings with normalized noise (N(0,1)) - labels unchanged")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Return embedding, rating, and optional genre.

        Returns:
            Dictionary with:
            - embedding: Embedding vector (embedding_dim,) or (4, embedding_dim)
            - rating: Rating value in [0, 1]
            - filename: Song filename
            - genre: (NUM_GENRES,) multi-hot, only if use_genre=True
        """
        item = self.data[idx]
        raw = np.asarray(item["embedding"], dtype=np.float32)
        if self.replace_embeddings_with_noise:
            # Normalized noise N(0,1) same shape; deterministic per sample for reproducibility
            rng = np.random.default_rng(self._noise_seed + idx)
            emb = rng.standard_normal(raw.shape).astype(np.float32)
        else:
            emb = raw
        out = {
            "embedding": torch.from_numpy(emb).float(),
            "rating": torch.tensor(item["rating"], dtype=torch.float32),
            "filename": item["filename"],
        }
        if "is_middle" in item:
            out["is_middle"] = torch.tensor(item["is_middle"], dtype=torch.bool)
        if "rating_continuous" in item:
            out["rating_continuous"] = torch.tensor(item["rating_continuous"], dtype=torch.float32)
        if self.use_genre:
            out["genre"] = torch.from_numpy(np.asarray(item["genre"], dtype=np.float32))
        return out

    def get_all_filenames(self) -> list[str]:
        """Get all filenames in the dataset.

        Returns:
            List of filenames
        """
        return [d["filename"] for d in self.data]

    def get_all_ratings(self) -> list[float]:
        """Get all ratings in the dataset.

        Returns:
            List of rating values
        """
        return [d["rating"] for d in self.data]

    def get_file_ratings_dict(self) -> dict[str, int]:
        """Get dict mapping filename to binary rating (0 or 1).

        Returns:
            Dict of {filename: binary_rating}
        """
        return {d["filename"]: int(d["rating"]) for d in self.data}

    def subset_by_filenames(self, filenames: set[str]) -> "EmbeddingDataset":
        """Create a subset containing only specified filenames.

        Args:
            filenames: Set of filenames to include

        Returns:
            New EmbeddingDataset with only the specified files
        """
        # Create a new dataset with filtered data
        subset = EmbeddingDataset.__new__(EmbeddingDataset)
        subset.classification_mode = self.classification_mode
        subset._binary_include_middle = self._binary_include_middle
        subset.use_genre = self.use_genre
        subset._genre_centroids = self._genre_centroids
        subset._genre_impute_top_k = self._genre_impute_top_k
        subset._genre_impute_min_votes = self._genre_impute_min_votes
        subset.replace_embeddings_with_noise = self.replace_embeddings_with_noise
        subset._noise_seed = self._noise_seed
        subset.data = [d for d in self.data if d["filename"] in filenames]
        return subset

    def split_by_filenames(
        self,
        train_files: list[str],
        val_files: list[str]
    ) -> tuple["EmbeddingDataset", "EmbeddingDataset"]:
        """Split dataset by filename lists.

        Args:
            train_files: List of filenames for training
            val_files: List of filenames for validation

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        train_set = set(train_files)
        val_set = set(val_files)

        return (
            self.subset_by_filenames(train_set),
            self.subset_by_filenames(val_set)
        )
