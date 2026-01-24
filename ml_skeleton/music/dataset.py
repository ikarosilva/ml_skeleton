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


class SimSiamMusicDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for SimSiam self-supervised learning.

    Returns two augmented views of the same audio for contrastive learning.
    Waveforms are returned directly - mel-spectrogram conversion happens on GPU
    in the encoder for much faster training.

    Args:
        songs: List of Song objects from Clementine DB
        sample_rate: Target sample rate (Hz)
        duration: Audio duration to extract (seconds)
        crop_position: Where to extract from - "start", "center", or "end"
        normalize: Apply z-normalization to waveform
        augmentor: AudioAugmentor instance for creating augmented views
        n_mels: Number of mel frequency bins (kept for compatibility, unused)
        n_fft: FFT window size (kept for compatibility, unused)
        hop_length: Hop length for STFT (kept for compatibility, unused)
        skip_unknown_metadata: If True, skip songs with all-unknown metadata
        speech_results: Optional speech detection scores for filtering
        speech_threshold: Threshold for speech filtering
        cache_dir: Directory for caching resampled waveforms (None = no caching)
        cache_max_gb: Maximum cache size in GB (deletes oldest files when exceeded)
    """

    def __init__(
        self,
        songs: list[Song],
        sample_rate: int = 16000,
        duration: float = 60.0,
        crop_position: str = "end",
        normalize: bool = True,
        augmentor=None,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        skip_unknown_metadata: bool = False,
        speech_results: Optional[dict[str, float]] = None,
        speech_threshold: float = 0.5,
        cache_dir: Optional[str] = None,
        cache_max_gb: float = 140.0
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.crop_position = crop_position
        self.normalize = normalize
        self.augmentor = augmentor
        # These are kept for compatibility but mel-spectrogram is computed on GPU
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Waveform caching setup
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_max_bytes = int(cache_max_gb * 1024 * 1024 * 1024)  # Convert GB to bytes
        self._cache_cleanup_counter = 0  # Only check size periodically
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Include settings in cache key to invalidate on config change
            self._cache_key = f"sr{sample_rate}_dur{duration}_{crop_position}_norm{normalize}"

        # Filter songs
        self.songs, self.filter_counts = self._filter_songs(
            songs,
            speech_results,
            speech_threshold,
            skip_unknown_metadata
        )

    def _filter_songs(
        self,
        songs: list[Song],
        speech_results: Optional[dict[str, float]],
        threshold: float,
        skip_unknown_metadata: bool
    ) -> tuple[list[Song], dict[str, int]]:
        """Filter songs by various criteria."""
        filtered = []
        counts = {
            "speech": 0,
            "missing_file": 0,
            "unknown_metadata": 0
        }

        # Load exclusion lists if filtering by metadata
        if skip_unknown_metadata:
            load_exclusion_lists()

        for song in songs:
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

            # Filter by metadata validity
            if skip_unknown_metadata:
                if has_excluded_metadata(song.artist, song.album):
                    counts["unknown_metadata"] += 1
                    continue
                if not has_valid_metadata(song.artist, song.album, song.title):
                    counts["unknown_metadata"] += 1
                    continue

            filtered.append(song)

        # Print filtering statistics
        if len(filtered) < len(songs):
            removed = len(songs) - len(filtered)
            print(f"SimSiam dataset: Filtered {removed} songs:")
            if counts["speech"] > 0:
                print(f"  - {counts['speech']} speech-detected songs")
            if counts["missing_file"] > 0:
                print(f"  - {counts['missing_file']} missing files")
            if counts["unknown_metadata"] > 0:
                print(f"  - {counts['unknown_metadata']} songs with excluded metadata")

        return filtered, counts

    def __len__(self) -> int:
        return len(self.songs)

    def _get_cache_path(self, filename: str) -> Optional[Path]:
        """Get cache file path for a song."""
        if self.cache_dir is None:
            return None
        # Create safe filename from original path
        safe_name = filename.replace("/", "_").replace("\\", "_")
        return self.cache_dir / self._cache_key / f"{safe_name}.npy"

    def _get_cache_size(self) -> int:
        """Get total size of cache directory in bytes."""
        if self.cache_dir is None or not self.cache_dir.exists():
            return 0
        total = 0
        for f in self.cache_dir.rglob("*.npy"):
            try:
                total += f.stat().st_size
            except OSError:
                pass
        return total

    def _cleanup_cache(self, target_bytes: int) -> None:
        """Delete oldest cache files until size is below target.

        Args:
            target_bytes: Target cache size in bytes
        """
        if self.cache_dir is None or not self.cache_dir.exists():
            return

        # Get all cache files with their modification times
        cache_files = []
        for f in self.cache_dir.rglob("*.npy"):
            try:
                stat = f.stat()
                cache_files.append((f, stat.st_mtime, stat.st_size))
            except OSError:
                pass

        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[1])

        # Calculate current size
        current_size = sum(f[2] for f in cache_files)

        # Delete oldest files until under target
        deleted_count = 0
        deleted_bytes = 0
        for filepath, _, size in cache_files:
            if current_size <= target_bytes:
                break
            try:
                filepath.unlink()
                current_size -= size
                deleted_count += 1
                deleted_bytes += size
            except OSError:
                pass

        if deleted_count > 0:
            print(f"  Cache cleanup: deleted {deleted_count} files ({deleted_bytes / 1e9:.1f} GB)")

    def _load_from_cache(self, cache_path: Path) -> Optional[torch.Tensor]:
        """Load waveform from cache (numpy .npy format for speed)."""
        try:
            if cache_path.exists():
                # Memory-map for faster loading
                arr = np.load(cache_path, mmap_mode='r')
                return torch.from_numpy(arr.copy()).float()
        except Exception:
            pass
        return None

    def _save_to_cache(self, cache_path: Path, waveform: torch.Tensor) -> None:
        """Save waveform to cache with periodic size check."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, waveform.numpy())

            # Check cache size every 100 saves to avoid I/O overhead
            self._cache_cleanup_counter += 1
            if self._cache_cleanup_counter >= 100:
                self._cache_cleanup_counter = 0
                current_size = self._get_cache_size()
                if current_size > self.cache_max_bytes:
                    # Delete oldest files to get to 90% of max (leave some headroom)
                    target = int(self.cache_max_bytes * 0.9)
                    self._cleanup_cache(target)
        except Exception:
            pass  # Silently fail on cache write errors

    def __getitem__(self, idx: int) -> dict:
        """Load audio and return two augmented spectrogram views.

        Returns:
            Dictionary with:
            - view1: First augmented spectrogram, shape (3, n_mels, time)
            - view2: Second augmented spectrogram, shape (3, n_mels, time)
            - filename: Song filename
        """
        song = self.songs[idx]
        audio = None

        # Try loading from cache first
        cache_path = self._get_cache_path(song.filename)
        if cache_path is not None:
            audio = self._load_from_cache(cache_path)

        # Load from original file if not cached
        if audio is None:
            audio = load_audio_file(
                str(song.filepath),
                sample_rate=self.sample_rate,
                mono=True,
                duration=self.duration,
                crop_position=self.crop_position,
                normalize=self.normalize
            )

            # Fallback to zeros if loading fails
            if audio is None:
                audio = torch.zeros(int(self.sample_rate * self.duration))
            elif cache_path is not None:
                # Save to cache for next time
                self._save_to_cache(cache_path, audio)

        # Create two augmented views
        if self.augmentor is not None:
            view1_waveform = self.augmentor(audio)
            view2_waveform = self.augmentor(audio)
        else:
            # No augmentation (for validation)
            view1_waveform = audio
            view2_waveform = audio

        # Convert to mel-spectrograms
        view1_spec = self._waveform_to_spec(view1_waveform)
        view2_spec = self._waveform_to_spec(view2_waveform)

        return {
            "view1": view1_spec,
            "view2": view2_spec,
            "filename": song.filename
        }

    def _waveform_to_spec(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel-spectrogram suitable for ResNet.

        Args:
            waveform: Audio waveform, shape (num_samples,) or (1, num_samples)

        Returns:
            Mel-spectrogram repeated to 3 channels, shape (3, n_mels, time)
        """
        # Ensure 2D for mel transform
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Convert to mel-spectrogram: (1, n_mels, time)
        mel_spec = self.mel_transform(waveform)

        # Log scale (more perceptually meaningful)
        mel_spec = torch.log(mel_spec + 1e-9)

        # Normalize to [0, 1] range for better ResNet compatibility
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-9)

        # Repeat to 3 channels for ResNet (pretrained on RGB images)
        mel_spec = mel_spec.repeat(3, 1, 1)

        return mel_spec


def simsiam_collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for SimSiamMusicDataset.

    Args:
        batch: List of dictionaries from SimSiamMusicDataset.__getitem__

    Returns:
        Batched dictionary with:
        - view1: Stacked tensor (batch_size, 3, n_mels, time)
        - view2: Stacked tensor (batch_size, 3, n_mels, time)
        - filename: List of strings
    """
    view1_list = []
    view2_list = []
    filename_list = []

    for item in batch:
        view1_list.append(item["view1"])
        view2_list.append(item["view2"])
        filename_list.append(item["filename"])

    # Stack spectrograms (need to handle variable time dimension)
    # Find min time dimension and crop all to that size
    min_time = min(v.shape[2] for v in view1_list)

    view1_cropped = [v[:, :, :min_time] for v in view1_list]
    view2_cropped = [v[:, :, :min_time] for v in view2_list]

    return {
        "view1": torch.stack(view1_cropped),
        "view2": torch.stack(view2_cropped),
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
