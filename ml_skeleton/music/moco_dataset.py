"""MoCo training dataset with chunk caching and augmentations.

Dataset for MoCo v2 + Genre BCE training:
- Loads pre-cached 30s chunks from .npy files
- Applies dynamic augmentations (time-crop, gain, noise)
- Creates positive pairs from same-song or same-album chunks
- Provides multi-label genre targets

Augmentation pipeline (applied to 30s cached chunks):
    Always:
    - Time-crop: Random 5-15s window from 30s chunk
    - Gain: ±2dB random volume adjustment
    50% probability:
    - Gaussian noise: SNR 25-35dB
    - Audio mixup: α=0.1 with same-genre chunk
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

from ml_skeleton.music.clementine_db import Song
from ml_skeleton.music.chunk_cache import (
    load_cached_chunk,
    get_cached_songs,
    DEFAULT_CACHE_DIR,
    DEFAULT_NUM_CHUNKS,
    DEFAULT_SAMPLE_RATE
)
from ml_skeleton.music.genre_mapper import (
    genre_to_multilabel,
    parse_genre_string,
    NUM_GENRES
)


class AudioAugmentor:
    """Audio augmentation pipeline for MoCo training.

    Applies dynamic augmentations to cached waveform chunks.
    All operations work on raw audio before CQT transform.

    Args:
        crop_duration_range: (min, max) crop duration in seconds
        gain_db_range: (min, max) gain adjustment in dB
        noise_prob: Probability of adding Gaussian noise
        noise_snr_range: (min, max) SNR in dB for noise
        mixup_prob: Probability of audio mixup
        mixup_alpha: Beta distribution alpha for mixup
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        crop_duration_range: Tuple[float, float] = (5.0, 15.0),
        gain_db_range: Tuple[float, float] = (-2.0, 2.0),
        noise_prob: float = 0.5,
        noise_snr_range: Tuple[float, float] = (25.0, 35.0),
        mixup_prob: float = 0.5,
        mixup_alpha: float = 0.1
    ):
        self.sample_rate = sample_rate
        self.crop_duration_range = crop_duration_range
        self.gain_db_range = gain_db_range
        self.noise_prob = noise_prob
        self.noise_snr_range = noise_snr_range
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha

    def random_crop(
        self,
        waveform: torch.Tensor,
        target_duration: Optional[float] = None
    ) -> torch.Tensor:
        """Extract random time crop from waveform.

        Args:
            waveform: Audio tensor of shape (T,)
            target_duration: Target duration in seconds (random if None)

        Returns:
            Cropped waveform
        """
        if target_duration is None:
            target_duration = random.uniform(*self.crop_duration_range)

        target_length = int(target_duration * self.sample_rate)
        current_length = waveform.shape[0]

        if current_length <= target_length:
            # Pad if too short
            padding = target_length - current_length
            return torch.nn.functional.pad(waveform, (0, padding))

        # Random start position
        max_start = current_length - target_length
        start = random.randint(0, max_start)
        return waveform[start:start + target_length]

    def apply_gain(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random gain adjustment.

        Args:
            waveform: Audio tensor

        Returns:
            Gain-adjusted waveform
        """
        gain_db = random.uniform(*self.gain_db_range)
        gain_linear = 10 ** (gain_db / 20)
        return waveform * gain_linear

    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise at random SNR.

        Args:
            waveform: Audio tensor

        Returns:
            Noisy waveform
        """
        if random.random() > self.noise_prob:
            return waveform

        snr_db = random.uniform(*self.noise_snr_range)

        # Calculate signal power
        signal_power = (waveform ** 2).mean()
        if signal_power < 1e-10:
            return waveform

        # Calculate noise power for target SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate and add noise
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

    def mixup(
        self,
        waveform: torch.Tensor,
        other_waveform: torch.Tensor
    ) -> torch.Tensor:
        """Apply audio mixup with another waveform.

        Args:
            waveform: Primary audio tensor
            other_waveform: Secondary audio tensor for mixing

        Returns:
            Mixed waveform
        """
        if random.random() > self.mixup_prob:
            return waveform

        # Sample mixing coefficient from Beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        lam = max(lam, 1 - lam)  # Ensure primary dominates

        # Match lengths
        min_len = min(waveform.shape[0], other_waveform.shape[0])
        waveform = waveform[:min_len]
        other_waveform = other_waveform[:min_len]

        return lam * waveform + (1 - lam) * other_waveform

    def __call__(
        self,
        waveform: torch.Tensor,
        mixup_waveform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply full augmentation pipeline.

        Args:
            waveform: Input audio tensor of shape (T,)
            mixup_waveform: Optional waveform for mixup

        Returns:
            Augmented waveform
        """
        # Always apply: crop and gain
        waveform = self.random_crop(waveform)
        waveform = self.apply_gain(waveform)

        # Probabilistic: noise
        waveform = self.add_noise(waveform)

        # Probabilistic: mixup (if provided)
        if mixup_waveform is not None:
            mixup_waveform = self.random_crop(mixup_waveform)
            waveform = self.mixup(waveform, mixup_waveform)

        return waveform


class MoCoDataset(Dataset):
    """Dataset for MoCo v2 training with cached chunks.

    Returns positive pairs and genre labels for each sample:
    - query: Augmented chunk from song
    - key: Different augmented chunk from same song OR same album
    - genre: Multi-hot genre label tensor

    Positive pair strategy:
    1. Same-song: Different chunks from same song (default)
    2. Same-album: Chunks from different songs in same album (30% prob)

    Args:
        songs: List of Song objects
        cache_dir: Directory with cached .npy chunks
        num_chunks: Number of chunks per song
        sample_rate: Audio sample rate
        augmentor: AudioAugmentor instance (created if None)
        same_album_prob: Probability of using same-album positive
        crop_duration: Fixed crop duration (random if None)
    """

    def __init__(
        self,
        songs: List[Song],
        cache_dir: str = DEFAULT_CACHE_DIR,
        num_chunks: int = DEFAULT_NUM_CHUNKS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        augmentor: Optional[AudioAugmentor] = None,
        same_album_prob: float = 0.3,
        crop_duration: Optional[float] = None
    ):
        self.cache_dir = cache_dir
        self.num_chunks = num_chunks
        self.sample_rate = sample_rate
        self.same_album_prob = same_album_prob
        self.crop_duration = crop_duration

        # Filter to songs with complete cache
        self.songs = get_cached_songs(songs, cache_dir, num_chunks)

        if len(self.songs) < len(songs):
            print(f"MoCoDataset: {len(self.songs)}/{len(songs)} songs have complete cache")

        # Create augmentor
        self.augmentor = augmentor or AudioAugmentor(sample_rate=sample_rate)

        # Build album index for same-album positives
        self._build_album_index()

        # Build genre index for mixup
        self._build_genre_index()

    def _build_album_index(self):
        """Build index of songs by album for same-album sampling."""
        self.album_to_songs: Dict[str, List[int]] = defaultdict(list)

        for idx, song in enumerate(self.songs):
            # Use artist|||album as key to handle same album name from different artists
            album_key = f"{song.artist}|||{song.album}"
            self.album_to_songs[album_key].append(idx)

        # Filter to albums with multiple songs
        self.multi_song_albums = {
            k: v for k, v in self.album_to_songs.items()
            if len(v) > 1
        }

    def _build_genre_index(self):
        """Build index of songs by genre for mixup sampling."""
        self.genre_to_songs: Dict[str, List[int]] = defaultdict(list)

        for idx, song in enumerate(self.songs):
            categories = parse_genre_string(song.genre)
            for cat in categories:
                self.genre_to_songs[cat].append(idx)

    def _load_chunk(self, song_idx: int, chunk_idx: int) -> Optional[torch.Tensor]:
        """Load a specific cached chunk."""
        song = self.songs[song_idx]
        return load_cached_chunk(song.rowid, chunk_idx, self.cache_dir)

    def _get_mixup_chunk(self, song_idx: int) -> Optional[torch.Tensor]:
        """Get a chunk from the same genre for mixup."""
        song = self.songs[song_idx]
        categories = parse_genre_string(song.genre)

        if not categories:
            return None

        # Pick random genre category
        category = random.choice(categories)
        candidates = self.genre_to_songs.get(category, [])

        if len(candidates) < 2:
            return None

        # Pick random different song from same genre
        other_idx = random.choice([i for i in candidates if i != song_idx])
        chunk_idx = random.randint(0, self.num_chunks - 1)

        return self._load_chunk(other_idx, chunk_idx)

    def _get_same_album_positive(self, song_idx: int) -> Optional[Tuple[int, int]]:
        """Get a chunk from a different song in the same album.

        Returns:
            Tuple of (song_idx, chunk_idx) or None if not possible
        """
        song = self.songs[song_idx]
        album_key = f"{song.artist}|||{song.album}"

        if album_key not in self.multi_song_albums:
            return None

        candidates = self.multi_song_albums[album_key]
        other_songs = [i for i in candidates if i != song_idx]

        if not other_songs:
            return None

        other_idx = random.choice(other_songs)
        chunk_idx = random.randint(0, self.num_chunks - 1)

        return (other_idx, chunk_idx)

    def __len__(self) -> int:
        return len(self.songs)

    def __getitem__(self, idx: int) -> dict:
        """Get a training sample.

        Returns:
            Dictionary with:
            - query: Augmented query audio (T,)
            - key: Augmented key audio (T,)
            - genre: Multi-hot genre labels (num_genres,)
            - song_id: Song row ID
        """
        song = self.songs[idx]

        # Load query chunk (random chunk from this song)
        query_chunk_idx = random.randint(0, self.num_chunks - 1)
        query_waveform = self._load_chunk(idx, query_chunk_idx)

        if query_waveform is None:
            # Fallback to zeros if cache corrupted
            target_len = int(self.crop_duration or 10.0) * self.sample_rate
            query_waveform = torch.zeros(target_len)

        # Decide positive pair strategy
        use_same_album = (
            random.random() < self.same_album_prob and
            self.multi_song_albums
        )

        if use_same_album:
            album_pos = self._get_same_album_positive(idx)
            if album_pos:
                key_song_idx, key_chunk_idx = album_pos
                key_waveform = self._load_chunk(key_song_idx, key_chunk_idx)
            else:
                use_same_album = False

        if not use_same_album:
            # Same-song positive: different chunk from same song
            available_chunks = [i for i in range(self.num_chunks) if i != query_chunk_idx]
            key_chunk_idx = random.choice(available_chunks) if available_chunks else query_chunk_idx
            key_waveform = self._load_chunk(idx, key_chunk_idx)

        if key_waveform is None:
            key_waveform = query_waveform.clone()

        # Get mixup chunk (same genre)
        mixup_waveform = self._get_mixup_chunk(idx)

        # Apply augmentations
        query_aug = self.augmentor(query_waveform, mixup_waveform)
        key_aug = self.augmentor(key_waveform, mixup_waveform)

        # Genre labels
        genre_labels = genre_to_multilabel(song.genre)

        return {
            "query": query_aug,
            "key": key_aug,
            "genre": genre_labels,
            "song_id": song.rowid
        }


class MoCoCollator:
    """Collator for MoCo dataset batches.

    Pads/truncates audio to consistent length and stacks tensors.

    Args:
        target_length: Target audio length in samples (computed from duration if None)
        sample_rate: Audio sample rate
        crop_duration: Crop duration in seconds
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        crop_duration: float = 10.0
    ):
        self.sample_rate = sample_rate
        self.target_length = int(crop_duration * sample_rate)

    def _pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Ensure waveform is exactly target_length."""
        current_length = waveform.shape[0]

        if current_length < self.target_length:
            padding = self.target_length - current_length
            return torch.nn.functional.pad(waveform, (0, padding))
        elif current_length > self.target_length:
            return waveform[:self.target_length]
        return waveform

    def __call__(self, batch: List[dict]) -> dict:
        """Collate batch of samples.

        Args:
            batch: List of sample dicts from MoCoDataset

        Returns:
            Batched dictionary with:
            - query: (B, T) stacked query audio
            - key: (B, T) stacked key audio
            - genre: (B, num_genres) stacked genre labels
            - song_ids: List of song IDs
        """
        queries = torch.stack([
            self._pad_or_truncate(sample["query"]) for sample in batch
        ])
        keys = torch.stack([
            self._pad_or_truncate(sample["key"]) for sample in batch
        ])
        genres = torch.stack([sample["genre"] for sample in batch])
        song_ids = [sample["song_id"] for sample in batch]

        return {
            "query": queries,
            "key": keys,
            "genre": genres,
            "song_ids": song_ids
        }


def create_moco_dataloader(
    songs: List[Song],
    cache_dir: str = DEFAULT_CACHE_DIR,
    batch_size: int = 128,
    num_workers: int = 4,
    crop_duration: float = 10.0,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """Create DataLoader for MoCo training.

    Args:
        songs: List of Song objects
        cache_dir: Cache directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        crop_duration: Crop duration in seconds
        **dataset_kwargs: Additional args for MoCoDataset

    Returns:
        PyTorch DataLoader
    """
    dataset = MoCoDataset(
        songs=songs,
        cache_dir=cache_dir,
        crop_duration=crop_duration,
        **dataset_kwargs
    )

    collator = MoCoCollator(crop_duration=crop_duration)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True  # MoCo needs consistent batch size for queue
    )
