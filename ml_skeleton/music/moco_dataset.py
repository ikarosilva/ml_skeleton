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
from ml_skeleton.music.chunk_fingerprinter import fingerprint_to_bits, CHROMAPRINT_BITS
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
    2. Same-album: Chunks from different songs in same album (when same_album_prob > 0)
    3. Far-apart chunks: With far_chunk_prob, key chunk is at least min_chunk_distance away
       from query chunk (e.g. verse vs chorus) to encourage section-invariant representations.

    Args:
        songs: List of Song objects
        cache_dir: Directory with cached .npy chunks
        num_chunks: Number of chunks per song
        sample_rate: Audio sample rate
        augmentor: AudioAugmentor instance (created if None)
        same_album_prob: Probability of using same-album positive (0 = disabled)
        far_chunk_prob: Probability of forcing key chunk to be far from query (e.g. 0.35 = 35%)
        min_chunk_distance: Min |key_idx - query_idx| when far_chunk_prob triggers (e.g. 4 for 8 chunks)
        crop_duration: Fixed crop duration (random if None)
        fp_db: Optional FingerprintDB for chromaprint regularization (loads full-file chromaprint per song)
        chromaprint_chunk_idx: Chunk index to load chromaprint from (0 = full-file from CLI fingerprint)
    """

    def __init__(
        self,
        songs: List[Song],
        cache_dir: str = DEFAULT_CACHE_DIR,
        num_chunks: int = DEFAULT_NUM_CHUNKS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        augmentor: Optional[AudioAugmentor] = None,
        same_album_prob: float = 0.0,
        far_chunk_prob: float = 0.0,
        min_chunk_distance: int = 2,
        crop_duration: Optional[float] = None,
        fp_db: Optional[object] = None,
        chromaprint_chunk_idx: int = 0,
        preload_chromaprint: bool = True,
    ):
        self.cache_dir = cache_dir
        self.num_chunks = num_chunks
        self.sample_rate = sample_rate
        self.same_album_prob = same_album_prob
        self.far_chunk_prob = far_chunk_prob
        self.min_chunk_distance = min_chunk_distance
        self.crop_duration = crop_duration
        self.fp_db = fp_db
        self.chromaprint_chunk_idx = chromaprint_chunk_idx

        # Filter to songs with complete cache
        self.songs = get_cached_songs(songs, cache_dir, num_chunks)

        if len(self.songs) < len(songs):
            print(f"MoCoDataset: {len(self.songs)}/{len(songs)} songs have complete cache")

        # Create augmentor
        self.augmentor = augmentor or AudioAugmentor(sample_rate=sample_rate)

        # Chromaprint: preload into memory so DataLoader workers never touch the DB (avoids locks, one connection).
        self._chromaprint_cache: Dict[int, Optional[np.ndarray]] = {}
        if fp_db is not None:
            if preload_chromaprint and self.songs:
                self._preload_chromaprint_cache(fp_db)
            else:
                by_chunk = fp_db.get_fingerprint_count_by_chunk()
                n_at_chunk = by_chunk.get(chromaprint_chunk_idx, 0)
                print(
                    f"MoCoDataset: chromaprint lazy-loaded (chunk {chromaprint_chunk_idx}, DB: {getattr(fp_db, 'db_path', '?')}); "
                    f"{n_at_chunk:,} fingerprints at this chunk in DB"
                )

        # Build album index for same-album positives
        self._build_album_index()

        # Build genre index for mixup
        self._build_genre_index()

    def _preload_chromaprint_cache(self, fp_db: object) -> None:
        """Load all chromaprint bits for self.songs in one batch; fill _chromaprint_cache. Workers then never touch the DB."""
        song_ids = [s.rowid for s in self.songs]
        batch = fp_db.get_fingerprints_batch(song_ids, self.chromaprint_chunk_idx)
        loaded = 0
        for song_idx, song in enumerate(self.songs):
            fp_obj = batch.get(song.rowid)
            if fp_obj is None:
                self._chromaprint_cache[song_idx] = None
                continue
            if getattr(fp_obj, "bits", None) is not None:
                arr = np.unpackbits(np.frombuffer(fp_obj.bits, dtype=np.uint8))
                bits = arr[:CHROMAPRINT_BITS].astype(np.float32)
                if len(bits) < CHROMAPRINT_BITS:
                    bits = np.pad(bits, (0, CHROMAPRINT_BITS - len(bits)), constant_values=0)
                self._chromaprint_cache[song_idx] = bits
                loaded += 1
            elif fp_obj.fingerprint:
                fp_str = (
                    fp_obj.fingerprint.decode("utf-8")
                    if isinstance(fp_obj.fingerprint, bytes)
                    else fp_obj.fingerprint
                )
                bits = fingerprint_to_bits(fp_str)
                self._chromaprint_cache[song_idx] = bits
                if bits is not None:
                    loaded += 1
            else:
                self._chromaprint_cache[song_idx] = None
        db_path = getattr(fp_db, "db_path", "?")
        print(
            f"MoCoDataset: chromaprint preloaded into memory (chunk {self.chromaprint_chunk_idx}, {loaded}/{len(self.songs)} from {db_path})"
        )

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

    def _get_chromaprint_bits(self, song_idx: int) -> Optional[np.ndarray]:
        """Lazy-load chromaprint bits. Uses precomputed bits from DB when present (no C decode)."""
        if self.fp_db is None:
            return None
        if song_idx in self._chromaprint_cache:
            return self._chromaprint_cache[song_idx]
        song = self.songs[song_idx]
        fp_obj = self.fp_db.get_fingerprint(song.rowid, self.chromaprint_chunk_idx)
        if fp_obj is None:
            self._chromaprint_cache[song_idx] = None
            return None
        # Precomputed bits in DB (stored at fingerprint time) → no chromaprint C decode
        if getattr(fp_obj, "bits", None) is not None:
            arr = np.unpackbits(np.frombuffer(fp_obj.bits, dtype=np.uint8))
            bits = arr[:CHROMAPRINT_BITS].astype(np.float32)
            if len(bits) < CHROMAPRINT_BITS:
                bits = np.pad(bits, (0, CHROMAPRINT_BITS - len(bits)), constant_values=0)
            self._chromaprint_cache[song_idx] = bits
            return self._chromaprint_cache[song_idx]
        # Fallback: decode from base64 (older DB rows without bits column)
        if fp_obj.fingerprint:
            fp_str = (
                fp_obj.fingerprint.decode("utf-8")
                if isinstance(fp_obj.fingerprint, bytes)
                else fp_obj.fingerprint
            )
            bits = fingerprint_to_bits(fp_str)
            self._chromaprint_cache[song_idx] = bits
        else:
            self._chromaprint_cache[song_idx] = None
        return self._chromaprint_cache[song_idx]

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
            # Optionally force far-apart chunks (e.g. verse vs chorus) 20–50% of the time
            if (
                available_chunks
                and self.far_chunk_prob > 0
                and random.random() < self.far_chunk_prob
            ):
                far_chunks = [
                    i for i in available_chunks
                    if abs(i - query_chunk_idx) >= self.min_chunk_distance
                ]
                if far_chunks:
                    available_chunks = far_chunks
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

        out = {
            "query": query_aug,
            "key": key_aug,
            "genre": genre_labels,
            "song_id": song.rowid,
            "filename": song.filename
        }
        if self.fp_db is not None:
            bits = self._get_chromaprint_bits(idx)
            if bits is not None:
                out["chromaprint"] = torch.from_numpy(bits)
                out["chromaprint_valid"] = True
            else:
                out["chromaprint"] = torch.zeros(CHROMAPRINT_BITS, dtype=torch.float32)
                out["chromaprint_valid"] = False
        return out


class ChunkExtractionDataset(Dataset):
    """Dataset for extracting embeddings for all num_chunks per song (no augmentation)."""

    def __init__(
        self,
        songs: List[Song],
        cache_dir: str = DEFAULT_CACHE_DIR,
        num_chunks: int = DEFAULT_NUM_CHUNKS,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        crop_duration: float = 10.0
    ):
        self.songs = get_cached_songs(songs, cache_dir, num_chunks)
        self.cache_dir = cache_dir
        self.num_chunks = num_chunks
        self.sample_rate = sample_rate
        self.crop_length = int(crop_duration * sample_rate)

    def __len__(self) -> int:
        return len(self.songs) * self.num_chunks

    def __getitem__(self, idx: int) -> dict:
        song_idx = idx // self.num_chunks
        chunk_idx = idx % self.num_chunks
        song = self.songs[song_idx]
        waveform = load_cached_chunk(song.rowid, chunk_idx, self.cache_dir)
        if waveform is None:
            waveform = torch.zeros(self.crop_length)
        elif waveform.shape[0] > self.crop_length:
            waveform = waveform[:self.crop_length]
        elif waveform.shape[0] < self.crop_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.crop_length - waveform.shape[0]))
        return {
            "query": waveform.float(),
            "filename": song.filename,
            "chunk_idx": chunk_idx
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
            batch: List of sample dicts from MoCoDataset or ChunkExtractionDataset

        Returns:
            Batched dictionary with query, filename; if ChunkExtractionDataset also chunk_idx.
            MoCoDataset batches also have key, genre, song_ids.
        """
        queries = torch.stack([
            self._pad_or_truncate(sample["query"]) for sample in batch
        ])
        filenames = [sample["filename"] for sample in batch]
        if "chunk_idx" in batch[0]:
            return {
                "query": queries,
                "filename": filenames,
                "chunk_idx": [sample["chunk_idx"] for sample in batch]
            }
        keys = torch.stack([
            self._pad_or_truncate(sample["key"]) for sample in batch
        ])
        genres = torch.stack([sample["genre"] for sample in batch])
        song_ids = [sample["song_id"] for sample in batch]
        result = {
            "query": queries,
            "key": keys,
            "genre": genres,
            "song_ids": song_ids,
            "filename": filenames
        }
        if "chromaprint" in batch[0]:
            result["chromaprint"] = torch.stack([s["chromaprint"] for s in batch])
            result["chromaprint_valid"] = torch.tensor([s["chromaprint_valid"] for s in batch], dtype=torch.bool)
        return result


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
