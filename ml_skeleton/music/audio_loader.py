"""Audio loading utilities with multiprocessing support.

CRITICAL: All operations are READ-ONLY. Audio files are NEVER modified.

Performance optimizations:
- Resampling uses cached torchaudio.transforms.Resample objects (LRU cache)
- PyTorch DataLoader workers handle parallel loading automatically
- prefetch_factor enables pre-loading multiple batches per worker
- persistent_workers keeps workers alive between epochs (reduces startup overhead)
- Librosa fallback for files that torchaudio can't handle

Example usage with DataLoader:
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,           # 4 parallel workers
        prefetch_factor=4,       # Pre-load 4 batches per worker
        persistent_workers=True, # Keep workers alive
        pin_memory=True          # Speed up GPU transfer
    )
"""

import os
import warnings

# Suppress ffmpeg mjpeg warnings (from embedded album art)
os.environ.setdefault('AV_LOG_FORCE_NOCOLOR', '1')
os.environ.setdefault('FFREPORT', 'level=error')

import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from typing import Optional
import multiprocessing
from functools import partial, lru_cache


def get_default_workers() -> int:
    """Get default number of workers (80% of CPU cores).

    Returns:
        Number of worker processes (minimum 1)
    """
    cpu_count = multiprocessing.cpu_count()
    return max(1, int(cpu_count * 0.8))


def _load_with_librosa(
    filepath: str,
    sample_rate: int,
    duration: Optional[float],
    crop_position: str,
    mono: bool
) -> Optional[tuple[torch.Tensor, int]]:
    """Fallback audio loader using librosa (more robust but slower).

    Args:
        filepath: Path to audio file
        sample_rate: Target sample rate (Hz)
        duration: Duration in seconds to extract (None = full file)
        crop_position: Where to extract from - "start", "center", or "end"
        mono: Convert to mono if True

    Returns:
        Tuple of (waveform tensor, sample_rate) or None if loading fails
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Get file duration first
            file_duration = librosa.get_duration(path=filepath)

            # Calculate offset based on crop position
            offset = 0.0
            if duration is not None and file_duration > duration:
                if crop_position == "center":
                    offset = (file_duration - duration) / 2
                elif crop_position == "end":
                    offset = file_duration - duration
                # "start" keeps offset = 0

            # Load with librosa (automatically resamples)
            y, sr = librosa.load(
                filepath,
                sr=sample_rate,
                mono=mono,
                offset=offset,
                duration=duration
            )

            # Convert to torch tensor
            waveform = torch.from_numpy(y.astype(np.float32))
            return waveform, sr

    except Exception:
        return None


@lru_cache(maxsize=8)
def get_resampler(orig_freq: int, new_freq: int) -> torchaudio.transforms.Resample:
    """Get cached resampler for given frequency pair.

    Caching resamplers improves performance when resampling many files
    with the same source/target sample rates.

    Args:
        orig_freq: Original sample rate (Hz)
        new_freq: Target sample rate (Hz)

    Returns:
        Cached Resample transform
    """
    return torchaudio.transforms.Resample(orig_freq, new_freq)


def load_audio_file(
    filepath: str,
    sample_rate: int = 16000,
    mono: bool = True,
    duration: Optional[float] = 60.0,
    crop_position: str = "end",
    normalize: bool = True,
    max_duration: float = 900.0,
    center_crop: Optional[bool] = None,
    noise_level: float = 0.0
) -> Optional[torch.Tensor]:
    """Load single audio file (READ-ONLY).

    Args:
        filepath: Path to audio file
        sample_rate: Target sample rate (Hz)
        mono: Convert to mono if True
        duration: Duration in seconds to extract (None = full song)
        crop_position: Where to extract from - "start", "center", or "end" (default: "end")
        normalize: Apply z-normalization (zero mean, unit variance) if True
        max_duration: Skip files longer than this (default: 900s = 15 minutes)
        center_crop: DEPRECATED - use crop_position instead (for backward compatibility)
        noise_level: Standard deviation of Gaussian noise to add (0.0 = no noise)

    Returns:
        Audio tensor of shape (num_samples,) if mono, or None if file invalid/too long

    Note:
        - This function is READ-ONLY - audio files are NEVER modified
        - Uses torchaudio.load() with librosa fallback for problematic files
        - Long files (>max_duration) are skipped to filter out live albums, DJ sets
        - Z-normalization: (audio - mean) / std, prevents volume differences from affecting features
    """
    # Handle deprecated center_crop parameter
    if center_crop is not None:
        crop_position = "center" if center_crop else "start"

    filepath = Path(filepath)

    # Check file exists
    if not filepath.exists():
        return None

    waveform = None
    sr = sample_rate
    use_librosa = False

    # Try torchaudio first (faster)
    try:
        # Get audio info first (faster than loading)
        info = torchaudio.info(str(filepath))
        file_duration = info.num_frames / info.sample_rate

        # Skip very long files (live albums, DJ sets, podcasts)
        if file_duration > max_duration:
            return None

        # Load audio (READ-ONLY operation)
        if duration is None:
            # Load full file
            waveform, sr = torchaudio.load(str(filepath))
        else:
            # Calculate frame offset based on crop position
            target_frames = int(duration * info.sample_rate)

            if file_duration > duration:
                # File is longer than target duration - extract segment
                if crop_position == "center":
                    # Extract from center
                    start_frame = (info.num_frames - target_frames) // 2
                elif crop_position == "end":
                    # Extract from end
                    start_frame = info.num_frames - target_frames
                else:  # "start" or default
                    # Extract from beginning
                    start_frame = 0

                waveform, sr = torchaudio.load(
                    str(filepath),
                    frame_offset=start_frame,
                    num_frames=target_frames
                )
            else:
                # File is shorter than target duration - load full file
                frames_to_load = info.num_frames
                waveform, sr = torchaudio.load(
                    str(filepath),
                    frame_offset=0,
                    num_frames=frames_to_load
                )

        # Convert to mono if requested
        if mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=False)
        elif mono:
            waveform = waveform.squeeze(0)

        # Resample if needed (uses cached resampler for efficiency)
        if sr != sample_rate:
            resampler = get_resampler(sr, sample_rate)
            waveform = resampler(waveform)

    except Exception:
        # Fallback to librosa (more robust for problematic files)
        use_librosa = True

    # Librosa fallback
    if use_librosa:
        result = _load_with_librosa(
            str(filepath), sample_rate, duration, crop_position, mono
        )
        if result is None:
            print(f"Warning: Could not load {filepath}: Failed to decode audio.")
            return None
        waveform, sr = result

    # Pad or truncate to exact duration if specified
    if duration is not None:
        target_length = int(duration * sample_rate)
        current_length = waveform.shape[-1] if waveform.dim() > 0 else len(waveform)

        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_length > target_length:
            # Truncate
            waveform = waveform[:target_length]

    # Apply z-normalization (standardization) if requested
    # This ensures consistent amplitude across different recordings
    if normalize:
        mean = waveform.mean()
        std = waveform.std()
        # Avoid division by zero for silent audio
        if std > 1e-8:
            waveform = (waveform - mean) / std
        else:
            # Silent audio - just center at zero
            waveform = waveform - mean

    # Add white noise (dynamic augmentation)
    if noise_level > 0.0:
        waveform = waveform + torch.randn_like(waveform) * noise_level

    return waveform


def load_audio_batch(
    filepaths: list[str],
    sample_rate: int = 16000,
    mono: bool = True,
    duration: Optional[float] = 60.0,
    crop_position: str = "end",
    normalize: bool = True,
    max_duration: float = 900.0,
    num_workers: Optional[int] = None
) -> list[Optional[torch.Tensor]]:
    """Load multiple audio files in parallel (READ-ONLY).

    Args:
        filepaths: List of audio file paths
        sample_rate: Target sample rate
        mono: Convert to mono
        duration: Duration to extract
        crop_position: Where to extract from - "start", "center", or "end"
        normalize: Apply z-normalization
        max_duration: Skip files longer than this
        num_workers: Number of parallel workers (default: 80% CPU cores)

    Returns:
        List of audio tensors (same order as filepaths)
        None entries for files that couldn't be loaded

    Note:
        - All operations are READ-ONLY
        - Uses multiprocessing.Pool for parallel loading
        - Progress can be tracked externally with tqdm if needed
    """
    if num_workers is None:
        num_workers = get_default_workers()

    # Create partial function with fixed parameters
    load_fn = partial(
        load_audio_file,
        sample_rate=sample_rate,
        mono=mono,
        duration=duration,
        crop_position=crop_position,
        normalize=normalize,
        max_duration=max_duration
    )
    
    # Load in parallel
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(load_fn, filepaths)
    
    return results


def load_audio_file_with_jitter(
    filepath: str,
    sample_rate: int = 16000,
    mono: bool = True,
    duration: float = 60.0,
    crop_position: str = "end",
    normalize: bool = True,
    max_duration: float = 900.0,
    jitter_seconds: float = 5.0,
    noise_level: float = 0.0
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load two different crops from the same audio file for contrastive learning.

    Returns two views of the same song with different random offsets.
    This creates guaranteed positive pairs for self-supervised learning.

    Args:
        filepath: Path to audio file
        sample_rate: Target sample rate (Hz)
        mono: Convert to mono if True
        duration: Duration in seconds to extract
        crop_position: Base position for extraction - "start", "center", or "end"
        normalize: Apply z-normalization if True
        max_duration: Skip files longer than this
        jitter_seconds: Random offset range for second crop (±jitter_seconds)
        noise_level: Standard deviation of Gaussian noise to add (0.0 = no noise)

    Returns:
        Tuple of (view1, view2) audio tensors, or (None, None) if loading fails

    Note:
        - view1: Standard crop at crop_position
        - view2: Same song with random offset (within ±jitter_seconds of view1)
    """
    import random

    try:
        filepath = Path(filepath)

        if not filepath.exists():
            return None, None

        # Get audio info
        info = torchaudio.info(str(filepath))
        file_duration = info.num_frames / info.sample_rate

        # Skip very long files
        if file_duration > max_duration:
            return None, None

        # Calculate target frames
        target_frames = int(duration * info.sample_rate)
        jitter_frames = int(jitter_seconds * info.sample_rate)

        # Determine base start position
        if file_duration > duration:
            if crop_position == "center":
                base_start = (info.num_frames - target_frames) // 2
            elif crop_position == "end":
                base_start = info.num_frames - target_frames
            else:  # "start"
                base_start = 0
        else:
            # File is shorter than target - use full file
            base_start = 0
            target_frames = info.num_frames

        # View 1: Standard crop
        waveform1, sr = torchaudio.load(
            str(filepath),
            frame_offset=base_start,
            num_frames=min(target_frames, info.num_frames - base_start)
        )

        # View 2: Jittered crop (different random offset)
        # Calculate valid jitter range
        max_jitter_back = min(jitter_frames, base_start)
        max_jitter_forward = min(jitter_frames, info.num_frames - base_start - target_frames)

        # Random jitter within valid range
        jitter_offset = random.randint(-max_jitter_back, max(0, max_jitter_forward))
        jittered_start = max(0, base_start + jitter_offset)

        # Ensure we don't exceed file bounds
        frames_available = info.num_frames - jittered_start
        frames_to_load = min(target_frames, frames_available)

        waveform2, _ = torchaudio.load(
            str(filepath),
            frame_offset=jittered_start,
            num_frames=frames_to_load
        )

        # Process both waveforms
        processed = []
        for waveform in [waveform1, waveform2]:
            # Convert to mono
            if mono and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=False)
            elif mono:
                waveform = waveform.squeeze(0)

            # Resample if needed
            if sr != sample_rate:
                resampler = get_resampler(sr, sample_rate)
                waveform = resampler(waveform)

            # Pad or truncate to exact duration
            target_length = int(duration * sample_rate)
            current_length = waveform.shape[-1] if waveform.dim() > 0 else len(waveform)

            if current_length < target_length:
                padding = target_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif current_length > target_length:
                waveform = waveform[:target_length]

            # Z-normalization
            if normalize:
                mean = waveform.mean()
                std = waveform.std()
                if std > 1e-8:
                    waveform = (waveform - mean) / std
                else:
                    waveform = waveform - mean

            # Add white noise (dynamic augmentation)
            if noise_level > 0.0:
                waveform = waveform + torch.randn_like(waveform) * noise_level

            processed.append(waveform)

        return processed[0], processed[1]

    except Exception:
        # Fallback to librosa for problematic files
        # Load two views with slightly different offsets
        import random

        result1 = _load_with_librosa(str(filepath), sample_rate, duration, crop_position, mono)
        if result1 is None:
            print(f"Warning: Could not load {filepath} for augmentation: Failed to decode audio.")
            return None, None

        waveform1, _ = result1

        # For view2, slightly shift the crop position
        alt_position = "center" if crop_position == "end" else "end"
        result2 = _load_with_librosa(str(filepath), sample_rate, duration, alt_position, mono)
        if result2 is None:
            # Use same view if alternate position fails
            waveform2 = waveform1.clone()
        else:
            waveform2, _ = result2

        # Process both waveforms
        processed = []
        for waveform in [waveform1, waveform2]:
            # Pad or truncate to exact duration
            target_length = int(duration * sample_rate)
            current_length = waveform.shape[-1] if waveform.dim() > 0 else len(waveform)

            if current_length < target_length:
                padding = target_length - current_length
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif current_length > target_length:
                waveform = waveform[:target_length]

            # Z-normalization
            if normalize:
                mean = waveform.mean()
                std = waveform.std()
                if std > 1e-8:
                    waveform = (waveform - mean) / std
                else:
                    waveform = waveform - mean

            # Add white noise (dynamic augmentation)
            if noise_level > 0.0:
                waveform = waveform + torch.randn_like(waveform) * noise_level

            processed.append(waveform)

        return processed[0], processed[1]


def get_audio_info(filepath: str) -> Optional[dict]:
    """Get audio file metadata without loading (READ-ONLY, fast).
    
    Args:
        filepath: Path to audio file
    
    Returns:
        Dictionary with metadata or None if file invalid:
        - duration: Duration in seconds
        - sample_rate: Sample rate in Hz
        - num_channels: Number of audio channels
        - num_frames: Total number of frames
        - bits_per_sample: Bits per sample
        - encoding: Audio encoding format
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            return None
        
        info = torchaudio.info(str(filepath))
        
        return {
            "duration": info.num_frames / info.sample_rate,
            "sample_rate": info.sample_rate,
            "num_channels": info.num_channels,
            "num_frames": info.num_frames,
            "bits_per_sample": info.bits_per_sample if hasattr(info, 'bits_per_sample') else None,
            "encoding": info.encoding if hasattr(info, 'encoding') else None
        }
    except Exception:
        return None


def validate_audio_files(
    filepaths: list[str],
    max_duration: float = 900.0,
    num_workers: Optional[int] = None
) -> tuple[list[str], list[str], list[str]]:
    """Validate audio files without loading them (READ-ONLY, fast).
    
    Args:
        filepaths: List of audio file paths to validate
        max_duration: Maximum allowed duration in seconds
        num_workers: Number of parallel workers
    
    Returns:
        Tuple of (valid_files, too_long_files, invalid_files)
        - valid_files: Files that can be loaded and are not too long
        - too_long_files: Files longer than max_duration
        - invalid_files: Files that don't exist or can't be read
    """
    if num_workers is None:
        num_workers = get_default_workers()
    
    with multiprocessing.Pool(num_workers) as pool:
        infos = pool.map(get_audio_info, filepaths)
    
    valid_files = []
    too_long_files = []
    invalid_files = []
    
    for filepath, info in zip(filepaths, infos):
        if info is None:
            invalid_files.append(filepath)
        elif info["duration"] > max_duration:
            too_long_files.append(filepath)
        else:
            valid_files.append(filepath)
    
    return valid_files, too_long_files, invalid_files
