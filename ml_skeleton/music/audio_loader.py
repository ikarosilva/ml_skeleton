"""Audio loading utilities with multiprocessing support.

CRITICAL: All operations are READ-ONLY. Audio files are NEVER modified.
"""

import torch
import torchaudio
from pathlib import Path
from typing import Optional
import multiprocessing
from functools import partial


def get_default_workers() -> int:
    """Get default number of workers (80% of CPU cores).
    
    Returns:
        Number of worker processes (minimum 1)
    """
    cpu_count = multiprocessing.cpu_count()
    return max(1, int(cpu_count * 0.8))


def load_audio_file(
    filepath: str,
    sample_rate: int = 22050,
    mono: bool = True,
    duration: Optional[float] = 30.0,
    center_crop: bool = True,
    max_duration: float = 900.0
) -> Optional[torch.Tensor]:
    """Load single audio file (READ-ONLY).
    
    Args:
        filepath: Path to audio file
        sample_rate: Target sample rate (Hz)
        mono: Convert to mono if True
        duration: Duration in seconds to extract (None = full song)
        center_crop: If True, extract from center; if False, from beginning
        max_duration: Skip files longer than this (default: 900s = 15 minutes)
    
    Returns:
        Audio tensor of shape (num_samples,) if mono, or None if file invalid/too long
        
    Note:
        - This function is READ-ONLY - audio files are NEVER modified
        - Uses torchaudio.load() which opens files in read-only mode
        - Long files (>max_duration) are skipped to filter out live albums, DJ sets
    """
    try:
        filepath = Path(filepath)
        
        # Check file exists
        if not filepath.exists():
            return None
        
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
            # Calculate frame offset for center crop
            target_frames = int(duration * info.sample_rate)
            
            if center_crop and file_duration > duration:
                # Extract from center
                start_frame = (info.num_frames - target_frames) // 2
                waveform, sr = torchaudio.load(
                    str(filepath),
                    frame_offset=start_frame,
                    num_frames=target_frames
                )
            else:
                # Extract from beginning or load full file if shorter than duration
                frames_to_load = min(target_frames, info.num_frames)
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
        
        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
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
        
        return waveform
        
    except Exception as e:
        # Return None for corrupted/unreadable files
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def load_audio_batch(
    filepaths: list[str],
    sample_rate: int = 22050,
    mono: bool = True,
    duration: Optional[float] = 30.0,
    center_crop: bool = True,
    max_duration: float = 900.0,
    num_workers: Optional[int] = None
) -> list[Optional[torch.Tensor]]:
    """Load multiple audio files in parallel (READ-ONLY).
    
    Args:
        filepaths: List of audio file paths
        sample_rate: Target sample rate
        mono: Convert to mono
        duration: Duration to extract
        center_crop: Extract from center or beginning
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
        center_crop=center_crop,
        max_duration=max_duration
    )
    
    # Load in parallel
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(load_fn, filepaths)
    
    return results


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
