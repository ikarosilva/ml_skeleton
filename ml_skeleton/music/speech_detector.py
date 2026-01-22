"""
This module implements the speech detection pipeline.
"""
import torch
import torchaudio
from pathlib import Path
from dataclasses import dataclass
import multiprocessing
from multiprocessing import Pool
from typing import List, Optional, Tuple
import sqlite3
import time

from .clementine_db import Song, get_default_workers

@dataclass
class SpeechDetectionResult:
    filename: str
    speech_probability: float

class SpeechDetector:
    """
    Analyzes audio files to detect the probability of speech using
    a pre-trained VAD model. Results are cached in an SQLite database.
    """

    def __init__(self, cache_path: str = "./speech_cache.db"):
        self.cache_path = cache_path
        self._cache_conn = self._init_db()
        self._model, self._utils = self._load_model()
        # Make model and utils available to child processes
        global _global_model, _global_utils
        _global_model = self._model
        _global_utils = self._utils


    def _load_model(self):
        """Loads the Silero VAD model from PyTorch Hub."""
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                verbose=False
            )
        except Exception as e:
            print(f"Error loading Silero VAD model: {e}")
            print("Please ensure you have an internet connection for the first run.")
            return None, None
        return model, utils

    def _init_db(self):
        """Initializes the SQLite database and creates the cache table."""
        conn = sqlite3.connect(self.cache_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS is_speech_cache (
                filename TEXT PRIMARY KEY,
                speech_probability REAL NOT NULL,
                mtime REAL NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_speech_probability 
            ON is_speech_cache(speech_probability);
        """)
        conn.commit()
        return conn

    def detect_speech_in_songs(
        self,
        songs: List[Song],
        num_workers: Optional[int] = None
    ) -> List[SpeechDetectionResult]:
        """
        Processes a list of songs in parallel to detect speech.
        """
        if self._model is None:
            print("VAD model not loaded. Skipping speech detection.")
            return []
            
        if num_workers is None:
            num_workers = get_default_workers()

        print(f"Starting speech detection for {len(songs)} songs using {num_workers} workers...")
        
        # Prepare arguments for multiprocessing
        tasks = [(song, self.cache_path) for song in songs]

        with Pool(num_workers) as pool:
            results = pool.map(_process_single_song_wrapper, tasks)

        print("Speech detection complete.")
        return [res for res in results if res is not None]

# These are global to be used by child processes
_global_model = None
_global_utils = None

def _process_single_song_wrapper(args: Tuple[Song, str]) -> Optional[SpeechDetectionResult]:
    """
    A top-level wrapper function to be used with multiprocessing.Pool.
    Initializes a worker-local cache connection.
    """
    song, cache_path = args
    
    # Each worker has its own connection to the DB
    if not hasattr(_process_single_song_wrapper, "cache_conn"):
        _process_single_song_wrapper.cache_conn = sqlite3.connect(cache_path, check_same_thread=False)

    conn = _process_single_song_wrapper.cache_conn
    
    filepath = song.filepath
    
    if not filepath.exists():
        return SpeechDetectionResult(song.filename, 0.0) # Skip non-existent files

    mtime = filepath.stat().st_mtime

    # 1. Check cache first
    cached_prob = _check_cache(conn, song.filename, mtime)
    if cached_prob is not None:
        return SpeechDetectionResult(song.filename, cached_prob)

    # 2. Skip files that are too long
    try:
        info = torchaudio.info(filepath)
    except Exception as e:
        # If we can't get info, we likely can't load it either.
        print(f"Could not get info for {filepath}: {e}")
        return None

    duration = info.num_frames / info.sample_rate
    if duration > 900: # 15 minutes
        _update_cache(conn, song.filename, 0.0, mtime)
        return SpeechDetectionResult(song.filename, 0.0)

    # 3. Load center 30s of audio
    try:
        waveform = _load_center_audio(filepath, duration, info.sample_rate)
    except Exception as e:
        print(f"Could not load audio for {filepath}: {e}")
        return None

    # 4. Run VAD model
    if _global_model is None or _global_utils is None:
        return None # Should not happen if main process loaded model

    get_speech_timestamps = _global_utils[0]
    
    # Silero VAD expects 16kHz
    resampler = torchaudio.transforms.Resample(info.sample_rate, 16000)
    resampled_waveform = resampler(waveform)

    # The model works on chunks, but for a 30s clip we can run it once.
    # It returns speech timestamps.
    speech_timestamps = get_speech_timestamps(resampled_waveform, _global_model, sampling_rate=16000)
    
    total_speech_duration = sum([d['end'] - d['start'] for d in speech_timestamps]) / 16000
    
    # We check speech duration relative to the actual clip duration, which might be < 30s
    clip_duration = len(resampled_waveform[0]) / 16000
    speech_probability = total_speech_duration / clip_duration if clip_duration > 0 else 0.0

    # 5. Update cache
    _update_cache(conn, song.filename, speech_probability, mtime)

    return SpeechDetectionResult(song.filename, speech_probability)


def _check_cache(conn: sqlite3.Connection, filename: str, mtime: float) -> Optional[float]:
    """Queries the cache for a non-stale entry."""
    cursor = conn.cursor()
    cursor.execute("SELECT speech_probability, mtime FROM is_speech_cache WHERE filename = ?", (filename,))
    result = cursor.fetchone()
    if result:
        cached_prob, cached_mtime = result
        if mtime == cached_mtime:
            return cached_prob
    return None

def _update_cache(conn: sqlite3.Connection, filename: str, probability: float, mtime: float):
    """Inserts or replaces a record in the cache."""
    cursor = conn.cursor()
    cursor.execute(
        "REPLACE INTO is_speech_cache (filename, speech_probability, mtime, updated_at) VALUES (?, ?, ?, ?)",
        (filename, probability, mtime, time.time())
    )
    conn.commit()

def _load_center_audio(filepath: Path, duration: float, sample_rate: int, target_duration: float = 30.0) -> torch.Tensor:
    """Loads a `target_duration` segment from the center of an audio file."""
    if duration <= target_duration:
        waveform, sr = torchaudio.load(filepath)
    else:
        offset_frames = int((duration - target_duration) / 2 * sample_rate)
        num_frames = int(target_duration * sample_rate)
        waveform, sr = torchaudio.load(filepath, frame_offset=offset_frames, num_frames=num_frames)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        
    return waveform
