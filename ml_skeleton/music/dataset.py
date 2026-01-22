"""
This module contains the PyTorch Dataset classes for handling music data.
"""
import torch
from typing import List, Dict, Optional

from .clementine_db import Song

class MusicDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading music files and their metadata.
    Handles filtering based on speech detection results.
    """
    def __init__(
        self, 
        songs: List[Song], 
        speech_results: Optional[Dict[str, float]] = None, 
        speech_threshold: float = 0.5
    ):
        super().__init__()
        self.all_songs = songs
        self.speech_results = speech_results
        self.speech_threshold = speech_threshold

        self.songs = self._filter_songs(self.all_songs, self.speech_results, self.speech_threshold)

    def _filter_songs(
        self, 
        songs: List[Song], 
        speech_results: Optional[Dict[str, float]], 
        threshold: float
    ) -> List[Song]:
        """
        Filters out songs that are likely to be speech.
        """
        if not speech_results:
            return songs

        original_count = len(songs)
        filtered_songs = []
        for song in songs:
            # Default to 0.0 (not speech) if not found in results
            prob = speech_results.get(song.filename, 0.0) 
            if prob <= threshold:
                filtered_songs.append(song)
        
        filtered_count = original_count - len(filtered_songs)
        if filtered_count > 0:
            print(f"Filtered {filtered_count} songs based on speech threshold > {threshold}")
        
        return filtered_songs

    def __len__(self) -> int:
        return len(self.songs)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single data point.
        This is a placeholder and needs to be implemented with actual audio loading.
        """
        song = self.songs[idx]
        # In a real implementation, you would load the audio here.
        # For now, just return metadata.
        return {
            "song": song,
            "audio": torch.zeros(1, 22050 * 30) # Placeholder for 30s audio
        }

