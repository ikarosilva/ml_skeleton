"""Simple baseline audio encoder implementations.

These are basic reference implementations that users can customize or replace.
All encoders conform to the AudioEncoder protocol.
"""

import torch
import torch.nn as nn
from typing import Optional


class SimpleAudioEncoder(nn.Module):
    """Simple 1D CNN encoder for raw audio waveforms.

    Architecture:
        - 4 conv1d blocks with pooling
        - Global average pooling
        - Linear projection to embedding space

    This is a minimal baseline. Users can replace with:
        - Deeper CNNs
        - Mel-spectrogram + 2D CNN
        - Pre-trained models (Wav2Vec2, CLAP, etc.)
        - Transformer architectures
        - Custom models

    Args:
        sample_rate: Audio sample rate (Hz)
        duration: Audio duration (seconds)
        embedding_dim: Output embedding dimension
        base_channels: Starting number of channels (doubles each block)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 30.0,
        embedding_dim: int = 512,
        base_channels: int = 32
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.duration = duration
        self.embedding_dim = embedding_dim
        self.num_samples = int(sample_rate * duration)

        # 4 conv blocks with increasing channels
        # Each reduces temporal dimension by ~16x (kernel=16, stride=8, pool=2)
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(1, base_channels),                    # 32
            self._make_conv_block(base_channels, base_channels * 2),    # 64
            self._make_conv_block(base_channels * 2, base_channels * 4), # 128
            self._make_conv_block(base_channels * 4, base_channels * 8), # 256
        ])

        # Global average pooling (reduces temporal dimension to 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Project to embedding space
        self.projection = nn.Linear(base_channels * 8, embedding_dim)

    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a conv1d block with batch norm, activation, and pooling."""
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=16,
                stride=8,
                padding=4
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveforms to embeddings.

        Args:
            audio: Raw audio waveform tensor
                   Shape: (batch_size, num_samples)
                   Example: (32, 480000) for 30s at 16000 Hz

        Returns:
            embeddings: Fixed-dimensional embedding vectors
                       Shape: (batch_size, embedding_dim)
                       Example: (32, 512)
        """
        # Add channel dimension: (B, num_samples) -> (B, 1, num_samples)
        x = audio.unsqueeze(1)

        # Apply conv blocks
        for block in self.conv_blocks:
            x = block(x)

        # Global pooling: (B, C, T) -> (B, C, 1) -> (B, C)
        x = self.global_pool(x).squeeze(-1)

        # Project to embedding space
        embeddings = self.projection(x)

        return embeddings

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim


class SpectrogramEncoder(nn.Module):
    """2D CNN encoder on mel-spectrograms.

    Alternative to raw waveform processing. Converts audio to mel-spectrogram
    then applies 2D CNN (similar to image processing).

    Args:
        sample_rate: Audio sample rate (Hz)
        n_mels: Number of mel frequency bins
        n_fft: FFT window size
        hop_length: Hop length for STFT
        embedding_dim: Output embedding dimension
        base_channels: Starting number of channels
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: Optional[int] = None,
        embedding_dim: int = 512,
        base_channels: int = 32
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4
        self.embedding_dim = embedding_dim

        # Mel-spectrogram transform
        self.mel_transform = nn.Sequential(
            # Note: torchaudio.transforms.MelSpectrogram should be used here
            # This is a placeholder showing the architecture
        )

        # 2D CNN (similar to image processing)
        self.conv_blocks = nn.ModuleList([
            self._make_conv2d_block(1, base_channels),                    # 32
            self._make_conv2d_block(base_channels, base_channels * 2),    # 64
            self._make_conv2d_block(base_channels * 2, base_channels * 4), # 128
            self._make_conv2d_block(base_channels * 4, base_channels * 8), # 256
        ])

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Project to embedding space
        self.projection = nn.Linear(base_channels * 8, embedding_dim)

    def _make_conv2d_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a conv2d block with batch norm, activation, and pooling."""
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveforms to embeddings via mel-spectrogram.

        Args:
            audio: Raw audio waveform tensor
                   Shape: (batch_size, num_samples)

        Returns:
            embeddings: Shape (batch_size, embedding_dim)
        """
        # Convert to mel-spectrogram: (B, num_samples) -> (B, 1, n_mels, time)
        # This is a placeholder - actual implementation would use:
        # x = self.mel_transform(audio).unsqueeze(1)
        raise NotImplementedError(
            "SpectrogramEncoder is a template. "
            "Add torchaudio.transforms.MelSpectrogram to complete implementation."
        )

        # Apply 2D conv blocks
        for block in self.conv_blocks:
            x = block(x)

        # Global pooling: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        x = self.global_pool(x).squeeze(-1).squeeze(-1)

        # Project to embedding space
        embeddings = self.projection(x)

        return embeddings

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim


class MultiTaskEncoder(nn.Module):
    """Multi-task encoder with album classification head.

    Extends base encoder with additional classification head for album prediction.
    This allows learning album-aware embeddings through multi-task learning.

    Args:
        base_encoder: Base audio encoder (SimpleAudioEncoder, etc.)
        num_albums: Number of unique albums in dataset
    """

    def __init__(self, base_encoder: nn.Module, num_albums: int):
        super().__init__()
        self.base_encoder = base_encoder
        self.num_albums = num_albums

        # Album classification head
        embedding_dim = base_encoder.get_embedding_dim()
        self.album_classifier = nn.Linear(embedding_dim, num_albums)

    def forward(
        self,
        audio: torch.Tensor,
        return_album_logits: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode audio and optionally predict album.

        Args:
            audio: Raw audio waveform tensor
                   Shape: (batch_size, num_samples)
            return_album_logits: If True, also return album logits

        Returns:
            embeddings: Shape (batch_size, embedding_dim)
            album_logits: Shape (batch_size, num_albums) if return_album_logits=True
                         None otherwise
        """
        # Get embeddings from base encoder
        embeddings = self.base_encoder(audio)

        # Optionally compute album logits
        album_logits = None
        if return_album_logits:
            album_logits = self.album_classifier(embeddings)

        return embeddings, album_logits

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.base_encoder.get_embedding_dim()
