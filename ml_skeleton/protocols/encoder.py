"""Audio encoder protocol definition.

This protocol defines the interface that user-provided audio encoders must implement.
Users can inject their own encoder architectures (CNN, Transformer, pre-trained models, etc.)
as long as they conform to this protocol.
"""

from typing import Protocol
import torch


class AudioEncoder(Protocol):
    """Protocol for audio encoder models.

    Users implement this interface to define how raw audio waveforms are encoded
    into fixed-dimensional embedding vectors.

    Example implementations:
    - 1D CNN on raw waveforms
    - Mel-spectrogram + 2D CNN
    - Pre-trained Wav2Vec2
    - Transformer encoder
    - Custom architectures
    """

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveforms to embeddings.

        Args:
            audio: Raw audio waveform tensor
                   Shape: (batch_size, num_samples)
                   Example: (32, 661500) for 30 seconds at 22050 Hz

        Returns:
            embeddings: Fixed-dimensional embedding vectors
                       Shape: (batch_size, embedding_dim)
                       Example: (32, 512)

        Note:
            - Input audio is normalized to [-1, 1] range
            - num_samples = sample_rate * duration (e.g., 22050 * 30 = 661500)
            - embedding_dim is user-defined (common: 128, 256, 512, 1024)
        """
        ...

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension (Z).

        Returns:
            embedding_dim: Dimension of output embeddings
                          Example: 512

        Note:
            This must match the second dimension of forward() output.
        """
        ...
