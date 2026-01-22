"""Rating classifier protocol definition.

This protocol defines the interface that user-provided rating classifiers must implement.
Users can inject their own classifier architectures (MLP, deeper networks, etc.)
as long as they conform to this protocol.
"""

from typing import Protocol
import torch


class RatingClassifier(Protocol):
    """Protocol for rating prediction models.

    Users implement this interface to define how song embeddings are mapped
    to predicted ratings.

    Example implementations:
    - Simple MLP (2-3 layers)
    - Deeper network with residual connections
    - Attention-based classifier
    - Custom architectures
    """

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict ratings from embeddings.

        Args:
            embeddings: Song embedding vectors from encoder
                       Shape: (batch_size, embedding_dim)
                       Example: (256, 512)

        Returns:
            ratings: Predicted rating values in range [0, 1]
                    Shape: (batch_size, 1)
                    Example: (256, 1)

        Note:
            - Ratings in [0, 1] range where:
              - 0.0 = lowest rating (dislike)
              - 1.0 = highest rating (like)
            - In Clementine DB: rating = -1 means unrated
            - Use sigmoid activation to ensure output in [0, 1]
        """
        ...
