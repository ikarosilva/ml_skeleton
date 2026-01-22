"""Simple baseline rating classifier implementations.

These are basic reference implementations that users can customize or replace.
All classifiers conform to the RatingClassifier protocol.
"""

import torch
import torch.nn as nn


class SimpleRatingClassifier(nn.Module):
    """Simple MLP classifier for rating prediction.

    Architecture:
        - 2-3 hidden layers with dropout
        - ReLU activations
        - Sigmoid output (ratings in [0, 1])

    This is a minimal baseline. Users can replace with:
        - Deeper networks
        - Residual connections
        - Attention mechanisms
        - Custom architectures

    Args:
        embedding_dim: Input embedding dimension (from encoder)
        hidden_dims: Hidden layer dimensions
        dropout: Dropout probability
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dims: list[int] = None,
        dropout: float = 0.3
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.embedding_dim = embedding_dim

        # Build MLP layers
        layers = []
        in_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        # Output layer with sigmoid (ratings in [0, 1])
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

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
        """
        return self.mlp(embeddings)


class DeepRatingClassifier(nn.Module):
    """Deeper rating classifier with residual connections.

    More sophisticated baseline with skip connections for better gradient flow.
    Useful when embedding dimension is large or dataset is complex.

    Args:
        embedding_dim: Input embedding dimension
        hidden_dims: Hidden layer dimensions
        dropout: Dropout probability
        use_batch_norm: Whether to use batch normalization
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.embedding_dim = embedding_dim
        self.use_batch_norm = use_batch_norm

        # Input projection (if first hidden dim != embedding dim)
        if hidden_dims[0] != embedding_dim:
            self.input_proj = nn.Linear(embedding_dim, hidden_dims[0])
        else:
            self.input_proj = nn.Identity()

        # Build residual blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(
                self._make_residual_block(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    dropout,
                    use_batch_norm
                )
            )

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )

    def _make_residual_block(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        use_batch_norm: bool
    ) -> nn.Module:
        """Create a residual block with optional batch norm."""
        layers = [nn.Linear(in_dim, out_dim)]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))

        layers.extend([
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        ])

        # Skip connection (if dimensions match)
        if in_dim == out_dim:
            return ResidualBlock(nn.Sequential(*layers))
        else:
            # Need projection for skip connection
            return ResidualBlock(
                nn.Sequential(*layers),
                skip_proj=nn.Linear(in_dim, out_dim)
            )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict ratings from embeddings.

        Args:
            embeddings: Shape (batch_size, embedding_dim)

        Returns:
            ratings: Shape (batch_size, 1) in [0, 1]
        """
        x = self.input_proj(embeddings)

        # Apply residual blocks
        for block in self.blocks:
            x = block(x)

        # Output
        return self.output(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, main_path: nn.Module, skip_proj: nn.Module = None):
        super().__init__()
        self.main_path = main_path
        self.skip_proj = skip_proj or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection."""
        return self.main_path(x) + self.skip_proj(x)


class EnsembleRatingClassifier(nn.Module):
    """Ensemble of multiple classifiers with voting.

    Combines predictions from multiple classifiers for potentially better
    generalization and robustness.

    Args:
        classifiers: List of rating classifier models
        weights: Optional weights for each classifier (default: equal weights)
    """

    def __init__(
        self,
        classifiers: list[nn.Module],
        weights: list[float] = None
    ):
        super().__init__()
        self.classifiers = nn.ModuleList(classifiers)

        if weights is None:
            weights = [1.0 / len(classifiers)] * len(classifiers)

        self.register_buffer(
            'weights',
            torch.tensor(weights, dtype=torch.float32)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict ratings using ensemble voting.

        Args:
            embeddings: Shape (batch_size, embedding_dim)

        Returns:
            ratings: Weighted average of all classifiers
                    Shape (batch_size, 1)
        """
        predictions = []

        for classifier in self.classifiers:
            pred = classifier(embeddings)
            predictions.append(pred)

        # Stack predictions: (num_classifiers, batch_size, 1)
        predictions = torch.stack(predictions, dim=0)

        # Weighted average: (batch_size, 1)
        weights = self.weights.view(-1, 1, 1)
        weighted_pred = (predictions * weights).sum(dim=0)

        return weighted_pred
