"""Simple baseline rating classifier implementations.

These are basic reference implementations that users can customize or replace.
All classifiers conform to the RatingClassifier protocol.
"""

import torch
import torch.nn as nn

from .genre_mapper import NUM_GENRES


class SimpleRatingClassifier(nn.Module):
    """MLP classifier for rating prediction.

    Architecture:
        - Hidden layers with optional BatchNorm and residual connections
        - Each block: Linear -> (BatchNorm) -> ReLU -> Dropout, then + skip(x)
        - Sigmoid output (ratings in [0, 1])
        - Optional: concat 7-dim genre multi-hot to input (genre as input feature)

    Args:
        embedding_dim: Input embedding dimension (from encoder)
        hidden_dims: Hidden layer dimensions
        dropout: Dropout probability
        use_genre: If True, forward expects (embeddings, genre) and input dim is embedding_dim + NUM_GENRES
        use_batch_norm: If True, BatchNorm1d after each hidden Linear
        use_residual: If True, residual connection per block (skip = Identity or Linear projection)
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
        use_genre: bool = False,
        use_batch_norm: bool = False,
        use_residual: bool = False,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.embedding_dim = embedding_dim
        self.hidden_dims = list(hidden_dims)
        self.use_genre = use_genre
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        in_dim = embedding_dim + (NUM_GENRES if use_genre else 0)

        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()

        for hidden_dim in hidden_dims:
            layer_list = [nn.Linear(in_dim, hidden_dim)]
            if use_batch_norm:
                layer_list.append(nn.BatchNorm1d(hidden_dim))
            layer_list.extend([nn.ReLU(inplace=True), nn.Dropout(dropout)])
            self.blocks.append(nn.Sequential(*layer_list))
            self.skips.append(nn.Identity() if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.output = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

    def forward(
        self,
        embeddings: torch.Tensor,
        genre: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict ratings from embeddings and optional genre.

        Args:
            embeddings: Song embedding vectors from encoder
                       Shape: (batch_size, embedding_dim) or (batch_size, num_chunks, embedding_dim)
            genre: Optional 7-dim genre multi-hot, shape (batch_size, NUM_GENRES).
                   Required when use_genre=True.

        Returns:
            ratings: Predicted rating values in range [0, 1]
                    Shape: (batch_size,)

        Note:
            - When use_genre=True, genre is concatenated to embeddings along last dim.
            - Ratings in [0, 1] range where:
              - 0.0 = lowest rating (dislike)
              - 1.0 = highest rating (like)
            - In Clementine DB: rating = -1 means unrated
        """
        if self.use_genre and genre is not None:
            # genre: (B, 7) or (B*C, 7) when chunks; must match embeddings batch dim
            x = torch.cat([embeddings, genre], dim=-1)
        else:
            x = embeddings

        if self.use_residual:
            for block, skip in zip(self.blocks, self.skips):
                x = block(x) + skip(x)
        else:
            for block in self.blocks:
                x = block(x)

        return self.output(x).squeeze(-1)


class LegacySimpleRatingClassifier(nn.Module):
    """Legacy MLP classifier using a single nn.Sequential named 'mlp'.

    Used to load old checkpoints whose state_dict uses keys mlp.0, mlp.3, ...
    (Linear, ReLU, Dropout per hidden layer, then Linear, Sigmoid). Same forward
    interface as SimpleRatingClassifier for A/B testing and inference.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: list[int],
        use_genre: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dims = list(hidden_dims)
        self.use_genre = use_genre
        in_dim = embedding_dim + (NUM_GENRES if use_genre else 0)
        layers = []
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(inplace=True),
                nn.Dropout(0.0),
            ])
            in_dim = h
        layers.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        embeddings: torch.Tensor,
        genre: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_genre and genre is not None:
            x = torch.cat([embeddings, genre], dim=-1)
        else:
            x = embeddings
        return self.mlp(x).squeeze(-1)


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
            ratings: Shape (batch_size,) in [0, 1]
        """
        x = self.input_proj(embeddings)

        # Apply residual blocks
        for block in self.blocks:
            x = block(x)

        # Output
        return self.output(x).squeeze(-1)


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
                    Shape (batch_size,)
        """
        predictions = []

        for classifier in self.classifiers:
            pred = classifier(embeddings)
            predictions.append(pred)

        # Stack predictions: (num_classifiers, batch_size)
        predictions = torch.stack(predictions, dim=0)

        # Weighted average: (batch_size,)
        weights = self.weights.view(-1, 1)
        weighted_pred = (predictions * weights).sum(dim=0)

        return weighted_pred


class AttentionRatingClassifier(nn.Module):
    """Rating classifier with attention over chunk sequence.

    Uses a single learned query to attend over all chunks (e.g. 8 segments per song),
    then an MLP on the resulting context vector. Interpretable (attention weights show
    which chunks matter) and uses full song structure.

    Architecture:
        (B, C, D) -> project to d_model -> optional pos encoding
        -> query attention (learned q, softmax over C) -> (B, d_model)
        -> concat genre if use_genre -> MLP (hidden_dims) -> sigmoid -> (B,)

    Same forward interface as SimpleRatingClassifier: (embeddings, genre?) -> (B,).
    Set handles_chunk_sequence = True so the trainer passes (B, C, D) and gets (B,) directly.
    """

    handles_chunk_sequence = True  # Trainer passes (B, C, D) and uses (B,) output without flatten+aggregate

    def __init__(
        self,
        embedding_dim: int = 2048,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
        use_genre: bool = False,
        use_batch_norm: bool = False,
        use_residual: bool = False,
        d_model: int = 512,
        num_heads: int = 4,
        max_chunks: int = 16,
        use_pos_encoding: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.embedding_dim = embedding_dim
        self.hidden_dims = list(hidden_dims)
        self.use_genre = use_genre
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_pos_encoding = use_pos_encoding
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.chunk_proj = nn.Linear(embedding_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_chunks, d_model)) if use_pos_encoding else None
        self.query = nn.Parameter(torch.randn(1, num_heads, self.head_dim) * 0.02)

        mlp_in_dim = d_model + (NUM_GENRES if use_genre else 0)
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for hidden_dim in hidden_dims:
            layer_list = [nn.Linear(mlp_in_dim, hidden_dim)]
            if use_batch_norm:
                layer_list.append(nn.BatchNorm1d(hidden_dim))
            layer_list.extend([nn.ReLU(inplace=True), nn.Dropout(dropout)])
            self.blocks.append(nn.Sequential(*layer_list))
            self.skips.append(
                nn.Identity() if mlp_in_dim == hidden_dim else nn.Linear(mlp_in_dim, hidden_dim)
            )
            mlp_in_dim = hidden_dim
        self.output = nn.Sequential(nn.Linear(mlp_in_dim, 1), nn.Sigmoid())

    def _attention_pool(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, d_model) -> (B, d_model) using multi-head query attention over C."""
        B, C, D = x.shape
        # x: (B, C, d_model) -> split to (B, C, num_heads, head_dim) -> (B, num_heads, C, head_dim)
        x_heads = x.view(B, C, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, C, head_dim)
        # query: (1, H, head_dim) -> (B, H, 1, head_dim)
        q = self.query.expand(B, -1, -1).unsqueeze(2)
        # scores: (B, H, 1, C)
        scores = torch.matmul(q, x_heads.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = scores.softmax(dim=-1)  # (B, H, 1, C)
        # context: (B, H, 1, head_dim) -> (B, H, head_dim) -> (B, d_model)
        context = torch.matmul(attn, x_heads).squeeze(2).reshape(B, -1)
        return context

    def forward(
        self,
        embeddings: torch.Tensor,
        genre: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict ratings. Accepts (B, D) or (B, C, D); returns (B,)."""
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)  # (B, D) -> (B, 1, D)
        # (B, C, D)
        x = self.chunk_proj(embeddings)  # (B, C, d_model)
        if self.pos_embed is not None:
            C = x.size(1)
            x = x + self.pos_embed[:, :C]
        pooled = self._attention_pool(x)  # (B, d_model)
        if self.use_genre and genre is not None:
            pooled = torch.cat([pooled, genre], dim=-1)
        if self.use_residual:
            for block, skip in zip(self.blocks, self.skips):
                pooled = block(pooled) + skip(pooled)
        else:
            for block in self.blocks:
                pooled = block(pooled)
        return self.output(pooled).squeeze(-1)
