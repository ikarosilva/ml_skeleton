"""MoCo v2 encoder with nnAudio CQT for music representation learning.

Architecture:
    Audio (B, T) → nnAudio CQT (B, 84, T') → ResNet-50 2D → 2048-dim embedding
    ↓
    ├── MoCo v2 projection head → contrastive loss (NT-Xent)
    └── Genre BCE head → multi-label classification

MoCo v2 features:
    - Momentum encoder with EMA (m=0.999)
    - Queue of 4096 negatives
    - MLP projection head (2048→2048→128)
    - Temperature τ=0.07 for NT-Xent

Reference:
    "Improved Baselines with Momentum Contrastive Learning" (Chen et al., 2020)
    https://arxiv.org/abs/2003.04297
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple
from nnAudio.features import CQT2010v2

from ml_skeleton.music.genre_mapper import NUM_GENRES


class CQTTransform(nn.Module):
    """GPU-accelerated CQT transform using nnAudio.

    Wraps nnAudio's CQT2010v2 for integration with the encoder pipeline.
    Outputs are ready for ResNet-50 2D (adds channel dimension, normalizes).

    Args:
        sr: Sample rate (Hz)
        n_bins: Number of CQT bins (default: 84 = 7 octaves × 12)
        fmin: Minimum frequency (default: 32.7 Hz = C1)
        hop_length: Hop length in samples
    """

    def __init__(
        self,
        sr: int = 16000,
        n_bins: int = 84,
        fmin: float = 32.7,
        hop_length: int = 512
    ):
        super().__init__()
        self.cqt = CQT2010v2(
            sr=sr,
            n_bins=n_bins,
            fmin=fmin,
            hop_length=hop_length,
            output_format='Magnitude',
            trainable=False
        )
        self.n_bins = n_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert audio to CQT spectrogram.

        Args:
            x: Audio tensor of shape (B, T) or (B, 1, T)

        Returns:
            CQT spectrogram of shape (B, 1, n_bins, T') ready for ResNet
        """
        # Handle different input shapes
        if x.dim() == 3:
            x = x.squeeze(1)  # (B, 1, T) → (B, T)

        # CQT transform: (B, T) → (B, n_bins, T')
        cqt = self.cqt(x)

        # Log magnitude (avoid log(0))
        cqt = torch.log(cqt + 1e-9)

        # Normalize to [0, 1] per sample
        # Using min-max normalization per batch item
        B = cqt.shape[0]
        cqt_flat = cqt.view(B, -1)
        min_vals = cqt_flat.min(dim=1, keepdim=True)[0]
        max_vals = cqt_flat.max(dim=1, keepdim=True)[0]
        cqt_flat = (cqt_flat - min_vals) / (max_vals - min_vals + 1e-9)
        cqt = cqt_flat.view_as(cqt)

        # Add channel dimension: (B, n_bins, T') → (B, 1, n_bins, T')
        cqt = cqt.unsqueeze(1)

        return cqt


class ProjectionMLP(nn.Module):
    """MoCo v2 projection MLP.

    Two-layer MLP with hidden layer BN+ReLU.
    Architecture: 2048 → 2048 (BN, ReLU) → 128

    Args:
        in_dim: Input dimension (backbone output)
        hidden_dim: Hidden layer dimension
        out_dim: Output projection dimension
    """

    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dim: int = 2048,
        out_dim: int = 128
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GenreHead(nn.Module):
    """Multi-label genre classification head.

    Simple linear layer for BCE classification.

    Args:
        in_dim: Input dimension (backbone output)
        num_genres: Number of genre categories
    """

    def __init__(self, in_dim: int = 2048, num_genres: int = NUM_GENRES):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_genres)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (not sigmoid) for BCEWithLogitsLoss."""
        return self.fc(x)


class MoCoEncoder(nn.Module):
    """MoCo v2 encoder with CQT and genre head for music.

    Full architecture:
        Audio → CQT → ResNet-50 → 2048-dim embedding
        ├── MoCo projection → 128-dim for contrastive
        └── Genre head → 7-dim logits for BCE

    Training mode returns all outputs for loss computation.
    Inference mode returns embeddings only.

    Args:
        sample_rate: Audio sample rate (Hz)
        n_bins: Number of CQT frequency bins
        fmin: Minimum CQT frequency (Hz)
        hop_length: CQT hop length
        embedding_dim: Backbone embedding dimension
        projection_dim: MoCo projection dimension
        num_genres: Number of genre categories
        queue_size: MoCo queue size (K)
        momentum: EMA momentum for momentum encoder
        temperature: Temperature for NT-Xent loss
        pretrained_backbone: Use ImageNet pretrained ResNet
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_bins: int = 84,
        fmin: float = 32.7,
        hop_length: int = 512,
        embedding_dim: int = 2048,
        projection_dim: int = 128,
        num_genres: int = NUM_GENRES,
        queue_size: int = 4096,
        momentum: float = 0.999,
        temperature: float = 0.07,
        pretrained_backbone: bool = True
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # CQT transform (GPU-accelerated)
        self.cqt_transform = CQTTransform(
            sr=sample_rate,
            n_bins=n_bins,
            fmin=fmin,
            hop_length=hop_length
        )

        # Query encoder (backbone + projection)
        self.backbone = self._create_backbone(pretrained_backbone)
        self.projector = ProjectionMLP(
            in_dim=embedding_dim,
            hidden_dim=embedding_dim,
            out_dim=projection_dim
        )

        # Key encoder (momentum-updated copy)
        self.backbone_k = copy.deepcopy(self.backbone)
        self.projector_k = copy.deepcopy(self.projector)

        # Freeze momentum encoder (updated via EMA only)
        for param in self.backbone_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False

        # Genre classification head (uses query encoder backbone)
        self.genre_head = GenreHead(in_dim=embedding_dim, num_genres=num_genres)

        # MoCo queue
        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _create_backbone(self, pretrained: bool) -> nn.Module:
        """Create ResNet-50 backbone for CQT spectrograms.

        Modifies first conv layer to accept 1-channel CQT input.
        """
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Modify first conv for 1-channel input (CQT)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # New: Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize new conv with mean of RGB channels if pretrained
        if pretrained:
            with torch.no_grad():
                backbone.conv1.weight = nn.Parameter(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )

        # Remove classification head
        backbone.fc = nn.Identity()

        return backbone

    @torch.no_grad()
    def _momentum_update(self):
        """Update momentum encoder via EMA."""
        for param_q, param_k in zip(
            self.backbone.parameters(), self.backbone_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

        for param_q, param_k in zip(
            self.projector.parameters(), self.projector_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update the queue with new keys."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Handle case where batch doesn't fit exactly
        if ptr + batch_size > self.queue_size:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
            ptr = batch_size - remaining
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through CQT and backbone only.

        Args:
            x: Audio tensor of shape (B, T)

        Returns:
            Embedding of shape (B, embedding_dim)
        """
        cqt = self.cqt_transform(x)
        return self.backbone(cqt)

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: Optional[torch.Tensor] = None,
        return_all: bool = False
    ) -> dict:
        """Forward pass for MoCo training or inference.

        Training mode (x_k provided):
            - Computes query and key embeddings
            - Updates momentum encoder
            - Computes contrastive logits
            - Returns genre logits

        Inference mode (x_k=None):
            - Returns embeddings only

        Args:
            x_q: Query audio tensor of shape (B, T)
            x_k: Key audio tensor of shape (B, T), optional for inference
            return_all: Return intermediate outputs for debugging

        Returns:
            Dictionary with:
            - embedding: (B, embedding_dim) backbone features
            - logits_contrastive: (B, 1+K) MoCo logits (training only)
            - logits_genre: (B, num_genres) genre classification logits
            - labels_contrastive: (B,) targets for contrastive loss (all zeros)
        """
        # Query forward
        cqt_q = self.cqt_transform(x_q)
        embedding_q = self.backbone(cqt_q)
        projection_q = self.projector(embedding_q)
        projection_q = F.normalize(projection_q, dim=1)

        # Genre prediction (always computed from query)
        logits_genre = self.genre_head(embedding_q)

        result = {
            "embedding": embedding_q,
            "logits_genre": logits_genre
        }

        # Training mode: compute contrastive loss components
        if x_k is not None:
            with torch.no_grad():
                # Update momentum encoder
                self._momentum_update()

                # Key forward (no gradient)
                cqt_k = self.cqt_transform(x_k)
                embedding_k = self.backbone_k(cqt_k)
                projection_k = self.projector_k(embedding_k)
                projection_k = F.normalize(projection_k, dim=1)

            # Positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', projection_q, projection_k).unsqueeze(-1)

            # Negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', projection_q, self.queue.clone().detach())

            # Logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits = logits / self.temperature

            # Labels: positives are at index 0
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

            # Update queue
            self._dequeue_and_enqueue(projection_k)

            result["logits_contrastive"] = logits
            result["labels_contrastive"] = labels

        if return_all:
            result["cqt"] = cqt_q
            result["projection"] = projection_q

        return result

    def get_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract embeddings for inference.

        Args:
            audio: Audio tensor of shape (B, T)

        Returns:
            Embeddings of shape (B, embedding_dim)
        """
        self.eval()
        with torch.no_grad():
            return self.forward_backbone(audio)

    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim


class MoCoLoss(nn.Module):
    """Combined loss for MoCo + Genre BCE training.

    Loss = moco_weight * NT-Xent + genre_weight * BCE

    Args:
        moco_weight: Weight for contrastive loss (default: 0.6)
        genre_weight: Weight for genre BCE loss (default: 0.4)
    """

    def __init__(
        self,
        moco_weight: float = 0.6,
        genre_weight: float = 0.4
    ):
        super().__init__()
        self.moco_weight = moco_weight
        self.genre_weight = genre_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits_contrastive: torch.Tensor,
        labels_contrastive: torch.Tensor,
        logits_genre: torch.Tensor,
        labels_genre: torch.Tensor
    ) -> dict:
        """Compute combined loss.

        Args:
            logits_contrastive: (B, 1+K) MoCo logits
            labels_contrastive: (B,) targets (all zeros)
            logits_genre: (B, num_genres) genre logits
            labels_genre: (B, num_genres) multi-hot genre labels

        Returns:
            Dictionary with:
            - loss: Combined weighted loss
            - loss_moco: Contrastive loss
            - loss_genre: Genre BCE loss
        """
        # MoCo contrastive loss (cross-entropy, positive at index 0)
        loss_moco = self.ce_loss(logits_contrastive, labels_contrastive)

        # Genre BCE loss (multi-label)
        loss_genre = self.bce_loss(logits_genre, labels_genre)

        # Combined loss
        loss = self.moco_weight * loss_moco + self.genre_weight * loss_genre

        return {
            "loss": loss,
            "loss_moco": loss_moco,
            "loss_genre": loss_genre
        }
