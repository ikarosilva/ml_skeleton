"""SimSiam encoder implementation for self-supervised audio representation learning.

SimSiam (Simple Siamese) learns representations without negative pairs, large
batch sizes, or momentum encoders. It uses:
1. A shared backbone (ResNet) for both views
2. A projection MLP (encoder head)
3. A predictor MLP (asymmetric, only on one branch)
4. Stop-gradient on one branch to prevent collapse

Architecture:
    Audio → Spectrogram → ResNet Backbone → Projection MLP → z (projection)
                                                    ↓
                                            Predictor MLP → p (prediction)

Training uses:
    L = D(p1, stopgrad(z2)) / 2 + D(p2, stopgrad(z1)) / 2
    where D = negative cosine similarity

Reference:
    "Exploring Simple Siamese Representation Learning" (Chen & He, 2020)
    https://arxiv.org/abs/2011.10566
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T
from torchvision import models
from typing import Optional

from ml_skeleton.music.augmentations import SpecAugment


class ProjectionMLP(nn.Module):
    """Projection MLP head for SimSiam.

    Maps backbone features to projection space. The output 'z' is used
    for the similarity computation.

    Architecture:
        Linear → BatchNorm → ReLU → Linear → BatchNorm → ReLU → Linear → BatchNorm

    Note:
        - Uses batch normalization throughout
        - No ReLU after the final layer (just BatchNorm)
        - 3-layer MLP as per the paper

    Args:
        in_dim: Input dimension (from backbone)
        hidden_dim: Hidden layer dimension
        out_dim: Output projection dimension
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        out_dim: int = 2048
    ):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim)
            # No ReLU after final layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through projection MLP.

        Args:
            x: Input features, shape (batch_size, in_dim)

        Returns:
            Projections z, shape (batch_size, out_dim)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class PredictorMLP(nn.Module):
    """Predictor MLP for SimSiam.

    Applied only to one branch to create asymmetry. Uses a bottleneck
    architecture which is crucial for preventing collapse.

    Architecture:
        Linear → BatchNorm → ReLU → Linear

    Note:
        - Bottleneck: in_dim → hidden_dim (smaller) → out_dim
        - The bottleneck (hidden_dim < in_dim) is important for SimSiam
        - 2-layer MLP as per the paper

    Args:
        in_dim: Input dimension (= projection dimension)
        hidden_dim: Bottleneck dimension (typically 1/4 of in_dim)
        out_dim: Output dimension (= projection dimension)
    """

    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dim: int = 512,
        out_dim: int = 2048
    ):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # Output layer has bias (unlike projection MLP)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through predictor MLP.

        Args:
            x: Projections z, shape (batch_size, in_dim)

        Returns:
            Predictions p, shape (batch_size, out_dim)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiamEncoder(nn.Module):
    """SimSiam encoder for audio representation learning.

    Combines:
    1. Mel-spectrogram transform (audio → 2D spectrogram)
    2. ResNet backbone (spectrogram → features)
    3. Projection MLP (features → projection z)
    4. Predictor MLP (projection → prediction p)

    For inference/embedding extraction, use get_embedding() which returns
    the projection z (before the predictor).

    Args:
        sample_rate: Audio sample rate (Hz)
        duration: Audio duration (seconds)
        embedding_dim: Output embedding dimension (used for compatibility)
        backbone: ResNet variant ("resnet18", "resnet34", "resnet50")
        pretrained: Use ImageNet pretrained weights
        projection_dim: Projection MLP output dimension
        predictor_hidden_dim: Predictor bottleneck dimension
        n_mels: Number of mel frequency bins
        n_fft: FFT window size
        hop_length: Hop length for STFT
        spec_augment: If True, apply SpecAugment on the GPU
        spec_augment_params: Parameters for SpecAugment
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        duration: float = 60.0,
        embedding_dim: int = 2048,
        backbone: str = "resnet50",
        pretrained: bool = False,
        projection_dim: int = 2048,
        predictor_hidden_dim: int = 512,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        spec_augment: bool = True,
        spec_augment_params: Optional[dict] = None
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.duration = duration
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim

        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )

        # SpecAugment (on-GPU)
        if spec_augment:
            params = spec_augment_params or {}
            self.spec_augment = SpecAugment(**params)
        else:
            self.spec_augment = None

        # Create backbone
        self.backbone, backbone_dim = self._create_backbone(backbone, pretrained)

        # Projection MLP (features → z)
        self.projector = ProjectionMLP(
            in_dim=backbone_dim,
            hidden_dim=projection_dim,
            out_dim=projection_dim
        )

        # Predictor MLP (z → p)
        self.predictor = PredictorMLP(
            in_dim=projection_dim,
            hidden_dim=predictor_hidden_dim,
            out_dim=projection_dim
        )

    def _create_backbone(
        self,
        backbone_name: str,
        pretrained: bool
    ) -> tuple[nn.Module, int]:
        """Create ResNet backbone.

        Args:
            backbone_name: ResNet variant name
            pretrained: Use ImageNet pretrained weights

        Returns:
            Tuple of (backbone module, output dimension)
        """
        weights = "IMAGENET1K_V1" if pretrained else None

        if backbone_name == "resnet18":
            backbone = models.resnet18(weights=weights)
            dim = 512
        elif backbone_name == "resnet34":
            backbone = models.resnet34(weights=weights)
            dim = 512
        elif backbone_name == "resnet50":
            backbone = models.resnet50(weights=weights)
            dim = 2048
        elif backbone_name == "resnet101":
            backbone = models.resnet101(weights=weights)
            dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        # Remove the classification head
        backbone.fc = nn.Identity()

        return backbone, dim

    def _audio_to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio waveform to mel-spectrogram for ResNet.

        Args:
            audio: Raw audio waveform, shape (batch, samples)

        Returns:
            Spectrogram suitable for ResNet, shape (batch, 3, n_mels, time)
        """
        # Add channel dimension if needed: (batch, samples) → (batch, 1, samples)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Compute mel spectrogram: (batch, 1, n_mels, time)
        mel_spec = self.mel_transform(audio.squeeze(1))

        # Add channel dimension: (batch, n_mels, time) → (batch, 1, n_mels, time)
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)

        # Log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Apply SpecAugment if enabled
        if self.spec_augment is not None and self.training:
            mel_spec = self.spec_augment(mel_spec)

        # Normalize to [0, 1]
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-9)

        # Repeat to 3 channels for ResNet
        mel_spec = mel_spec.repeat(1, 3, 1, 1)

        return mel_spec

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through backbone only.

        Args:
            x: Spectrogram tensor, shape (batch, 3, n_mels, time)

        Returns:
            Backbone features, shape (batch, backbone_dim)
        """
        return self.backbone(x)

    def forward_projection(self, features: torch.Tensor) -> torch.Tensor:
        """Forward through projection MLP.

        Args:
            features: Backbone features, shape (batch, backbone_dim)

        Returns:
            Projections z, shape (batch, projection_dim)
        """
        return self.projector(features)

    def forward_predictor(self, z: torch.Tensor) -> torch.Tensor:
        """Forward through predictor MLP.

        Args:
            z: Projections, shape (batch, projection_dim)

        Returns:
            Predictions p, shape (batch, projection_dim)
        """
        return self.predictor(z)

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a single view.

        For training, use forward_simsiam() which handles both views.
        This method is for embedding extraction.

        Args:
            x: Input tensor - either:
               - Raw audio waveform, shape (batch, samples)
               - Spectrogram, shape (batch, 3, n_mels, time)
            return_all: If True, return (z, p, features) for debugging

        Returns:
            If return_all=False: Embeddings z (projections), shape (batch, projection_dim)
            If return_all=True: Tuple of (z, p, features)
        """
        # Check if input is audio or spectrogram
        # Spectrogram has 4 dims: (batch, channels, freq, time)
        # Audio has 2 dims: (batch, samples)
        if x.dim() == 2:
            x = self._audio_to_spectrogram(x)

        # Backbone
        features = self.forward_backbone(x)

        # Projection
        z = self.forward_projection(features)

        if return_all:
            p = self.forward_predictor(z)
            return z, p, features

        return z

    def forward_simsiam(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for SimSiam training with two views.

        Args:
            view1: First augmented view (audio or spectrogram)
            view2: Second augmented view (audio or spectrogram)

        Returns:
            Tuple of (p1, p2, z1, z2) for SimSiam loss computation:
            - p1: Predictions from view 1
            - p2: Predictions from view 2
            - z1: Projections from view 1
            - z2: Projections from view 2
        """
        # Convert audio to spectrogram if needed
        if view1.dim() == 2:
            view1 = self._audio_to_spectrogram(view1)
        if view2.dim() == 2:
            view2 = self._audio_to_spectrogram(view2)

        # Forward both views through shared backbone and projector
        f1 = self.forward_backbone(view1)
        f2 = self.forward_backbone(view2)

        z1 = self.forward_projection(f1)
        z2 = self.forward_projection(f2)

        # Predictions (only used for loss, not for embedding)
        p1 = self.forward_predictor(z1)
        p2 = self.forward_predictor(z2)

        return p1, p2, z1, z2

    def get_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract embeddings for inference.

        Uses the projection z as the embedding (not the prediction p).

        Args:
            audio: Raw audio waveform, shape (batch, samples)

        Returns:
            Embeddings, shape (batch, projection_dim)
        """
        return self.forward(audio)

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension.

        Returns projection_dim as that's what get_embedding() returns.
        """
        return self.projection_dim