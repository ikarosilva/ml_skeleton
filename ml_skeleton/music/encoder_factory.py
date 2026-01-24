"""Factory functions for creating encoders, losses, and datasets.

This module provides a unified interface for creating encoder-related components
based on configuration. Supports A/B testing between different encoder types.

Encoder Types:
- "simple": Current 1D CNN on raw waveform (baseline)
- "simsiam": SimSiam with ResNet backbone on spectrograms (NEW)

Usage:
    from ml_skeleton.music.encoder_factory import (
        create_encoder,
        create_loss_fn,
        create_dataset,
        get_encoder_type
    )

    encoder = create_encoder(config)
    loss_fn = create_loss_fn(config)
    dataset = create_dataset(config, songs, album_to_idx, filename_to_albums)
"""

from typing import Optional, Any
import torch.nn as nn

from ..music.clementine_db import Song


def get_encoder_type(config: dict) -> str:
    """Get encoder type from configuration.

    Args:
        config: Configuration dictionary with 'encoder' section

    Returns:
        Encoder type string ("simple" or "simsiam")
    """
    return config.get('encoder', {}).get('encoder_type', 'simple')


def create_encoder(config: dict) -> nn.Module:
    """Factory function to create encoder based on config.

    Args:
        config: Configuration dictionary with 'encoder' and 'music' sections

    Returns:
        Encoder module (SimpleAudioEncoder or SimSiamEncoder)

    Raises:
        ValueError: If encoder_type is not recognized
    """
    encoder_type = get_encoder_type(config)
    encoder_config = config['encoder']
    music_config = config['music']

    if encoder_type == "simple":
        from .baseline_encoder import SimpleAudioEncoder

        # Get base_channels from nested config or fallback to top-level
        base_channels = encoder_config.get('simple', {}).get(
            'base_channels',
            encoder_config.get('base_channels', 32)
        )

        return SimpleAudioEncoder(
            sample_rate=music_config['sample_rate'],
            duration=music_config['audio_duration'],
            embedding_dim=encoder_config['embedding_dim'],
            base_channels=base_channels
        )

    elif encoder_type == "simsiam":
        from .simsiam_encoder import SimSiamEncoder

        simsiam_config = encoder_config.get('simsiam', {})

        return SimSiamEncoder(
            sample_rate=music_config['sample_rate'],
            duration=music_config['audio_duration'],
            embedding_dim=encoder_config['embedding_dim'],
            backbone=simsiam_config.get('backbone', 'resnet50'),
            pretrained=simsiam_config.get('pretrained_backbone', False),
            projection_dim=simsiam_config.get('projection_dim', 2048),
            predictor_hidden_dim=simsiam_config.get('predictor_hidden_dim', 512),
            n_mels=simsiam_config.get('n_mels', 128),
            n_fft=simsiam_config.get('n_fft', 2048),
            hop_length=simsiam_config.get('hop_length', 512)
        )

    else:
        raise ValueError(
            f"Unknown encoder_type: {encoder_type}. "
            f"Valid options: 'simple', 'simsiam'"
        )


def create_loss_fn(config: dict) -> nn.Module:
    """Factory function to create loss based on encoder type.

    Args:
        config: Configuration dictionary with 'encoder' section

    Returns:
        Loss function module

    Raises:
        ValueError: If encoder_type or loss_type is not recognized
    """
    encoder_type = get_encoder_type(config)
    encoder_config = config['encoder']

    if encoder_type == "simple":
        # Use existing loss selection logic for simple encoder
        loss_type = encoder_config.get('loss_type', 'metadata_contrastive')
        use_augmentation = encoder_config.get('use_augmentation', False)

        # When augmentation is enabled, use NTXentLoss
        if use_augmentation:
            from .losses import NTXentLoss
            return NTXentLoss(
                temperature=encoder_config.get('contrastive_temperature', 0.5)
            )

        if loss_type == 'metadata_contrastive':
            from .losses import MetadataContrastiveLoss
            return MetadataContrastiveLoss(
                temperature=encoder_config.get('contrastive_temperature', 0.5),
                year_threshold=encoder_config.get('year_threshold', 5),
                use_artist=encoder_config.get('use_artist', True),
                use_album=encoder_config.get('use_album', True),
                use_year=encoder_config.get('use_year', False)
            )
        elif loss_type == 'supervised_contrastive':
            from .losses import SupervisedContrastiveLoss
            return SupervisedContrastiveLoss(
                temperature=encoder_config.get('contrastive_temperature', 0.5),
                rating_threshold=encoder_config.get('rating_threshold', 0.2)
            )
        elif loss_type == 'contrastive':
            from .losses import NTXentLoss
            return NTXentLoss(
                temperature=encoder_config.get('contrastive_temperature', 0.5)
            )
        else:
            return nn.MSELoss()

    elif encoder_type == "simsiam":
        from .losses import SimSiamLoss
        return SimSiamLoss()

    else:
        raise ValueError(
            f"Unknown encoder_type: {encoder_type}. "
            f"Valid options: 'simple', 'simsiam'"
        )


def create_dataset(
    config: dict,
    songs: list[Song],
    album_to_idx: dict[str, int],
    filename_to_albums: dict[str, list[str]],
    is_training: bool = True,
    speech_results: Optional[dict[str, float]] = None
):
    """Factory function to create dataset based on encoder type.

    Args:
        config: Configuration dictionary
        songs: List of Song objects from Clementine DB
        album_to_idx: Mapping from album key to integer index
        filename_to_albums: Mapping from filename to list of album keys
        is_training: If True, applies augmentations (for SimSiam)
        speech_results: Optional speech detection scores for filtering

    Returns:
        Dataset instance (MusicDataset or SimSiamMusicDataset)
    """
    encoder_type = get_encoder_type(config)
    encoder_config = config['encoder']
    music_config = config['music']

    if encoder_type == "simple":
        from .dataset import MusicDataset

        use_augmentation = encoder_config.get('use_augmentation', False)
        crop_jitter = encoder_config.get('crop_jitter', 5.0)
        noise_level = encoder_config.get('noise_level', 0.0)

        return MusicDataset(
            songs=songs,
            album_to_idx=album_to_idx,
            filename_to_albums=filename_to_albums,
            sample_rate=music_config['sample_rate'],
            duration=music_config['audio_duration'],
            crop_position=music_config.get('crop_position', 'end'),
            normalize=music_config.get('normalize', True),
            only_rated=False,
            skip_unknown_metadata=music_config.get('skip_unknown_metadata', True),
            use_augmentation=use_augmentation,
            crop_jitter=crop_jitter,
            noise_level=noise_level,
            speech_results=speech_results,
            speech_threshold=music_config.get('speech_threshold', 0.5)
        )

    elif encoder_type == "simsiam":
        from .dataset import SimSiamMusicDataset
        from .augmentations import create_audio_augmentor

        simsiam_config = encoder_config.get('simsiam', {})

        # Create augmentor for training, None for validation/inference
        augmentor = None
        if is_training:
            augmentor = create_audio_augmentor(
                config=simsiam_config.get('augmentation', {}),
                sample_rate=music_config['sample_rate']
            )

        return SimSiamMusicDataset(
            songs=songs,
            sample_rate=music_config['sample_rate'],
            duration=music_config['audio_duration'],
            crop_position=music_config.get('crop_position', 'end'),
            normalize=music_config.get('normalize', True),
            augmentor=augmentor,
            n_mels=simsiam_config.get('n_mels', 128),
            n_fft=simsiam_config.get('n_fft', 2048),
            hop_length=simsiam_config.get('hop_length', 512),
            skip_unknown_metadata=music_config.get('skip_unknown_metadata', False),
            speech_results=speech_results,
            speech_threshold=music_config.get('speech_threshold', 0.5)
        )

    else:
        raise ValueError(
            f"Unknown encoder_type: {encoder_type}. "
            f"Valid options: 'simple', 'simsiam'"
        )


def create_optimizer(config: dict, model: nn.Module):
    """Factory function to create optimizer based on encoder type.

    Args:
        config: Configuration dictionary
        model: Model to optimize

    Returns:
        PyTorch optimizer
    """
    import torch.optim as optim

    encoder_type = get_encoder_type(config)
    encoder_config = config['encoder']

    if encoder_type == "simsiam":
        simsiam_config = encoder_config.get('simsiam', {})
        optimizer_type = simsiam_config.get('optimizer', 'sgd')

        if optimizer_type == 'sgd':
            return optim.SGD(
                model.parameters(),
                lr=simsiam_config.get('sgd_learning_rate', 0.03),
                momentum=simsiam_config.get('sgd_momentum', 0.9),
                weight_decay=simsiam_config.get('sgd_weight_decay', 0.0005)
            )
        else:
            # Fall through to Adam
            pass

    # Default: Adam optimizer (used for simple encoder or simsiam with adam)
    if 'adam_beta1' in encoder_config and 'adam_beta2' in encoder_config:
        betas = (encoder_config['adam_beta1'], encoder_config['adam_beta2'])
    else:
        betas = tuple(encoder_config.get('adam_betas', [0.9, 0.999]))

    use_adamw = encoder_config.get('adam_decoupled_weight_decay', False)
    optimizer_cls = optim.AdamW if use_adamw else optim.Adam

    return optimizer_cls(
        model.parameters(),
        lr=encoder_config['learning_rate'],
        betas=betas,
        eps=encoder_config.get('adam_eps', 1e-08),
        weight_decay=encoder_config.get('adam_weight_decay', 0.0),
        amsgrad=encoder_config.get('adam_amsgrad', False)
    )


def get_mlflow_tags(config: dict) -> dict[str, str]:
    """Get MLflow tags for A/B testing based on encoder type.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of MLflow tags for the current encoder configuration
    """
    encoder_type = get_encoder_type(config)
    encoder_config = config['encoder']

    tags = {
        'encoder_type': encoder_type,
    }

    if encoder_type == "simple":
        tags['loss_type'] = encoder_config.get('loss_type', 'metadata_contrastive')
        tags['use_augmentation'] = str(encoder_config.get('use_augmentation', False))

    elif encoder_type == "simsiam":
        simsiam_config = encoder_config.get('simsiam', {})
        tags['loss_type'] = 'simsiam'
        tags['backbone'] = simsiam_config.get('backbone', 'resnet50')
        tags['pretrained'] = str(simsiam_config.get('pretrained_backbone', False))
        tags['projection_dim'] = str(simsiam_config.get('projection_dim', 2048))

    # Combined experiment variant for easy filtering
    if encoder_type == "simsiam":
        tags['experiment_variant'] = f"simsiam_{tags.get('backbone', 'resnet50')}"
    else:
        tags['experiment_variant'] = f"simple_{tags.get('loss_type', 'unknown')}"

    return tags
