"""Factory functions for creating encoders, losses, and datasets.

This module provides a unified interface for creating encoder-related components
based on configuration. Currently supports MoCo v2 architecture.

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

from typing import Optional
import torch.nn as nn

from ..music.clementine_db import Song


def get_encoder_type(config: dict) -> str:
    """Get encoder type from configuration.

    Args:
        config: Configuration dictionary with 'encoder' section

    Returns:
        Encoder type string ("moco")
    """
    return config.get('encoder', {}).get('encoder_type', 'moco')


def create_encoder(config: dict) -> nn.Module:
    """Factory function to create encoder based on config.

    Args:
        config: Configuration dictionary with 'encoder' and 'music' sections

    Returns:
        MoCoEncoder module
    """
    encoder_config = config['encoder']
    music_config = config['music']
    encoder_type = get_encoder_type(config)

    if encoder_type != 'moco':
        raise ValueError(f"Unsupported encoder type: {encoder_type}. Only 'moco' is supported.")

    from .moco_encoder import MoCoEncoder
    moco_config = encoder_config.get('moco', {})
    cqt_config = config.get('cqt', {})

    return MoCoEncoder(
        sample_rate=music_config['sample_rate'],
        embedding_dim=encoder_config.get('embedding_dim', 2048),
        pretrained_backbone=encoder_config.get('pretrained_backbone', True),
        queue_size=moco_config.get('queue_size', 4096),
        momentum=moco_config.get('momentum', 0.999),
        temperature=moco_config.get('temperature', 0.07),
        projection_dim=moco_config.get('projection_dim', 128),
        num_genres=encoder_config.get('genre', {}).get('num_categories', 7),
        n_bins=cqt_config.get('n_bins', 84),
        fmin=cqt_config.get('fmin', 32.7),
        hop_length=cqt_config.get('hop_length', 512)
    )


def create_loss_fn(config: dict) -> nn.Module:
    """Factory function to create loss function.

    Args:
        config: Configuration dictionary with 'encoder' section

    Returns:
        MoCoLoss module
    """
    encoder_type = get_encoder_type(config)

    if encoder_type != 'moco':
        raise ValueError(f"Unsupported encoder type: {encoder_type}. Only 'moco' is supported.")

    from .moco_encoder import MoCoLoss
    encoder_config = config['encoder']
    loss_weights = encoder_config.get('loss_weights', {})
    return MoCoLoss(
        moco_weight=loss_weights.get('moco', 0.6),
        genre_weight=loss_weights.get('genre_bce', 0.4)
    )


def create_dataset(
    config: dict,
    songs: list[Song],
    album_to_idx: dict[str, int],
    filename_to_albums: dict[str, list[str]],
    is_training: bool = True,
    speech_results: Optional[dict[str, float]] = None
):
    """Factory function to create dataset.

    Args:
        config: Configuration dictionary
        songs: List of Song objects from Clementine DB
        album_to_idx: Mapping from album key to integer index
        filename_to_albums: Mapping from filename to list of album keys
        is_training: If True, applies augmentations
        speech_results: Optional speech detection scores for filtering

    Returns:
        MoCoDataset instance
    """
    encoder_config = config['encoder']
    music_config = config['music']
    encoder_type = get_encoder_type(config)

    if encoder_type != 'moco':
        raise ValueError(f"Unsupported encoder type: {encoder_type}. Only 'moco' is supported.")

    from .moco_dataset import MoCoDataset, AudioAugmentor
    chunk_cache_config = music_config.get('chunk_cache', {})
    aug_config = encoder_config.get('augmentation', {})

    augmentor = AudioAugmentor(
        sample_rate=music_config['sample_rate'],
        crop_duration_range=(
            aug_config.get('crop_duration_min', 5.0),
            aug_config.get('crop_duration_max', 15.0)
        ),
        gain_db_range=(
            aug_config.get('gain_db_min', -2.0),
            aug_config.get('gain_db_max', 2.0)
        ),
        noise_prob=aug_config.get('noise_prob', 0.5),
        noise_snr_range=(
            aug_config.get('noise_snr_min', 25.0),
            aug_config.get('noise_snr_max', 35.0)
        ),
        mixup_prob=aug_config.get('mixup_prob', 0.5),
        mixup_alpha=aug_config.get('mixup_alpha', 0.1)
    )

    return MoCoDataset(
        songs=songs,
        cache_dir=chunk_cache_config.get('directory', './cache/chunks'),
        num_chunks=chunk_cache_config.get('num_chunks', 4),
        sample_rate=music_config['sample_rate'],
        augmentor=augmentor,
        same_album_prob=aug_config.get('same_album_positive_prob', 0.3)
    )


def create_optimizer(config: dict, model: nn.Module):
    """Factory function to create optimizer.

    Args:
        config: Configuration dictionary
        model: Model to optimize

    Returns:
        PyTorch optimizer (Adam/AdamW for MoCo)
    """
    import torch.optim as optim

    encoder_config = config['encoder']

    # Get beta values
    if 'adam_beta1' in encoder_config and 'adam_beta2' in encoder_config:
        betas = (encoder_config['adam_beta1'], encoder_config['adam_beta2'])
    else:
        betas = tuple(encoder_config.get('adam_betas', [0.9, 0.999]))

    # Use AdamW if decoupled weight decay is enabled
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
    """Get MLflow tags for experiment tracking.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of MLflow tags
    """
    encoder_config = config['encoder']
    moco_config = encoder_config.get('moco', {})

    return {
        'encoder_type': 'moco',
        'loss_type': 'moco_genre',
        'backbone': encoder_config.get('backbone', 'resnet50'),
        'pretrained': str(encoder_config.get('pretrained_backbone', True)),
        'queue_size': str(moco_config.get('queue_size', 4096)),
        'temperature': str(moco_config.get('temperature', 0.07)),
        'experiment_variant': f"moco_{encoder_config.get('backbone', 'resnet50')}"
    }
