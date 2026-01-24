"""Factory functions for creating encoders, losses, and datasets.

This module provides a unified interface for creating encoder-related components
based on configuration.

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
        Encoder type string (currently only "simsiam" supported)
    """
    return config.get('encoder', {}).get('encoder_type', 'simsiam')


def create_encoder(config: dict) -> nn.Module:
    """Factory function to create encoder based on config.

    Args:
        config: Configuration dictionary with 'encoder' and 'music' sections

    Returns:
        SimSiamEncoder module
    """
    encoder_config = config['encoder']
    music_config = config['music']
    simsiam_config = encoder_config.get('simsiam', {})

    from .simsiam_encoder import SimSiamEncoder

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


def create_loss_fn(config: dict) -> nn.Module:
    """Factory function to create loss function.

    Args:
        config: Configuration dictionary with 'encoder' section

    Returns:
        SimSiamLoss module
    """
    from .losses import SimSiamLoss
    return SimSiamLoss()


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
        SimSiamMusicDataset instance
    """
    encoder_config = config['encoder']
    music_config = config['music']
    simsiam_config = encoder_config.get('simsiam', {})

    from .dataset import SimSiamMusicDataset
    from .augmentations import create_audio_augmentor

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


def create_optimizer(config: dict, model: nn.Module):
    """Factory function to create optimizer.

    Args:
        config: Configuration dictionary
        model: Model to optimize

    Returns:
        PyTorch optimizer (SGD for SimSiam)
    """
    import torch.optim as optim

    encoder_config = config['encoder']
    simsiam_config = encoder_config.get('simsiam', {})
    optimizer_type = simsiam_config.get('optimizer', 'sgd')

    if optimizer_type == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=simsiam_config.get('sgd_learning_rate', 0.03),
            momentum=simsiam_config.get('sgd_momentum', 0.9),
            weight_decay=simsiam_config.get('sgd_weight_decay', 0.0005)
        )

    # Adam fallback
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
    """Get MLflow tags for experiment tracking.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of MLflow tags
    """
    encoder_config = config['encoder']
    simsiam_config = encoder_config.get('simsiam', {})

    return {
        'encoder_type': 'simsiam',
        'loss_type': 'simsiam',
        'backbone': simsiam_config.get('backbone', 'resnet50'),
        'pretrained': str(simsiam_config.get('pretrained_backbone', False)),
        'projection_dim': str(simsiam_config.get('projection_dim', 2048)),
        'experiment_variant': f"simsiam_{simsiam_config.get('backbone', 'resnet50')}"
    }
