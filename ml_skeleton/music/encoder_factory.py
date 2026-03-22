"""Factory functions for creating encoders, losses, and datasets.

This module provides a unified interface for creating encoder-related components
based on configuration. Currently supports MoCo v2 architecture.

Usage:
    from ml_skeleton.music.encoder_factory import (
        create_encoder,
        create_loss_fn,
        create_dataset,
        get_encoder_type,
        get_fingerprint_db_path,
    )

    encoder = create_encoder(config)
    loss_fn = create_loss_fn(config)
    dataset = create_dataset(config, songs, album_to_idx, filename_to_albums)
"""

import os
from typing import Optional
import torch.nn as nn

from ..music.clementine_db import Song

# Default root for all cache stores (fingerprints DB, chunk cache)
DEFAULT_CACHE_ROOT = "./cache"
DEFAULT_FINGERPRINT_DB_PATH = "./cache/fingerprints.db"


def get_cache_root(config: dict, resolve_absolute: bool = True) -> str:
    """Return the root directory for all cache stores (fingerprints DB, chunks, etc.).

    Args:
        config: Full config dict; music.cache_root used if set.
        resolve_absolute: If True, return absolute path.

    Returns:
        Path to the cache root (e.g. ./cache or /path/to/cache).
    """
    root = config.get("music", {}).get("cache_root", DEFAULT_CACHE_ROOT)
    if resolve_absolute:
        root = os.path.abspath(root)
    return root


def get_chunk_cache_dir(config: dict, resolve_absolute: bool = True) -> str:
    """Return the chunk cache directory (under music.cache_root when path is relative)."""
    music_config = config.get("music", {})
    chunk_config = music_config.get("chunk_cache", {})
    raw = chunk_config.get("directory", os.path.join(DEFAULT_CACHE_ROOT, "chunks"))
    if not os.path.isabs(raw):
        root = get_cache_root(config, resolve_absolute=False)
        path = os.path.normpath(os.path.join(root, raw.lstrip("./")))
    else:
        path = raw
    if resolve_absolute:
        path = os.path.abspath(path)
    return path


def get_fingerprint_db_path(config: dict, resolve_absolute: bool = True) -> str:
    """Return the canonical fingerprint DB path from config (same DB for encoder, classifier, fingerprint, enrich).

    Path is under music.cache_root when fingerprint_db_path is relative.

    Args:
        config: Full config dict (fingerprinting.fingerprint_db_path or derived from cache_root).
        resolve_absolute: If True, return absolute path so cwd does not change which file is used.

    Returns:
        Path to the fingerprint SQLite database.
    """
    raw = config.get("fingerprinting", {}).get("fingerprint_db_path") or "fingerprints.db"
    if not os.path.isabs(raw):
        root = get_cache_root(config, resolve_absolute=False)
        path = os.path.normpath(os.path.join(root, raw.lstrip("./")))
    else:
        path = raw
    if resolve_absolute:
        path = os.path.abspath(path)
    return path


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
        MoCoEncoder or FingerprintEncoder module
    """
    encoder_config = config['encoder']
    music_config = config['music']
    encoder_type = get_encoder_type(config)

    if encoder_type == 'fingerprint_baseline':
        from .fingerprint_encoder import FingerprintEncoder
        fp_config = config.get('fingerprinting', {})
        bl_config = encoder_config.get('fingerprint_baseline', {})
        return FingerprintEncoder(
            fp_db_path=get_fingerprint_db_path(config),
            embedding_dim=bl_config.get('embedding_dim'),
            project_dim=bl_config.get('project_dim'),
            chromaprint_chunk_idx=fp_config.get('chunk_for_fingerprinting', 1),
        )

    if encoder_type != 'moco':
        raise ValueError(f"Unsupported encoder type: {encoder_type}. Only 'moco' and 'fingerprint_baseline' are supported.")

    from .moco_encoder import MoCoEncoder
    moco_config = encoder_config.get('moco', {})
    cqt_config = config.get('cqt', {})

    chromaprint_loss_weight = encoder_config.get('chromaprint_loss_weight', 0.0)

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
        hop_length=cqt_config.get('hop_length', 512),
        use_chromaprint=(chromaprint_loss_weight > 0)
    )


def create_loss_fn(config: dict) -> nn.Module:
    """Factory function to create loss function.

    Args:
        config: Configuration dictionary with 'encoder' section

    Returns:
        MoCoLoss module (or dummy for fingerprint_baseline; not used for training)
    """
    encoder_type = get_encoder_type(config)

    if encoder_type == 'fingerprint_baseline':
        # Not used; fingerprint baseline only does extraction, no training
        from .fingerprint_encoder import _DummyLoss
        return _DummyLoss()

    if encoder_type != 'moco':
        raise ValueError(f"Unsupported encoder type: {encoder_type}. Only 'moco' and 'fingerprint_baseline' are supported.")

    from .moco_encoder import MoCoLoss
    encoder_config = config['encoder']
    loss_weights = encoder_config.get('loss_weights', {})
    chromaprint_loss_weight = encoder_config.get('chromaprint_loss_weight', 0.0)
    return MoCoLoss(
        moco_weight=loss_weights.get('moco', 0.6),
        genre_weight=loss_weights.get('genre_bce', 0.4),
        chromaprint_weight=chromaprint_loss_weight
    )


def create_dataset(
    config: dict,
    songs: list[Song],
    album_to_idx: dict[str, int],
    filename_to_albums: dict[str, list[str]],
    is_training: bool = True,
    speech_results: Optional[dict[str, float]] = None,
    chunk_indices_override: Optional[list[int]] = None,
):
    """Factory function to create dataset.

    Args:
        config: Configuration dictionary
        songs: List of Song objects from Clementine DB
        album_to_idx: Mapping from album key to integer index
        filename_to_albums: Mapping from filename to list of album keys
        is_training: If True, applies augmentations
        speech_results: Optional speech detection scores for filtering
        chunk_indices_override: If set (e.g. for HPO), MoCo uses only these chunk indices
            so fewer chunks are loaded per song (e.g. 4 instead of 8) to fit in RAM.

    Returns:
        MoCoDataset or FingerprintBaselineDataset instance
    """
    encoder_config = config['encoder']
    music_config = config['music']
    encoder_type = get_encoder_type(config)

    if encoder_type == 'fingerprint_baseline':
        from .fingerprint_encoder import FingerprintBaselineDataset
        fp_config = config.get('fingerprinting', {})
        return FingerprintBaselineDataset(
            songs=songs,
            fp_db_path=fp_config.get('fingerprint_db_path', './cache/fingerprints.db'),
            chromaprint_chunk_idx=fp_config.get('chunk_for_fingerprinting', 1),
        )

    if encoder_type != 'moco':
        raise ValueError(f"Unsupported encoder type: {encoder_type}. Only 'moco' and 'fingerprint_baseline' are supported.")

    from .moco_dataset import MoCoDataset, AudioAugmentor
    from .fingerprint_db import FingerprintDB

    chunk_cache_config = music_config.get('chunk_cache', {})
    aug_config = encoder_config.get('augmentation', {})

    chromaprint_loss_weight = encoder_config.get('chromaprint_loss_weight', 0.0)
    fp_db = None
    if chromaprint_loss_weight > 0:
        fp_db = FingerprintDB(get_fingerprint_db_path(config))

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

    fp_config = config.get('fingerprinting', {})
    chromaprint_chunk_idx = fp_config.get('chunk_for_fingerprinting', 1)
    preload_chromaprint = encoder_config.get('preload_chromaprint', True)
    min_chunks = chunk_cache_config.get('min_chunks_for_training', 3)  # exclude 1–2 chunk songs for MoCo
    if fp_db is not None:
        print(f"  Chromaprint chunk index: {chromaprint_chunk_idx} (config: chunk_for_fingerprinting)")
    return MoCoDataset(
        songs=songs,
        cache_dir=get_chunk_cache_dir(config),
        num_chunks=chunk_cache_config.get('num_chunks', 8),
        sample_rate=music_config['sample_rate'],
        augmentor=augmentor,
        same_album_prob=aug_config.get('same_album_positive_prob', 0.0),
        far_chunk_prob=aug_config.get('far_chunk_prob', 0.0),
        min_chunk_distance=aug_config.get('min_chunk_distance', 2),
        fp_db=fp_db,
        chromaprint_chunk_idx=chromaprint_chunk_idx,
        preload_chromaprint=preload_chromaprint,
        chunk_indices=chunk_indices_override,
        min_chunks=min_chunks,
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
