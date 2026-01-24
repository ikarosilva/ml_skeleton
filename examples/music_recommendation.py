"""Music Recommendation System - Complete Example.

This example demonstrates the full two-phase training pipeline:
1. Stage 1: Train audio encoder (audio -> embeddings)
2. Stage 2: Train rating classifier (embeddings -> ratings)
3. Generate recommendations
4. Model card generation (HuggingFace-compatible)
5. Hyperparameter tuning with Optuna/Ray Tune

Usage:
    # Run complete pipeline (encoder + classifier + model card)
    python examples/music_recommendation.py --stage all --config configs/music_recommendation.yaml

    # Run stages individually:
    python examples/music_recommendation.py --stage encoder --config configs/music_recommendation.yaml
    python examples/music_recommendation.py --stage classifier --config configs/music_recommendation.yaml
    python examples/music_recommendation.py --stage recommend --config configs/music_recommendation.yaml

    # Hyperparameter tuning (uses search space from YAML config):
    python examples/music_recommendation.py --stage tune-encoder --config configs/music_recommendation.yaml --n-trials 30
    python examples/music_recommendation.py --stage tune-classifier --config configs/music_recommendation.yaml --n-trials 20

    # Final training with best hyperparameters (50 epochs instead of 20):
    python examples/music_recommendation.py --stage encoder --config configs/music_recommendation.yaml --final-training
    python examples/music_recommendation.py --stage all --config configs/music_recommendation.yaml --final-training

    # Automated HPO pipeline (no manual intervention):
    ./run_music_pipeline.sh hpo

    # Or manually apply best params from tuning:
    python examples/music_recommendation.py --stage encoder --config configs/music_recommendation.yaml \
        --final-training --best-params checkpoints/best_encoder_params.json
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_skeleton.music.clementine_db import ClementineDB
from ml_skeleton.music.dataset import MusicDataset, EmbeddingDataset, music_collate_fn
from ml_skeleton.music.embedding_store import EmbeddingStore
from ml_skeleton.music.losses import RatingLoss, build_album_mapping
from ml_skeleton.music.encoder_factory import (
    create_encoder,
    create_loss_fn,
    create_optimizer,
    get_encoder_type,
    get_mlflow_tags
)
from ml_skeleton.music.dataset import SimSiamMusicDataset, simsiam_collate_fn
from ml_skeleton.music.augmentations import create_audio_augmentor
from ml_skeleton.music.baseline_classifier import SimpleRatingClassifier
from ml_skeleton.music.xspf_playlist import generate_human_feedback_playlists
from ml_skeleton.training.encoder_trainer import EncoderTrainer
from ml_skeleton.training.classifier_trainer import (
    ClassifierTrainer,
    get_encoder_version_from_checkpoint,
    validate_model_compatibility
)
from ml_skeleton.music.model_card import ModelCardGenerator
from ml_skeleton.music.dataset_stats import (
    collect_preprocessing_stats,
    collect_dataset_stats,
    collect_training_stats
)

# Framework imports for hyperparameter tuning and MLflow tracking
from ml_skeleton import TrainingContext, TrainingResult, ExperimentConfig, run_experiment
from ml_skeleton.core.config import TunerType
from ml_skeleton.tracking import ExplrTracker, MLflowServer
from ml_skeleton.utils.memory import cleanup_memory


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Environment variable override:
    - CLEMENTINE_DB_PATH: Override database path from config
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Allow environment variable override for database path
    env_db_path = os.getenv('CLEMENTINE_DB_PATH')
    if env_db_path:
        print(f"Using database path from environment: {env_db_path}")
        config['music']['database_path'] = env_db_path

    return config


def apply_hyperparameters_to_config(config: dict, hyperparameters: dict, stage: str) -> dict:
    """Apply hyperparameters from tuning to config dict.

    Args:
        config: Base configuration dictionary
        hyperparameters: Hyperparameters from TrainingContext
        stage: 'encoder' or 'classifier'

    Returns:
        Updated config with hyperparameters applied
    """
    import copy
    config = copy.deepcopy(config)

    # Apply hyperparameters to the appropriate stage
    stage_config = config[stage]
    for key, value in hyperparameters.items():
        if key in stage_config:
            stage_config[key] = value
            print(f"  Tuning: {key} = {value}")

    return config


# Global variable to store model card across training runs
_global_model_card: Optional[ModelCardGenerator] = None

# Global tracking for HPO best trial (encoder)
_hpo_best_value: float = float('inf')
_hpo_best_trial: int = -1

# Global tracking for HPO best trial (classifier)
_hpo_classifier_best_value: float = float('inf')
_hpo_classifier_best_trial: int = -1


def create_encoder_training_fn(base_config: dict, n_trials: int = None):
    """Create encoder training function for hyperparameter tuning.

    Args:
        base_config: Base configuration dictionary
        n_trials: Total number of HPO trials (for progress logging)

    Returns:
        Training function that accepts TrainingContext and returns TrainingResult
    """
    def train_encoder_fn(ctx: TrainingContext) -> TrainingResult:
        """Encoder training function for framework integration.

        Args:
            ctx: TrainingContext with hyperparameters, tracker, device

        Returns:
            TrainingResult with primary metric (validation loss)
        """
        global _global_model_card

        # Get trial info for logging
        trial_info = None
        if ctx.trial_number is not None and n_trials is not None:
            # Optuna uses 0-indexed trials, display as 1-indexed
            trial_info = (ctx.trial_number + 1, n_trials)

        # Apply hyperparameters from tuning
        config = apply_hyperparameters_to_config(
            base_config,
            ctx.hyperparameters,
            stage='encoder'
        )

        # Override device if provided by context
        if ctx.device:
            config['device'] = ctx.device

        # Run training (skip embeddings during HPO to save ~10 min per trial)
        model_card = train_encoder(
            config,
            model_card=_global_model_card,
            skip_embeddings=True,  # Skip during HPO
            trial_info=trial_info,
            verbose=False  # Minimal logging during HPO
        )
        _global_model_card = model_card

        # Get best validation loss from encoder stats
        encoder_stats = model_card.encoder_stats
        best_val_loss = encoder_stats.get('best_val_loss', encoder_stats.get('final_val_loss', float('inf')))

        # Track and report new best trials
        global _hpo_best_value, _hpo_best_trial
        if best_val_loss < _hpo_best_value:
            _hpo_best_value = best_val_loss
            _hpo_best_trial = ctx.trial_number + 1 if ctx.trial_number is not None else 0
            print(f"  ★ NEW BEST (Trial {_hpo_best_trial}): val_loss={best_val_loss:.6f}")
            print(f"    Parameters: lr={ctx.hyperparameters.get('learning_rate', 'N/A'):.2e}, "
                  f"eps={ctx.hyperparameters.get('adam_eps', 'N/A'):.2e}, "
                  f"wd={ctx.hyperparameters.get('adam_weight_decay', 'N/A'):.2e}, "
                  f"amsgrad={ctx.hyperparameters.get('adam_amsgrad', 'N/A')}")

        # Log to MLflow
        ctx.tracker.log_params(ctx.hyperparameters)
        ctx.tracker.log_metric('best_val_loss', best_val_loss)
        ctx.tracker.log_metric('epochs_run', encoder_stats.get('epochs_run', 0))
        ctx.tracker.log_metric('training_time', encoder_stats.get('training_time_seconds', 0))

        return TrainingResult(
            primary_metric=best_val_loss,
            primary_metric_name='val_loss',
            minimize=True,
            metrics={
                'final_train_loss': encoder_stats.get('final_train_loss', 0),
                'final_val_loss': encoder_stats.get('final_val_loss', 0),
                'best_val_loss': best_val_loss,
                'best_epoch': encoder_stats.get('best_epoch', 0)
            },
            best_model_path=str(Path(config['checkpoint_dir']) / 'encoder_best.pt'),
            epochs_completed=encoder_stats.get('epochs_run', 0)
        )

    return train_encoder_fn


def create_classifier_training_fn(base_config: dict, n_trials: int = None):
    """Create classifier training function for hyperparameter tuning.

    Args:
        base_config: Base configuration dictionary
        n_trials: Total number of HPO trials (for progress logging)

    Returns:
        Training function that accepts TrainingContext and returns TrainingResult
    """
    def train_classifier_fn(ctx: TrainingContext) -> TrainingResult:
        """Classifier training function for framework integration.

        Args:
            ctx: TrainingContext with hyperparameters, tracker, device

        Returns:
            TrainingResult with primary metric (validation MAE)
        """
        global _global_model_card

        # Get trial info for logging
        trial_info = None
        if ctx.trial_number is not None and n_trials is not None:
            # Optuna uses 0-indexed trials, display as 1-indexed
            trial_info = (ctx.trial_number + 1, n_trials)

        # Apply hyperparameters from tuning
        config = apply_hyperparameters_to_config(
            base_config,
            ctx.hyperparameters,
            stage='classifier'
        )

        # Override device if provided by context
        if ctx.device:
            config['device'] = ctx.device

        # Run training
        model_card = train_classifier(config, model_card=_global_model_card, trial_info=trial_info, verbose=False)
        _global_model_card = model_card

        # Get metrics from classifier stats
        classifier_stats = model_card.classifier_stats
        best_val_loss = classifier_stats.get('best_val_loss', classifier_stats.get('final_val_loss', float('inf')))

        # For classifier, we optimize MAE instead of loss
        # Calculate best MAE from history if available
        best_val_mae = best_val_loss  # Fallback
        if 'val_mae' in classifier_stats:
            best_val_mae = classifier_stats['val_mae']

        # Track and report new best trials
        global _hpo_classifier_best_value, _hpo_classifier_best_trial
        if best_val_mae < _hpo_classifier_best_value:
            _hpo_classifier_best_value = best_val_mae
            _hpo_classifier_best_trial = ctx.trial_number + 1 if ctx.trial_number is not None else 0
            print(f"  ★ NEW BEST (Trial {_hpo_classifier_best_trial}): val_mae={best_val_mae:.6f}")
            print(f"    Parameters: lr={ctx.hyperparameters.get('learning_rate', 'N/A'):.2e}, "
                  f"dropout={ctx.hyperparameters.get('dropout', 'N/A')}, "
                  f"wd={ctx.hyperparameters.get('adam_weight_decay', 'N/A'):.2e}")

        # Log to MLflow
        ctx.tracker.log_params(ctx.hyperparameters)
        ctx.tracker.log_metric('best_val_loss', best_val_loss)
        ctx.tracker.log_metric('best_val_mae', best_val_mae)
        ctx.tracker.log_metric('epochs_run', classifier_stats.get('epochs_run', 0))
        ctx.tracker.log_metric('training_time', classifier_stats.get('training_time_seconds', 0))

        return TrainingResult(
            primary_metric=best_val_mae,
            primary_metric_name='val_mae',
            minimize=True,
            metrics={
                'final_train_loss': classifier_stats.get('final_train_loss', 0),
                'final_val_loss': classifier_stats.get('final_val_loss', 0),
                'best_val_loss': best_val_loss,
                'best_val_mae': best_val_mae,
                'best_epoch': classifier_stats.get('best_epoch', 0)
            },
            best_model_path=str(Path(config['checkpoint_dir']) / 'classifier_best.pt'),
            epochs_completed=classifier_stats.get('epochs_run', 0)
        )

    return train_classifier_fn


def train_encoder(
    config: dict,
    model_card: ModelCardGenerator = None,
    final_training: bool = False,
    skip_embeddings: bool = False,
    trial_info: tuple[int, int] = None,
    verbose: bool = True,
    resume_checkpoint: str = None,
    model_version_override: str = None
):
    """Stage 1: Train audio encoder.

    Args:
        config: Configuration dictionary
        model_card: Optional ModelCardGenerator to collect statistics
        final_training: If True, uses final_training_epochs (50) instead of epochs (20)
        skip_embeddings: If True, skip embedding extraction (useful during HPO)
        trial_info: Optional tuple of (trial_number, n_trials) for HPO logging
        verbose: If True, print detailed setup info (set False during HPO to reduce noise)
        resume_checkpoint: Path to checkpoint to resume training from
        model_version_override: Override model version for embeddings (e.g., 'v2')

    Returns:
        ModelCardGenerator with encoder statistics
    """
    import time

    # Ensure clean memory state at start of stage
    cleanup_memory()

    print("=" * 80)
    if trial_info:
        trial_num, n_trials = trial_info
        print(f"STAGE 1: ENCODER TRAINING (Optuna Trial {trial_num}/{n_trials})")
    else:
        print("STAGE 1: ENCODER TRAINING")
    print("=" * 80)

    # Create model card generator if not provided
    if model_card is None:
        model_card = ModelCardGenerator()

    # Load configuration
    music_config = config['music']
    encoder_config = config['encoder']
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Select appropriate epoch count based on training mode
    if final_training and 'final_training_epochs' in encoder_config:
        num_epochs = encoder_config['final_training_epochs']
        if verbose:
            print(f"  Using final_training_epochs={num_epochs} for training with best hyperparameters")
    else:
        num_epochs = encoder_config['epochs']
        if verbose:
            print(f"  Using epochs={num_epochs} (HPO/regular training mode)")

    # Store config in model card
    model_card.set_config(config)

    # Initialize MLflow tracking
    mlflow_config = config.get('mlflow', {})
    mlflow_enabled = mlflow_config.get('auto_start', True)

    if False and mlflow_enabled:  #TODO DEBUG
        # Start MLflow server if configured
        mlflow_server = MLflowServer(
            tracking_uri=mlflow_config.get('tracking_uri', 'http://localhost:5000'),
            backend_store_uri=mlflow_config.get('backend_store_uri', 'sqlite:///mlflow.db'),
            artifact_location=mlflow_config.get('artifact_location', './mlruns')
        )
        mlflow_server.ensure_started()

        # Create run name
        run_name = f"encoder_{'final' if final_training else 'regular'}_{int(time.time())}"

        tracker = ExplrTracker(
            tracking_uri=mlflow_config.get('tracking_uri', 'http://localhost:5000'),
            experiment_name=mlflow_config.get('experiment_name', 'music_recommendation'),
            run_name=run_name
        )
    else:
        tracker = None

    # Connect to Clementine database
    if verbose:
        print("\n[1/7] Loading Clementine database...")
    db = ClementineDB(music_config['database_path'])
    all_songs = db.get_all_songs()
    total_loaded = len(all_songs)
    if verbose:
        print(f"  Found {total_loaded} songs in database")

    # Count rated vs unrated (encoder will train on all, loss only on rated)
    rated_count = sum(1 for s in all_songs if s.is_rated)
    unrated_count = len(all_songs) - rated_count
    if verbose:
        print(f"  - Rated: {rated_count} songs")
        print(f"  - Unrated: {unrated_count} songs")
        print(f"  Note: Encoder sees all songs, loss computed only on rated songs")

    # Build album mappings
    if verbose:
        print("\n[2/7] Building album mappings...")
    album_to_idx, filename_to_albums = build_album_mapping(all_songs)
    if verbose:
        print(f"  Found {len(album_to_idx)} unique albums")

    # Create datasets
    if verbose:
        print("\n[3/7] Creating datasets...")
    # For encoder training, we can skip songs with all-unknown metadata (artist, album, title)
    # to avoid learning from garbage metadata. These songs will still be used for classifier
    # training since rating prediction doesn't require metadata.
    skip_unknown = music_config.get('skip_unknown_metadata', True)

    # Augmentation settings for contrastive learning
    use_augmentation = encoder_config.get('use_augmentation', False)
    crop_jitter = encoder_config.get('crop_jitter', 5.0)
    noise_level = encoder_config.get('noise_level', 0.0)

    if verbose and use_augmentation:
        print(f"  Audio augmentation ENABLED: crop_jitter={crop_jitter}s, noise_level={noise_level}")

    # Determine encoder type early (needed for dataset creation)
    encoder_type = get_encoder_type(config)
    use_simsiam = (encoder_type == "simsiam")

    if use_simsiam:
        # Create SimSiam dataset with augmentations
        simsiam_config = encoder_config.get('simsiam', {})
        augmentor = create_audio_augmentor(simsiam_config.get('augmentation', {}))

        # Waveform cache for faster loading (skips audio decode + resample)
        cache_dir = music_config.get('waveform_cache_dir')
        cache_max_gb = music_config.get('waveform_cache_max_gb', 140.0)
        if verbose and cache_dir:
            print(f"  Waveform cache: {cache_dir} (max {cache_max_gb} GB)")

        full_dataset = SimSiamMusicDataset(
            songs=all_songs,
            sample_rate=music_config['sample_rate'],
            duration=music_config['audio_duration'],
            crop_position=music_config.get('crop_position', 'end'),
            normalize=music_config.get('normalize', True),
            skip_unknown_metadata=skip_unknown,
            augmentor=augmentor,
            n_mels=simsiam_config.get('n_mels', 128),
            n_fft=simsiam_config.get('n_fft', 2048),
            hop_length=simsiam_config.get('hop_length', 512),
            cache_dir=cache_dir,
            cache_max_gb=cache_max_gb
        )
        if verbose:
            print(f"  Using SimSiamMusicDataset with dual views")
    else:
        full_dataset = MusicDataset(
            songs=all_songs,
            album_to_idx=album_to_idx,
            filename_to_albums=filename_to_albums,
            sample_rate=music_config['sample_rate'],
            duration=music_config['audio_duration'],
            crop_position=music_config.get('crop_position', 'end'),
            normalize=music_config.get('normalize', True),
            only_rated=False,  # Include all songs; loss functions handle rated/unrated
            skip_unknown_metadata=skip_unknown,  # Skip songs with all-unknown metadata for encoder
            use_augmentation=use_augmentation,  # Enable dual-crop for contrastive learning
            crop_jitter=crop_jitter,  # Random offset for second crop
            noise_level=noise_level   # Add white noise
        )

    # Collect preprocessing stats from dataset filtering
    filter_counts = full_dataset.filter_counts
    preprocessing_stats = collect_preprocessing_stats(
        total_loaded=total_loaded,
        excluded_missing=filter_counts.get('missing_file', 0),
        excluded_duration=0,  # Would be tracked during duration filtering (not yet implemented)
        excluded_speech=filter_counts.get('speech', 0),
        excluded_duplicates=0,  # Would be tracked during deduplication (not yet implemented)
        excluded_unknown_metadata=filter_counts.get('unknown_metadata', 0),
        final_songs=len(full_dataset),
        rated_count=rated_count - filter_counts.get('rating', 0),  # Adjust for filtered rated songs
        unrated_count=unrated_count
    )
    model_card.set_preprocessing_stats(preprocessing_stats)

    # Train/val split
    train_split = music_config.get('train_split', 0.8)
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )

    if verbose:
        print(f"  Train: {len(train_dataset)} songs")
        print(f"  Val: {len(val_dataset)} songs")

    # Create data loaders with optimized parallel loading
    num_workers = music_config.get('dataloader_workers', 4)
    collate_fn = simsiam_collate_fn if use_simsiam else music_collate_fn

    # DataLoader kwargs - prefetch and persistent_workers only work with num_workers > 0
    loader_kwargs = {
        'batch_size': encoder_config['batch_size'],
        'num_workers': num_workers,
        'pin_memory': True,
        'collate_fn': collate_fn,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 8  # Increased for better GPU utilization
        loader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Multi-task is only for simple encoder (incompatible with SimSiam)
    use_multi_task = music_config.get('use_multi_task', False) and not use_simsiam

    # Create model using factory
    if verbose:
        print("\n[4/7] Creating encoder model...")

    # Create SimSiam encoder using factory
    encoder = create_encoder(config)
    if verbose:
        print(f"  Using SimSiamEncoder")
        simsiam_config = encoder_config.get('simsiam', {})
        print(f"  Backbone: {simsiam_config.get('backbone', 'resnet50')}")
        print(f"  Projection dim: {simsiam_config.get('projection_dim', 2048)}")
        print(f"  Predictor hidden dim: {simsiam_config.get('predictor_hidden_dim', 512)}")

    if verbose:
        print(f"  Embedding dim: {encoder.get_embedding_dim()}")

    # Create loss function
    if verbose:
        print("\n[5/7] Creating loss function...")

    # SimSiam loss
    from ml_skeleton.music.losses import SimSiamLoss
    loss_fn = SimSiamLoss()
    if verbose:
        print(f"  Using SimSiamLoss (self-supervised)")
        print(f"    - No negative pairs needed")
        print(f"    - Stop-gradient prevents collapse")

    # Create optimizer with full Adam parameters
    # Handle betas - can be list from config or separate beta1/beta2 from HPO
    if 'adam_beta1' in encoder_config and 'adam_beta2' in encoder_config:
        betas = (encoder_config['adam_beta1'], encoder_config['adam_beta2'])
    else:
        betas = tuple(encoder_config.get('adam_betas', [0.9, 0.999]))

    # Check if using decoupled weight decay (AdamW)
    use_adamw = encoder_config.get('adam_decoupled_weight_decay', False)

    optimizer_cls = torch.optim.AdamW if use_adamw else torch.optim.Adam
    optimizer = optimizer_cls(
        encoder.parameters(),
        lr=encoder_config['learning_rate'],
        betas=betas,
        eps=encoder_config.get('adam_eps', 1e-08),
        weight_decay=encoder_config.get('adam_weight_decay', 0.0),
        amsgrad=encoder_config.get('adam_amsgrad', False)
    )

    if verbose:
        print(f"  Using {'AdamW' if use_adamw else 'Adam'} optimizer:")
        print(f"    lr={encoder_config['learning_rate']}")
        print(f"    betas={betas}")
        print(f"    eps={encoder_config.get('adam_eps', 1e-08)}")
        print(f"    weight_decay={encoder_config.get('adam_weight_decay', 0.0)}")
        print(f"    amsgrad={encoder_config.get('adam_amsgrad', False)}")

    # Create embedding store
    embedding_store = EmbeddingStore(music_config['embedding_db_path'])

    # Collect dataset statistics for model card
    dataset_stats = collect_dataset_stats(all_songs, only_rated=False)
    if verbose:
        print(f"\n  Dataset Statistics:")
        print(f"    Total songs: {dataset_stats['total_songs']}")
        print(f"    Total artists: {dataset_stats['total_artists']}")
        print(f"    Total albums: {dataset_stats['total_albums']}")

    # Apply model version override if provided
    model_version = model_version_override if model_version_override else music_config['encoder_version']

    # Create trainer
    if verbose:
        print("\n[6/7] Creating trainer...")

    trainer = EncoderTrainer(
        encoder=encoder,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        embedding_store=embedding_store,
        model_version=model_version,
        tracker=tracker  # Pass tracker for MLflow learning curves
    )

    # Load checkpoint if resuming from previous training
    start_epoch = 0
    if resume_checkpoint:
        checkpoint_path = Path(resume_checkpoint)
        if checkpoint_path.exists():
            print(f"\n  Resuming from checkpoint: {resume_checkpoint}")
            trainer.load_checkpoint(checkpoint_path)
            start_epoch = trainer.current_epoch + 1  # Start from next epoch
            print(f"  Resuming training from epoch {start_epoch + 1}")
            if model_version_override:
                print(f"  Model version updated to: {model_version}")
        else:
            print(f"\n  WARNING: Checkpoint not found: {resume_checkpoint}")
            print("  Starting from scratch...")

    # Train with time tracking and MLflow logging
    if verbose:
        print("\n[7/7] Training...")
    training_start_time = time.time()

    if tracker:
        with tracker:
            # Log configuration and hyperparameters
            tracker.log_params({
                'stage': 'encoder',
                'final_training': final_training,
                'embedding_dim': encoder_config['embedding_dim'],
                'base_channels': encoder_config.get('base_channels', 32),
                'batch_size': encoder_config['batch_size'],
                'learning_rate': encoder_config['learning_rate'],
                'num_epochs': num_epochs,
                'optimizer': encoder_config.get('optimizer', 'adam'),
                'scheduler': encoder_config.get('scheduler', 'cosine'),
                'loss_type': encoder_config.get('loss_type', 'metadata_contrastive'),
                'audio_duration': music_config['audio_duration'],
                'sample_rate': music_config['sample_rate'],
                'crop_position': music_config.get('crop_position', 'end'),
                'normalize': music_config.get('normalize', True),
            })

            # Train
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                checkpoint_dir=config['checkpoint_dir'],
                use_multi_task=use_multi_task,
                use_augmentation=use_augmentation,
                use_simsiam=use_simsiam,
                save_best_only=True,
                early_stopping_patience=encoder_config.get('early_stopping_patience'),
                early_stopping_min_delta=encoder_config.get('early_stopping_min_delta', 0.0),
                verbose=verbose,
                start_epoch=start_epoch
            )
            training_time = time.time() - training_start_time

            # Log final metrics to MLflow
            best_val_loss_mlflow = min(history['val_loss']) if history['val_loss'] else float('inf')
            final_train_loss_mlflow = history['train_loss'][-1] if history['train_loss'] else float('inf')

            tracker.log_metric('best_val_loss', best_val_loss_mlflow)
            tracker.log_metric('final_train_loss', final_train_loss_mlflow)
            tracker.log_metric('training_time_seconds', training_time)
            tracker.log_metric('epochs_completed', len(history['train_loss']))

            # Log checkpoint as artifact
            checkpoint_path = Path(config['checkpoint_dir']) / 'encoder_best.pt'
            if checkpoint_path.exists():
                tracker.log_artifact(str(checkpoint_path))
    else:
        # Train without MLflow
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            checkpoint_dir=config['checkpoint_dir'],
            use_multi_task=use_multi_task,
            use_augmentation=use_augmentation,
            use_simsiam=use_simsiam,
            save_best_only=True,
            early_stopping_patience=encoder_config.get('early_stopping_patience'),
            early_stopping_min_delta=encoder_config.get('early_stopping_min_delta', 0.0),
            verbose=verbose,
            start_epoch=start_epoch
        )
        training_time = time.time() - training_start_time

    # Calculate metrics
    best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
    final_train_loss = history['train_loss'][-1] if history['train_loss'] else float('inf')
    epochs_run = len(history['train_loss'])

    if verbose:
        print("\n" + "=" * 80)
        print("ENCODER TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Checkpoint saved to: {config['checkpoint_dir']}/encoder_best.pt")
    else:
        # Concise HPO trial summary
        print(f"  Epochs: {epochs_run} | Train: {final_train_loss:.4f} | Val: {best_val_loss:.4f} | Time: {training_time:.1f}s")

    # Collect training statistics for model card
    encoder_stats = collect_training_stats(
        trainer=trainer,
        training_time_seconds=training_time,
        dataset_stats=dataset_stats
    )
    model_card.set_encoder_stats(encoder_stats)

    # Extract embeddings (skip during HPO to save time)
    if skip_embeddings:
        pass  # Silent skip during HPO
    else:
        # Free up training memory before extraction to avoid OOM
        del train_loader, val_loader, train_dataset, val_dataset
        cleanup_memory()

        print("\n" + "=" * 80)
        print("EXTRACTING EMBEDDINGS")
        print("=" * 80)

        # Load best model
        best_checkpoint = Path(config['checkpoint_dir']) / "encoder_best.pt"
        trainer.load_checkpoint(best_checkpoint)

        # Create a non-augmented dataset for embedding extraction
        # (we need single embeddings per song, not dual views)
        # For SimSiam and augmentation mode, we need a regular dataset
        if use_simsiam or use_augmentation:
            extraction_dataset = MusicDataset(
                songs=all_songs,
                album_to_idx=album_to_idx,
                filename_to_albums=filename_to_albums,
                sample_rate=music_config['sample_rate'],
                duration=music_config['audio_duration'],
                crop_position=music_config.get('crop_position', 'end'),
                normalize=music_config.get('normalize', True),
                only_rated=False,
                skip_unknown_metadata=skip_unknown,
                use_augmentation=False  # No augmentation for extraction
            )
        else:
            extraction_dataset = full_dataset

        # Extract embeddings for all songs (optimized for throughput)
        all_loader = DataLoader(
            extraction_dataset,
            batch_size=encoder_config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            collate_fn=music_collate_fn,
            prefetch_factor=4,  # Pre-load batches for better throughput
            pin_memory=True  # Speed up GPU transfer
        )

        embeddings = trainer.extract_embeddings(
            all_loader,
            save_to_store=True
        )

        print(f"\nExtracted {len(embeddings)} embeddings")
        print(f"Saved to: {music_config['embedding_db_path']}")

        # Print embedding store stats
        stats = embedding_store.get_stats()
        print(f"\nEmbedding Store Stats:")
        print(f"  Total embeddings: {stats['total_embeddings']}")
        print(f"  Unique songs: {stats['unique_songs']}")
        print(f"  Model versions: {stats['model_versions']}")
        print(f"  DB size: {stats['db_size_mb']:.2f} MB")

    return model_card


def train_classifier(
    config: dict,
    model_card: ModelCardGenerator = None,
    final_training: bool = False,
    trial_info: tuple[int, int] = None,
    verbose: bool = True,
    classifier_version_override: str = None
):
    """Stage 2: Train rating classifier.

    Args:
        config: Configuration dictionary
        model_card: Optional ModelCardGenerator with encoder statistics
        final_training: If True, uses final_training_epochs (50) instead of epochs (20)
        trial_info: Optional tuple of (trial_number, n_trials) for HPO logging
        verbose: If True, print detailed setup info (set False during HPO to reduce noise)
        classifier_version_override: Override classifier version (e.g., 'v2')

    Returns:
        ModelCardGenerator with complete statistics
    """
    import time

    # Ensure clean memory state at start of stage
    cleanup_memory()

    print("=" * 80)
    if trial_info:
        trial_num, n_trials = trial_info
        print(f"STAGE 2: CLASSIFIER TRAINING (Optuna Trial {trial_num}/{n_trials})")
    else:
        print("STAGE 2: CLASSIFIER TRAINING")
    print("=" * 80)

    # Create or verify model card generator
    if model_card is None:
        print("  WARNING: No model card from encoder stage. Creating new one.")
        model_card = ModelCardGenerator()
        model_card.set_config(config)

    # Load configuration
    music_config = config['music']
    classifier_config = config['classifier']
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Select appropriate epoch count based on training mode
    if final_training and 'final_training_epochs' in classifier_config:
        num_epochs = classifier_config['final_training_epochs']
        if verbose:
            print(f"  Using final_training_epochs={num_epochs} for training with best hyperparameters")
    else:
        num_epochs = classifier_config['epochs']
        if verbose:
            print(f"  Using epochs={num_epochs} (HPO/regular training mode)")

    # Initialize MLflow tracking
    mlflow_config = config.get('mlflow', {})
    mlflow_enabled = mlflow_config.get('auto_start', True)

    if mlflow_enabled:
        # Start MLflow server if configured
        mlflow_server = MLflowServer(
            tracking_uri=mlflow_config.get('tracking_uri', 'http://localhost:5000'),
            backend_store_uri=mlflow_config.get('backend_store_uri', 'sqlite:///mlflow.db'),
            artifact_location=mlflow_config.get('artifact_location', './mlruns')
        )
        mlflow_server.ensure_started()

        # Create run name
        run_name = f"classifier_{'final' if final_training else 'regular'}_{int(time.time())}"

        tracker = ExplrTracker(
            tracking_uri=mlflow_config.get('tracking_uri', 'http://localhost:5000'),
            experiment_name=mlflow_config.get('experiment_name', 'music_recommendation'),
            run_name=run_name
        )
    else:
        tracker = None

    # Connect to database
    if verbose:
        print("\n[1/6] Loading Clementine database...")
    db = ClementineDB(music_config['database_path'])
    all_songs = db.get_all_songs()

    # Filter rated songs
    if music_config.get('only_rated', True):
        all_songs = [s for s in all_songs if s.is_rated]

    if verbose:
        print(f"  Found {len(all_songs)} rated songs")

    # Collect dataset statistics for classifier (only rated songs)
    classifier_dataset_stats = collect_dataset_stats(all_songs, only_rated=True)
    if verbose:
        print(f"\n  Classifier Dataset Statistics:")
        print(f"    Total rated songs: {classifier_dataset_stats['total_songs']}")
        print(f"    Total artists: {classifier_dataset_stats['total_artists']}")
        print(f"    Total albums: {classifier_dataset_stats['total_albums']}")

    # Load embeddings
    if verbose:
        print("\n[2/6] Loading embeddings...")
    embedding_store = EmbeddingStore(music_config['embedding_db_path'])

    # Get embeddings for all songs
    filenames = [s.filename for s in all_songs]
    embeddings_dict = embedding_store.get_embeddings_batch(
        filenames,
        model_version=music_config['encoder_version']
    )

    if verbose:
        print(f"  Loaded {len(embeddings_dict)} embeddings")

    # Check embedding dimension
    first_embedding = next(iter(embeddings_dict.values()))
    embedding_dim = len(first_embedding)
    if verbose:
        print(f"  Embedding dimension: {embedding_dim}")

    # Create dataset
    if verbose:
        print("\n[3/6] Creating dataset...")
    full_dataset = EmbeddingDataset(
        embeddings=embeddings_dict,
        songs=all_songs,
        only_rated=True
    )

    # Train/val split
    train_split = music_config.get('train_split', 0.8)
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )

    if verbose:
        print(f"  Train: {len(train_dataset)} songs")
        print(f"  Val: {len(val_dataset)} songs")

    # Create data loaders (embeddings are cheap to load, so smaller prefetch)
    train_loader = DataLoader(
        train_dataset,
        batch_size=classifier_config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,  # Embeddings load fast, so 2 batches is enough
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=classifier_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    # Create model
    if verbose:
        print("\n[4/6] Creating classifier model...")
    classifier = SimpleRatingClassifier(
        embedding_dim=embedding_dim,
        hidden_dims=classifier_config['hidden_dims'],
        dropout=classifier_config['dropout']
    )

    if verbose:
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Hidden dims: {classifier_config['hidden_dims']}")
        print(f"  Dropout: {classifier_config['dropout']}")

    # Create loss and optimizer with full Adam parameters
    loss_fn = RatingLoss()

    # Handle betas - can be list from config or separate beta1/beta2 from HPO
    if 'adam_beta1' in classifier_config and 'adam_beta2' in classifier_config:
        betas = (classifier_config['adam_beta1'], classifier_config['adam_beta2'])
    else:
        betas = tuple(classifier_config.get('adam_betas', [0.9, 0.999]))

    # Check if using decoupled weight decay (AdamW)
    use_adamw = classifier_config.get('adam_decoupled_weight_decay', False)

    optimizer_cls = torch.optim.AdamW if use_adamw else torch.optim.Adam
    optimizer = optimizer_cls(
        classifier.parameters(),
        lr=classifier_config['learning_rate'],
        betas=betas,
        eps=classifier_config.get('adam_eps', 1e-08),
        weight_decay=classifier_config.get('adam_weight_decay', 0.0),
        amsgrad=classifier_config.get('adam_amsgrad', False)
    )

    if verbose:
        print(f"  Using {'AdamW' if use_adamw else 'Adam'} optimizer:")
        print(f"    lr={classifier_config['learning_rate']}")
        print(f"    betas={betas}")
        print(f"    eps={classifier_config.get('adam_eps', 1e-08)}")
        print(f"    weight_decay={classifier_config.get('adam_weight_decay', 0.0)}")
        print(f"    amsgrad={classifier_config.get('adam_amsgrad', False)}")

    # Get version information
    encoder_checkpoint_path = Path(config['checkpoint_dir']) / "encoder_best.pt"
    if encoder_checkpoint_path.exists():
        encoder_version = get_encoder_version_from_checkpoint(str(encoder_checkpoint_path))
        if verbose:
            print(f"\n  Encoder version (from checkpoint): {encoder_version}")
    else:
        encoder_version = music_config.get('encoder_version', 'v1')
        if verbose:
            print(f"\n  Encoder version (from config): {encoder_version}")
            print(f"  WARNING: No encoder checkpoint found at {encoder_checkpoint_path}")

    classifier_version = classifier_version_override if classifier_version_override else music_config.get('classifier_version', 'v1')
    if verbose:
        print(f"  Classifier version: {classifier_version}")

    # Create trainer
    if verbose:
        print("\n[5/6] Creating trainer...")
    trainer = ClassifierTrainer(
        classifier=classifier,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        tracker=tracker,  # Pass tracker for MLflow learning curves
        encoder_version=encoder_version,
        classifier_version=classifier_version
    )

    # Train with time tracking and MLflow logging
    if verbose:
        print("\n[6/6] Training...")
    training_start_time = time.time()

    if tracker:
        with tracker:
            # Log configuration and hyperparameters
            tracker.log_params({
                'stage': 'classifier',
                'final_training': final_training,
                'hidden_dims': str(classifier_config.get('hidden_dims', [256, 128])),
                'dropout': classifier_config.get('dropout', 0.3),
                'batch_size': classifier_config['batch_size'],
                'learning_rate': classifier_config['learning_rate'],
                'num_epochs': num_epochs,
                'optimizer': classifier_config.get('optimizer', 'adam'),
                'scheduler': classifier_config.get('scheduler', 'cosine'),
                'loss_type': classifier_config.get('loss_type', 'mse'),
            })

            # Train
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                checkpoint_dir=config['checkpoint_dir'],
                save_best_only=True,
                early_stopping_patience=classifier_config.get('early_stopping_patience'),
                early_stopping_min_delta=classifier_config.get('early_stopping_min_delta', 0.0)
            )
            training_time = time.time() - training_start_time

            # Log final metrics
            best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
            best_val_mae = min(history['val_mae']) if history['val_mae'] else float('inf')
            final_train_loss = history['train_loss'][-1] if history['train_loss'] else float('inf')

            tracker.log_metric('best_val_loss', best_val_loss)
            tracker.log_metric('best_val_mae', best_val_mae)
            tracker.log_metric('final_train_loss', final_train_loss)
            tracker.log_metric('training_time_seconds', training_time)
            tracker.log_metric('epochs_completed', len(history['train_loss']))

            # Log checkpoint as artifact
            checkpoint_path = Path(config['checkpoint_dir']) / 'classifier_best.pt'
            if checkpoint_path.exists():
                tracker.log_artifact(str(checkpoint_path))
    else:
        # Train without MLflow
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            checkpoint_dir=config['checkpoint_dir'],
            save_best_only=True,
            early_stopping_patience=classifier_config.get('early_stopping_patience'),
            early_stopping_min_delta=classifier_config.get('early_stopping_min_delta', 0.0)
        )
        training_time = time.time() - training_start_time

    # Calculate metrics
    best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
    best_val_mae = min(history['val_mae']) if history['val_mae'] else float('inf')
    final_train_loss = history['train_loss'][-1] if history['train_loss'] else float('inf')
    epochs_run = len(history['train_loss'])

    if verbose:
        print("\n" + "=" * 80)
        print("CLASSIFIER TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Best val MAE: {best_val_mae:.4f}")
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Checkpoint saved to: {config['checkpoint_dir']}/classifier_best.pt")
    else:
        # Concise HPO trial summary
        print(f"  Epochs: {epochs_run} | Train: {final_train_loss:.4f} | Val: {best_val_loss:.4f} | MAE: {best_val_mae:.4f} | Time: {training_time:.1f}s")

    # Collect training statistics for model card
    classifier_stats = collect_training_stats(
        trainer=trainer,
        training_time_seconds=training_time,
        dataset_stats=classifier_dataset_stats
    )
    model_card.set_classifier_stats(classifier_stats)

    # Generate model card (only during final training, not HPO)
    if verbose:
        print("\n" + "=" * 80)
        print("GENERATING MODEL CARD")
        print("=" * 80)
        model_card_path = Path(config.get('checkpoint_dir', './checkpoints')) / "MODEL_CARD.md"
        model_card.generate(model_card_path)
        print(f"Model card saved to: {model_card_path}")
    # Skip model card generation during HPO (verbose=False)

    return model_card


def generate_recommendations(config: dict):
    """Generate recommendations for unrated songs.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If classifier was trained with a different encoder version
    """
    # Ensure clean memory state at start of stage
    cleanup_memory()

    print("=" * 80)
    print("GENERATING RECOMMENDATIONS")
    print("=" * 80)

    # Load configuration
    music_config = config['music']
    rec_config = config.get('recommendations', {})
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Validate model compatibility BEFORE proceeding
    encoder_checkpoint = Path(config['checkpoint_dir']) / "encoder_best.pt"
    classifier_checkpoint = Path(config['checkpoint_dir']) / "classifier_best.pt"

    if encoder_checkpoint.exists() and classifier_checkpoint.exists():
        print("\n[0/5] Validating model compatibility...")
        validate_model_compatibility(
            str(encoder_checkpoint),
            str(classifier_checkpoint)
        )
    else:
        if not encoder_checkpoint.exists():
            print(f"\n  WARNING: Encoder checkpoint not found: {encoder_checkpoint}")
        if not classifier_checkpoint.exists():
            print(f"\n  WARNING: Classifier checkpoint not found: {classifier_checkpoint}")

    # Connect to database
    print("\n[1/5] Loading Clementine database...")
    db = ClementineDB(music_config['database_path'])
    all_songs = db.get_all_songs()

    # Get unrated songs
    unrated_songs = [s for s in all_songs if not s.is_rated]
    print(f"  Found {len(unrated_songs)} unrated songs")

    if len(unrated_songs) == 0:
        print("  No unrated songs to recommend!")
        return

    # Load embeddings
    print("\n[2/5] Loading embeddings...")
    embedding_store = EmbeddingStore(music_config['embedding_db_path'])

    filenames = [s.filename for s in unrated_songs]
    # Use encoder_version for embedding lookup (with fallback to model_version for backwards compatibility)
    encoder_version = music_config.get('encoder_version', music_config.get('model_version', 'v1'))
    embeddings_dict = embedding_store.get_embeddings_batch(
        filenames,
        model_version=encoder_version
    )

    print(f"  Loaded {len(embeddings_dict)} embeddings")

    if len(embeddings_dict) == 0:
        print("  No embeddings found! Run encoder training first.")
        return

    # Create dataset
    dataset = EmbeddingDataset(
        embeddings=embeddings_dict,
        songs=unrated_songs,
        only_rated=False
    )

    data_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )

    # Load classifier
    print("\n[3/5] Loading classifier...")
    embedding_dim = len(next(iter(embeddings_dict.values())))

    classifier = SimpleRatingClassifier(
        embedding_dim=embedding_dim,
        hidden_dims=config['classifier']['hidden_dims'],
        dropout=config['classifier']['dropout']
    )

    checkpoint_path = Path(config['checkpoint_dir']) / "classifier_best.pt"
    if not checkpoint_path.exists():
        print(f"  Classifier checkpoint not found: {checkpoint_path}")
        print("  Run classifier training first!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.to(device)

    # Generate predictions
    print("\n[4/5] Generating predictions...")
    loss_fn = RatingLoss()
    trainer = ClassifierTrainer(
        classifier=classifier,
        device=device,
        loss_fn=loss_fn,
        optimizer=torch.optim.Adam(classifier.parameters())  # Dummy optimizer
    )

    predictions, pred_filenames = trainer.predict(data_loader)

    # Sort by predicted rating
    results = list(zip(predictions, pred_filenames))
    results.sort(reverse=True)  # Highest ratings first

    # Apply threshold
    min_threshold = rec_config.get('min_rating_threshold', 0.0)
    results = [(r, f) for r, f in results if r >= min_threshold]

    # Get top-N
    top_n = rec_config.get('top_n', 100)
    results = results[:top_n]

    print(f"  Generated {len(results)} recommendations")

    # Save recommendations
    print("\n[5/5] Saving recommendations...")
    output_path = Path(rec_config.get('output_path', './recommendations.txt'))

    with open(output_path, 'w') as f:
        f.write(f"Top {len(results)} Recommendations\n")
        f.write("=" * 80 + "\n\n")

        for i, (rating, filename) in enumerate(results, 1):
            # Find song metadata
            song = next((s for s in unrated_songs if s.filename == filename), None)
            if song:
                f.write(f"{i}. [{rating:.3f}] {song.artist} - {song.title}\n")
                f.write(f"   Album: {song.album} ({song.year})\n")
                f.write(f"   Path: {filename}\n\n")

    print(f"  Saved to: {output_path}")

    # Print top 10
    print("\nTop 10 Recommendations:")
    for i, (rating, filename) in enumerate(results[:10], 1):
        song = next((s for s in unrated_songs if s.filename == filename), None)
        if song:
            print(f"  {i}. [{rating:.3f}] {song.artist} - {song.title}")

    # Generate human feedback playlists (for reinforcement learning loop)
    print("\n" + "=" * 80)
    print("GENERATING HUMAN FEEDBACK PLAYLISTS")
    print("=" * 80)

    # Prepare full list of songs and predictions for playlist generation
    full_predictions = [r for r, _ in results]
    full_songs = []
    for _, filename in results:
        song = next((s for s in unrated_songs if s.filename == filename), None)
        if song:
            full_songs.append(song)

    # Generate both uncertainty and best-predictions playlists
    top_n_uncertain = rec_config.get('human_feedback_uncertain', 100)
    top_n_best = rec_config.get('human_feedback_best', 50)
    playlist_output_dir = Path(rec_config.get('output_dir', './'))

    playlist_stats = generate_human_feedback_playlists(
        songs=full_songs,
        predictions=full_predictions,
        output_dir=playlist_output_dir,
        top_n_uncertain=top_n_uncertain,
        top_n_best=top_n_best,
        uncertainty_method="distance_from_middle"
    )

    print("\n" + "=" * 80)
    print("RECOMMENDATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {output_path} (text recommendations)")
    print(f"  - {playlist_output_dir / 'recommender_help.xspf'} (high uncertainty - maximize learning)")
    print(f"  - {playlist_output_dir / 'recommender_best.xspf'} (top predictions - validate quality)")
    print(f"\nNext steps for human-in-the-loop training:")
    print(f"  1. Open XSPF playlists in Clementine")
    print(f"  2. Listen and rate songs")
    print(f"  3. Re-run training with updated ratings")
    print(f"  4. Repeat for continuous improvement!")


def build_waveform_cache(config: dict):
    """Pre-populate waveform cache for consistent training speed.

    Iterates through all audio files once to ensure they are cached.
    This eliminates variable iteration times during training caused by
    cache misses when loading uncached files.

    Args:
        config: Configuration dictionary
    """
    from tqdm import tqdm
    from ml_skeleton.music.clementine_db import ClementineDB
    from ml_skeleton.music.audio_loader import load_audio_file
    from ml_skeleton.music.dataset import SimSiamMusicDataset

    music_config = config['music']

    # Load database
    print("\n[1/3] Loading song database...")
    db = ClementineDB(music_config.get('database_path'))
    all_songs = db.get_all_songs()
    print(f"  Found {len(all_songs)} songs")

    # Get cache settings from config
    sample_rate = music_config.get('sample_rate', 16000)
    duration = music_config.get('audio_duration', 60.0)
    crop_position = music_config.get('crop_position', 'end')
    normalize = music_config.get('normalize', True)
    cache_dir = music_config.get('waveform_cache_dir', './cache')
    cache_max_gb = music_config.get('waveform_cache_max_gb', 140.0)

    print(f"\n[2/3] Cache configuration:")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration}s")
    print(f"  Crop position: {crop_position}")
    print(f"  Max cache size: {cache_max_gb} GB")

    # Create dataset to leverage its caching logic
    print("\n[3/3] Building cache...")
    print("  This iterates through all audio files to populate the cache.")
    print("  Files that fail to load will be logged but won't stop the process.")
    print("")

    # Create dataset (this handles filtering and caching)
    # Note: No augmentor needed for cache building - we just want to cache the raw waveforms
    dataset = SimSiamMusicDataset(
        songs=all_songs,
        sample_rate=sample_rate,
        duration=duration,
        crop_position=crop_position,
        normalize=normalize,
        augmentor=None,  # No augmentation for cache building
        skip_unknown_metadata=music_config.get('skip_unknown_metadata', True),
        cache_dir=cache_dir,
        cache_max_gb=cache_max_gb
    )

    # Iterate through dataset to populate cache
    cached = 0
    failed = 0
    for i in tqdm(range(len(dataset)), desc="Caching audio files"):
        try:
            _ = dataset[i]
            cached += 1
        except Exception:
            failed += 1

    print(f"\n  Cached: {cached} files")
    print(f"  Failed: {failed} files")
    if cache_dir:
        import os
        cache_size = sum(
            os.path.getsize(os.path.join(cache_dir, f))
            for f in os.listdir(cache_dir) if f.endswith('.npy')
        ) / (1024 ** 3)
        print(f"  Cache size: {cache_size:.1f} GB")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Music Recommendation System with Hyperparameter Tuning"
    )
    parser.add_argument(
        '--stage',
        type=str,
        required=True,
        choices=['encoder', 'classifier', 'recommend', 'all', 'tune-encoder', 'tune-classifier', 'build-cache'],
        help='Training stage, recommendation generation, cache building, or hyperparameter tuning'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/music_recommendation.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=None,
        help='Number of trials for hyperparameter tuning (overrides config)'
    )
    parser.add_argument(
        '--tuner',
        type=str,
        choices=['optuna', 'ray_tune'],
        default='optuna',
        help='Tuner backend (default: optuna)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout in seconds for hyperparameter tuning'
    )
    parser.add_argument(
        '--final-training',
        action='store_true',
        help='Use final_training_epochs (50) instead of epochs (20) for training with best hyperparameters'
    )
    parser.add_argument(
        '--best-params',
        type=str,
        default=None,
        help='Path to best parameters JSON file (from HPO) to override config values'
    )
    parser.add_argument(
        '--encoder-type',
        type=str,
        choices=['simple', 'simsiam'],
        default=None,
        help='Override encoder type from config (for A/B testing)'
    )
    parser.add_argument(
        '--resume-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from (e.g., checkpoints/encoder_best.pt)'
    )
    parser.add_argument(
        '--encoder-version',
        type=str,
        default=None,
        help='Override encoder version for embeddings (e.g., v2, v3). Defaults to config value.'
    )
    parser.add_argument(
        '--model-version',
        type=str,
        default=None,
        help='DEPRECATED: Use --encoder-version instead. Kept for backwards compatibility.'
    )
    parser.add_argument(
        '--classifier-version',
        type=str,
        default=None,
        help='Override classifier version (e.g., v2, v3). Defaults to config value.'
    )

    args = parser.parse_args()

    # Handle backwards compatibility for --model-version
    if args.model_version and not args.encoder_version:
        args.encoder_version = args.model_version
        print(f"NOTE: --model-version is deprecated, use --encoder-version instead")

    # Load configuration
    config = load_config(args.config)

    # Apply encoder type override if provided
    if args.encoder_type:
        config['encoder']['encoder_type'] = args.encoder_type
        print(f"Encoder type overridden to: {args.encoder_type}")

    # Apply best parameters if provided
    if args.best_params:
        import json
        print(f"\nLoading best parameters from: {args.best_params}")
        with open(args.best_params, 'r') as f:
            best_params = json.load(f)

        # Determine which section to update (encoder or classifier)
        # For 'all' stage, assume it's encoder params (classifier would be separate run)
        if args.stage in ['encoder', 'all']:
            print("Applying best parameters to encoder config:")
            for key, value in best_params.items():
                config['encoder'][key] = value
                print(f"  {key} = {value}")
        elif args.stage == 'classifier':
            print("Applying best parameters to classifier config:")
            for key, value in best_params.items():
                config['classifier'][key] = value
                print(f"  {key} = {value}")
        print("")

    # Run stage
    if args.stage == 'encoder':
        model_card = train_encoder(
            config,
            final_training=args.final_training,
            resume_checkpoint=args.resume_checkpoint,
            model_version_override=args.encoder_version
        )
        cleanup_memory()
        print("\nNext step: Run with --stage classifier to train the rating predictor")

    elif args.stage == 'classifier':
        model_card = train_classifier(
            config,
            final_training=args.final_training,
            classifier_version_override=args.classifier_version
        )
        cleanup_memory()
        print("\nNext step: Run with --stage recommend to generate recommendations")

    elif args.stage == 'recommend':
        generate_recommendations(config)
        cleanup_memory()

    elif args.stage == 'build-cache':
        # Pre-populate waveform cache for consistent training speed
        print("\n" + "=" * 80)
        print("BUILDING WAVEFORM CACHE")
        print("=" * 80)
        print("Pre-populating cache to ensure consistent training iteration times.")
        print("This avoids slowdowns during training when loading uncached files.")
        print("")

        build_waveform_cache(config)
        cleanup_memory()
        print("\nCache build complete! Training will now have consistent speed.")

    elif args.stage == 'all':
        # Run complete pipeline: encoder -> classifier -> model card
        print("\n" + "=" * 80)
        print("RUNNING COMPLETE PIPELINE")
        print("=" * 80)
        print("This will run:")
        print("  1. Encoder training (Stage 1)")
        print("  2. Classifier training (Stage 2)")
        print("  3. Model card generation")
        print("")

        # Stage 1: Train encoder
        model_card = train_encoder(
            config,
            final_training=args.final_training,
            resume_checkpoint=args.resume_checkpoint,
            model_version_override=args.encoder_version
        )
        cleanup_memory()

        # Stage 2: Train classifier (with encoder statistics)
        model_card = train_classifier(
            config,
            model_card=model_card,
            final_training=args.final_training,
            classifier_version_override=args.classifier_version
        )
        cleanup_memory()

        print("\n" + "=" * 80)
        print("COMPLETE PIPELINE FINISHED")
        print("=" * 80)
        print("Next step: Run with --stage recommend to generate recommendations")

    elif args.stage == 'tune-encoder':
        # Hyperparameter tuning for encoder
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING: ENCODER")
        print("=" * 80)

        # Create ExperimentConfig manually (our music config has custom structure)
        from ml_skeleton.core.config import TuningConfig, SearchSpaceConfig, MLflowConfig

        # Extract tuning config
        tuning_dict = config.get('tuning', {})
        n_trials = args.n_trials if args.n_trials else tuning_dict.get('n_trials', 30)

        # Create experiment config
        exp_config = ExperimentConfig(
            name=config.get('name', 'music_recommendation_encoder'),
            framework=config.get('framework', 'pytorch'),
            hyperparameters=config['encoder'].copy(),
            seed=config.get('seed', 42),
            checkpoint_dir=config.get('checkpoint_dir', './checkpoints'),
            artifact_dir=config.get('artifact_dir', './artifacts')
        )

        # Configure MLflow
        if 'mlflow' in config:
            exp_config.mlflow = MLflowConfig(**config['mlflow'])

        # Configure tuning
        exp_config.tuning = TuningConfig(
            tuner_type=TunerType.OPTUNA if args.tuner == 'optuna' else TunerType.RAY_TUNE,
            n_trials=n_trials,
            timeout=args.timeout,
            sampler=tuning_dict.get('sampler', 'TPESampler'),
            pruner=tuning_dict.get('pruner', 'MedianPruner')
        )

        # Set encoder search space
        if 'encoder_search_space' in tuning_dict:
            exp_config.tuning.search_space = SearchSpaceConfig(
                parameters=tuning_dict['encoder_search_space']['parameters']
            )

        print(f"Tuner: {args.tuner}")
        print(f"Trials: {n_trials}")
        print(f"Search space parameters: {list(exp_config.tuning.search_space.parameters.keys())}")
        print("")

        # Create training function (pass n_trials for progress logging)
        train_fn = create_encoder_training_fn(config, n_trials=n_trials)

        # Run hyperparameter tuning
        results = run_experiment(train_fn, exp_config, tune=True)

        print("\n" + "=" * 80)
        print("ENCODER TUNING COMPLETE")
        print("=" * 80)
        print(f"Best value: {results['best_value']:.6f}")
        print(f"Best parameters:")
        for key, value in results['best_params'].items():
            print(f"  {key}: {value}")

        # Save best parameters to file for automated pipeline
        import json
        best_params_file = Path(config.get('checkpoint_dir', './checkpoints')) / 'best_encoder_params.json'
        best_params_file.parent.mkdir(parents=True, exist_ok=True)
        with open(best_params_file, 'w') as f:
            json.dump(results['best_params'], f, indent=2)
        print(f"\nBest parameters saved to: {best_params_file}")

        print("\nUpdate your config file with these parameters and run:")
        print("  python examples/music_recommendation.py --stage encoder --config configs/music_recommendation.yaml --final-training")

    elif args.stage == 'tune-classifier':
        # Hyperparameter tuning for classifier
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING: CLASSIFIER")
        print("=" * 80)

        # Verify embeddings exist
        embedding_db_path = config['music']['embedding_db_path']
        if not Path(embedding_db_path).exists():
            print(f"ERROR: Embeddings database not found: {embedding_db_path}")
            print("Run encoder training first:")
            print("  python examples/music_recommendation.py --stage encoder --config configs/music_recommendation.yaml")
            sys.exit(1)

        # Create ExperimentConfig manually (our music config has custom structure)
        from ml_skeleton.core.config import TuningConfig, SearchSpaceConfig, MLflowConfig

        # Extract tuning config
        tuning_dict = config.get('tuning', {})
        n_trials = args.n_trials if args.n_trials else tuning_dict.get('n_trials', 20)

        # Create experiment config
        exp_config = ExperimentConfig(
            name=config.get('name', 'music_recommendation_classifier'),
            framework=config.get('framework', 'pytorch'),
            hyperparameters=config['classifier'].copy(),
            seed=config.get('seed', 42),
            checkpoint_dir=config.get('checkpoint_dir', './checkpoints'),
            artifact_dir=config.get('artifact_dir', './artifacts')
        )

        # Configure MLflow
        if 'mlflow' in config:
            exp_config.mlflow = MLflowConfig(**config['mlflow'])

        # Configure tuning
        exp_config.tuning = TuningConfig(
            tuner_type=TunerType.OPTUNA if args.tuner == 'optuna' else TunerType.RAY_TUNE,
            n_trials=n_trials,
            timeout=args.timeout,
            sampler=tuning_dict.get('sampler', 'TPESampler'),
            pruner=tuning_dict.get('pruner', 'MedianPruner')
        )

        # Set classifier search space
        if 'classifier_search_space' in tuning_dict:
            exp_config.tuning.search_space = SearchSpaceConfig(
                parameters=tuning_dict['classifier_search_space']['parameters']
            )

        print(f"Tuner: {args.tuner}")
        print(f"Trials: {n_trials}")
        print(f"Search space parameters: {list(exp_config.tuning.search_space.parameters.keys())}")
        print("")

        # Create training function (pass n_trials for progress logging)
        train_fn = create_classifier_training_fn(config, n_trials=n_trials)

        # Run hyperparameter tuning
        results = run_experiment(train_fn, exp_config, tune=True)

        print("\n" + "=" * 80)
        print("CLASSIFIER TUNING COMPLETE")
        print("=" * 80)
        print(f"Best value: {results['best_value']:.6f}")
        print(f"Best parameters:")
        for key, value in results['best_params'].items():
            print(f"  {key}: {value}")

        # Save best parameters to file for automated pipeline
        import json
        best_params_file = Path(config.get('checkpoint_dir', './checkpoints')) / 'best_classifier_params.json'
        best_params_file.parent.mkdir(parents=True, exist_ok=True)
        with open(best_params_file, 'w') as f:
            json.dump(results['best_params'], f, indent=2)
        print(f"\nBest parameters saved to: {best_params_file}")

        print("\nUpdate your config file with these parameters and run:")
        print("  python examples/music_recommendation.py --stage classifier --config configs/music_recommendation.yaml --final-training")


if __name__ == '__main__':
    main()
