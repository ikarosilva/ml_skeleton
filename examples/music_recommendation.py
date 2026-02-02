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
import shutil
import numpy as np
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
from ml_skeleton.music.losses import RatingLoss, WeightedRatingLoss, BinaryRatingLoss, compute_class_weights, build_album_mapping
from ml_skeleton.music.encoder_factory import (
    create_encoder,
    create_loss_fn,
    create_dataset,
    create_optimizer,
    get_encoder_type,
    get_mlflow_tags
)
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
from ml_skeleton.music.training_manifest import TrainingManifest
from ml_skeleton.music.ab_testing import run_ab_test, format_ab_test_summary

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
        # Special handling for encoder loss weights
        if stage == 'encoder' and key == 'loss_weights_moco':
            stage_config['loss_weights']['moco'] = value
            stage_config['loss_weights']['genre_bce'] = 1.0 - value
            print(f"  Tuning: loss_weights.moco = {value:.3f}, genre_bce = {1.0-value:.3f}")
        # Special handling for encoder augmentation parameters
        elif stage == 'encoder' and key in ['gain_db_max', 'noise_prob', 'mixup_alpha']:
            if 'augmentation' not in stage_config:
                stage_config['augmentation'] = {}
            stage_config['augmentation'][key] = value
            # For symmetric gain, also set min
            if key == 'gain_db_max':
                stage_config['augmentation']['gain_db_min'] = -value
            print(f"  Tuning: augmentation.{key} = {value}")
        # Direct parameter mapping
        elif key in stage_config:
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


def create_encoder_training_fn(base_config: dict, n_trials: int = None, hpo_runs: int = 1):
    """Create encoder training function for hyperparameter tuning.

    Args:
        base_config: Base configuration dictionary
        n_trials: Total number of HPO trials (for progress logging)
        hpo_runs: Number of runs per trial with different seeds (default: 1)

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
        import numpy as np
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

        # Multi-run HPO: run multiple times with different seeds, use best (min) loss
        # NOTE: config['seed'] stays constant for train/val split consistency across trials
        if hpo_runs > 1:
            base_seed = config.get('seed', 42)
            run_losses = []
            best_run_idx = 0
            best_run_loss = float('inf')

            for run_idx in range(hpo_runs):
                training_seed = base_seed + run_idx * 1000  # Different seed for model init/training

                if run_idx == 0:
                    print(f"  HPO multi-run: {hpo_runs} runs per trial (objective = min loss, split fixed)")

                model_card = train_encoder(
                    config,
                    model_card=_global_model_card,
                    skip_embeddings=True,
                    trial_info=trial_info,
                    verbose=False,
                    training_seed=training_seed  # Only model init varies, split stays constant
                )

                encoder_stats = model_card.encoder_stats
                run_loss = encoder_stats.get('best_val_loss', encoder_stats.get('final_val_loss', float('inf')))
                run_losses.append(run_loss)
                print(f"    Run {run_idx + 1}/{hpo_runs} (seed={training_seed}): val_loss={run_loss:.6f}")

                # Track best run
                if run_loss < best_run_loss:
                    best_run_loss = run_loss
                    best_run_idx = run_idx

            # Use minimum loss as objective (find best single model)
            best_val_loss = np.min(run_losses)
            print(f"    Best: {best_val_loss:.6f} (run {best_run_idx + 1}), Mean: {np.mean(run_losses):.6f} +/- {np.std(run_losses):.6f}")
            _global_model_card = model_card
        else:
            # Single run (original behavior)
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

            # Save best HPO model checkpoint for resume in final training
            checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
            current_checkpoint = checkpoint_dir / 'encoder_best.pt'
            hpo_best_checkpoint = checkpoint_dir / 'encoder_hpo_best.pt'
            if current_checkpoint.exists():
                shutil.copy(current_checkpoint, hpo_best_checkpoint)
                print(f"    Saved best HPO model to: {hpo_best_checkpoint}")

        # Log to MLflow (skip params during HPO - optuna_tuner already logged them)
        # Only log metrics here
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


def create_classifier_training_fn(base_config: dict, n_trials: int = None, hpo_runs: int = 1):
    """Create classifier training function for hyperparameter tuning.

    Args:
        base_config: Base configuration dictionary
        n_trials: Total number of HPO trials (for progress logging)
        hpo_runs: Number of runs per trial with different seeds (default: 1)

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
        import numpy as np
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

        # Multi-run HPO: run multiple times with different seeds, use best (min) MAE
        # NOTE: config['seed'] stays constant for train/val split consistency across trials
        if hpo_runs > 1:
            base_seed = config.get('seed', 42)
            run_maes = []
            best_run_idx = 0
            best_run_mae = float('inf')

            for run_idx in range(hpo_runs):
                training_seed = base_seed + run_idx * 1000  # Different seed for model init/training

                if run_idx == 0:
                    print(f"  HPO multi-run: {hpo_runs} runs per trial (objective = min MAE, split fixed)")

                model_card = train_classifier(
                    config,
                    model_card=_global_model_card,
                    trial_info=trial_info,
                    verbose=False,
                    training_seed=training_seed  # Only model init varies, split stays constant
                )

                classifier_stats = model_card.classifier_stats
                run_mae = classifier_stats.get('val_mae', classifier_stats.get('best_val_loss', float('inf')))
                run_maes.append(run_mae)
                print(f"    Run {run_idx + 1}/{hpo_runs} (seed={training_seed}): val_mae={run_mae:.6f}")

                # Track best run
                if run_mae < best_run_mae:
                    best_run_mae = run_mae
                    best_run_idx = run_idx

            # Use minimum MAE as objective (find best single model)
            best_val_mae = np.min(run_maes)
            print(f"    Best: {best_val_mae:.6f} (run {best_run_idx + 1}), Mean: {np.mean(run_maes):.6f} +/- {np.std(run_maes):.6f}")
            _global_model_card = model_card
            best_val_loss = best_val_mae  # Use MAE as loss for consistency
        else:
            # Single run (original behavior)
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

        # Log to MLflow (skip params during HPO - optuna_tuner already logged them)
        # Only log metrics here
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


def _perform_encoder_training_run(
    config: dict,
    full_dataset,
    all_songs: list,
    album_to_idx: dict,
    filename_to_albums: dict,
    music_config: dict,
    encoder_config: dict,
    tracker,
    num_epochs: int,
    use_augmentation: bool,
    device: str,
    model_version: str,
    run_seed: int,
    checkpoint_dir: Path,
    checkpoint_suffix: str = "",
    verbose: bool = True,
    resume_checkpoint: str = None
) -> dict:
    """Perform a single encoder training run with a specific seed.

    Returns:
        dict with keys: best_val_loss, best_epoch, history, checkpoint_path, training_time
    """
    import time
    import random
    import numpy as np

    # Set random seeds for this run
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Train/val split with this run's seed
    train_split = music_config.get('train_split', 0.8)
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(run_seed)
    )

    if verbose:
        print(f"  Train: {len(train_dataset)} songs")
        print(f"  Val: {len(val_dataset)} songs")

    # Create data loaders
    num_workers = music_config.get('dataloader_workers', 4)

    def worker_init_fn(worker_id):
        worker_seed = run_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    from ml_skeleton.music.moco_dataset import MoCoCollator
    crop_duration_max = encoder_config.get('augmentation', {}).get('crop_duration_max', 15.0)
    moco_collator = MoCoCollator(
        sample_rate=music_config.get('sample_rate', 16000),
        crop_duration=crop_duration_max
    )

    loader_kwargs = {
        'batch_size': encoder_config['batch_size'],
        'num_workers': num_workers,
        'pin_memory': True,
        'collate_fn': moco_collator,
        'worker_init_fn': worker_init_fn if num_workers > 0 else None,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 8
        loader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Create model
    encoder = create_encoder(config)

    # Create loss function
    loss_fn = create_loss_fn(config)

    # Create optimizer
    if 'adam_beta1' in encoder_config and 'adam_beta2' in encoder_config:
        betas = (encoder_config['adam_beta1'], encoder_config['adam_beta2'])
    else:
        betas = tuple(encoder_config.get('adam_betas', [0.9, 0.999]))

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

    # Create scheduler
    scheduler = None
    scheduler_type = encoder_config.get('scheduler', 'cosine')
    if scheduler_type == 'cosine':
        t_max = encoder_config.get('cosine_t_max', num_epochs)
        eta_min = encoder_config.get('cosine_eta_min', 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )

    # Create embedding store and trainer
    embedding_store = EmbeddingStore(music_config['embedding_db_path'])
    trainer = EncoderTrainer(
        encoder=encoder,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        embedding_store=embedding_store,
        model_version=model_version,
        tracker=tracker,
        scheduler=scheduler
    )

    # Load checkpoint if resuming
    start_epoch = 0
    if resume_checkpoint:
        checkpoint_path = Path(resume_checkpoint)
        if checkpoint_path.exists():
            trainer.load_checkpoint(checkpoint_path)
            start_epoch = trainer.current_epoch + 1

    # Train
    training_start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        checkpoint_dir=str(checkpoint_dir),
        use_multi_task=False,
        use_augmentation=use_augmentation,
        use_moco=True,
        save_best_only=True,
        early_stopping_patience=encoder_config.get('early_stopping_patience'),
        early_stopping_min_delta=encoder_config.get('early_stopping_min_delta', 0.0),
        verbose=verbose,
        start_epoch=start_epoch
    )
    training_time = time.time() - training_start_time

    # Calculate metrics
    best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
    best_epoch = history['val_loss'].index(best_val_loss) + 1 if history['val_loss'] else 0
    epochs_run = len(history['train_loss'])

    # Copy checkpoint with suffix if specified
    src_checkpoint = checkpoint_dir / "encoder_best.pt"
    if checkpoint_suffix and src_checkpoint.exists():
        dst_checkpoint = checkpoint_dir / f"encoder_best{checkpoint_suffix}.pt"
        shutil.copy(src_checkpoint, dst_checkpoint)
        checkpoint_path = dst_checkpoint
    else:
        checkpoint_path = src_checkpoint

    # Cleanup
    del train_loader, val_loader, train_dataset, val_dataset, trainer, encoder, optimizer
    cleanup_memory()

    return {
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'epochs_run': epochs_run,
        'history': history,
        'checkpoint_path': checkpoint_path,
        'training_time': training_time
    }


def train_encoder(
    config: dict,
    model_card: ModelCardGenerator = None,
    final_training: bool = False,
    skip_embeddings: bool = False,
    trial_info: tuple[int, int] = None,
    verbose: bool = True,
    resume_checkpoint: str = None,
    model_version_override: str = None,
    num_runs: int = 1,
    training_seed: int = None
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
        num_runs: Number of training runs with different seeds (default: 1)
        training_seed: Seed for model init/training (if None, uses config seed).
                      Split always uses config seed for consistency across HPO trials.

    Returns:
        ModelCardGenerator with encoder statistics
    """
    import time
    import random
    import numpy as np

    # Ensure clean memory state at start of stage
    cleanup_memory()

    # Seed for data split (always constant from config for fair HPO comparison)
    split_seed = config.get('seed', 42)

    # Seed for model init/training (can be overridden for multi-run)
    model_seed = training_seed if training_seed is not None else split_seed

    # Set random seeds for model initialization and training
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
        # Make CuDNN deterministic (may reduce performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

    if False and mlflow_enabled:  # TODO DEBUG - disabled for now
        # Start MLflow server if configured
        tracking_uri = mlflow_config.get('tracking_uri', 'http://localhost:5000')
        # Extract port from tracking_uri
        port = int(tracking_uri.split(':')[-1].rstrip('/'))

        mlflow_server = MLflowServer.ensure_running(
            port=port,
            backend_store_uri=mlflow_config.get('backend_store_uri', 'sqlite:///mlflow.db'),
            artifact_root=mlflow_config.get('artifact_location', './mlruns')
        )

        # Create run name
        run_name = f"encoder_{'final' if final_training else 'regular'}_{int(time.time())}"

        tracker = ExplrTracker(
            tracking_uri=tracking_uri,
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
    use_moco = (encoder_type == "moco")

    # MoCo uses MoCoDataset (created by factory)
    if use_moco:
        from ml_skeleton.music.moco_dataset import MoCoDataset
        full_dataset = create_dataset(
            config=config,
            songs=all_songs,
            album_to_idx=album_to_idx,
            filename_to_albums=filename_to_albums,
            is_training=True
        )
        if verbose:
            print(f"  Using MoCoDataset with chunk cache")
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
    # MoCoDataset doesn't track filter_counts, so we use defaults
    filter_counts = getattr(full_dataset, 'filter_counts', {})
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

    # Multi-run training for statistical validation
    if num_runs > 1:
        import numpy as np
        base_seed = config.get('seed', 42)
        checkpoint_dir = Path(config['checkpoint_dir'])
        run_results = []

        # Apply model version override if provided
        model_version = model_version_override if model_version_override else music_config['encoder_version']

        print(f"\n{'='*60}")
        print(f"MULTI-RUN TRAINING: {num_runs} runs with different seeds")
        print(f"{'='*60}")

        for run_idx in range(num_runs):
            run_seed = base_seed + run_idx * 1000  # e.g., 42, 1042, 2042...
            print(f"\n{'='*60}")
            print(f"RUN {run_idx + 1}/{num_runs} (seed={run_seed})")
            print(f"{'='*60}")

            result = _perform_encoder_training_run(
                config=config,
                full_dataset=full_dataset,
                all_songs=all_songs,
                album_to_idx=album_to_idx,
                filename_to_albums=filename_to_albums,
                music_config=music_config,
                encoder_config=encoder_config,
                tracker=tracker,
                num_epochs=num_epochs,
                use_augmentation=use_augmentation,
                device=device,
                model_version=model_version,
                run_seed=run_seed,
                checkpoint_dir=checkpoint_dir,
                checkpoint_suffix=f"_run{run_idx + 1}",
                verbose=verbose,
                resume_checkpoint=resume_checkpoint if run_idx == 0 else None
            )

            run_results.append({
                'run': run_idx + 1,
                'seed': run_seed,
                'best_val_loss': result['best_val_loss'],
                'best_epoch': result['best_epoch'],
                'epochs_run': result['epochs_run'],
                'checkpoint_path': result['checkpoint_path'],
                'training_time': result['training_time']
            })

            print(f"\n  Run {run_idx + 1} complete: best_val_loss={result['best_val_loss']:.6f} (epoch {result['best_epoch']})")

        # Report multi-run statistics
        losses = [r['best_val_loss'] for r in run_results]
        total_time = sum(r['training_time'] for r in run_results)

        print(f"\n{'='*60}")
        print("MULTI-RUN STATISTICS")
        print(f"{'='*60}")
        for r in run_results:
            print(f"  Run {r['run']}: val_loss={r['best_val_loss']:.6f} (epoch {r['best_epoch']}, seed={r['seed']})")
        print(f"\n  Mean: {np.mean(losses):.6f} +/- {np.std(losses):.6f}")
        print(f"  Min:  {np.min(losses):.6f}")
        print(f"  Max:  {np.max(losses):.6f}")

        # Identify and copy best model
        best_run = min(run_results, key=lambda x: x['best_val_loss'])
        print(f"\n  Best: Run {best_run['run']} (val_loss={best_run['best_val_loss']:.6f}, seed={best_run['seed']})")

        # Copy best run's checkpoint to encoder_best.pt
        best_checkpoint = checkpoint_dir / f"encoder_best.pt"
        shutil.copy(best_run['checkpoint_path'], best_checkpoint)
        print(f"  Best model saved to: {best_checkpoint}")
        print(f"\n  Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

        # Extract embeddings using best model
        if not skip_embeddings:
            print(f"\n{'='*60}")
            print("EXTRACTING EMBEDDINGS (using best model)")
            print(f"{'='*60}")

            # Create embedding store and load best model
            embedding_store = EmbeddingStore(music_config['embedding_db_path'])
            encoder = create_encoder(config)
            loss_fn = create_loss_fn(config)

            # Minimal optimizer just for trainer initialization
            optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

            trainer = EncoderTrainer(
                encoder=encoder,
                device=device,
                loss_fn=loss_fn,
                optimizer=optimizer,
                embedding_store=embedding_store,
                model_version=model_version,
                tracker=None,
                scheduler=None
            )
            trainer.load_checkpoint(best_checkpoint)

            # Create extraction dataset
            from ml_skeleton.music.moco_dataset import MoCoCollator
            crop_duration_max = encoder_config.get('augmentation', {}).get('crop_duration_max', 15.0)
            moco_collator = MoCoCollator(
                sample_rate=music_config.get('sample_rate', 16000),
                crop_duration=crop_duration_max
            )

            extraction_dataset = create_dataset(
                config=config,
                songs=all_songs,
                album_to_idx=album_to_idx,
                filename_to_albums=filename_to_albums,
                is_training=False
            )

            num_workers = music_config.get('dataloader_workers', 4)
            all_loader = DataLoader(
                extraction_dataset,
                batch_size=encoder_config['batch_size'],
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=4,
                pin_memory=True,
                collate_fn=moco_collator
            )

            embeddings = trainer.extract_embeddings(
                all_loader,
                save_to_store=True,
                use_moco=True
            )

            print(f"\nExtracted {len(embeddings)} embeddings")
            print(f"Saved to: {music_config['embedding_db_path']}")

            stats = embedding_store.get_stats()
            print(f"\nEmbedding Store Stats:")
            print(f"  Total embeddings: {stats['total_embeddings']}")
            print(f"  Unique songs: {stats['unique_songs']}")
            print(f"  Model versions: {stats['model_versions']}")
            print(f"  DB size: {stats['db_size_mb']:.2f} MB")

        return model_card

    # Single run (existing code path)
    # Train/val split
    train_split = music_config.get('train_split', 0.8)
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(split_seed)
    )

    if verbose:
        print(f"  Train: {len(train_dataset)} songs")
        print(f"  Val: {len(val_dataset)} songs")

    # Create data loaders with optimized parallel loading
    num_workers = music_config.get('dataloader_workers', 4)

    # Worker initialization function for reproducible DataLoader workers (use model_seed)
    def worker_init_fn(worker_id):
        worker_seed = model_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # MoCo needs custom collator to handle memory-mapped arrays
    from ml_skeleton.music.moco_dataset import MoCoCollator

    # Get crop duration from augmentation config
    crop_duration_max = encoder_config.get('augmentation', {}).get('crop_duration_max', 15.0)
    moco_collator = MoCoCollator(
        sample_rate=music_config.get('sample_rate', 16000),
        crop_duration=crop_duration_max
    )

    loader_kwargs = {
        'batch_size': encoder_config['batch_size'],
        'num_workers': num_workers,
        'pin_memory': True,
        'collate_fn': moco_collator,
        'worker_init_fn': worker_init_fn if num_workers > 0 else None,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 8  # Increased for better GPU utilization
        loader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # Multi-task is only for simple encoder (not used with MoCo)
    use_multi_task = False

    # Create model using factory
    if verbose:
        print("\n[4/7] Creating encoder model...")

    # Create MoCo encoder using factory
    encoder = create_encoder(config)
    if verbose:
        print(f"  Using MoCoEncoder")
        moco_config = encoder_config.get('moco', {})
        print(f"  Backbone: {encoder_config.get('backbone', 'resnet50')}")
        print(f"  Queue size: {moco_config.get('queue_size', 4096)}")
        print(f"  Temperature: {moco_config.get('temperature', 0.07)}")
        print(f"  Projection dim: {moco_config.get('projection_dim', 128)}")

    if verbose:
        print(f"  Embedding dim: {encoder_config.get('embedding_dim', 2048)}")

    # Create loss function
    if verbose:
        print("\n[5/7] Creating loss function...")

    # MoCo loss (using factory)
    loss_fn = create_loss_fn(config)
    if verbose:
        print(f"  Using MoCoLoss")
        print(f"    - Contrastive loss with queue")
        print(f"    - Genre BCE auxiliary task")

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

    # Create learning rate scheduler
    scheduler = None
    scheduler_type = encoder_config.get('scheduler', 'cosine')
    if scheduler_type == 'cosine':
        t_max = encoder_config.get('cosine_t_max', num_epochs)
        eta_min = encoder_config.get('cosine_eta_min', 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min
        )
        if verbose:
            print(f"  Using CosineAnnealingLR scheduler:")
            print(f"    T_max={t_max}")
            print(f"    eta_min={eta_min}")

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
        tracker=tracker,  # Pass tracker for MLflow learning curves
        scheduler=scheduler  # Pass learning rate scheduler
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
                use_moco=True,
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
            use_moco=True,
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
        # MoCo dataset is suitable for extraction (no augmentation during inference)
        extraction_dataset = create_dataset(
            config=config,
            songs=all_songs,
            album_to_idx=album_to_idx,
            filename_to_albums=filename_to_albums,
            is_training=False  # No augmentation for extraction
        )

        # Extract embeddings for all songs (optimized for throughput)
        all_loader = DataLoader(
            extraction_dataset,
            batch_size=encoder_config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=4,  # Pre-load batches for better throughput
            pin_memory=True,  # Speed up GPU transfer
            collate_fn=moco_collator  # Need custom collator for memory-mapped arrays
        )

        embeddings = trainer.extract_embeddings(
            all_loader,
            save_to_store=True,
            use_moco=True
        )

        # Extract all 4 chunks per song into embedding_chunks (for classifier average)
        from ml_skeleton.music.moco_dataset import ChunkExtractionDataset
        chunk_cache_config = music_config.get('chunk_cache', {})
        num_chunks = chunk_cache_config.get('num_chunks', 4)
        crop_duration = encoder_config.get('augmentation', {}).get('crop_duration_max', 15.0)
        chunk_extraction_dataset = ChunkExtractionDataset(
            songs=all_songs,
            cache_dir=chunk_cache_config.get('directory', './cache/chunks'),
            num_chunks=num_chunks,
            sample_rate=music_config['sample_rate'],
            crop_duration=crop_duration
        )
        chunk_extraction_loader = DataLoader(
            chunk_extraction_dataset,
            batch_size=encoder_config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=4,
            pin_memory=True,
            collate_fn=moco_collator
        )
        trainer.extract_embeddings(chunk_extraction_loader, save_to_store=True, use_moco=True)
        print(f"Extracted all {num_chunks} chunks per song to embedding_chunks")

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


def _get_ratings_from_dataset(dataset) -> list:
    """Get ratings from an EmbeddingDataset or a Subset.

    Args:
        dataset: Either an EmbeddingDataset or a torch Subset

    Returns:
        List of ratings
    """
    from torch.utils.data import Subset

    if isinstance(dataset, Subset):
        # For Subset, access underlying dataset via indices
        underlying = dataset.dataset
        return [underlying.data[i]["rating"] for i in dataset.indices]
    else:
        # For EmbeddingDataset, use the method directly
        return dataset.get_all_ratings()


def _perform_classifier_training_run(
    config: dict,
    full_dataset,
    classifier_config: dict,
    music_config: dict,
    embedding_dim: int,
    encoder_version: str,
    classifier_version: str,
    tracker,
    num_epochs: int,
    device: str,
    run_seed: int,
    checkpoint_dir: Path,
    checkpoint_suffix: str = "",
    verbose: bool = True,
    class_weight_strategy: str = "none",
    classification_mode: str = "regression",
    init_from_prod: bool = True
) -> dict:
    """Perform a single classifier training run with a specific seed.

    Args:
        class_weight_strategy: Strategy for class weighting to handle imbalance.
        classification_mode: "regression" (MSE loss) or "binary" (BCE loss).
            - "none": No weighting (standard MSE)
            - "inverse": Weight = N / (n_classes * count_i)
            - "sqrt_inverse": Weight = sqrt(N / (n_classes * count_i))
        init_from_prod: If True, initialize from prod/classifier_best.pt if architecture matches.

    Returns:
        dict with keys: best_val_loss, best_val_mae, best_epoch, history, checkpoint_path, training_time
    """
    import time
    import random
    import numpy as np

    # Set random seeds for this run
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Train/val split with this run's seed
    train_split = music_config.get('train_split', 0.8)
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(run_seed)
    )

    if verbose:
        print(f"  Train: {len(train_dataset)} songs")
        print(f"  Val: {len(val_dataset)} songs")

    # Compute class weights from training data if requested
    class_weights = None
    if class_weight_strategy != "none":
        # Extract ratings from training dataset
        train_ratings = []
        for idx in train_dataset.indices:
            item = full_dataset[idx]
            train_ratings.append(item['rating'].item())

        class_weights = compute_class_weights(train_ratings, strategy=class_weight_strategy)
        if verbose:
            print(f"  Class weight strategy: {class_weight_strategy}")
            for bucket, weight in sorted(class_weights.items()):
                print(f"    {bucket} stars: weight={weight:.3f}")

    # Worker init function
    def worker_init_fn(worker_id):
        worker_seed = run_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=classifier_config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=classifier_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )

    # Create model
    classifier = SimpleRatingClassifier(
        embedding_dim=embedding_dim,
        hidden_dims=classifier_config['hidden_dims'],
        dropout=classifier_config['dropout']
    )

    # Try to initialize from production model if requested
    prod_checkpoint_path = Path("prod") / "classifier_best.pt"
    if init_from_prod and prod_checkpoint_path.exists():
        try:
            prod_checkpoint = torch.load(prod_checkpoint_path, map_location='cpu', weights_only=False)
            prod_state_dict = prod_checkpoint['model_state_dict']

            # Check if architectures match by comparing state dict keys and shapes
            current_state = classifier.state_dict()
            arch_matches = True
            for key in current_state:
                if key not in prod_state_dict:
                    arch_matches = False
                    break
                if current_state[key].shape != prod_state_dict[key].shape:
                    arch_matches = False
                    break

            if arch_matches:
                classifier.load_state_dict(prod_state_dict)
                if verbose:
                    print(f"  Initialized from production model: {prod_checkpoint_path}")
            else:
                if verbose:
                    print(f"  Architecture mismatch with prod model - using random init")
        except Exception as e:
            if verbose:
                print(f"  Could not load prod model ({e}) - using random init")
    elif init_from_prod and verbose:
        print(f"  No production model found - using random init")

    # Create loss function based on classification mode
    if classification_mode == "binary":
        # Compute pos_weight from training data for class imbalance
        train_labels = _get_ratings_from_dataset(train_dataset)
        pos_weight = BinaryRatingLoss.compute_pos_weight(train_labels)
        loss_fn = BinaryRatingLoss(pos_weight=pos_weight)
        if verbose:
            print(f"  Binary classification mode - pos_weight: {pos_weight:.3f}")
    elif class_weights is not None:
        loss_fn = WeightedRatingLoss(class_weights=class_weights)
    else:
        loss_fn = RatingLoss()

    if 'adam_beta1' in classifier_config and 'adam_beta2' in classifier_config:
        betas = (classifier_config['adam_beta1'], classifier_config['adam_beta2'])
    else:
        betas = tuple(classifier_config.get('adam_betas', [0.9, 0.999]))

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

    # Create trainer
    trainer = ClassifierTrainer(
        classifier=classifier,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        tracker=tracker,
        encoder_version=encoder_version,
        classifier_version=classifier_version,
        classification_mode=classification_mode
    )

    # Train
    training_start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        checkpoint_dir=str(checkpoint_dir),
        save_best_only=True,
        early_stopping_patience=classifier_config.get('early_stopping_patience'),
        early_stopping_min_delta=classifier_config.get('early_stopping_min_delta', 0.0)
    )
    training_time = time.time() - training_start_time

    # Calculate metrics
    best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
    best_val_mae = min(history['val_mae']) if history['val_mae'] else float('inf')
    best_epoch = history['val_mae'].index(best_val_mae) + 1 if history['val_mae'] else 0
    epochs_run = len(history['train_loss'])

    # Copy checkpoint with suffix if specified
    src_checkpoint = checkpoint_dir / "classifier_best.pt"
    if checkpoint_suffix and src_checkpoint.exists():
        dst_checkpoint = checkpoint_dir / f"classifier_best{checkpoint_suffix}.pt"
        shutil.copy(src_checkpoint, dst_checkpoint)
        checkpoint_path = dst_checkpoint
    else:
        checkpoint_path = src_checkpoint

    # Cleanup
    del train_loader, val_loader, train_dataset, val_dataset, trainer, classifier, optimizer
    cleanup_memory()

    return {
        'best_val_loss': best_val_loss,
        'best_val_mae': best_val_mae,
        'best_epoch': best_epoch,
        'epochs_run': epochs_run,
        'history': history,
        'checkpoint_path': checkpoint_path,
        'training_time': training_time
    }


def train_classifier(
    config: dict,
    model_card: ModelCardGenerator = None,
    final_training: bool = False,
    trial_info: tuple[int, int] = None,
    verbose: bool = True,
    classifier_version_override: str = None,
    num_runs: int = 1,
    training_seed: int = None,
    init_from_prod: bool = True,
    vault_size: int = 200
):
    """Stage 2: Train rating classifier.

    Args:
        config: Configuration dictionary
        model_card: Optional ModelCardGenerator with encoder statistics
        final_training: If True, uses final_training_epochs (50) instead of epochs (20)
        trial_info: Optional tuple of (trial_number, n_trials) for HPO logging
        verbose: If True, print detailed setup info (set False during HPO to reduce noise)
        classifier_version_override: Override classifier version (e.g., 'v2')
        num_runs: Number of training runs with different seeds (default: 1)
        training_seed: Seed for model init/training (if None, uses config seed).
                      Split always uses config seed for consistency across HPO trials.
        init_from_prod: If True (default), initialize from prod/classifier_best.pt if
                       architecture matches. If False, use random initialization.
        vault_size: Number of ratings to reserve for A/B testing vault (default: 200).
                   Vault files are never used for training, only for comparing models.

    Returns:
        ModelCardGenerator with complete statistics
    """
    import time
    import random
    import numpy as np

    # Ensure clean memory state at start of stage
    cleanup_memory()

    # Seed for data split (always constant from config for fair HPO comparison)
    split_seed = config.get('seed', 42)

    # Seed for model init/training (can be overridden for multi-run)
    model_seed = training_seed if training_seed is not None else split_seed

    # Set random seeds for model initialization and training
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
        # Make CuDNN deterministic (may reduce performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

    if False and mlflow_enabled:  # TODO DEBUG - disabled for now like encoder
        # Start MLflow server if configured
        tracking_uri = mlflow_config.get('tracking_uri', 'http://localhost:5000')
        # Extract port from tracking_uri
        port = int(tracking_uri.split(':')[-1].rstrip('/'))

        mlflow_server = MLflowServer.ensure_running(
            port=port,
            backend_store_uri=mlflow_config.get('backend_store_uri', 'sqlite:///mlflow.db'),
            artifact_root=mlflow_config.get('artifact_location', './mlruns')
        )

        # Create run name
        run_name = f"classifier_{'final' if final_training else 'regular'}_{int(time.time())}"

        tracker = ExplrTracker(
            tracking_uri=tracking_uri,
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

    # Get embeddings for all songs (prefer 4 chunks per song for classifier average)
    filenames = [s.filename for s in all_songs]
    embeddings_dict = embedding_store.get_embeddings_batch_all_chunks(
        filenames,
        model_version=music_config['encoder_version'],
        num_chunks=4
    )
    if not embeddings_dict:
        embeddings_dict = embedding_store.get_embeddings_batch(
            filenames,
            model_version=music_config['encoder_version']
        )

    if verbose:
        print(f"  Loaded {len(embeddings_dict)} embeddings")

    # Check embedding dimension (support (4, D) per song)
    first_embedding = next(iter(embeddings_dict.values()))
    arr = np.asarray(first_embedding)
    embedding_dim = int(arr.shape[-1]) if arr.ndim > 1 else len(arr)
    if verbose:
        print(f"  Embedding dimension: {embedding_dim}")

    # Create dataset
    if verbose:
        print("\n[3/6] Creating dataset...")

    # Get classification mode from config
    classification_mode = classifier_config.get('classification_mode', 'regression')
    binary_positive_threshold = classifier_config.get('binary_positive_threshold', 4.0)
    binary_negative_threshold = classifier_config.get('binary_negative_threshold', 2.0)

    full_dataset = EmbeddingDataset(
        embeddings=embeddings_dict,
        songs=all_songs,
        only_rated=True,
        classification_mode=classification_mode,
        binary_positive_threshold=binary_positive_threshold,
        binary_negative_threshold=binary_negative_threshold
    )

    # Get version information (needed for multi-run)
    encoder_checkpoint_path = Path(config['checkpoint_dir']) / "encoder_best.pt"
    if encoder_checkpoint_path.exists():
        encoder_version = get_encoder_version_from_checkpoint(str(encoder_checkpoint_path))
    else:
        encoder_version = music_config.get('encoder_version', 'v1')

    classifier_version = classifier_version_override if classifier_version_override else music_config.get('classifier_version', 'v1')

    # Multi-run training for statistical validation
    if num_runs > 1:
        base_seed = config.get('seed', 42)
        checkpoint_dir = Path(config['checkpoint_dir'])
        run_results = []

        print(f"\n{'='*60}")
        print(f"MULTI-RUN TRAINING: {num_runs} runs with different seeds")
        print(f"{'='*60}")

        for run_idx in range(num_runs):
            run_seed = base_seed + run_idx * 1000
            print(f"\n{'='*60}")
            print(f"RUN {run_idx + 1}/{num_runs} (seed={run_seed})")
            print(f"{'='*60}")

            result = _perform_classifier_training_run(
                config=config,
                full_dataset=full_dataset,
                classifier_config=classifier_config,
                music_config=music_config,
                embedding_dim=embedding_dim,
                encoder_version=encoder_version,
                classifier_version=classifier_version,
                tracker=tracker,
                num_epochs=num_epochs,
                device=device,
                run_seed=run_seed,
                checkpoint_dir=checkpoint_dir,
                checkpoint_suffix=f"_run{run_idx + 1}",
                verbose=verbose,
                class_weight_strategy=classifier_config.get('class_weight_strategy', 'none'),
                classification_mode=classification_mode,
                init_from_prod=init_from_prod
            )

            run_results.append({
                'run': run_idx + 1,
                'seed': run_seed,
                'best_val_loss': result['best_val_loss'],
                'best_val_mae': result['best_val_mae'],
                'best_epoch': result['best_epoch'],
                'epochs_run': result['epochs_run'],
                'checkpoint_path': result['checkpoint_path'],
                'training_time': result['training_time']
            })

            print(f"\n  Run {run_idx + 1} complete: best_val_mae={result['best_val_mae']:.6f} (epoch {result['best_epoch']})")

        # Report multi-run statistics
        maes = [r['best_val_mae'] for r in run_results]
        losses = [r['best_val_loss'] for r in run_results]
        total_time = sum(r['training_time'] for r in run_results)

        print(f"\n{'='*60}")
        print("MULTI-RUN STATISTICS")
        print(f"{'='*60}")
        for r in run_results:
            print(f"  Run {r['run']}: val_mae={r['best_val_mae']:.6f}, val_loss={r['best_val_loss']:.6f} (epoch {r['best_epoch']}, seed={r['seed']})")
        print(f"\n  MAE  - Mean: {np.mean(maes):.6f} +/- {np.std(maes):.6f}, Min: {np.min(maes):.6f}, Max: {np.max(maes):.6f}")
        print(f"  Loss - Mean: {np.mean(losses):.6f} +/- {np.std(losses):.6f}")

        # Identify and copy best model (by MAE)
        best_run = min(run_results, key=lambda x: x['best_val_mae'])
        print(f"\n  Best: Run {best_run['run']} (val_mae={best_run['best_val_mae']:.6f}, seed={best_run['seed']})")

        # Copy best run's checkpoint to classifier_best.pt
        best_checkpoint = checkpoint_dir / "classifier_best.pt"
        shutil.copy(best_run['checkpoint_path'], best_checkpoint)
        print(f"  Best model saved to: {best_checkpoint}")
        print(f"\n  Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

        # Save training manifest for multi-run (use consistent split with vault for A/B testing)
        manifest_path = checkpoint_dir / "training_manifest.json"
        manifest = TrainingManifest.load_or_create(str(manifest_path))

        # Get all filenames and do manifest-based split with vault for A/B testing
        all_filenames = full_dataset.get_all_filenames()
        file_ratings = full_dataset.get_file_ratings_dict()
        train_files, val_files, vault_files = manifest.split_with_vault(
            all_filenames,
            train_ratio=music_config.get('train_split', 0.8),
            vault_size=vault_size,
            seed=base_seed,
            file_ratings=file_ratings
        )

        manifest.set_version_info(
            encoder_version=encoder_version,
            classifier_version=classifier_version,
            classification_mode=classification_mode
        )
        manifest.save()

        # Collect stats for model card (use best run's stats)
        classifier_stats = {
            'best_val_loss': best_run['best_val_loss'],
            'best_val_mae': best_run['best_val_mae'],
            'val_mae': best_run['best_val_mae'],
            'best_epoch': best_run['best_epoch'],
            'epochs_run': best_run['epochs_run'],
            'training_time_seconds': total_time,
            'num_runs': num_runs,
            'mae_mean': np.mean(maes),
            'mae_std': np.std(maes),
            'train_size': len(train_files),
            'val_size': len(val_files),
            'vault_size': len(vault_files)
        }
        model_card.set_classifier_stats(classifier_stats)

        if verbose:
            print(f"\nTraining manifest saved to: {manifest_path}")
            print(f"  Vault files (A/B test only): {len(vault_files)}")

        # A/B test against production model using vault files (always available)
        prod_classifier_path = Path("prod") / "classifier_best.pt"
        prod_manifest_path = Path("prod") / "training_manifest.json"
        new_classifier_path = checkpoint_dir / "classifier_best.pt"

        if prod_classifier_path.exists() and len(vault_files) > 0 and verbose:
            print("\n" + "=" * 60)
            print("A/B TEST: New Model vs Production (using vault)")
            print("=" * 60)

            # Use CURRENT vault for A/B testing - fair for both models:
            # - NEW model: hasn't seen vault files (held out by definition)
            # - PROD model: hasn't seen vault files either:
            #   * Files in both vaults: PROD held these out
            #   * Files only in current vault: new ratings that didn't exist when PROD was trained
            current_vault_set = set(vault_files)
            print(f"  Using current vault for A/B test: {len(current_vault_set)} files")

            # Create test dataset from current vault
            test_dataset = full_dataset.subset_by_filenames(current_vault_set)

            if len(test_dataset) >= 10:
                ab_result = run_ab_test(
                    new_classifier_path=str(new_classifier_path),
                    prod_classifier_path=str(prod_classifier_path),
                    test_dataset=test_dataset,
                    classification_mode=classification_mode,
                    device=device,
                    verbose=True
                )
                manifest.set_metadata('ab_test_result', {
                    'n_samples': int(ab_result.get('n_samples', 0)),
                    'new_accuracy': float(ab_result.get('new_accuracy', 0)),
                    'prod_accuracy': float(ab_result.get('prod_accuracy', 0)),
                    'improvement': float(ab_result.get('improvement', 0)),
                    'p_value': float(ab_result.get('p_value', 1.0)),
                    'significant': bool(ab_result.get('significant', False))
                })
                manifest.save()
            else:
                print(f"  Skipping A/B test: only {len(test_dataset)} samples in vault (need >= 10)")
                print(f"  Rate more songs to build up a stable A/B test vault")
        elif verbose and len(vault_files) == 0:
            print(f"\n  No vault files available for A/B testing")
            print(f"  Need at least {vault_size} rated files to create vault")
        elif verbose:
            print(f"\n  No production model found at {prod_classifier_path}")
            print(f"  Run 'promote-to-prod' after initial training to enable A/B testing")

        print("\nNext step: Run with --stage recommend to generate recommendations")

        return model_card

    # Single run (existing code path)
    # Train/val split using manifest to track files with vault for A/B testing
    train_split = music_config.get('train_split', 0.8)
    checkpoint_dir = Path(config['checkpoint_dir'])

    # Load or create training manifest
    manifest_path = checkpoint_dir / "training_manifest.json"
    manifest = TrainingManifest.load_or_create(str(manifest_path))

    # Get all filenames from dataset
    all_filenames = full_dataset.get_all_filenames()

    if verbose:
        print(f"\n[4/6] Splitting dataset with manifest tracking (vault_size={vault_size})...")

    # Get file ratings for class-balanced vault
    file_ratings = full_dataset.get_file_ratings_dict()

    # Split using manifest with vault for A/B testing
    train_files, val_files, vault_files = manifest.split_with_vault(
        all_filenames,
        train_ratio=train_split,
        vault_size=vault_size,
        seed=split_seed,
        file_ratings=file_ratings
    )

    # Update manifest with version info
    manifest.set_version_info(
        encoder_version=encoder_version,
        classifier_version=classifier_version,
        classification_mode=classification_mode
    )

    # Create train/val datasets from filename lists
    train_dataset, val_dataset = full_dataset.split_by_filenames(train_files, val_files)

    if verbose:
        print(f"  Train: {len(train_dataset)} songs")
        print(f"  Val: {len(val_dataset)} songs")
        print(f"  Vault (A/B test only): {len(vault_files)} songs")

    # Compute class weights from training data if requested
    class_weight_strategy = classifier_config.get('class_weight_strategy', 'none')
    class_weights = None
    if class_weight_strategy != "none":
        train_ratings = _get_ratings_from_dataset(train_dataset)

        class_weights = compute_class_weights(train_ratings, strategy=class_weight_strategy)
        if verbose:
            print(f"  Class weight strategy: {class_weight_strategy}")
            for bucket, weight in sorted(class_weights.items()):
                print(f"    {bucket} stars: weight={weight:.3f}")

    # Worker initialization function for reproducible DataLoader workers (use model_seed)
    def worker_init_fn(worker_id):
        worker_seed = model_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Create data loaders (embeddings are cheap to load, so smaller prefetch)
    train_loader = DataLoader(
        train_dataset,
        batch_size=classifier_config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,  # Embeddings load fast, so 2 batches is enough
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=classifier_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
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

    # Create loss function based on classification mode
    if classification_mode == "binary":
        # Compute pos_weight from training data for class imbalance
        train_labels = _get_ratings_from_dataset(train_dataset)
        pos_weight = BinaryRatingLoss.compute_pos_weight(train_labels)
        loss_fn = BinaryRatingLoss(pos_weight=pos_weight)
        if verbose:
            print(f"  Binary classification mode - pos_weight: {pos_weight:.3f}")
    elif class_weights is not None:
        loss_fn = WeightedRatingLoss(class_weights=class_weights)
    else:
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
        print(f"\n  Encoder version: {encoder_version}")
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
        classifier_version=classifier_version,
        classification_mode=classification_mode
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
    # Add train/val/vault sizes for model card
    classifier_stats['train_size'] = len(train_dataset)
    classifier_stats['val_size'] = len(val_dataset)
    classifier_stats['vault_size'] = len(vault_files)
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

    # Save training manifest (tracks which files were used for training)
    manifest.set_metadata('best_val_loss', best_val_loss)
    manifest.set_metadata('best_val_mae', best_val_mae)
    manifest.set_metadata('training_time_seconds', training_time)
    manifest.set_metadata('epochs_run', epochs_run)
    manifest.set_metadata('train_size', len(train_dataset))
    manifest.set_metadata('val_size', len(val_dataset))
    manifest.set_metadata('vault_size', len(vault_files))
    manifest.save()

    if verbose:
        print(f"\nTraining manifest saved to: {manifest_path}")
        print(f"  Training files: {len(manifest.training_files)}")
        print(f"  Validation files: {len(manifest.validation_files)}")
        print(f"  Vault files (A/B test): {len(manifest.vault_files)}")

    # A/B test against production model using vault files (always available)
    prod_classifier_path = Path("prod") / "classifier_best.pt"
    prod_manifest_path = Path("prod") / "training_manifest.json"
    new_classifier_path = checkpoint_dir / "classifier_best.pt"

    if prod_classifier_path.exists() and len(vault_files) > 0 and verbose:
        print("\n" + "=" * 60)
        print("A/B TEST: New Model vs Production (using vault)")
        print("=" * 60)

        # Use CURRENT vault for A/B testing - fair for both models:
        # - NEW model: hasn't seen vault files (held out by definition)
        # - PROD model: hasn't seen vault files either:
        #   * Files in both vaults: PROD held these out
        #   * Files only in current vault: new ratings that didn't exist when PROD was trained
        current_vault_set = set(vault_files)
        print(f"  Using current vault for A/B test: {len(current_vault_set)} files")

        # Create test dataset from current vault
        test_dataset = full_dataset.subset_by_filenames(current_vault_set)

        if len(test_dataset) >= 10:  # Need minimum samples for meaningful test
            ab_result = run_ab_test(
                new_classifier_path=str(new_classifier_path),
                prod_classifier_path=str(prod_classifier_path),
                test_dataset=test_dataset,
                classification_mode=classification_mode,
                device=device,
                verbose=True
            )

            # Store A/B test result in manifest (include accuracies for debugging)
            manifest.set_metadata('ab_test_result', {
                'n_samples': int(ab_result.get('n_samples', 0)),
                'new_accuracy': float(ab_result.get('new_accuracy', 0)),
                'prod_accuracy': float(ab_result.get('prod_accuracy', 0)),
                'improvement': float(ab_result.get('improvement', 0)),
                'p_value': float(ab_result.get('p_value', 1.0)),
                'significant': bool(ab_result.get('significant', False))
            })
            manifest.save()
        else:
            print(f"  Skipping A/B test: only {len(test_dataset)} samples in vault (need >= 10)")
            print(f"  Rate more songs to build up a stable A/B test vault")
    elif verbose and len(vault_files) == 0:
        print(f"\n  No vault files available for A/B testing")
        print(f"  Need at least {vault_size} rated files to create vault")
    elif verbose:
        print(f"\n  No production model found at {prod_classifier_path}")
        print(f"  Run 'promote-to-prod' after initial training to enable A/B testing")

    return model_card


def generate_recommendations(
    config: dict,
    prod_dir: str = None,
    low_rating_ratio: float = 0.0,
    genre_filter: str = None
):
    """Generate recommendations for unrated songs.

    Args:
        config: Configuration dictionary
        prod_dir: Optional path to production models directory. If specified,
                  loads models from this directory instead of checkpoint_dir.
        low_rating_ratio: Ratio of low-ranked (predicted dislike) songs to include
                         in recommendations (0.0-1.0). Useful for A/B testing to
                         ensure negative labels and force careful listening.
        genre_filter: Optional genre category to filter recommendations by.
                     Valid categories: rock, pop, electronic, hiphop, jazz_classical,
                     country, latin_world

    Raises:
        ValueError: If classifier was trained with a different encoder version
    """
    import json
    import numpy as np

    # Ensure clean memory state at start of stage
    cleanup_memory()

    print("=" * 80)
    if prod_dir:
        print("GENERATING RECOMMENDATIONS (PRODUCTION)")
    else:
        print("GENERATING RECOMMENDATIONS")
    print("=" * 80)

    # Load configuration
    music_config = config['music']
    rec_config = config.get('recommendations', {})
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Determine model directory (prod or dev)
    if prod_dir:
        model_dir = Path(prod_dir)
        embeddings_db_path = model_dir / "embeddings.db"
        print(f"\n  Using production models from: {model_dir}")
    else:
        model_dir = Path(config['checkpoint_dir'])
        embeddings_db_path = Path(music_config['embedding_db_path'])

    # Validate model compatibility BEFORE proceeding
    encoder_checkpoint = model_dir / "encoder_best.pt"
    classifier_checkpoint = model_dir / "classifier_best.pt"

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

    # Filter by genre if specified
    if genre_filter:
        from ml_skeleton.music.genre_mapper import GENRE_CATEGORIES, parse_genre_string
        genre_filter_lower = genre_filter.lower()

        # Validate genre category
        if genre_filter_lower not in GENRE_CATEGORIES:
            print(f"\n  ERROR: Invalid genre '{genre_filter}'")
            print(f"  Valid categories: {', '.join(GENRE_CATEGORIES)}")
            return

        # Filter songs that have this genre
        unrated_songs_filtered = [
            s for s in unrated_songs
            if genre_filter_lower in parse_genre_string(s.genre)
        ]
        print(f"  Filtered to {len(unrated_songs_filtered)} '{genre_filter}' songs")
        unrated_songs = unrated_songs_filtered

    if len(unrated_songs) == 0:
        print("  No unrated songs to recommend!")
        return

    # Load embeddings
    print("\n[2/5] Loading embeddings...")
    embedding_store = EmbeddingStore(str(embeddings_db_path))

    filenames = [s.filename for s in unrated_songs]
    # Use encoder_version for embedding lookup (with fallback to model_version for backwards compatibility)
    encoder_version = music_config.get('encoder_version', music_config.get('model_version', 'v1'))
    embeddings_dict = embedding_store.get_embeddings_batch_all_chunks(
        filenames,
        model_version=encoder_version,
        num_chunks=4
    )
    if not embeddings_dict:
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
    _first_emb = next(iter(embeddings_dict.values()))
    _arr = np.asarray(_first_emb)
    embedding_dim = int(_arr.shape[-1]) if _arr.ndim > 1 else len(_arr)

    checkpoint_path = classifier_checkpoint  # Already set above based on model_dir
    if not checkpoint_path.exists():
        print(f"  Classifier checkpoint not found: {checkpoint_path}")
        if prod_dir:
            print("  Run 'promote-to-prod' first to deploy models!")
        else:
            print("  Run classifier training first!")
        return

    # Load checkpoint first to infer architecture from state dict
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Infer hidden_dims from checkpoint state dict
    # Layer weights are named mlp.0.weight, mlp.3.weight, mlp.6.weight, etc.
    # Each weight shape is [out_features, in_features]
    hidden_dims = []
    layer_idx = 0
    while f"mlp.{layer_idx}.weight" in state_dict:
        weight = state_dict[f"mlp.{layer_idx}.weight"]
        out_features = weight.shape[0]
        # Skip the final output layer (size 1)
        if out_features > 1:
            hidden_dims.append(out_features)
        layer_idx += 3  # Each block is Linear + ReLU + Dropout (3 layers)

    dropout = config['classifier'].get('dropout', 0.3)  # Dropout doesn't affect loading
    print(f"  Inferred architecture from checkpoint: hidden_dims={hidden_dims}")

    classifier = SimpleRatingClassifier(
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    )

    classifier.load_state_dict(state_dict)
    classifier = classifier.to(device)

    # Get classification mode from config
    classifier_config = config['classifier']
    classification_mode = classifier_config.get('classification_mode', 'regression')

    # Generate predictions
    print("\n[4/5] Generating predictions...")
    if classification_mode == "binary":
        loss_fn = BinaryRatingLoss()
        print(f"  Mode: Binary classification (like/dislike)")
    else:
        loss_fn = RatingLoss()
        print(f"  Mode: Regression (continuous ratings)")

    trainer = ClassifierTrainer(
        classifier=classifier,
        device=device,
        loss_fn=loss_fn,
        optimizer=torch.optim.Adam(classifier.parameters()),  # Dummy optimizer
        classification_mode=classification_mode
    )

    predictions, pred_filenames = trainer.predict(data_loader)

    # Diagnostic: Show prediction statistics
    if predictions:
        pred_array = np.array(predictions)
        if classification_mode == "binary":
            print(f"  Prediction statistics (probability of 'like', where 1.0 = definitely like):")
            print(f"    Min: {pred_array.min():.4f}")
            print(f"    Max: {pred_array.max():.4f}")
            print(f"    Mean: {pred_array.mean():.4f}")
            print(f"    Std: {pred_array.std():.4f}")
            print(f"    Predicted 'likes' (>0.5): {(pred_array > 0.5).sum()} songs")
        else:
            print(f"  Prediction statistics (normalized 0-1 scale, where 1.0 = 5 stars):")
            print(f"    Min: {pred_array.min():.4f} ({pred_array.min() * 5:.2f} stars)")
            print(f"    Max: {pred_array.max():.4f} ({pred_array.max() * 5:.2f} stars)")
            print(f"    Mean: {pred_array.mean():.4f} ({pred_array.mean() * 5:.2f} stars)")
            print(f"    Std: {pred_array.std():.4f}")

    # Sort by predicted rating
    results = list(zip(predictions, pred_filenames))
    results.sort(reverse=True)  # Highest ratings first

    # Apply threshold (only for high-predicted songs)
    min_threshold = rec_config.get('min_rating_threshold', 0.0)
    above_threshold = sum(1 for r, _ in results if r >= min_threshold)
    print(f"  Threshold: {min_threshold} ({min_threshold * 5:.1f} stars) - {above_threshold}/{len(results)} songs pass")

    # Get top-N with optional low-rating ratio
    top_n = rec_config.get('top_n', 100)

    if low_rating_ratio > 0.0:
        # Include some predicted "dislikes" for A/B testing balance
        n_low = int(top_n * low_rating_ratio)
        n_high = top_n - n_low

        # High-predicted songs (above threshold)
        high_results = [(r, f) for r, f in results if r >= min_threshold][:n_high]

        # Low-predicted songs (from the bottom, below threshold preferred)
        results_sorted_low = sorted(results, key=lambda x: x[0])  # Lowest first
        low_results = results_sorted_low[:n_low]

        # Combine: high predictions + low predictions (marked for identification)
        print(f"\n  Low-rating ratio: {low_rating_ratio:.1%}")
        print(f"    High-predicted songs: {len(high_results)}")
        print(f"    Low-predicted songs: {len(low_results)}")

        # Mark low-predicted songs with negative rating for identification
        # (The actual prediction value is stored, sign indicates category)
        final_results = high_results + low_results
        # Sort by absolute prediction for display, but keep track of which are "low"
        low_filenames = {f for _, f in low_results}
        results = final_results
    else:
        results = [(r, f) for r, f in results if r >= min_threshold][:top_n]
        low_filenames = set()

    print(f"  Generated {len(results)} recommendations (top {top_n})")

    # Save recommendations
    print("\n[5/5] Saving recommendations...")

    # Determine filename prefix for genre-filtered playlists
    filename_prefix = f"{genre_filter}_" if genre_filter else ""

    # Apply genre prefix to output path
    base_output_path = rec_config.get('output_path', './recommendations.txt')
    if genre_filter:
        output_dir = Path(base_output_path).parent
        output_name = f"{filename_prefix}recommendations.txt"
        output_path = output_dir / output_name
    else:
        output_path = Path(base_output_path)

    with open(output_path, 'w') as f:
        f.write(f"Top {len(results)} Recommendations")
        if genre_filter:
            f.write(f" (Genre: {genre_filter})")
        f.write("\n")
        if low_rating_ratio > 0.0:
            f.write(f"(includes {low_rating_ratio:.0%} predicted dislikes marked with [LOW])\n")
        f.write("=" * 80 + "\n\n")

        for i, (rating, filename) in enumerate(results, 1):
            # Find song metadata
            song = next((s for s in unrated_songs if s.filename == filename), None)
            if song:
                is_low = filename in low_filenames
                marker = " [LOW]" if is_low else ""
                f.write(f"{i}. [{rating:.3f}]{marker} {song.artist} - {song.title}\n")
                f.write(f"   Album: {song.album} ({song.year})\n")
                f.write(f"   Path: {filename}\n\n")

    print(f"  Saved to: {output_path}")

    # Print top 10
    print("\nTop 10 Recommendations:")
    for i, (rating, filename) in enumerate(results[:10], 1):
        song = next((s for s in unrated_songs if s.filename == filename), None)
        if song:
            is_low = filename in low_filenames
            marker = " [LOW]" if is_low else ""
            print(f"  {i}. [{rating:.3f}]{marker} {song.artist} - {song.title}")

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
        uncertainty_method="distance_from_middle",
        filename_prefix=filename_prefix
    )

    print("\n" + "=" * 80)
    print("RECOMMENDATION COMPLETE")
    if genre_filter:
        print(f"(Genre filter: {genre_filter})")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {output_path} (text recommendations)")
    print(f"  - {playlist_output_dir / f'{filename_prefix}recommender_help.xspf'} (high uncertainty - maximize learning)")
    print(f"  - {playlist_output_dir / f'{filename_prefix}recommender_best.xspf'} (top predictions - validate quality)")
    print(f"\nNext steps for human-in-the-loop training:")
    print(f"  1. Open XSPF playlists in Clementine")
    print(f"  2. Listen and rate songs")
    print(f"  3. Re-run training with updated ratings")
    print(f"  4. Repeat for continuous improvement!")


def build_waveform_cache(config: dict):
    """Pre-populate waveform cache for consistent training speed.

    Supports two caching modes:
    1. MoCo mode (chunk_cache enabled): 4 evenly-spaced 30s chunks per song
    2. Legacy mode: Single crop from specified position

    Args:
        config: Configuration dictionary
    """
    from ml_skeleton.music.clementine_db import ClementineDB

    music_config = config['music']

    # Load database
    print("\n[1/3] Loading song database...")
    db = ClementineDB(music_config.get('database_path'))
    all_songs = db.get_all_songs()
    print(f"  Found {len(all_songs)} songs")

    # Check if using MoCo chunk cache (new 4-chunk strategy)
    chunk_cache_config = music_config.get('chunk_cache', {})
    use_chunk_cache = chunk_cache_config.get('enabled', False)

    if use_chunk_cache:
        # MoCo mode: Use new 4-chunk cache builder
        from ml_skeleton.music.chunk_cache import build_chunk_cache, get_cache_stats

        cache_dir = chunk_cache_config.get('directory', './cache/chunks')
        num_chunks = chunk_cache_config.get('num_chunks', 4)
        chunk_duration = chunk_cache_config.get('chunk_duration', 30.0)
        num_workers = chunk_cache_config.get('num_workers', None)  # None = 80% CPU
        sample_rate = music_config.get('sample_rate', 16000)
        max_duration = music_config.get('max_duration', 900.0)

        print(f"\n[2/3] MoCo Chunk Cache configuration:")
        print(f"  Cache dir: {cache_dir}")
        print(f"  Chunks per song: {num_chunks}")
        print(f"  Chunk duration: {chunk_duration}s")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Max file duration: {max_duration}s")
        print(f"  Workers: {num_workers if num_workers else 'auto (80% CPU)'}")

        print("\n[3/3] Building chunk cache...")
        stats = build_chunk_cache(
            songs=all_songs,
            cache_dir=cache_dir,
            num_chunks=num_chunks,
            chunk_duration=chunk_duration,
            sample_rate=sample_rate,
            max_duration=max_duration,
            num_workers=num_workers,
            overwrite=False,
            show_progress=True
        )

        # Show final stats
        final_stats = get_cache_stats(cache_dir)
        print(f"\nCache complete:")
        print(f"  Total songs: {final_stats['num_songs']}")
        print(f"  Total files: {final_stats['num_files']}")
        print(f"  Cache size: {final_stats['size_gb']:.1f} GB")

    else:
        # No legacy mode - MoCo chunk cache is the only supported method
        print("\nERROR: chunk_cache.enabled must be True in config")
        print("MoCo v2 encoder requires 4-chunk cache. Update your config:")
        print("  music:")
        print("    chunk_cache:")
        print("      enabled: true")
        print("      directory: ./cache/chunks")
        print("      num_chunks: 4")
        print("      chunk_duration: 30.0")
        return


def fingerprint_songs_stage(config: dict, exhaust: bool = False, workers: Optional[int] = None):
    """Extract acoustic fingerprints from original audio files.

    Args:
        config: Configuration dictionary with music and fingerprinting settings
        exhaust: If True, process up to daily API limit (500 for free tier)
        workers: Number of parallel workers (default: from config or 4)
    """
    from ml_skeleton.music.file_fingerprinter import fingerprint_songs_from_files
    from ml_skeleton.music.fingerprint_db import FingerprintDB

    print("=" * 80)
    print("EXTRACTING ACOUSTIC FINGERPRINTS")
    print("=" * 80)

    # Get configurations
    music_config = config.get('music', {})
    fp_config = config.get('fingerprinting', {})

    # Check if fingerprinting is enabled
    if not fp_config.get('enabled', False):
        print("\nWARNING: Fingerprinting is disabled in config")
        print("Enable in config:")
        print("  fingerprinting:")
        print("    enabled: true")
        return

    # Load songs from Clementine database
    print("\n[1/4] Loading songs from database...")
    from ml_skeleton.music.clementine_db import ClementineDB

    # Support both 'database_path' (new) and 'clementine_db_path' (legacy) config keys
    db_path = os.getenv('CLEMENTINE_DB_PATH',
                       music_config.get('database_path') or music_config.get('clementine_db_path'))
    clementine_db = ClementineDB(db_path)
    all_songs = clementine_db.get_all_songs()
    print(f"  Loaded {len(all_songs)} songs")

    # Initialize fingerprint database
    print("\n[2/4] Initializing fingerprint database...")
    fp_db_path = fp_config.get('fingerprint_db_path', './cache/fingerprints.db')
    fp_db = FingerprintDB(fp_db_path)
    print(f"  Database: {fp_db_path}")

    # Get existing stats
    existing_stats = fp_db.get_stats()
    if existing_stats['total_fingerprints'] > 0:
        print(f"  Existing fingerprints: {existing_stats['total_fingerprints']}")
        print(f"  Existing songs: {existing_stats['unique_songs']}")

    # Get fingerprinting parameters
    # Use CLI override if provided, otherwise config (default: 4)
    num_workers = workers if workers is not None else fp_config.get('num_workers', 4)
    skip_existing = fp_config.get('skip_existing', True)
    max_duration = fp_config.get('max_duration', 300)  # Default: 5 minutes

    # Get prioritization and limit settings
    prioritize_missing = fp_config.get('prioritize_missing_metadata', True)

    # Determine max_songs based on exhaust flag
    if exhaust:
        # Use daily API limit for the tier
        tier = fp_config.get('acoustid_tier', 'free')
        max_songs = 500 if tier == 'free' else None  # 500 for free tier, unlimited for paid
        print("\n⚡ EXHAUST MODE: Processing up to daily API limit")
        if tier == 'free':
            print(f"  Free tier: 500 songs/day max")
        else:
            print(f"  Paid tier: unlimited (rate-limited to 3 req/s)")
    else:
        max_songs = fp_config.get('max_songs', 10)  # Default: 10 for testing

    print(f"\n[3/4] Fingerprinting configuration:")
    print(f"  Source: Original audio files (full duration)")
    print(f"  Workers: {num_workers}")
    print(f"  Skip existing: {skip_existing}")
    print(f"  Max duration: {max_duration}s (filters out long tracks)")
    print(f"  Prioritize missing metadata: {prioritize_missing}")
    print(f"  Max songs: {max_songs if max_songs else 'unlimited'}")

    # Extract fingerprints from original audio files
    print(f"\n[4/4] Extracting fingerprints from original files...")
    print(f"  NOTE: This fingerprints FULL audio files, not 30s chunks")
    print(f"  This is required for AcoustID API matching")
    stats = fingerprint_songs_from_files(
        songs=all_songs,
        fp_db=fp_db,
        num_workers=num_workers,
        skip_existing=skip_existing,
        prioritize_missing_metadata=prioritize_missing,
        max_songs=max_songs,
        max_duration=max_duration,
        verbose=True
    )

    # Display results
    print("\n" + "=" * 80)
    print("FINGERPRINTING COMPLETE")
    print("=" * 80)
    print(f"  Total songs processed: {stats['total_songs']}")
    print(f"  Successfully fingerprinted: {stats['fingerprinted']}")
    print(f"  Skipped (existing): {stats['skipped']}")
    if stats.get('skipped_duration', 0) > 0:
        print(f"  Skipped (too long): {stats['skipped_duration']}")
    print(f"  Failed: {stats['failed']}")
    if stats['processed'] > 0:
        print(f"  Rated songs processed: {stats['rated']}")
        print(f"  Unrated songs processed: {stats['unrated']}")

    if stats['failed'] > 0 and stats['errors']:
        print(f"\n  First 5 errors:")
        for error in stats['errors'][:5]:
            print(f"    - {error}")

    # Show final database stats
    final_stats = fp_db.get_stats()
    print(f"\nFingerprint Database Statistics:")
    print(f"  Total fingerprints: {final_stats['total_fingerprints']}")
    print(f"  Unique songs: {final_stats['unique_songs']}")
    print(f"  Complete fingerprints: {final_stats['songs_with_complete_fingerprints']}")
    print(f"  Database size: {final_stats['db_size_mb']:.2f} MB")
    print(f"  Database path: {fp_db_path}")
    print("")


def enrich_metadata_stage(config: dict, exhaust: bool = False):
    """Enrich song metadata using AcoustID and MusicBrainz APIs.

    Args:
        config: Configuration dictionary with music and fingerprinting settings
        exhaust: If True, process up to daily API limit (500 for free tier)
    """
    from ml_skeleton.music.metadata_enrichment import enrich_songs_metadata
    from ml_skeleton.music.fingerprint_db import FingerprintDB
    from ml_skeleton.music.musicbrainz_db import MusicBrainzDB

    print("=" * 80)
    print("ENRICHING METADATA VIA ACOUSTID/MUSICBRAINZ")
    print("=" * 80)

    # Get configurations
    music_config = config.get('music', {})
    fp_config = config.get('fingerprinting', {})

    # Check if external lookup is enabled
    if not fp_config.get('enable_external_lookup', False):
        print("\nWARNING: External lookup is disabled in config")
        print("Enable in config:")
        print("  fingerprinting:")
        print("    enable_external_lookup: true")
        print("    acoustid_api_key: YOUR_API_KEY")
        return

    # Get API key from environment variable
    acoustid_api_key = os.getenv('ACOUSTID_API_KEY', fp_config.get('acoustid_api_key'))
    if not acoustid_api_key or acoustid_api_key.startswith('${'):
        print("\nERROR: AcoustID API key not configured")
        print("Set environment variable: export ACOUSTID_API_KEY=your_key_here")
        print("Or add to config: fingerprinting.acoustid_api_key")
        print("\nRegister for free API key at: https://acoustid.org/new-application")
        return

    # Load songs from Clementine database
    print("\n[1/5] Loading songs from database...")
    from ml_skeleton.music.clementine_db import ClementineDB

    # Support both 'database_path' (new) and 'clementine_db_path' (legacy) config keys
    db_path = os.getenv('CLEMENTINE_DB_PATH',
                       music_config.get('database_path') or music_config.get('clementine_db_path'))
    clementine_db = ClementineDB(db_path)
    all_songs = clementine_db.get_all_songs()
    print(f"  Loaded {len(all_songs)} songs")

    # Initialize databases
    print("\n[2/5] Initializing databases...")
    fp_db_path = fp_config.get('fingerprint_db_path', './cache/fingerprints.db')
    mb_db_path = fp_config.get('musicbrainz_db_path', './musicbrainz_metadata.db')

    fp_db = FingerprintDB(fp_db_path)
    mb_db = MusicBrainzDB(mb_db_path)

    print(f"  Fingerprint DB: {fp_db_path}")
    print(f"  MusicBrainz DB: {mb_db_path}")

    # Get existing stats
    fp_stats = fp_db.get_stats()
    mb_stats = mb_db.get_stats()

    print(f"  Songs with fingerprints: {fp_stats['unique_songs']}")
    print(f"  Songs with enrichment: {mb_stats['total_songs']}")

    # Get enrichment parameters
    chunk_idx = fp_config.get('chunk_for_fingerprinting', 1)
    acoustid_rate_limit = fp_config.get('acoustid_rate_limit', 3.0)
    mb_rate_limit = fp_config.get('musicbrainz_rate_limit', 1.0)
    skip_existing = fp_config.get('skip_existing', True)

    # Determine max_songs based on exhaust flag
    if exhaust:
        # Use daily API limit for the tier
        tier = fp_config.get('acoustid_tier', 'free')
        max_songs = 500 if tier == 'free' else None  # 500 for free tier, unlimited for paid
        print("\n⚡ EXHAUST MODE: Processing up to daily API limit")
        if tier == 'free':
            print(f"  Free tier: 500 songs/day max")
        else:
            print(f"  Paid tier: unlimited (rate-limited to 3 req/s)")
    else:
        max_songs = fp_config.get('max_songs', 10)  # Default: 10 for testing

    print(f"\n[3/5] Enrichment configuration:")
    print(f"  Chunk index: {chunk_idx}")
    print(f"  AcoustID rate limit: {acoustid_rate_limit} req/s")
    print(f"  MusicBrainz rate limit: {mb_rate_limit} req/s")
    print(f"  Skip existing: {skip_existing}")
    print(f"  Max songs: {max_songs if max_songs else 'unlimited'}")

    # Estimate time for free tier
    if max_songs and max_songs <= 500:
        estimated_time_sec = max_songs / min(acoustid_rate_limit, mb_rate_limit)
        estimated_time_min = estimated_time_sec / 60
        print(f"  Estimated time: ~{estimated_time_min:.1f} minutes")

    # Enrich metadata
    print(f"\n[4/5] Enriching metadata...")
    print("NOTE: This queries external APIs (AcoustID + MusicBrainz)")
    print("Free tier limit: 500 lookups/day")
    print("")

    stats = enrich_songs_metadata(
        songs=all_songs,
        fp_db=fp_db,
        mb_db=mb_db,
        acoustid_api_key=acoustid_api_key,
        chunk_idx=chunk_idx,
        acoustid_rate_limit=acoustid_rate_limit,
        musicbrainz_rate_limit=mb_rate_limit,
        skip_existing=skip_existing,
        max_songs=max_songs,
        verbose=True
    )

    # Display results
    print("\n" + "=" * 80)
    print("METADATA ENRICHMENT COMPLETE")
    print("=" * 80)
    print(f"  Total songs: {stats['total_songs']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Successfully enriched: {stats['enriched']}")
    print(f"  Skipped (existing): {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  No fingerprint: {stats['no_fingerprint']}")
    print(f"  API lookups performed: {stats['api_lookups']}")
    if stats['processed'] > 0:
        print(f"  Rated songs processed: {stats['rated']}")
        print(f"  Unrated songs processed: {stats['unrated']}")

    if stats['failed'] > 0 and stats['errors']:
        print(f"\n  First 5 errors:")
        for error in stats['errors'][:5]:
            print(f"    - {error}")

    # Show API usage summary for exhaust mode
    if exhaust:
        tier = fp_config.get('acoustid_tier', 'free')
        if tier == 'free':
            remaining = 500 - stats['api_lookups']
            print(f"\n  Daily API quota (free tier):")
            print(f"    Used today: {stats['api_lookups']}/500 lookups")
            print(f"    Remaining: {remaining} lookups")
            if remaining > 0:
                print(f"    ℹ️  You can run enrichment again today to use remaining quota")
        else:
            print(f"\n  API usage (paid tier):")
            print(f"    Lookups performed: {stats['api_lookups']} (unlimited)")

    # Show final MusicBrainz database stats
    print("\n[5/5] MusicBrainz Database Statistics:")
    final_mb_stats = mb_db.get_stats()
    print(f"  Total enriched songs: {final_mb_stats['total_songs']}")
    print(f"  With AcoustID: {final_mb_stats['with_acoustid']}")
    print(f"  With MusicBrainz ID: {final_mb_stats['with_musicbrainz']}")
    print(f"  High confidence artist: {final_mb_stats['high_confidence_artist']}")
    print(f"  High confidence album: {final_mb_stats['high_confidence_album']}")
    print(f"  Avg artist confidence: {final_mb_stats['avg_artist_confidence']:.3f}")
    print(f"  Avg album confidence: {final_mb_stats['avg_album_confidence']:.3f}")
    print(f"  Database size: {final_mb_stats['db_size_mb']:.2f} MB")
    print(f"  Database path: {mb_db_path}")
    print("")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Music Recommendation System with Hyperparameter Tuning"
    )
    parser.add_argument(
        '--stage',
        type=str,
        required=True,
        choices=['encoder', 'classifier', 'recommend', 'all', 'tune-encoder', 'tune-classifier', 'build-cache', 'fingerprint', 'enrich-metadata', 'init-baseline', 'generate-model-card'],
        help='Training stage, recommendation generation, cache building, fingerprinting, metadata enrichment, or hyperparameter tuning'
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
        choices=['moco'],
        default=None,
        help='Override encoder type from config (only moco supported)'
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
    parser.add_argument(
        '--exhaust',
        action='store_true',
        help='Process maximum songs for the day (500 for free tier, respects API rate limits)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers for fingerprinting (default: 4, or from config)'
    )
    parser.add_argument(
        '-N', '--num-runs',
        type=int,
        default=1,
        dest='num_runs',
        help='Number of training runs with different seeds (reports mean/std, saves best model)'
    )
    parser.add_argument(
        '--prod-dir',
        type=str,
        default=None,
        help='Use production models from specified directory (e.g., prod/) for recommendations'
    )
    parser.add_argument(
        '--low-rating-ratio',
        type=float,
        default=0.0,
        dest='low_rating_ratio',
        help='Ratio of low-ranked (predicted dislike) songs to include in recommendations (e.g., 0.1 for 10%%). '
             'Useful for A/B testing to ensure negative labels and force careful listening.'
    )
    parser.add_argument(
        '--genre',
        type=str,
        default=None,
        help='Filter recommendations by genre category. '
             'Categories: rock, pop, electronic, hiphop, jazz_classical, country, latin_world'
    )
    parser.add_argument(
        '--random-init',
        action='store_true',
        dest='random_init',
        help='Initialize classifier with random weights instead of loading from production model. '
             'Default is to warm-start from prod/classifier_best.pt if architecture matches.'
    )
    parser.add_argument(
        '--vault-size',
        type=int,
        default=200,
        dest='vault_size',
        help='Number of ratings to reserve in vault for A/B testing only (default: 200). '
             'Vault files are never used for training, only for comparing models.'
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
            model_version_override=args.encoder_version,
            num_runs=args.num_runs
        )
        cleanup_memory()
        print("\nNext step: Run with --stage classifier to train the rating predictor")

    elif args.stage == 'classifier':
        model_card = train_classifier(
            config,
            final_training=args.final_training,
            classifier_version_override=args.classifier_version,
            num_runs=args.num_runs,
            init_from_prod=not args.random_init,
            vault_size=args.vault_size
        )
        cleanup_memory()
        print("\nNext step: Run with --stage recommend to generate recommendations")

    elif args.stage == 'recommend':
        generate_recommendations(
            config,
            prod_dir=args.prod_dir,
            low_rating_ratio=args.low_rating_ratio,
            genre_filter=args.genre
        )
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

    elif args.stage == 'fingerprint':
        # Extract acoustic fingerprints from original files
        fingerprint_songs_stage(config, exhaust=args.exhaust, workers=args.workers)
        cleanup_memory()

    elif args.stage == 'enrich-metadata':
        # Enrich metadata using AcoustID/MusicBrainz APIs
        enrich_metadata_stage(config, exhaust=args.exhaust)
        cleanup_memory()

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
            model_version_override=args.encoder_version,
            num_runs=args.num_runs
        )
        cleanup_memory()

        # Stage 2: Train classifier (with encoder statistics)
        model_card = train_classifier(
            config,
            model_card=model_card,
            final_training=args.final_training,
            classifier_version_override=args.classifier_version,
            num_runs=args.num_runs,
            vault_size=args.vault_size
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
        if args.num_runs > 1:
            print(f"Runs per trial: {args.num_runs} (min loss used as objective)")
        print(f"Search space parameters: {list(exp_config.tuning.search_space.parameters.keys())}")
        print("")

        # Create training function (pass n_trials for progress logging)
        train_fn = create_encoder_training_fn(config, n_trials=n_trials, hpo_runs=args.num_runs)

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
        checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        best_params_file = checkpoint_dir / 'best_encoder_params.json'
        best_params_file.parent.mkdir(parents=True, exist_ok=True)
        with open(best_params_file, 'w') as f:
            json.dump(results['best_params'], f, indent=2)
        print(f"\nBest parameters saved to: {best_params_file}")

        # Check if HPO best checkpoint exists
        hpo_best_checkpoint = checkpoint_dir / 'encoder_hpo_best.pt'
        if hpo_best_checkpoint.exists():
            print(f"Best HPO model saved to: {hpo_best_checkpoint}")
            print("\nTo continue training from best HPO model with best parameters:")
            print(f"  python examples/music_recommendation.py --stage encoder --config configs/music_recommendation.yaml \\")
            print(f"      --final-training --best-params {best_params_file} --resume-checkpoint {hpo_best_checkpoint}")
        else:
            print("\nTo run final training with best parameters:")
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
        if args.num_runs > 1:
            print(f"Runs per trial: {args.num_runs} (min MAE used as objective)")
        print(f"Search space parameters: {list(exp_config.tuning.search_space.parameters.keys())}")
        print("")

        # Create training function (pass n_trials for progress logging)
        train_fn = create_classifier_training_fn(config, n_trials=n_trials, hpo_runs=args.num_runs)

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

    elif args.stage == 'init-baseline':
        # Create a random baseline classifier for A/B testing workflow testing
        print("\n" + "=" * 80)
        print("CREATING RANDOM BASELINE FOR A/B TESTING")
        print("=" * 80)

        music_config = config['music']
        classifier_config = config['classifier']

        # Load embeddings to get dimension
        embedding_store = EmbeddingStore(music_config['embedding_db_path'])
        encoder_version = music_config.get('encoder_version', 'v1')

        # Get one embedding to determine dimension
        db = ClementineDB(music_config['database_path'])
        all_songs = db.get_all_songs()
        rated_songs = [s for s in all_songs if s.is_rated]

        if not rated_songs:
            print("ERROR: No rated songs found")
            sys.exit(1)

        embeddings_dict = embedding_store.get_embeddings_batch(
            [rated_songs[0].filename],
            model_version=encoder_version
        )

        if not embeddings_dict:
            print("ERROR: No embeddings found. Run encoder training first.")
            sys.exit(1)

        embedding_dim = len(next(iter(embeddings_dict.values())))
        print(f"  Embedding dimension: {embedding_dim}")

        # Create random classifier
        hidden_dims = classifier_config.get('hidden_dims', [256, 128])
        classifier = SimpleRatingClassifier(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=0.0
        )

        # Random initialization is default - just save it
        prod_dir = Path("prod")
        prod_dir.mkdir(exist_ok=True)

        checkpoint_path = prod_dir / "classifier_best.pt"
        torch.save({
            'model_state_dict': classifier.state_dict(),
            'encoder_version': encoder_version,
            'classifier_version': 'random_baseline',
            'embedding_dim': embedding_dim,
            'hidden_dims': hidden_dims,
        }, checkpoint_path)

        print(f"  Random baseline classifier saved to: {checkpoint_path}")
        print(f"  Architecture: {hidden_dims}")

        # Also copy embeddings.db if not exists
        embeddings_src = Path(music_config['embedding_db_path'])
        embeddings_dst = prod_dir / "embeddings.db"
        if embeddings_src.exists() and not embeddings_dst.exists():
            import shutil
            shutil.copy(embeddings_src, embeddings_dst)
            print(f"  Embeddings copied to: {embeddings_dst}")

        # Copy encoder if exists
        encoder_src = Path(config['checkpoint_dir']) / "encoder_best.pt"
        encoder_dst = prod_dir / "encoder_best.pt"
        if encoder_src.exists() and not encoder_dst.exists():
            import shutil
            shutil.copy(encoder_src, encoder_dst)
            print(f"  Encoder copied to: {encoder_dst}")

        # Create initial training manifest with vault for A/B testing
        # Vault contains vault_size files reserved exclusively for A/B testing
        import random
        random.seed(42)

        # Get all embeddings for rated songs
        all_rated_filenames = [s.filename for s in rated_songs]
        all_embeddings = embedding_store.get_embeddings_batch(
            all_rated_filenames,
            model_version=encoder_version
        )

        # Apply the same filtering as classifier training
        classification_mode = classifier_config.get('classification_mode', 'binary')
        binary_positive_threshold = classifier_config.get('binary_positive_threshold', 4.0)
        binary_negative_threshold = classifier_config.get('binary_negative_threshold', 2.0)

        valid_songs = []
        for song in rated_songs:
            # Must have embedding
            if song.filename not in all_embeddings:
                continue
            # For binary mode, exclude ambiguous ratings
            if classification_mode == 'binary':
                if song.rating >= binary_positive_threshold:
                    valid_songs.append(song)
                elif song.rating <= binary_negative_threshold:
                    valid_songs.append(song)
                # Skip songs between thresholds (ambiguous)
            else:
                valid_songs.append(song)

        print(f"  Classification mode: {classification_mode}")
        print(f"  Rated songs: {len(rated_songs)}")
        print(f"  With embeddings: {len(all_embeddings)}")
        if classification_mode == 'binary':
            print(f"  After binary filtering (rating >= {binary_positive_threshold} or <= {binary_negative_threshold}): {len(valid_songs)}")
        else:
            print(f"  Valid for training: {len(valid_songs)}")

        # Create binary rating lookup for class balancing
        file_ratings = {}
        for song in valid_songs:
            if classification_mode == 'binary':
                file_ratings[song.filename] = 1 if song.rating >= binary_positive_threshold else 0
            else:
                file_ratings[song.filename] = 1 if song.rating >= 3.0 else 0

        all_filenames = [s.filename for s in valid_songs]
        random.shuffle(all_filenames)

        # Split: vault_size for A/B testing, remainder split 80/20 train/val
        vault_size = args.vault_size
        n_total = len(all_filenames)
        actual_vault_size = min(vault_size, n_total)

        # Create class-balanced vault (50/50 likes/dislikes)
        likes = [f for f in all_filenames if file_ratings.get(f, 0) == 1]
        dislikes = [f for f in all_filenames if file_ratings.get(f, 0) == 0]
        half_vault = actual_vault_size // 2
        vault_likes = likes[:min(half_vault, len(likes))]
        vault_dislikes = dislikes[:min(half_vault, len(dislikes))]
        # Fill remaining if one class is short
        remaining_slots = actual_vault_size - len(vault_likes) - len(vault_dislikes)
        if remaining_slots > 0:
            if len(vault_likes) < half_vault:
                vault_dislikes.extend(dislikes[len(vault_dislikes):len(vault_dislikes) + remaining_slots])
            else:
                vault_likes.extend(likes[len(vault_likes):len(vault_likes) + remaining_slots])
        vault_files = vault_likes + vault_dislikes

        # Remaining files (not in vault) for train/val
        vault_set = set(vault_files)
        remaining = [f for f in all_filenames if f not in vault_set]

        n_train = int(len(remaining) * 0.8)
        train_files = remaining[:n_train]
        val_files = remaining[n_train:]

        # Create manifest in CHECKPOINTS dir (not prod) - this is where classifier training looks
        checkpoint_dir = Path(config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        manifest = TrainingManifest(str(checkpoint_dir / "training_manifest.json"))
        manifest.training_files = set(train_files)
        manifest.validation_files = set(val_files)
        manifest.vault_files = set(vault_files)  # Reserved for A/B testing
        manifest.set_version_info(
            encoder_version=encoder_version,
            classifier_version='random_baseline',
            classification_mode=classifier_config.get('classification_mode', 'binary')
        )
        manifest.save()

        # Also copy to prod for reference
        prod_manifest = TrainingManifest(str(prod_dir / "training_manifest.json"))
        prod_manifest.training_files = set(train_files)
        prod_manifest.validation_files = set(val_files)
        prod_manifest.vault_files = set(vault_files)
        prod_manifest.set_version_info(
            encoder_version=encoder_version,
            classifier_version='random_baseline',
            classification_mode=classifier_config.get('classification_mode', 'binary')
        )
        prod_manifest.save()

        print(f"  Training manifest created:")
        print(f"    Training files: {len(train_files)}")
        print(f"    Validation files: {len(val_files)}")
        vault_pos = sum(1 for f in vault_files if file_ratings.get(f, 0) == 1)
        vault_neg = len(vault_files) - vault_pos
        print(f"    Vault files: {len(vault_files)} ({vault_pos} likes, {vault_neg} dislikes) - reserved for A/B testing")

        print("\n" + "=" * 80)
        print("RANDOM BASELINE CREATED")
        print("=" * 80)
        print("\nNow train a real classifier and A/B test against this baseline:")
        print("  ./run_music_pipeline.sh classifier --final-training -N 3")
        print(f"\nThe {len(vault_files)} vault files will be used for A/B testing (never for training).")

    elif args.stage == 'generate-model-card':
        # Generate production model card with A/B test results
        from datetime import datetime
        import json

        print("\n" + "=" * 80)
        print("GENERATING PRODUCTION MODEL CARD")
        print("=" * 80)

        prod_dir = Path("prod")
        checkpoint_dir = Path(config['checkpoint_dir'])
        music_config = config['music']
        classifier_config = config['classifier']

        # Load training manifest for A/B test results
        manifest_path = checkpoint_dir / "training_manifest.json"
        ab_result = None
        manifest_data = {}

        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest_data = json.load(f)
            ab_result = manifest_data.get('metadata', {}).get('ab_test_result')

        # Load classifier checkpoint for architecture info
        classifier_path = checkpoint_dir / "classifier_best.pt"
        classifier_info = {}
        if classifier_path.exists():
            checkpoint = torch.load(classifier_path, map_location='cpu', weights_only=False)
            classifier_info = {
                'encoder_version': checkpoint.get('encoder_version', 'unknown'),
                'classifier_version': checkpoint.get('classifier_version', 'unknown'),
                'embedding_dim': checkpoint.get('embedding_dim', 'unknown'),
                'hidden_dims': checkpoint.get('hidden_dims', []),
            }

        # Count training data
        n_train = len(manifest_data.get('training_files', []))
        n_val = len(manifest_data.get('validation_files', []))

        # Generate markdown model card
        model_card_md = f"""# Music Recommendation Model Card

## Model Overview

- **Model Type**: Binary Rating Classifier (like/dislike)
- **Encoder Version**: {classifier_info.get('encoder_version', music_config.get('encoder_version', 'v1'))}
- **Classifier Version**: {classifier_info.get('classifier_version', music_config.get('classifier_version', 'v1'))}
- **Promoted to Production**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Architecture

- **Encoder**: MoCo v2 + Genre BCE (ResNet-50 backbone)
- **Embedding Dimension**: {classifier_info.get('embedding_dim', 2048)}
- **Classifier Hidden Layers**: {classifier_info.get('hidden_dims', classifier_config.get('hidden_dims', []))}
- **Classification Mode**: {classifier_config.get('classification_mode', 'binary')}

## Training Data

- **Training Songs**: {n_train:,}
- **Validation Songs**: {n_val:,}
- **Total Rated Songs**: {n_train + n_val:,}

## A/B Test Results

"""
        if ab_result:
            model_card_md += f"""| Metric | New Model | Production | Improvement |
|--------|-----------|------------|-------------|
| Accuracy | {ab_result.get('improvement', 0) + 0.5:.1%} | 50.0% (baseline) | {ab_result.get('improvement', 0):+.1%} |

- **Test Samples**: {ab_result.get('n_samples', 'N/A')} (new ratings since last training)
- **Statistical Test**: McNemar's test
- **p-value**: {ab_result.get('p_value', 'N/A'):.4f}
- **Significant**: {'✓ Yes (p < 0.05)' if ab_result.get('significant') else 'No'}

"""
        else:
            model_card_md += "*No A/B test results available*\n\n"

        model_card_md += f"""## Usage

```bash
# Generate recommendations using this model
./run_music_pipeline.sh recommend --prod

# Include 10% predicted dislikes for balanced feedback
./run_music_pipeline.sh recommend --prod --low-rating-ratio 0.1
```

## Files

- `encoder_best.pt` - Audio encoder (MoCo v2 + Genre BCE)
- `classifier_best.pt` - Rating classifier
- `embeddings.db` - Pre-computed embeddings for all songs
- `training_manifest.json` - Training/validation split tracking

## Iteration Workflow

1. Listen to recommended songs and rate them in Clementine
2. Re-run classifier training: `./run_music_pipeline.sh classifier --final-training -N 3`
3. Review A/B test results
4. If improved, promote: `./run_music_pipeline.sh promote-to-prod`

---
*Generated by ml_skeleton music recommendation pipeline*
"""

        # Save model card
        model_card_path = prod_dir / "MODEL_CARD.md"
        prod_dir.mkdir(exist_ok=True)
        with open(model_card_path, 'w') as f:
            f.write(model_card_md)

        print(f"  Model card saved to: {model_card_path}")

        # Get vault size from manifest
        n_vault = len(manifest_data.get('vault_files', []))

        # Also save as JSON for programmatic access
        model_card_json = {
            'model_type': 'binary_rating_classifier',
            'encoder_version': classifier_info.get('encoder_version', music_config.get('encoder_version', 'v1')),
            'classifier_version': classifier_info.get('classifier_version', music_config.get('classifier_version', 'v1')),
            'promoted_at': datetime.now().isoformat(),
            'architecture': {
                'encoder': 'moco_v2_genre_bce',
                'backbone': 'resnet50',
                'embedding_dim': classifier_info.get('embedding_dim', 2048),
                'hidden_dims': classifier_info.get('hidden_dims', classifier_config.get('hidden_dims', [])),
                'classification_mode': classifier_config.get('classification_mode', 'binary'),
            },
            'training_data': {
                'n_training': n_train,
                'n_validation': n_val,
                'total': n_train + n_val,
            },
            'ab_test': ab_result,
            # classifier_stats format for A/B history display compatibility
            'classifier_stats': {
                'train_size': n_train,
                'val_size': n_val,
                'vault_size': n_vault,
                'metadata': {
                    'ab_test_result': ab_result
                }
            }
        }

        model_card_json_path = prod_dir / "model_card.json"
        with open(model_card_json_path, 'w') as f:
            json.dump(model_card_json, f, indent=2)

        print(f"  Model card JSON saved to: {model_card_json_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("MODEL CARD SUMMARY")
        print("=" * 60)
        if ab_result:
            print(f"  A/B Test Results:")
            print(f"    Samples: {ab_result.get('n_samples', 'N/A')}")
            print(f"    Improvement: {ab_result.get('improvement', 0):+.1%}")
            print(f"    p-value: {ab_result.get('p_value', 'N/A'):.4f}")
            print(f"    Significant: {'✓ Yes' if ab_result.get('significant') else 'No'}")
        else:
            print("  No A/B test results available")
        print("=" * 60)


if __name__ == '__main__':
    main()
