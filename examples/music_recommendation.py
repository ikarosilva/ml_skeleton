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

    # Classifier: load hyperparameters from an MLflow run (e.g. classifier HPO parent run ID):
    python examples/music_recommendation.py --stage classifier --config configs/music_recommendation.yaml \
        --final-training --mlflow-run-id a985c318e2b1434abd04864cdcdaa4c4
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
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
    get_fingerprint_db_path,
    get_chunk_cache_dir,
    get_mlflow_tags
)
from ml_skeleton.music.baseline_classifier import SimpleRatingClassifier
from ml_skeleton.music.xspf_playlist import generate_human_feedback_playlists, export_to_xspf
from ml_skeleton.training.encoder_trainer import EncoderTrainer
from ml_skeleton.training.classifier_trainer import (
    ClassifierTrainer,
    get_encoder_version_from_checkpoint,
    validate_model_compatibility
)
from ml_skeleton.training.joint_finetune_trainer import JointFinetuneTrainer
from ml_skeleton.music.model_card import ModelCardGenerator
from ml_skeleton.music.dataset_stats import (
    collect_preprocessing_stats,
    collect_dataset_stats,
    collect_training_stats
)
from ml_skeleton.music.training_manifest import TrainingManifest
from ml_skeleton.music.ab_testing import run_ab_test, format_ab_test_summary, ab_result_to_mlflow_metrics

# Framework imports for hyperparameter tuning and MLflow tracking
from ml_skeleton import TrainingContext, TrainingResult, ExperimentConfig, run_experiment
from ml_skeleton.core.config import TunerType
from ml_skeleton.tracking import ExplrTracker, MLflowServer
from ml_skeleton.utils.memory import cleanup_memory, limit_gpu_memory


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


def _parse_mlflow_param_value(value: str):
    """Parse MLflow param string to int/float/bool/list/dict or leave as str."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return value


def get_mlflow_run_name(tracking_uri: str, run_id: str) -> str:
    """Return display name for an MLflow run (run_name, tag, or run_id prefix)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    run = client.get_run(run_id)
    if getattr(run.info, "run_name", None):
        return run.info.run_name or ""
    return run.data.tags.get("mlflow.runName", run_id[:8] if run_id else "")


def load_classifier_params_from_mlflow_run(tracking_uri: str, run_id: str) -> tuple[dict, str]:
    """Load classifier hyperparameters from an MLflow run (e.g. classifier HPO parent run).

    Supports both HPO parent runs (params like best_learning_rate, best_dropout, ...)
    and trial runs (learning_rate, dropout, ...). Skips non-hyperparameter keys.

    Returns:
        Tuple of (params_dict, run_name). params_dict is suitable for config['classifier'].
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params or {}
    skip_keys = {"n_trials", "sampler", "pruner", "n_completed_trials", "best_trial_number", "hpo_mlflow_run_id", "hpo_mlflow_run_name"}
    result = {}
    for key, value in params.items():
        if key in skip_keys:
            continue
        if key.startswith("best_"):
            config_key = key[5:]
        else:
            config_key = key
        result[config_key] = _parse_mlflow_param_value(value)
    run_name = getattr(run.info, "run_name", None) or run.data.tags.get("mlflow.runName", run_id[:8] if run_id else "")
    return result, run_name


def get_hpo_val_roc_auc(config: dict, tracking_uri: Optional[str] = None) -> Optional[float]:
    """Get HPO best val ROC AUC from config (from best params JSON) or from MLflow.

    Tries in order: (1) config key hpo_val_roc_auc, (2) run metric best_val_roc_auc,
    (3) max roc_auc among child runs, (4) run's own roc_auc (when run_id is a trial run).
    Returns None if not available.
    """
    clf = config.get("classifier", {})
    v = clf.get("hpo_val_roc_auc")
    if v is not None:
        return float(v)
    run_id = clf.get("hpo_mlflow_run_id")
    if not run_id:
        return None
    uri = tracking_uri or config.get("mlflow", {}).get("tracking_uri", "http://localhost:5000")
    err_msg = None
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(uri)
        client = MlflowClient(tracking_uri=uri)
        run = client.get_run(run_id)
        m = run.data.metrics or {}
        v = m.get("best_val_roc_auc")
        if v is not None:
            return float(v)
        # Fallback: get max roc_auc from child runs (HPO parent run; children are trials)
        from mlflow.entities import ViewType
        child_runs = client.search_runs(
            experiment_ids=[run.info.experiment_id],
            filter_string=f'tags."mlflow.parentRunId" = "{run_id}"',
            run_view_type=ViewType.ALL,
            max_results=500,
        )
        roc_aucs = []
        for r in child_runs:
            roc = (r.data.metrics or {}).get("roc_auc")
            if roc is not None:
                roc_aucs.append(float(roc))
        if roc_aucs:
            return max(roc_aucs)
        # Fallback: run_id may be a trial run (no children); use this run's roc_auc
        v = m.get("roc_auc")
        if v is not None:
            return float(v)
        return None
    except Exception as e:
        err_msg = str(e)
        return None
    finally:
        if err_msg is not None and run_id:
            print(f"  (Could not fetch HPO ROC AUC from MLflow run {run_id[:8]}...: {err_msg})")


def _report_and_log_hpo_vs_final_roc_auc(
    config: dict,
    hpo_val_roc_auc: Optional[float],
    final_val_roc_auc: Optional[float],
    classifier_mlflow_run_id: Optional[str],
) -> None:
    """Report HPO vs final val ROC AUC at end of classifier --final-training; log to MLflow."""
    if hpo_val_roc_auc is None and final_val_roc_auc is None:
        return
    tracking_uri = config.get("mlflow", {}).get("tracking_uri", "http://localhost:5000")
    print("\n" + "=" * 60)
    print("HPO vs FINAL TRAINING (val ROC AUC)")
    print("=" * 60)
    if hpo_val_roc_auc is not None:
        print(f"  HPO best val ROC AUC:  {hpo_val_roc_auc:.6f}")
    else:
        print("  HPO best val ROC AUC:  (not available — use --best-params or --mlflow-run-id from tune-classifier)")
    if final_val_roc_auc is not None:
        print(f"  Final val ROC AUC:     {final_val_roc_auc:.6f}")
    if hpo_val_roc_auc is not None and final_val_roc_auc is not None:
        diff = final_val_roc_auc - hpo_val_roc_auc
        print(f"  Difference (final−HPO): {diff:+.6f}")
        if final_val_roc_auc < hpo_val_roc_auc:
            print("\n  ⚠ WARNING: Final model has LOWER val ROC AUC than the HPO best run.")
            print("  Consider re-running HPO or checking data/split consistency.")
        if classifier_mlflow_run_id:
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient(tracking_uri=tracking_uri)
                client.log_metric(classifier_mlflow_run_id, "hpo_val_roc_auc", hpo_val_roc_auc)
                client.log_metric(classifier_mlflow_run_id, "final_val_roc_auc", final_val_roc_auc)
                client.log_metric(classifier_mlflow_run_id, "roc_auc_diff_final_minus_hpo", diff)
            except Exception:
                pass
    print("=" * 60)


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
# Per-trial best init seed when reps > 1 (trial_number -> seed)
_trial_best_reps_seed: dict = {}


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

        # Optional: disable chromaprint loss during HPO to avoid malloc/C library crash in first batch
        if os.environ.get("EXPLR_HPO_DISABLE_CHROMAPRINT") == "1":
            config["encoder"]["chromaprint_loss_weight"] = 0
            print("  HPO: chromaprint loss disabled (EXPLR_HPO_DISABLE_CHROMAPRINT=1)")

        # Optional: smaller encoder batch size during HPO to reduce chromaprint/malloc pressure in first batch
        _hpo_batch = os.environ.get("EXPLR_HPO_ENCODER_BATCH_SIZE")
        if _hpo_batch is not None:
            try:
                config["encoder"]["batch_size"] = int(_hpo_batch)
                print(f"  HPO: encoder batch_size overridden to {config['encoder']['batch_size']} (EXPLR_HPO_ENCODER_BATCH_SIZE)")
            except ValueError:
                pass

        # If chromaprint loss is on but fingerprint DB has no precomputed bits, disable chromaprint to avoid malloc/free crash
        if config["encoder"].get("chromaprint_loss_weight", 0) > 0:
            from ml_skeleton.music.fingerprint_db import FingerprintDB
            from ml_skeleton.music.encoder_factory import get_fingerprint_db_path
            fp_path = get_fingerprint_db_path(config)
            if Path(fp_path).exists():
                try:
                    _fp_db = FingerprintDB(fp_path)
                    _missing = _fp_db.count_missing_bits()
                    if _missing > 0:
                        config["encoder"]["chromaprint_loss_weight"] = 0
                        print(f"  HPO: chromaprint loss disabled: {_missing} fingerprints have no precomputed 'bits' (run: ./run_music_pipeline.sh backfill-fingerprint-bits)")
                except Exception:
                    pass

        # Override device if provided by context
        if ctx.device:
            config['device'] = ctx.device

        # Multi-run HPO: run multiple times with different seeds, nested as child runs under this trial
        # NOTE: config['seed'] stays constant for train/val split consistency across trials
        if hpo_runs > 1:
            base_seed = config.get('seed', 42)
            run_losses = []
            best_run_idx = 0
            best_run_loss = float('inf')

            for run_idx in range(hpo_runs):
                training_seed = base_seed + run_idx * 1000  # Different seed for model init/training

                if run_idx == 0:
                    print(f"  HPO multi-run: {hpo_runs} runs per trial (nested under trial, objective = min loss)")

                # Nest each seed run as a child MLflow run under this trial
                child_tracker = ctx.tracker.create_child_tracker(run_name=f"seed_{training_seed}")
                with child_tracker:
                    model_card = train_encoder(
                        config,
                        model_card=_global_model_card,
                        skip_embeddings=True,
                        trial_info=trial_info,
                        verbose=False,
                        training_seed=training_seed,
                        mlflow_tracker=child_tracker,
                    )

                encoder_stats = model_card.encoder_stats
                run_loss = encoder_stats.get('val_loss', encoder_stats.get('final_val_loss', float('inf')))
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
            best_val_loss = encoder_stats.get('val_loss', encoder_stats.get('final_val_loss', float('inf')))

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
        # Log seed used for this run; then metrics
        seed_used = encoder_stats.get('training_seed')
        if seed_used is not None:
            ctx.tracker.log_param('training_seed', str(seed_used))
        ctx.tracker.log_metric('val_loss', best_val_loss)
        ctx.tracker.log_metric('epochs_run', encoder_stats.get('epochs_run', 0))
        ctx.tracker.log_metric('training_time', encoder_stats.get('training_time_seconds', 0))

        return TrainingResult(
            primary_metric=best_val_loss,
            primary_metric_name='val_loss',
            minimize=True,
            metrics={
                'final_train_loss': encoder_stats.get('final_train_loss', 0),
                'final_val_loss': encoder_stats.get('final_val_loss', 0),
                'val_loss': best_val_loss,
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

        encoder_version = config.get('music', {}).get('encoder_version', 'default')
        effective_batch = _effective_classifier_batch_size(config['classifier'], encoder_version)
        suffix = " (fingerprint_baseline)" if encoder_version == 'fingerprint_baseline' else ""
        print(f"  Effective batch size: {effective_batch}{suffix}")

        # Override device if provided by context
        if ctx.device:
            config['device'] = ctx.device

        # When init_seed is in the search space, run a single trial with that seed (Optuna explores seed)
        classifier_config = config.get('classifier', {})
        search_space_params = config.get('tuning', {}).get('classifier_search_space', {}).get('parameters', {})
        init_seed_in_space = 'init_seed' in search_space_params
        hpo_init_seed = classifier_config.get('init_seed') if isinstance(classifier_config.get('init_seed'), (int, float)) else None
        if init_seed_in_space and hpo_init_seed is not None:
            training_seed = int(hpo_init_seed)
            print(f"  HPO init_seed: {training_seed} (single run)")
            model_card = train_classifier(
                config,
                model_card=_global_model_card,
                trial_info=trial_info,
                verbose=False,
                training_seed=training_seed,
                mlflow_tracker=ctx.tracker,
            )
            _global_model_card = model_card
            classifier_stats = model_card.classifier_stats
            best_val_loss = classifier_stats.get('val_loss', classifier_stats.get('final_val_loss', float('inf')))
            best_val_mae = classifier_stats.get('val_mae', best_val_loss)
            best_val_f1 = classifier_stats.get('val_f1')
            best_val_rating_mse = classifier_stats.get('val_rating_mse')
            best_val_rating_corr = classifier_stats.get('val_rating_corr')
        # Multi-rep HPO: run multiple times with different init seeds, nested as child runs under this trial
        # NOTE: config['seed'] stays constant for train/val split consistency across trials
        elif hpo_runs > 1:
            global _trial_best_reps_seed
            base_seed = config.get('seed', 42)
            hpo_metric = config.get('classifier', {}).get('hpo_metric', 'val_roc_auc')
            use_f1_for_best = (hpo_metric == 'val_f1')
            use_roc_auc_for_best = (hpo_metric == 'val_roc_auc')
            run_maes = []
            run_f1s = []
            run_roc_aucs = []
            best_run_idx = 0
            best_run_mae = float('inf')
            best_run_f1 = -1.0
            best_run_roc_auc = -1.0
            best_run_model_card = None

            for run_idx in range(hpo_runs):
                training_seed = base_seed + run_idx * 1000  # Different seed for model init/training

                if run_idx == 0:
                    print(f"  HPO reps: {hpo_runs} reps per trial (nested under trial, best by {hpo_metric})")

                # Nest each seed run as a child MLflow run under this trial
                child_tracker = ctx.tracker.create_child_tracker(run_name=f"seed_{training_seed}")
                with child_tracker:
                    model_card = train_classifier(
                        config,
                        model_card=_global_model_card,
                        trial_info=trial_info,
                        verbose=False,
                        training_seed=training_seed,
                        mlflow_tracker=child_tracker,
                    )

                classifier_stats = model_card.classifier_stats
                run_mae = classifier_stats.get('val_mae', classifier_stats.get('val_loss', float('inf')))
                run_f1 = classifier_stats.get('val_f1') or 0.0
                run_roc_auc = classifier_stats.get('val_roc_auc') or 0.0
                run_maes.append(run_mae)
                run_f1s.append(run_f1)
                run_roc_aucs.append(run_roc_auc)
                if use_f1_for_best:
                    print(f"    Rep {run_idx + 1}/{hpo_runs} (seed={training_seed}): val_f1={run_f1:.6f}")
                elif use_roc_auc_for_best:
                    print(f"    Rep {run_idx + 1}/{hpo_runs} (seed={training_seed}): val_roc_auc={run_roc_auc:.6f}")
                else:
                    print(f"    Rep {run_idx + 1}/{hpo_runs} (seed={training_seed}): val_mae={run_mae:.6f}")

                # Track best run by hpo_metric (max F1, max ROC AUC, or min MAE)
                if use_f1_for_best:
                    if run_f1 > best_run_f1:
                        best_run_f1 = run_f1
                        best_run_idx = run_idx
                        best_run_model_card = model_card
                elif use_roc_auc_for_best:
                    if run_roc_auc > best_run_roc_auc:
                        best_run_roc_auc = run_roc_auc
                        best_run_idx = run_idx
                        best_run_model_card = model_card
                else:
                    if run_mae < best_run_mae:
                        best_run_mae = run_mae
                        best_run_idx = run_idx
                        best_run_model_card = model_card

            best_val_mae = run_maes[best_run_idx]
            best_seed = base_seed + best_run_idx * 1000
            if ctx.trial_number is not None:
                _trial_best_reps_seed[ctx.trial_number] = best_seed
            if use_f1_for_best:
                print(f"    Best: val_f1={best_run_f1:.6f} (rep {best_run_idx + 1}, init_seed={best_seed}), Mean F1: {np.mean(run_f1s):.6f} +/- {np.std(run_f1s):.6f}")
            elif use_roc_auc_for_best:
                print(f"    Best: val_roc_auc={best_run_roc_auc:.6f} (rep {best_run_idx + 1}, init_seed={best_seed}), Mean: {np.mean(run_roc_aucs):.6f} +/- {np.std(run_roc_aucs):.6f}")
            else:
                print(f"    Best: {best_val_mae:.6f} (rep {best_run_idx + 1}, init_seed={best_seed}), Mean: {np.mean(run_maes):.6f} +/- {np.std(run_maes):.6f}")
            _global_model_card = best_run_model_card if best_run_model_card is not None else model_card
            best_val_loss = best_run_model_card.classifier_stats.get('val_loss', best_val_mae) if best_run_model_card else best_val_mae
            classifier_stats = _global_model_card.classifier_stats
            best_val_f1 = classifier_stats.get('val_f1')
            best_val_roc_auc = classifier_stats.get('val_roc_auc')
            best_val_rating_mse = classifier_stats.get('val_rating_mse')
            best_val_rating_corr = classifier_stats.get('val_rating_corr')
        else:
            # Single run (original behavior)
            model_card = train_classifier(
                config,
                model_card=_global_model_card,
                trial_info=trial_info,
                verbose=False,
                mlflow_tracker=ctx.tracker,
            )
            _global_model_card = model_card

            # Get metrics from classifier stats
            classifier_stats = model_card.classifier_stats
            best_val_loss = classifier_stats.get('val_loss', classifier_stats.get('final_val_loss', float('inf')))
            best_val_mae = classifier_stats.get('val_mae', best_val_loss)
            best_val_f1 = classifier_stats.get('val_f1')
            best_val_roc_auc = classifier_stats.get('val_roc_auc')
            best_val_rating_mse = classifier_stats.get('val_rating_mse')
            best_val_rating_corr = classifier_stats.get('val_rating_corr')

        # HPO objective: configurable via classifier.hpo_metric (default val_roc_auc)
        hpo_metric = config.get('classifier', {}).get('hpo_metric', 'val_roc_auc')
        if hpo_metric == 'val_loss':
            # Minimize validation BCE loss (binary classifier)
            primary_metric = best_val_loss
            primary_metric_name = 'val_loss'
            minimize = True
            best_display = best_val_loss
        elif hpo_metric == 'val_rating_mse' and best_val_rating_mse is not None:
            primary_metric = best_val_rating_mse
            primary_metric_name = 'val_rating_mse'
            minimize = True
            best_display = best_val_rating_mse
        elif hpo_metric == 'val_rating_corr' and best_val_rating_corr is not None:
            # Maximize correlation: Optuna minimizes, so primary_metric = 1 - corr
            primary_metric = 1.0 - best_val_rating_corr
            primary_metric_name = 'optuna_objective'  # Log minimized value; actual metric is val_rating_corr
            minimize = True
            best_display = best_val_rating_corr
        elif hpo_metric == 'val_rating_corr_and_f1':
            # Linear combination: maximize (w_corr * norm_corr + w_f1 * f1), minimize 1 - that
            clf_cfg = config.get('classifier', {})
            w_corr = clf_cfg.get('hpo_metric_corr_weight', 0.5)
            w_f1 = clf_cfg.get('hpo_metric_f1_weight', 0.5)
            total_w = w_corr + w_f1
            if total_w > 0:
                w_corr, w_f1 = w_corr / total_w, w_f1 / total_w
            norm_corr = (best_val_rating_corr + 1.0) / 2.0 if best_val_rating_corr is not None else None  # [-1,1] -> [0,1]
            combined = 0.0
            if norm_corr is not None and best_val_f1 is not None:
                combined = w_corr * norm_corr + w_f1 * best_val_f1
                primary_metric = 1.0 - combined
                primary_metric_name = 'optuna_objective'
                minimize = True
                best_display = combined
            elif norm_corr is not None:
                primary_metric = 1.0 - norm_corr
                primary_metric_name = 'optuna_objective'
                minimize = True
                best_display = norm_corr
            elif best_val_f1 is not None:
                primary_metric = 1.0 - best_val_f1
                primary_metric_name = 'optuna_objective'
                minimize = True
                best_display = best_val_f1
            else:
                primary_metric = best_val_mae
                primary_metric_name = 'val_mae'
                minimize = True
                best_display = best_val_mae
        elif hpo_metric == 'val_roc_auc' and best_val_roc_auc is not None:
            # Maximize ROC AUC: Optuna minimizes, so primary_metric = 1 - ROC AUC. Log as optuna_objective.
            primary_metric = 1.0 - best_val_roc_auc
            primary_metric_name = 'optuna_objective'
            minimize = True
            best_display = best_val_roc_auc
        elif hpo_metric == 'val_f1' and best_val_f1 is not None:
            # Maximize F1: Optuna minimizes, so primary_metric = 1 - F1. Log as optuna_objective
            # so MLflow doesn't show "val_f1" = (1 - F1); actual F1 is in val_f1.
            primary_metric = 1.0 - best_val_f1
            primary_metric_name = 'optuna_objective'
            minimize = True
            best_display = best_val_f1
        else:
            primary_metric = best_val_mae
            primary_metric_name = 'val_mae'
            minimize = True
            best_display = best_val_mae

        # Track and report new best trials (compare using same metric we optimize)
        global _hpo_classifier_best_value, _hpo_classifier_best_trial
        if primary_metric < _hpo_classifier_best_value:
            _hpo_classifier_best_value = primary_metric
            _hpo_classifier_best_trial = ctx.trial_number + 1 if ctx.trial_number is not None else 0
            best_seed_for_trial = _trial_best_reps_seed.get(ctx.trial_number) if hpo_runs > 1 and ctx.trial_number is not None else None
            seed_info = f", init_seed={best_seed_for_trial}" if best_seed_for_trial is not None else ""
            display_name = hpo_metric if primary_metric_name == 'optuna_objective' else primary_metric_name
            print(f"  ★ NEW BEST (Trial {_hpo_classifier_best_trial}): {display_name}={best_display:.6f}{seed_info}")
            print(f"    Parameters: lr={ctx.hyperparameters.get('learning_rate', 'N/A'):.2e}, "
                  f"dropout={ctx.hyperparameters.get('dropout', 'N/A')}, "
                  f"wd={ctx.hyperparameters.get('adam_weight_decay', 'N/A'):.2e}")

        # Log key fixed config and seed so runs are comparable in MLflow
        cc = config.get('classifier', {})
        seed_val = classifier_stats.get('training_seed')
        ctx.tracker.log_params({
            'training_label_noise': cc.get('training_label_noise', 0),
            'fingerprint_noise_variance': cc.get('fingerprint_noise_variance', 0),
            'hpo_metric': cc.get('hpo_metric', 'val_roc_auc'),
            'training_seed': str(seed_val) if seed_val is not None else '',
        })
        # Log to MLflow (tuner already logged sampled hyperparameters) — final metrics at saved checkpoint only
        ctx.tracker.log_metric('val_loss', best_val_loss)
        ctx.tracker.log_metric('val_mae', best_val_mae)
        if best_val_f1 is not None:
            ctx.tracker.log_metric('val_f1', best_val_f1)
        if best_val_rating_mse is not None:
            ctx.tracker.log_metric('val_rating_mse', best_val_rating_mse)
        if best_val_rating_corr is not None:
            ctx.tracker.log_metric('val_rating_corr', best_val_rating_corr)
        if primary_metric_name == 'val_rating_corr_and_f1':
            ctx.tracker.log_metric('val_rating_corr_and_f1', best_display)
        ctx.tracker.log_metric('epochs_run', classifier_stats.get('epochs_run', 0))
        ctx.tracker.log_metric('training_time', classifier_stats.get('training_time_seconds', 0))

        # Only include numeric metrics (MLflow rejects None)
        metrics_dict = {
            'final_train_loss': classifier_stats.get('final_train_loss', 0),
            'final_val_loss': classifier_stats.get('final_val_loss', 0),
            'val_loss': best_val_loss,
            'val_mae': best_val_mae,
            'best_epoch': classifier_stats.get('best_epoch', 0),
        }
        best_val_accuracy = classifier_stats.get('val_accuracy')
        if best_val_accuracy is not None:
            ctx.tracker.log_metric('val_accuracy', best_val_accuracy)
            metrics_dict['val_accuracy'] = best_val_accuracy
        best_val_precision = classifier_stats.get('val_precision')
        if best_val_precision is not None:
            ctx.tracker.log_metric('val_precision', best_val_precision)
            ctx.tracker.log_metric('val_ppv', best_val_precision)
            metrics_dict['val_precision'] = best_val_precision
            metrics_dict['val_ppv'] = best_val_precision
        best_val_recall = classifier_stats.get('val_recall')
        if best_val_recall is not None:
            ctx.tracker.log_metric('val_recall', best_val_recall)
            metrics_dict['val_recall'] = best_val_recall
        roc_auc = classifier_stats.get('val_roc_auc')  # ROC AUC at saved checkpoint
        if roc_auc is not None:
            ctx.tracker.log_metric('roc_auc', roc_auc)
            metrics_dict['roc_auc'] = roc_auc
        if best_val_f1 is not None:
            metrics_dict['val_f1'] = best_val_f1
        if best_val_rating_mse is not None:
            metrics_dict['val_rating_mse'] = best_val_rating_mse
        if best_val_rating_corr is not None:
            metrics_dict['val_rating_corr'] = best_val_rating_corr

        return TrainingResult(
            primary_metric=primary_metric,
            primary_metric_name=primary_metric_name,
            minimize=minimize,
            metrics=metrics_dict,
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
        dict with keys: val_loss, best_epoch, history, checkpoint_path, training_time
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
        'val_loss': best_val_loss,
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
    training_seed: int = None,
    mlflow_tracker=None,
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
        mlflow_tracker: Optional MLflow tracker from HPO context (e.g. nested seed run).
                       When set, logging uses this run; no new run is started.

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

    # Initialize MLflow tracking (use provided tracker when in HPO nested seed run)
    mlflow_config = config.get('mlflow', {})
    mlflow_enabled = mlflow_config.get('auto_start', True)

    if mlflow_tracker is not None:
        tracker = mlflow_tracker
    elif False and mlflow_enabled:  # TODO DEBUG - disabled for now
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

    # Fingerprint-baseline: extraction only (no training). Saves chromaprint-derived "embeddings" for classifier ablation.
    if encoder_type == "fingerprint_baseline":
        if verbose:
            print("\n[3/7] Fingerprint-baseline: extraction only (no training)...")
        encoder = create_encoder(config)
        loss_fn = create_loss_fn(config)
        bl_config = encoder_config.get('fingerprint_baseline', {})
        model_version = model_version_override or bl_config.get('encoder_version', 'fingerprint_baseline')
        embedding_store = EmbeddingStore(music_config['embedding_db_path'])
        extraction_dataset = create_dataset(
            config=config,
            songs=all_songs,
            album_to_idx=album_to_idx,
            filename_to_albums=filename_to_albums,
            is_training=False,
        )
        from ml_skeleton.music.fingerprint_encoder import collate_fingerprint_baseline
        num_workers = music_config.get('dataloader_workers', 4)
        all_loader = DataLoader(
            extraction_dataset,
            batch_size=encoder_config.get('batch_size', 64),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fingerprint_baseline,
        )
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
        trainer = EncoderTrainer(
            encoder=encoder,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            embedding_store=embedding_store,
            model_version=model_version,
            tracker=None,
            scheduler=None,
        )
        if verbose:
            print("  Extracting chromaprint-derived embeddings...")
        trainer.extract_embeddings(all_loader, save_to_store=True, use_moco=False)
        if verbose:
            print(f"  Saved to {music_config['embedding_db_path']} (model_version={model_version})")
        return model_card

    # MoCo uses MoCoDataset (created by factory)
    if use_moco:
        from ml_skeleton.music.moco_dataset import MoCoDataset
        if trial_info is not None:
            print("[HPO] Creating MoCo dataset...", flush=True)
        full_dataset = create_dataset(
            config=config,
            songs=all_songs,
            album_to_idx=album_to_idx,
            filename_to_albums=filename_to_albums,
            is_training=True
        )
        if trial_info is not None:
            print("[HPO] MoCo dataset created.", flush=True)
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
                'val_loss': result['val_loss'],
                'best_epoch': result['best_epoch'],
                'epochs_run': result['epochs_run'],
                'checkpoint_path': result['checkpoint_path'],
                'training_time': result['training_time']
            })

            print(f"\n  Run {run_idx + 1} complete: val_loss={result['val_loss']:.6f} (epoch {result['best_epoch']})")

        # Report multi-run statistics
        losses = [r['val_loss'] for r in run_results]
        total_time = sum(r['training_time'] for r in run_results)

        print(f"\n{'='*60}")
        print("MULTI-RUN STATISTICS")
        print(f"{'='*60}")
        for r in run_results:
            print(f"  Run {r['run']}: val_loss={r['val_loss']:.6f} (epoch {r['best_epoch']}, seed={r['seed']})")
        print(f"\n  Mean: {np.mean(losses):.6f} +/- {np.std(losses):.6f}")
        print(f"  Min:  {np.min(losses):.6f}")
        print(f"  Max:  {np.max(losses):.6f}")

        # Identify and copy best model
        best_run = min(run_results, key=lambda x: x['val_loss'])
        print(f"\n  Best: Run {best_run['run']} (val_loss={best_run['val_loss']:.6f}, seed={best_run['seed']})")

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
    # Avoid fork-related malloc crashes in Docker/HPO: set EXPLR_HPO_DATALOADER_WORKERS=0
    if trial_info is not None:
        _hpo_workers = os.environ.get("EXPLR_HPO_DATALOADER_WORKERS")
        if _hpo_workers is not None:
            num_workers = int(_hpo_workers)
            if verbose:
                print(f"  HPO: dataloader_workers overridden to {num_workers} (EXPLR_HPO_DATALOADER_WORKERS)")

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

    if trial_info is not None:
        print("[HPO] Creating data loaders...", flush=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    if trial_info is not None:
        print("[HPO] Data loaders created.", flush=True)

    # Multi-task is only for simple encoder (not used with MoCo)
    use_multi_task = False

    # Create model using factory
    if verbose:
        print("\n[4/7] Creating encoder model...")
    if trial_info is not None:
        print("[HPO] Creating encoder model...", flush=True)

    # Create MoCo encoder using factory
    encoder = create_encoder(config)
    if trial_info is not None:
        print("[HPO] Encoder model created.", flush=True)
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
    if trial_info is not None:
        print("[HPO] Starting training loop...", flush=True)
    training_start_time = time.time()

    tracker_already_active = mlflow_tracker is not None

    def _encoder_train_and_log():
        nonlocal history, training_time
        if tracker:
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
                'training_seed': str(model_seed),
            })
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
        if tracker:
            best_val_loss_mlflow = min(history['val_loss']) if history['val_loss'] else float('inf')
            final_train_loss_mlflow = history['train_loss'][-1] if history['train_loss'] else float('inf')
            tracker.log_metric('val_loss', best_val_loss_mlflow)
            tracker.log_metric('final_train_loss', final_train_loss_mlflow)
            tracker.log_metric('training_time_seconds', training_time)
            tracker.log_metric('epochs_completed', len(history['train_loss']))
            checkpoint_path = Path(config['checkpoint_dir']) / 'encoder_best.pt'
            if checkpoint_path.exists():
                tracker.log_artifact(str(checkpoint_path))

    if tracker:
        if tracker_already_active:
            _encoder_train_and_log()
        else:
            with tracker:
                _encoder_train_and_log()
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
    encoder_stats['training_seed'] = model_seed
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

        # Extract all chunks per song into embedding_chunks (for classifier average)
        from ml_skeleton.music.moco_dataset import ChunkExtractionDataset
        chunk_cache_config = music_config.get('chunk_cache', {})
        num_chunks = chunk_cache_config.get('num_chunks', 8)
        crop_duration = encoder_config.get('augmentation', {}).get('crop_duration_max', 15.0)
        chunk_extraction_dataset = ChunkExtractionDataset(
            songs=all_songs,
            cache_dir=get_chunk_cache_dir(config),
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

    if isinstance(dataset, NoisyEmbeddingWrapper):
        dataset = dataset.dataset
    if isinstance(dataset, Subset):
        # For Subset, access underlying dataset via indices
        underlying = dataset.dataset
        return [underlying.data[i]["rating"] for i in dataset.indices]
    else:
        # For EmbeddingDataset, use the method directly
        return dataset.get_all_ratings()


def _effective_classifier_batch_size(classifier_config: dict, encoder_version: str) -> int:
    """Use fingerprint_baseline_batch_size when training on chromaprint embeddings only."""
    if encoder_version == "fingerprint_baseline" and classifier_config.get("fingerprint_baseline_batch_size") is not None:
        return classifier_config["fingerprint_baseline_batch_size"]
    return classifier_config["batch_size"]


class NoisyEmbeddingWrapper(Dataset):
    """Wraps a dataset and adds Gaussian noise N(0, variance) to the 'embedding' field in __getitem__.

    Used so that noise is applied only to training data, not validation.
    """

    def __init__(self, dataset: Dataset, variance: float = 0.1, seed: Optional[int] = None):
        self.dataset = dataset
        self.variance = variance
        self.seed = seed
        self._std = np.sqrt(variance)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        out = {k: (v.clone() if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in item.items()}
        emb = out["embedding"]
        if self.seed is not None:
            np.random.seed(self.seed + idx)
        noise = np.random.normal(0, self._std, size=emb.shape).astype(np.float32)
        out["embedding"] = emb + torch.from_numpy(noise)
        return out


def _zscore_normalize_embeddings(embeddings_dict: dict) -> dict:
    """Z-score normalize embeddings per dimension (mean=0, std=1) to improve classifier training.

    Fingerprint (chromaprint) inputs are 0/1 binary with possibly low variance per dimension;
    normalization gives the classifier inputs with consistent scale and can help avoid collapse.
    Handles both (D,) and (num_chunks, D) shapes per key.
    """
    if not embeddings_dict:
        return embeddings_dict
    arrs = []
    for v in embeddings_dict.values():
        a = np.asarray(v, dtype=np.float32)
        if a.ndim == 1:
            arrs.append(a)
        else:
            arrs.append(a.reshape(-1, a.shape[-1]))
    stack = np.vstack(arrs)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)
    std[std == 0] = 1.0
    out = {}
    for k, v in embeddings_dict.items():
        a = np.asarray(v, dtype=np.float32)
        out[k] = ((a - mean) / std).astype(np.float32)
    return out


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
    init_from_prod: bool = True,
    use_genre: bool = False,
    genre_centroids: Optional[np.ndarray] = None,
) -> dict:
    """Perform a single classifier training run with a specific seed.

    Args:
        class_weight_strategy: Strategy for class weighting to handle imbalance.
        classification_mode: "regression" (MSE loss) or "binary" (BCE loss).
            - "none": No weighting (standard MSE)
            - "inverse": Weight = N / (n_classes * count_i)
            - "sqrt_inverse": Weight = sqrt(N / (n_classes * count_i))
        init_from_prod: If True, initialize from prod/classifier_best.pt if architecture matches.
        use_genre: If True, classifier receives 7-dim genre multi-hot (real or centroid-imputed).
        genre_centroids: (7, D) centroids for imputing missing genre; saved in checkpoint.

    Returns:
        dict with keys: val_loss, val_mae, best_epoch, history, checkpoint_path, training_time
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

    # Fingerprint baseline: add Gaussian noise to both train and val (val sees same regime as train)
    noise_variance = classifier_config.get("fingerprint_noise_variance", 0.0)
    if encoder_version == "fingerprint_baseline" and noise_variance > 0:
        train_dataset = NoisyEmbeddingWrapper(
            train_dataset, variance=noise_variance, seed=run_seed
        )
        val_dataset = NoisyEmbeddingWrapper(
            val_dataset, variance=noise_variance, seed=run_seed + 1000000
        )
        if verbose:
            print(f"  Fingerprint: Gaussian noise (variance={noise_variance}) applied to train and val")

    if verbose:
        print(f"  Train: {len(train_dataset)} songs")
        print(f"  Val: {len(val_dataset)} songs")

    # Compute class weights from training data if requested
    class_weights = None
    if class_weight_strategy != "none":
        # Extract ratings from training dataset (unwrap if NoisyEmbeddingWrapper)
        train_subset = train_dataset.dataset if isinstance(train_dataset, NoisyEmbeddingWrapper) else train_dataset
        train_ratings = []
        for idx in train_subset.indices:
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

    batch_size = _effective_classifier_batch_size(classifier_config, encoder_version)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
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
        dropout=classifier_config['dropout'],
        use_genre=use_genre,
        use_batch_norm=classifier_config.get('use_batch_norm', False),
        use_residual=classifier_config.get('use_residual', False),
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
        train_labels = _get_ratings_from_dataset(train_dataset)
        binary_pos_weight = classifier_config.get('binary_pos_weight')
        if binary_pos_weight is not None and isinstance(binary_pos_weight, (int, float)):
            pos_weight = float(binary_pos_weight)
            pos_weight_note = " (explicit binary_pos_weight)"
        else:
            use_pos_weight = classifier_config.get('binary_use_pos_weight', True)
            pos_weight = BinaryRatingLoss.compute_pos_weight(train_labels) if use_pos_weight else 1.0
            pos_weight_note = (" (no upweight)" if not use_pos_weight else "")
        middle_weight = classifier_config.get('binary_middle_loss_weight', 0.1)
        loss_fn = BinaryRatingLoss(pos_weight=pos_weight, middle_weight=middle_weight)
        if verbose:
            print(f"  Binary classification mode - pos_weight: {pos_weight:.3f}, middle_weight: {middle_weight}{pos_weight_note}")
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
        classification_mode=classification_mode,
        genre_centroids=genre_centroids,
        chunk_aggregation=classifier_config.get('chunk_aggregation', 'mean'),
        clip_grad=classifier_config.get('clip_grad', False),
        clip_grad_norm=classifier_config.get('clip_grad_norm', 1.0),
        training_label_noise=classifier_config.get('training_label_noise', 0),
        hpo_mlflow_run_id=classifier_config.get('hpo_mlflow_run_id'),
        hpo_mlflow_run_name=classifier_config.get('hpo_mlflow_run_name'),
    )

    # Train (use HPO metric for early stop/checkpoint when set)
    hpo_metric = classifier_config.get('hpo_metric', 'val_roc_auc')
    train_kwargs = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'num_epochs': num_epochs,
        'checkpoint_dir': str(checkpoint_dir),
        'save_best_only': True,
        'early_stopping_patience': classifier_config.get('early_stopping_patience'),
        'early_stopping_min_delta': classifier_config.get('early_stopping_min_delta', 0.0),
    }
    if hpo_metric == 'val_rating_corr_and_f1':
        train_kwargs['monitor_metric'] = 'val_rating_corr_and_f1'
        train_kwargs['hpo_metric_corr_weight'] = classifier_config.get('hpo_metric_corr_weight', 0.5)
        train_kwargs['hpo_metric_f1_weight'] = classifier_config.get('hpo_metric_f1_weight', 0.5)
    elif hpo_metric == 'val_f1':
        train_kwargs['monitor_metric'] = 'val_f1'
    elif hpo_metric == 'val_roc_auc':
        train_kwargs['monitor_metric'] = 'val_roc_auc'
    training_start_time = time.time()
    history = trainer.train(**train_kwargs)
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
        'val_loss': best_val_loss,
        'val_mae': best_val_mae,
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
    vault_size: int = 1000,
    mlflow_tracker=None,
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
        vault_size: Number of ratings to reserve for A/B testing vault (default: 1000).
                   Vault files are never used for training, only for comparing models.
        mlflow_tracker: Optional MLflow tracker from HPO context. When set, logging uses
                       this run (caller already started it); no new run is started.

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

    classifier_mlflow_run_id = None  # Set when single-run uses tracker; used to log A/B test metrics

    # Create or verify model card generator
    if model_card is None:
        print("  WARNING: No model card from encoder stage. Creating new one.")
        model_card = ModelCardGenerator()
        model_card.set_config(config)

    # Load configuration
    music_config = config['music']
    classifier_config = config['classifier']
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Select appropriate epoch count (fingerprint_baseline can use longer training)
    encoder_version_for_epochs = music_config.get('encoder_version', 'v1')
    if encoder_version_for_epochs == 'fingerprint_baseline' and (
        classifier_config.get('fingerprint_baseline_epochs') is not None
        or classifier_config.get('fingerprint_baseline_final_training_epochs') is not None
    ):
        if final_training and classifier_config.get('fingerprint_baseline_final_training_epochs') is not None:
            num_epochs = classifier_config['fingerprint_baseline_final_training_epochs']
            if verbose:
                print(f"  Using fingerprint_baseline_final_training_epochs={num_epochs}")
        else:
            num_epochs = classifier_config.get('fingerprint_baseline_epochs', classifier_config['epochs'])
            if verbose:
                print(f"  Using fingerprint_baseline_epochs={num_epochs} (fingerprint_baseline)")
    elif final_training and 'final_training_epochs' in classifier_config:
        num_epochs = classifier_config['final_training_epochs']
        if verbose:
            print(f"  Using final_training_epochs={num_epochs} for training with best hyperparameters")
    else:
        num_epochs = classifier_config['epochs']
        if verbose:
            print(f"  Using epochs={num_epochs} (HPO/regular training mode)")

    # Initialize MLflow tracking (use provided tracker when in HPO to avoid starting a second run)
    mlflow_config = config.get('mlflow', {})
    mlflow_enabled = mlflow_config.get('auto_start', True)

    if mlflow_tracker is not None:
        tracker = mlflow_tracker
    elif mlflow_enabled:
        # Start MLflow server if configured
        tracking_uri = mlflow_config.get('tracking_uri', 'http://localhost:5000')
        port = int(tracking_uri.split(':')[-1].rstrip('/'))
        MLflowServer.ensure_running(
            port=port,
            backend_store_uri=mlflow_config.get('backend_store_uri', 'sqlite:///mlflow.db'),
            artifact_root=mlflow_config.get('artifact_location', './mlruns')
        )
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

    # Get embeddings for all songs (num_chunks per song, default 8, for classifier average)
    num_chunks = music_config.get('chunk_cache', {}).get('num_chunks', 8)
    filenames = [s.filename for s in all_songs]
    embeddings_dict = embedding_store.get_embeddings_batch_all_chunks(
        filenames,
        model_version=music_config['encoder_version'],
        num_chunks=num_chunks
    )
    if not embeddings_dict:
        embeddings_dict = embedding_store.get_embeddings_batch(
            filenames,
            model_version=music_config['encoder_version']
        )

    if verbose:
        print(f"  Loaded {len(embeddings_dict)} embeddings")

    # For fingerprint baseline: z-score normalize (noise is added only to training data in the train step)
    encoder_version_for_norm = music_config.get("encoder_version", "v1")
    if encoder_version_for_norm == "fingerprint_baseline" and classifier_config.get("normalize_fingerprint_embeddings", True):
        embeddings_dict = _zscore_normalize_embeddings(embeddings_dict)
        if verbose:
            print("  Fingerprint embeddings: z-score normalized (per dimension)")

    # Check embedding dimension (support (4, D) per song)
    first_embedding = next(iter(embeddings_dict.values()))
    arr = np.asarray(first_embedding)
    embedding_dim = int(arr.shape[-1]) if arr.ndim > 1 else len(arr)
    if verbose:
        print(f"  Embedding dimension: {embedding_dim} (chunks per song: {num_chunks})")

    # Create dataset
    if verbose:
        print("\n[3/6] Creating dataset...")

    # Get classification mode and genre options from config
    classification_mode = classifier_config.get('classification_mode', 'regression')
    binary_positive_threshold = classifier_config.get('binary_positive_threshold', 4.0)
    binary_negative_threshold = classifier_config.get('binary_negative_threshold', 2.0)
    use_genre = classifier_config.get('use_genre', False)
    genre_impute_top_k = classifier_config.get('genre_impute_top_k', 2)
    genre_impute_min_votes = classifier_config.get('genre_impute_min_votes', 1)

    # When use_genre: compute 7 category centroids from songs with genre (for imputing missing)
    genre_centroids = None
    if use_genre:
        from ml_skeleton.music.genre_centroids import compute_genre_centroids
        genre_centroids = compute_genre_centroids(
            embeddings_dict,
            all_songs,
            encoder_version=music_config.get('encoder_version', 'v1'),
        )
        if verbose:
            print(f"  Genre centroids: computed from songs with metadata (shape {genre_centroids.shape})")

    binary_include_middle = classifier_config.get('binary_include_middle', False)
    replace_embeddings_with_noise = classifier_config.get('replace_embeddings_with_noise', False)
    full_dataset = EmbeddingDataset(
        embeddings=embeddings_dict,
        songs=all_songs,
        only_rated=True,
        classification_mode=classification_mode,
        binary_positive_threshold=binary_positive_threshold,
        binary_negative_threshold=binary_negative_threshold,
        use_genre=use_genre,
        genre_centroids=genre_centroids,
        genre_impute_top_k=genre_impute_top_k,
        genre_impute_min_votes=genre_impute_min_votes,
        binary_include_middle=binary_include_middle,
        replace_embeddings_with_noise=replace_embeddings_with_noise,
        noise_seed=config.get('seed', 42),
    )

    # Get version information (needed for multi-run and checkpoint)
    # When using fingerprint_baseline there is no encoder checkpoint; always use config so HPO works.
    encoder_checkpoint_path = Path(config['checkpoint_dir']) / "encoder_best.pt"
    if music_config.get('encoder_version') == 'fingerprint_baseline':
        encoder_version = 'fingerprint_baseline'
    elif encoder_checkpoint_path.exists():
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
                init_from_prod=init_from_prod,
                use_genre=use_genre,
                genre_centroids=genre_centroids,
            )

            run_results.append({
                'run': run_idx + 1,
                'seed': run_seed,
                'val_loss': result['val_loss'],
                'val_mae': result['val_mae'],
                'best_epoch': result['best_epoch'],
                'epochs_run': result['epochs_run'],
                'checkpoint_path': result['checkpoint_path'],
                'training_time': result['training_time']
            })

            print(f"\n  Run {run_idx + 1} complete: val_mae={result['val_mae']:.6f} (epoch {result['best_epoch']})")

        # Report multi-run statistics
        maes = [r['val_mae'] for r in run_results]
        losses = [r['val_loss'] for r in run_results]
        total_time = sum(r['training_time'] for r in run_results)

        print(f"\n{'='*60}")
        print("MULTI-RUN STATISTICS")
        print(f"{'='*60}")
        for r in run_results:
            print(f"  Run {r['run']}: val_mae={r['val_mae']:.6f}, val_loss={r['val_loss']:.6f} (epoch {r['best_epoch']}, seed={r['seed']})")
        print(f"\n  MAE  - Mean: {np.mean(maes):.6f} +/- {np.std(maes):.6f}, Min: {np.min(maes):.6f}, Max: {np.max(maes):.6f}")
        print(f"  Loss - Mean: {np.mean(losses):.6f} +/- {np.std(losses):.6f}")

        # Identify and copy best model (by MAE)
        best_run = min(run_results, key=lambda x: x['val_mae'])
        print(f"\n  Best: Run {best_run['run']} (val_mae={best_run['val_mae']:.6f}, seed={best_run['seed']})")

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
        train_pos = sum(1 for f in train_files if file_ratings.get(f, 0) == 1)
        val_pos = sum(1 for f in val_files if file_ratings.get(f, 0) == 1)
        train_prev = train_pos / len(train_files) if train_files else 0.0
        val_prev = val_pos / len(val_files) if val_files else 0.0
        classifier_stats = {
            'val_loss': best_run['val_loss'],
            'val_mae': best_run['val_mae'],
            'best_epoch': best_run['best_epoch'],
            'epochs_run': best_run['epochs_run'],
            'training_time_seconds': total_time,
            'num_runs': num_runs,
            'mae_mean': np.mean(maes),
            'mae_std': np.std(maes),
            'train_size': len(train_files),
            'val_size': len(val_files),
            'vault_size': len(vault_files),
            'train_prevalence': train_prev,
            'val_prevalence': val_prev,
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

            # Create test dataset from current vault (fingerprint_baseline: add noise to avoid collapse)
            test_dataset = full_dataset.subset_by_filenames(current_vault_set)
            if encoder_version == "fingerprint_baseline":
                noise_variance = classifier_config.get("fingerprint_noise_variance", 0.0)
                if noise_variance > 0:
                    test_dataset = NoisyEmbeddingWrapper(test_dataset, variance=noise_variance, seed=99999)

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
                if classifier_mlflow_run_id:
                    from mlflow.tracking import MlflowClient
                    tracking_uri = config.get('mlflow', {}).get('tracking_uri', 'http://localhost:5000')
                    client = MlflowClient(tracking_uri=tracking_uri)
                    for k, v in ab_result_to_mlflow_metrics(ab_result).items():
                        client.log_metric(classifier_mlflow_run_id, k, v)
            else:
                print(f"  Skipping A/B test: only {len(test_dataset)} samples in vault (need >= 10)")
                print(f"  Rate more songs to build up a stable A/B test vault")
        elif verbose and len(vault_files) == 0:
            print(f"\n  No vault files available for A/B testing")
            print(f"  Need at least {vault_size} rated files to create vault")
        elif verbose:
            print(f"\n  No production model found at {prod_classifier_path}")
            print(f"  Run 'promote-to-prod' after initial training to enable A/B testing")

        # Final training: report HPO vs final val ROC AUC and log to MLflow
        if verbose and final_training:
            hpo_auc = get_hpo_val_roc_auc(config)
            final_auc = model_card.classifier_stats.get("val_roc_auc") if model_card and model_card.classifier_stats else None
            _report_and_log_hpo_vs_final_roc_auc(
                config=config,
                hpo_val_roc_auc=hpo_auc,
                final_val_roc_auc=final_auc,
                classifier_mlflow_run_id=classifier_mlflow_run_id,
            )

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

    # Fingerprint baseline: add Gaussian noise to both train and val (val sees same regime as train)
    noise_variance = classifier_config.get("fingerprint_noise_variance", 0.0)
    if encoder_version == "fingerprint_baseline" and noise_variance > 0:
        train_dataset = NoisyEmbeddingWrapper(
            train_dataset, variance=noise_variance, seed=model_seed
        )
        val_dataset = NoisyEmbeddingWrapper(
            val_dataset, variance=noise_variance, seed=model_seed + 1000000
        )
        if verbose:
            print(f"  Fingerprint: Gaussian noise (variance={noise_variance}) applied to train and val")

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

    batch_size = _effective_classifier_batch_size(classifier_config, encoder_version)
    # Generator for reproducible shuffle (same model_seed => same batch order as HPO best run)
    train_shuffle_generator = torch.Generator().manual_seed(model_seed)

    # Create data loaders (embeddings are cheap to load, so smaller prefetch)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_shuffle_generator,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,  # Embeddings load fast, so 2 batches is enough
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
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
        dropout=classifier_config['dropout'],
        use_genre=use_genre,
        use_batch_norm=classifier_config.get('use_batch_norm', False),
        use_residual=classifier_config.get('use_residual', False),
    )

    if verbose:
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Use genre: {use_genre}")
        print(f"  Hidden dims: {classifier_config['hidden_dims']}")
        print(f"  Dropout: {classifier_config['dropout']}")

    # Create loss function based on classification mode
    if classification_mode == "binary":
        train_labels = _get_ratings_from_dataset(train_dataset)
        binary_pos_weight = classifier_config.get('binary_pos_weight')
        if binary_pos_weight is not None and isinstance(binary_pos_weight, (int, float)):
            pos_weight = float(binary_pos_weight)
            pos_weight_note = " (explicit binary_pos_weight)"
        else:
            use_pos_weight = classifier_config.get('binary_use_pos_weight', True)
            pos_weight = BinaryRatingLoss.compute_pos_weight(train_labels) if use_pos_weight else 1.0
            pos_weight_note = (" (no upweight)" if not use_pos_weight else "")
        middle_weight = classifier_config.get('binary_middle_loss_weight', 0.1)
        loss_fn = BinaryRatingLoss(pos_weight=pos_weight, middle_weight=middle_weight)
        if verbose:
            print(f"  Binary classification mode - pos_weight: {pos_weight:.3f}, middle_weight: {middle_weight}{pos_weight_note}")
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
        classification_mode=classification_mode,
        genre_centroids=genre_centroids,
        chunk_aggregation=classifier_config.get('chunk_aggregation', 'mean'),
        clip_grad=classifier_config.get('clip_grad', False),
        clip_grad_norm=classifier_config.get('clip_grad_norm', 1.0),
        training_label_noise=classifier_config.get('training_label_noise', 0),
        hpo_mlflow_run_id=classifier_config.get('hpo_mlflow_run_id'),
        hpo_mlflow_run_name=classifier_config.get('hpo_mlflow_run_name'),
    )

    # Train with time tracking and MLflow logging
    if verbose:
        print("\n[6/6] Training...")
    training_start_time = time.time()

    # When HPO optimizes a metric (e.g. val_roc_auc, val_f1), use it for early stopping and best checkpoint.
    hpo_metric = classifier_config.get('hpo_metric', 'val_roc_auc')
    train_kwargs = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'num_epochs': num_epochs,
        'checkpoint_dir': config['checkpoint_dir'],
        'save_best_only': True,
        'early_stopping_patience': classifier_config.get('early_stopping_patience'),
        'early_stopping_min_delta': classifier_config.get('early_stopping_min_delta', 0.0),
    }
    if hpo_metric == 'val_rating_corr_and_f1':
        train_kwargs['monitor_metric'] = 'val_rating_corr_and_f1'
        train_kwargs['hpo_metric_corr_weight'] = classifier_config.get('hpo_metric_corr_weight', 0.5)
        train_kwargs['hpo_metric_f1_weight'] = classifier_config.get('hpo_metric_f1_weight', 0.5)
    elif hpo_metric == 'val_f1':
        train_kwargs['monitor_metric'] = 'val_f1'
    elif hpo_metric == 'val_roc_auc':
        train_kwargs['monitor_metric'] = 'val_roc_auc'

    # When tracker was passed from HPO (mlflow_tracker), we're already inside that run; don't start another
    tracker_already_active = mlflow_tracker is not None
    if tracker:
        def _run_with_tracker():
            nonlocal classifier_mlflow_run_id
            # Log params only when we own the run (not HPO). In HPO the tuner already logged trial params,
            # and with multi-rep we'd re-log training_seed (42 then 1042) which MLflow forbids.
            if not tracker_already_active:
                params_to_log = {
                    'stage': 'classifier',
                    'final_training': final_training,
                    'hidden_dims': str(classifier_config.get('hidden_dims', [256, 128])),
                    'dropout': classifier_config.get('dropout', 0.3),
                    'batch_size': batch_size,
                    'learning_rate': classifier_config['learning_rate'],
                    'num_epochs': num_epochs,
                    'optimizer': classifier_config.get('optimizer', 'adam'),
                    'scheduler': classifier_config.get('scheduler', 'cosine'),
                    'loss_type': classifier_config.get('loss_type', 'mse'),
                    'chunk_aggregation': classifier_config.get('chunk_aggregation', 'mean'),
                    'training_label_noise': classifier_config.get('training_label_noise', 0),
                    'fingerprint_noise_variance': classifier_config.get('fingerprint_noise_variance', 0),
                    'training_seed': str(model_seed),
                }
                if classifier_config.get('hpo_mlflow_run_id'):
                    params_to_log['hpo_mlflow_run_id'] = classifier_config['hpo_mlflow_run_id']
                if classifier_config.get('hpo_mlflow_run_name'):
                    params_to_log['hpo_mlflow_run_name'] = classifier_config['hpo_mlflow_run_name']
                tracker.log_params(params_to_log)

            # Train
            history = trainer.train(**train_kwargs)
            training_time = time.time() - training_start_time

            # Log final metrics
            best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
            best_val_mae = min(history['val_mae']) if history['val_mae'] else float('inf')
            final_train_loss = history['train_loss'][-1] if history['train_loss'] else float('inf')
            best_val_accuracy = max(history['val_accuracy']) if history.get('val_accuracy') else None
            best_val_precision = max(history['val_precision']) if history.get('val_precision') else None
            best_val_recall = max(history['val_recall']) if history.get('val_recall') else None
            roc_auc_saved = history.get('roc_auc') or (max(history['val_roc_auc']) if history.get('val_roc_auc') else None)

            tracker.log_metric('val_loss', best_val_loss)
            tracker.log_metric('val_mae', best_val_mae)
            tracker.log_metric('final_train_loss', final_train_loss)
            tracker.log_metric('training_time_seconds', training_time)
            tracker.log_metric('epochs_completed', len(history['train_loss']))
            if best_val_accuracy is not None:
                tracker.log_metric('val_accuracy', best_val_accuracy)
            if best_val_precision is not None:
                tracker.log_metric('val_precision', best_val_precision)
                tracker.log_metric('val_ppv', best_val_precision)
            if best_val_recall is not None:
                tracker.log_metric('val_recall', best_val_recall)
            if roc_auc_saved is not None:
                tracker.log_metric('roc_auc', roc_auc_saved)

            # Log checkpoint as artifact
            checkpoint_path = Path(config['checkpoint_dir']) / 'classifier_best.pt'
            if checkpoint_path.exists():
                tracker.log_artifact(str(checkpoint_path))
            classifier_mlflow_run_id = tracker.run_id
            return history, training_time

        if tracker_already_active:
            history, training_time = _run_with_tracker()
        else:
            with tracker:
                history, training_time = _run_with_tracker()
    else:
        # Train without MLflow
        history = trainer.train(**train_kwargs)
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
        if classifier_mlflow_run_id:
            print(f"MLflow run ID: {classifier_mlflow_run_id}")
            print(f"MLflow run hash: {classifier_mlflow_run_id[:8]}")
    else:
        # Concise HPO trial summary
        print(f"  Epochs: {epochs_run} | Train: {final_train_loss:.4f} | Val: {best_val_loss:.4f} | MAE: {best_val_mae:.4f} | Time: {training_time:.1f}s")

    # Collect training statistics for model card
    classifier_stats = collect_training_stats(
        trainer=trainer,
        training_time_seconds=training_time,
        dataset_stats=classifier_dataset_stats
    )
    classifier_stats['val_mae'] = best_val_mae
    if history.get('val_accuracy'):
        classifier_stats['val_accuracy'] = max(history['val_accuracy'])
    else:
        classifier_stats['val_accuracy'] = None
    if history.get('val_precision'):
        classifier_stats['val_precision'] = max(history['val_precision'])
        classifier_stats['val_ppv'] = classifier_stats['val_precision']  # PPV = precision
    else:
        classifier_stats['val_precision'] = None
        classifier_stats['val_ppv'] = None
    if history.get('val_recall'):
        classifier_stats['val_recall'] = max(history['val_recall'])
    else:
        classifier_stats['val_recall'] = None
    if history.get('val_f1'):
        classifier_stats['val_f1'] = max(history['val_f1'])
        if config.get('classifier', {}).get('hpo_metric') == 'val_f1':
            classifier_stats['best_epoch'] = history['val_f1'].index(max(history['val_f1'])) + 1
    else:
        classifier_stats['val_f1'] = None
    if history.get('roc_auc') is not None:
        classifier_stats['val_roc_auc'] = history['roc_auc']  # ROC AUC of saved best model (early stopping)
    elif history.get('val_roc_auc'):
        classifier_stats['val_roc_auc'] = max(history['val_roc_auc'])
    else:
        classifier_stats['val_roc_auc'] = None
    if history.get('best_epoch') is not None:
        classifier_stats['best_epoch'] = history['best_epoch']  # Epoch when best checkpoint was saved
    elif history.get('val_roc_auc') and config.get('classifier', {}).get('hpo_metric') == 'val_roc_auc':
        classifier_stats['best_epoch'] = history['val_roc_auc'].index(max(history['val_roc_auc'])) + 1
    if history.get('val_rating_mse'):
        classifier_stats['val_rating_mse'] = min(history['val_rating_mse'])
    else:
        classifier_stats['val_rating_mse'] = None
    if history.get('val_rating_corr'):
        valid_corr = [x for x in history['val_rating_corr'] if not (isinstance(x, float) and np.isnan(x))]
        classifier_stats['val_rating_corr'] = max(valid_corr) if valid_corr else None
    else:
        classifier_stats['val_rating_corr'] = None
    # Add train/val/vault sizes and prevalence for model card
    classifier_stats['train_size'] = len(train_dataset)
    classifier_stats['val_size'] = len(val_dataset)
    classifier_stats['vault_size'] = len(vault_files)
    train_pos = sum(1 for f in train_files if file_ratings.get(f, 0) == 1)
    val_pos = sum(1 for f in val_files if file_ratings.get(f, 0) == 1)
    classifier_stats['train_prevalence'] = train_pos / len(train_files) if train_files else 0.0
    classifier_stats['val_prevalence'] = val_pos / len(val_files) if val_files else 0.0
    classifier_stats['training_seed'] = model_seed
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
    manifest.set_metadata('val_loss', best_val_loss)
    manifest.set_metadata('val_mae', best_val_mae)
    manifest.set_metadata('training_time_seconds', training_time)
    manifest.set_metadata('epochs_run', epochs_run)
    manifest.set_metadata('train_size', len(train_dataset))
    manifest.set_metadata('val_size', len(val_dataset))
    manifest.set_metadata('vault_size', len(vault_files))
    manifest.set_metadata('train_prevalence', classifier_stats['train_prevalence'])
    manifest.set_metadata('val_prevalence', classifier_stats['val_prevalence'])
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

        # Create test dataset from current vault (fingerprint_baseline: add noise to avoid collapse)
        test_dataset = full_dataset.subset_by_filenames(current_vault_set)
        if encoder_version == "fingerprint_baseline":
            noise_variance = classifier_config.get("fingerprint_noise_variance", 0.0)
            if noise_variance > 0:
                test_dataset = NoisyEmbeddingWrapper(test_dataset, variance=noise_variance, seed=99999)

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
            # Log A/B test metrics to MLflow (classifier run) if we have a run id
            if classifier_mlflow_run_id:
                from mlflow.tracking import MlflowClient
                tracking_uri = config.get('mlflow', {}).get('tracking_uri', 'http://localhost:5000')
                client = MlflowClient(tracking_uri=tracking_uri)
                for k, v in ab_result_to_mlflow_metrics(ab_result).items():
                    client.log_metric(classifier_mlflow_run_id, k, v)
        else:
            print(f"  Skipping A/B test: only {len(test_dataset)} samples in vault (need >= 10)")
            print(f"  Rate more songs to build up a stable A/B test vault")
    elif verbose and len(vault_files) == 0:
        print(f"\n  No vault files available for A/B testing")
        print(f"  Need at least {vault_size} rated files to create vault")
    elif verbose:
        print(f"\n  No production model found at {prod_classifier_path}")
        print(f"  Run 'promote-to-prod' after initial training to enable A/B testing")

    # Final training: report HPO vs final val ROC AUC and log to MLflow
    if verbose and final_training:
        hpo_auc = get_hpo_val_roc_auc(config)
        final_auc = classifier_stats.get("val_roc_auc")
        _report_and_log_hpo_vs_final_roc_auc(
            config=config,
            hpo_val_roc_auc=hpo_auc,
            final_val_roc_auc=final_auc,
            classifier_mlflow_run_id=classifier_mlflow_run_id,
        )

    return model_card


def train_joint_finetune(
    config: dict,
    verbose: bool = True,
):
    """Joint fine-tune: unfreeze encoder + classifier, train on audio → rating.

    Run after encoder and classifier training. Loads both checkpoints, builds
    audio dataset for train+val files from manifest, and trains with two learning
    rates (encoder lower, classifier higher). Saves updated encoder and classifier,
    then re-extracts embeddings for the fine-tuned encoder.

    Requires config with encoder_type 'moco' (create_encoder supports only MoCo).
    """
    import time

    cleanup_memory()

    music_config = config["music"]
    classifier_config = config.get("classifier", {})
    jf_config = config.get("joint_finetune", {})
    if not jf_config.get("enabled", False):
        if verbose:
            print("Joint fine-tune is disabled (joint_finetune.enabled: false). Skipping.")
        return None

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(config["checkpoint_dir"])
    encoder_path = checkpoint_dir / "encoder_best.pt"
    classifier_path = checkpoint_dir / "classifier_best.pt"

    if not encoder_path.exists() or not classifier_path.exists():
        if verbose:
            print("Joint fine-tune requires encoder_best.pt and classifier_best.pt. Run encoder then classifier first.")
        return None

    encoder_type = config.get("encoder", {}).get("encoder_type", "")
    if encoder_type != "moco":
        if verbose:
            print(f"Joint fine-tune currently only supports encoder_type 'moco' (got '{encoder_type}'). Skipping.")
        return None

    if verbose:
        print("=" * 80)
        print("JOINT FINE-TUNE (encoder + classifier on audio → rating)")
        print("=" * 80)

    # Load manifest for train/val split
    manifest_path = checkpoint_dir / "training_manifest.json"
    if not manifest_path.exists():
        if verbose:
            print("No training_manifest.json found. Run classifier stage first.")
        return None
    manifest = TrainingManifest.load_or_create(str(manifest_path))
    train_files = list(manifest.training_files)
    val_files = list(manifest.validation_files)
    vault_files = list(manifest.vault_files)
    if not train_files and not val_files:
        if verbose:
            print("Manifest has no training or validation files.")
        return None

    # DB and songs
    db = ClementineDB(music_config["database_path"])
    all_songs = db.get_all_songs()
    album_to_idx, filename_to_albums = build_album_mapping(all_songs)
    train_val_vault = set(train_files) | set(val_files) | set(vault_files)
    songs_for_audio = [s for s in all_songs if s.filename in train_val_vault and s.is_rated]
    if not songs_for_audio:
        if verbose:
            print("No rated songs in train/val/vault.")
        return None

    # MusicDataset for audio (single crop, no augmentation)
    sample_rate = music_config.get("sample_rate", 16000)
    duration = music_config.get("audio_duration", 60.0)
    crop_position = music_config.get("crop_position", "end")
    normalize = music_config.get("normalize", True)
    music_dataset = MusicDataset(
        songs=songs_for_audio,
        album_to_idx=album_to_idx,
        filename_to_albums=filename_to_albums,
        sample_rate=sample_rate,
        duration=duration,
        crop_position=crop_position,
        normalize=normalize,
        only_rated=True,
        use_augmentation=False,
    )
    train_indices = [i for i in range(len(music_dataset)) if music_dataset.songs[i].filename in set(train_files)]
    val_indices = [i for i in range(len(music_dataset)) if music_dataset.songs[i].filename in set(val_files)]
    if not train_indices:
        if verbose:
            print("No training samples in audio dataset.")
        return None
    train_ds = torch.utils.data.Subset(music_dataset, train_indices)
    val_ds = torch.utils.data.Subset(music_dataset, val_indices) if val_indices else None

    num_workers = music_config.get("dataloader_workers", 4)
    batch_size = jf_config.get("batch_size", classifier_config.get("batch_size", 32))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=music_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=music_collate_fn,
    ) if val_ds and len(val_ds) > 0 else None

    # Load encoder
    encoder = create_encoder(config)
    enc_ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(enc_ckpt["model_state_dict"])
    encoder_version = enc_ckpt.get("encoder_version", music_config.get("encoder_version", "v1"))

    # Load classifier
    clf_ckpt = torch.load(classifier_path, map_location="cpu", weights_only=False)
    state_dict = clf_ckpt["model_state_dict"]
    embedding_dim = int(clf_ckpt.get("embedding_dim", 2048))
    hidden_dims = clf_ckpt.get("hidden_dims", classifier_config.get("hidden_dims", [512, 256, 128]))
    dropout = classifier_config.get("dropout", 0.3)
    use_genre = clf_ckpt.get("use_genre", False)
    use_batch_norm = clf_ckpt.get("use_batch_norm", False)
    use_residual = clf_ckpt.get("use_residual", False)
    genre_centroids = clf_ckpt.get("genre_centroids")
    classifier_version = clf_ckpt.get("classifier_version", "v1")
    classification_mode = classifier_config.get("classification_mode", "regression")

    classifier = SimpleRatingClassifier(
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_genre=use_genre,
        use_batch_norm=use_batch_norm,
        use_residual=use_residual,
    )
    classifier.load_state_dict(state_dict)

    # Genre dict for train+val (from current embeddings + centroids)
    genre_dict = {}
    if use_genre and genre_centroids is not None:
        from ml_skeleton.music.genre_centroids import get_genre_features
        embedding_store = EmbeddingStore(music_config["embedding_db_path"])
        all_filenames = train_files + val_files
        embeddings_dict = embedding_store.get_embeddings_batch(
            all_filenames,
            model_version=encoder_version,
        )
        for song in songs_for_audio:
            if song.filename not in embeddings_dict:
                continue
            emb = embeddings_dict[song.filename]
            if isinstance(emb, np.ndarray):
                pass
            else:
                emb = np.asarray(emb)
            genre_dict[song.filename] = get_genre_features(
                song,
                emb,
                genre_centroids,
                top_k=classifier_config.get("genre_impute_top_k", 2),
                min_votes=classifier_config.get("genre_impute_min_votes", 1),
            )

    # Loss and optimizer (two param groups)
    if classification_mode == "binary":
        loss_fn = BinaryRatingLoss()
    else:
        loss_fn = RatingLoss()
    encoder_lr = jf_config.get("encoder_lr", 1e-5)
    classifier_lr = jf_config.get("classifier_lr", 1e-4)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": encoder_lr},
            {"params": classifier.parameters(), "lr": classifier_lr},
        ],
        weight_decay=classifier_config.get("adam_weight_decay", 0.0),
    )

    num_epochs = jf_config.get("epochs", 5)
    early_stopping_patience = jf_config.get("early_stopping_patience", 3)
    early_stopping_min_delta = jf_config.get("early_stopping_min_delta", 0.0)

    trainer = JointFinetuneTrainer(
        encoder=encoder,
        classifier=classifier,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        encoder_version=encoder_version,
        classifier_version=classifier_version,
        classification_mode=classification_mode,
        use_genre=use_genre,
        genre_dict=genre_dict,
        genre_centroids=genre_centroids,
        tracker=None,
    )
    start = time.time()
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        checkpoint_dir=str(checkpoint_dir),
        save_best_only=True,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        verbose=verbose,
    )
    elapsed = time.time() - start
    if verbose:
        print(f"\nJoint fine-tune completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    prod_classifier_path = Path("prod") / "classifier_best.pt"
    prod_encoder_path = Path("prod") / "encoder_best.pt"
    current_vault_set = set(vault_files)

    # Re-extract embeddings for fine-tuned encoder (train+val+vault)
    if verbose:
        print("Re-extracting embeddings for fine-tuned encoder...")
    encoder.eval()
    embedding_store = EmbeddingStore(music_config["embedding_db_path"])
    reembed_loader = DataLoader(
        music_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=music_collate_fn,
    )
    enc_trainer = EncoderTrainer(
        encoder=encoder,
        device=device,
        loss_fn=loss_fn,
        optimizer=torch.optim.Adam(encoder.parameters(), lr=1e-5),
        embedding_store=embedding_store,
        model_version=encoder_version,
    )
    enc_trainer.extract_embeddings(reembed_loader, save_to_store=True, use_moco=True)
    if verbose:
        print("Embeddings updated.")

    # A/B test: new model on NEW embeddings, prod on embeddings from PROD encoder (prod pipeline never changes)
    new_classifier_path = checkpoint_dir / "classifier_best.pt"
    if prod_classifier_path.exists() and prod_encoder_path.exists() and len(vault_files) > 0 and verbose:
        print("\n" + "=" * 60)
        print("A/B TEST: New Model (joint-finetuned) vs Production (using vault)")
        print("=" * 60)
        print(f"  Using current vault for A/B test: {len(current_vault_set)} files")
        print("  Prod: embeddings from prod encoder (prod pipeline unchanged). New: fine-tuned embeddings.")
        # New embeddings (from store after re-extract)
        new_embeddings_dict = enc_trainer.embedding_store.get_embeddings_batch(
            list(current_vault_set),
            model_version=encoder_version,
        )
        vault_songs = [s for s in all_songs if s.filename in current_vault_set and s.is_rated]
        vault_songs = sorted(vault_songs, key=lambda s: s.filename)
        vault_songs_new = [s for s in vault_songs if s.filename in new_embeddings_dict]
        if new_embeddings_dict and vault_songs_new:
            test_dataset_new = EmbeddingDataset(
                embeddings=new_embeddings_dict,
                songs=vault_songs_new,
                only_rated=True,
                classification_mode=classification_mode,
                binary_positive_threshold=classifier_config.get("binary_positive_threshold", 4.0),
                binary_negative_threshold=classifier_config.get("binary_negative_threshold", 2.0),
                use_genre=use_genre,
                genre_centroids=genre_centroids,
                genre_impute_top_k=classifier_config.get("genre_impute_top_k", 2),
                genre_impute_min_votes=classifier_config.get("genre_impute_min_votes", 1),
            )
            # Prod embeddings: generate from prod encoder on vault audio (keeps prod pipeline intact)
            prod_embeddings_dict = {}
            try:
                prod_encoder = create_encoder(config)
                prod_ckpt = torch.load(prod_encoder_path, map_location="cpu", weights_only=False)
                prod_encoder.load_state_dict(prod_ckpt["model_state_dict"])
                prod_encoder = prod_encoder.to(device)
                prod_encoder.eval()
                vault_indices = [i for i in range(len(music_dataset)) if music_dataset.songs[i].filename in current_vault_set]
                vault_indices = sorted(vault_indices, key=lambda i: music_dataset.songs[i].filename)
                vault_audio_subset = torch.utils.data.Subset(music_dataset, vault_indices)
                vault_loader = DataLoader(
                    vault_audio_subset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=music_collate_fn,
                )
                with torch.no_grad():
                    for batch in vault_loader:
                        audio_batch = batch["audio"].to(device)
                        out = prod_encoder(audio_batch)
                        emb = out["embedding"] if isinstance(out, dict) else out
                        emb_cpu = emb.cpu().numpy()
                        for i, fn in enumerate(batch["filename"]):
                            prod_embeddings_dict[fn] = emb_cpu[i]
            except Exception as e:
                if verbose:
                    print(f"  Could not generate prod vault embeddings: {e}")
            prod_test_dataset = None
            if prod_embeddings_dict:
                vault_songs_both = [s for s in vault_songs_new if s.filename in prod_embeddings_dict]
                if len(vault_songs_both) >= 10:
                    prod_test_dataset = EmbeddingDataset(
                        embeddings=prod_embeddings_dict,
                        songs=vault_songs_both,
                        only_rated=True,
                        classification_mode=classification_mode,
                        binary_positive_threshold=classifier_config.get("binary_positive_threshold", 4.0),
                        binary_negative_threshold=classifier_config.get("binary_negative_threshold", 2.0),
                        use_genre=use_genre,
                        genre_centroids=genre_centroids,
                        genre_impute_top_k=classifier_config.get("genre_impute_top_k", 2),
                        genre_impute_min_votes=classifier_config.get("genre_impute_min_votes", 1),
                    )
                    # New model dataset must match same songs (same order and length for A/B test)
                    test_dataset_new = EmbeddingDataset(
                        embeddings=new_embeddings_dict,
                        songs=vault_songs_both,
                        only_rated=True,
                        classification_mode=classification_mode,
                        binary_positive_threshold=classifier_config.get("binary_positive_threshold", 4.0),
                        binary_negative_threshold=classifier_config.get("binary_negative_threshold", 2.0),
                        use_genre=use_genre,
                        genre_centroids=genre_centroids,
                        genre_impute_top_k=classifier_config.get("genre_impute_top_k", 2),
                        genre_impute_min_votes=classifier_config.get("genre_impute_min_votes", 1),
                    )
            if prod_test_dataset is None:
                prod_test_dataset = test_dataset_new
            if len(test_dataset_new) >= 10:
                ab_result = run_ab_test(
                    new_classifier_path=str(new_classifier_path),
                    prod_classifier_path=str(prod_classifier_path),
                    test_dataset=test_dataset_new,
                    classification_mode=classification_mode,
                    device=device,
                    verbose=True,
                    prod_test_dataset=prod_test_dataset,
                )
                manifest.set_metadata("ab_test_result", {
                    "n_samples": int(ab_result.get("n_samples", 0)),
                    "new_accuracy": float(ab_result.get("new_accuracy", 0)),
                    "prod_accuracy": float(ab_result.get("prod_accuracy", 0)),
                    "improvement": float(ab_result.get("improvement", 0)),
                    "p_value": float(ab_result.get("p_value", 1.0)),
                    "significant": bool(ab_result.get("significant", False)),
                })
                manifest.save()
            else:
                print(f"  Skipping A/B test: only {len(test_dataset_new)} samples in vault (need >= 10)")
        else:
            print("  Skipping A/B test: could not load vault embeddings")
    elif verbose and len(vault_files) == 0:
        print("  No vault files available for A/B testing")
    elif verbose and (not prod_classifier_path.exists() or not prod_encoder_path.exists()):
        print("  No production model in prod/ (need encoder_best.pt and classifier_best.pt); run promote-to-prod to enable A/B testing")

    if verbose:
        print("\nNext: --stage recommend or promote-to-prod")
    return None


def generate_recommendations(
    config: dict,
    prod_dir: str = None,
    low_rating_ratio: float = 0.0,
    genre_filter: str = None,
    error_playlist_size: int = None
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
        error_playlist_size: Max songs per false_positives/false_negatives playlist.
                            None = use config; 0 = do not generate error playlists.

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
    classifier_config = config.get('classifier', {})
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
    num_chunks = music_config.get('chunk_cache', {}).get('num_chunks', 8)
    embeddings_dict = embedding_store.get_embeddings_batch_all_chunks(
        filenames,
        model_version=encoder_version,
        num_chunks=num_chunks
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

    checkpoint_path = classifier_checkpoint
    if not checkpoint_path.exists():
        print(f"  Classifier checkpoint not found: {checkpoint_path}")
        if prod_dir:
            print("  Run 'promote-to-prod' first to deploy models!")
        else:
            print("  Run classifier training first!")
        return

    # Load checkpoint once (used for dataset genre setup and for classifier weights)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    use_genre_recommend = checkpoint.get("use_genre", False)
    genre_centroids_recommend = checkpoint.get("genre_centroids")

    # Create dataset (with genre when classifier was trained with genre)
    dataset = EmbeddingDataset(
        embeddings=embeddings_dict,
        songs=unrated_songs,
        only_rated=False,
        use_genre=use_genre_recommend,
        genre_centroids=genre_centroids_recommend,
        genre_impute_top_k=classifier_config.get('genre_impute_top_k', 2),
        genre_impute_min_votes=classifier_config.get('genre_impute_min_votes', 1),
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
    embedding_dim_from_emb = int(_arr.shape[-1]) if _arr.ndim > 1 else len(_arr)

    # Checkpoint already loaded above; use it for classifier state and architecture
    state_dict = checkpoint['model_state_dict']
    use_genre_ck = checkpoint.get('use_genre', False)
    embedding_dim_ck = checkpoint.get('embedding_dim')
    embedding_dim = int(embedding_dim_ck) if embedding_dim_ck is not None and int(embedding_dim_ck) > 0 else embedding_dim_from_emb

    # Get hidden_dims from checkpoint or infer from state dict (old: mlp.*, new: blocks.*)
    hidden_dims = checkpoint.get('hidden_dims')
    if hidden_dims is None:
        hidden_dims = []
        if "blocks.0.0.weight" in state_dict:
            i = 0
            while f"blocks.{i}.0.weight" in state_dict:
                hidden_dims.append(state_dict[f"blocks.{i}.0.weight"].shape[0])
                i += 1
        else:
            layer_idx = 0
            while f"mlp.{layer_idx}.weight" in state_dict:
                weight = state_dict[f"mlp.{layer_idx}.weight"]
                out_features = weight.shape[0]
                if out_features > 1:
                    hidden_dims.append(out_features)
                layer_idx += 3

    dropout = config['classifier'].get('dropout', 0.3)  # Dropout doesn't affect loading
    use_batch_norm_ck = checkpoint.get('use_batch_norm', False)
    use_residual_ck = checkpoint.get('use_residual', False)
    print(f"  Inferred architecture from checkpoint: hidden_dims={hidden_dims}, use_genre={use_genre_ck}")

    classifier = SimpleRatingClassifier(
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_genre=use_genre_ck,
        use_batch_norm=use_batch_norm_ck,
        use_residual=use_residual_ck,
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

    chunk_aggregation = checkpoint.get(
        'chunk_aggregation', classifier_config.get('chunk_aggregation', 'mean')
    )
    trainer = ClassifierTrainer(
        classifier=classifier,
        device=device,
        loss_fn=loss_fn,
        optimizer=torch.optim.Adam(classifier.parameters()),  # Dummy optimizer
        classification_mode=classification_mode,
        chunk_aggregation=chunk_aggregation,
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

    # Generate false_positives and false_negatives playlists (train+val errors for label correction)
    effective_error_size = (
        error_playlist_size if error_playlist_size is not None
        else rec_config.get('error_playlist_size', 100)
    )
    if effective_error_size <= 0:
        print("\n  Skipping error playlists (error_playlist_size=0 or disabled)")
        fp_path = None
        fn_path = None
    else:
        print("\n" + "=" * 80)
        print("GENERATING ERROR PLAYLISTS (FALSE POSITIVES / FALSE NEGATIVES)")
        print("=" * 80)
        rated_songs = [s for s in all_songs if s.is_rated]
        manifest_path = model_dir / "training_manifest.json"
        if manifest_path.exists():
            manifest = TrainingManifest.load_or_create(str(manifest_path))
            manifest.load()
            train_val_files = manifest.training_files | manifest.validation_files
            train_val_songs = [s for s in rated_songs if s.filename in train_val_files]
            print(f"  Using train+val from manifest: {len(train_val_songs)} rated songs")
        else:
            train_val_songs = rated_songs
            print(f"  No manifest found; using all rated songs: {len(train_val_songs)}")

        if train_val_songs:
            filenames_tv = [s.filename for s in train_val_songs]
            num_chunks = music_config.get('chunk_cache', {}).get('num_chunks', 8)
            embeddings_tv = embedding_store.get_embeddings_batch_all_chunks(
                filenames_tv,
                model_version=encoder_version,
                num_chunks=num_chunks
            )
            if not embeddings_tv:
                embeddings_tv = embedding_store.get_embeddings_batch(
                    filenames_tv,
                    model_version=encoder_version
                )
            train_val_with_emb = [s for s in train_val_songs if s.filename in embeddings_tv]
            if train_val_with_emb:
                binary_pos = classifier_config.get('binary_positive_threshold', 4.0)
                binary_neg = classifier_config.get('binary_negative_threshold', 2.0)
                dataset_tv = EmbeddingDataset(
                    embeddings=embeddings_tv,
                    songs=train_val_with_emb,
                    only_rated=True,
                    classification_mode=classification_mode,
                    binary_positive_threshold=binary_pos,
                    binary_negative_threshold=binary_neg,
                    use_genre=use_genre_recommend,
                    genre_centroids=genre_centroids_recommend,
                    genre_impute_top_k=classifier_config.get('genre_impute_top_k', 2),
                    genre_impute_min_votes=classifier_config.get('genre_impute_min_votes', 1),
                )
                loader_tv = DataLoader(
                    dataset_tv,
                    batch_size=256,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False
                )
                preds_tv, pred_filenames_tv = trainer.predict(loader_tv)
                filename_to_song = {s.filename: s for s in train_val_with_emb}
                false_positives = []
                false_negatives = []
                for filename, pred in zip(pred_filenames_tv, preds_tv):
                    song = filename_to_song.get(filename)
                    if not song or song.rating is None:
                        continue
                    r = song.rating
                    true_pos = r >= binary_pos
                    true_neg = r <= binary_neg
                    if not (true_pos or true_neg):
                        continue
                    pred_pos = pred > 0.5
                    if true_neg and pred_pos:
                        false_positives.append((song, pred))
                    elif true_pos and not pred_pos:
                        false_negatives.append((song, pred))
                false_positives.sort(key=lambda x: x[1], reverse=True)
                false_negatives.sort(key=lambda x: x[1], reverse=False)
                fp_songs = [s for s, _ in false_positives[:effective_error_size]]
                fp_preds = [p for _, p in false_positives[:effective_error_size]]
                fn_songs = [s for s, _ in false_negatives[:effective_error_size]]
                fn_preds = [p for _, p in false_negatives[:effective_error_size]]
                fp_path = playlist_output_dir / f"{filename_prefix}false_positives.xspf"
                fn_path = playlist_output_dir / f"{filename_prefix}false_negatives.xspf"
                if fp_songs:
                    export_to_xspf(
                        songs=fp_songs,
                        predictions=fp_preds,
                        output_path=fp_path,
                        playlist_title="False Positives (true dislike, predicted like) - correct labels",
                        annotation_prefix="Predicted"
                    )
                    print(f"  Exported {len(fp_songs)} false positives -> {fp_path}")
                else:
                    print("  No false positives (train+val)")
                if fn_songs:
                    export_to_xspf(
                        songs=fn_songs,
                        predictions=fn_preds,
                        output_path=fn_path,
                        playlist_title="False Negatives (true like, predicted dislike) - correct labels",
                        annotation_prefix="Predicted"
                    )
                    print(f"  Exported {len(fn_songs)} false negatives -> {fn_path}")
                else:
                    print("  No false negatives (train+val)")
            else:
                print("  No train+val songs with embeddings; skipping error playlists")
        else:
            print("  No rated songs in train+val; skipping error playlists")

    print("\n" + "=" * 80)
    print("RECOMMENDATION COMPLETE")
    if genre_filter:
        print(f"(Genre filter: {genre_filter})")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {output_path} (text recommendations)")
    print(f"  - {playlist_output_dir / f'{filename_prefix}recommender_help.xspf'} (high uncertainty - maximize learning)")
    print(f"  - {playlist_output_dir / f'{filename_prefix}recommender_best.xspf'} (top predictions - validate quality)")
    if effective_error_size > 0:
        print(f"  - {playlist_output_dir / f'{filename_prefix}false_positives.xspf'} (worst false positives - true dislike, predicted like)")
        print(f"  - {playlist_output_dir / f'{filename_prefix}false_negatives.xspf'} (worst false negatives - true like, predicted dislike)")
    print(f"\nNext steps for human-in-the-loop training:")
    print(f"  1. Open XSPF playlists in Clementine")
    print(f"  2. Listen and rate songs")
    print(f"  3. Re-run training with updated ratings")
    print(f"  4. Repeat for continuous improvement!")


def build_waveform_cache(config: dict, overwrite: Optional[bool] = None):
    """Pre-populate waveform cache for consistent training speed.

    MoCo mode (chunk_cache enabled): evenly-spaced 30s chunks per song (num_chunks from config).

    Args:
        config: Configuration dictionary
        overwrite: If True, re-extract and overwrite existing chunk files (e.g. when changing num_chunks).
                  If None, uses config chunk_cache.overwrite. CLI: --overwrite with build-cache.
    """
    from ml_skeleton.music.clementine_db import ClementineDB

    music_config = config['music']

    # Load database
    print("\n[1/3] Loading song database...")
    db = ClementineDB(music_config.get('database_path'))
    all_songs = db.get_all_songs()
    print(f"  Found {len(all_songs)} songs")

    # Check if using MoCo chunk cache (num_chunks per song, default 8)
    chunk_cache_config = music_config.get('chunk_cache', {})
    use_chunk_cache = chunk_cache_config.get('enabled', False)

    if use_chunk_cache:
        # MoCo mode: Use new 4-chunk cache builder
        from ml_skeleton.music.chunk_cache import build_chunk_cache, get_cache_stats

        cache_dir = os.environ.get('CHUNK_CACHE_DIR') or get_chunk_cache_dir(config)
        num_chunks = chunk_cache_config.get('num_chunks', 8)
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

        overwrite_val = overwrite if overwrite is not None else chunk_cache_config.get('overwrite', False)
        if overwrite_val:
            print("  Overwrite: True (re-extracting all chunks)")
        print("\n[3/3] Building chunk cache...")
        stats = build_chunk_cache(
            songs=all_songs,
            cache_dir=cache_dir,
            num_chunks=num_chunks,
            chunk_duration=chunk_duration,
            sample_rate=sample_rate,
            max_duration=max_duration,
            num_workers=num_workers,
            overwrite=overwrite_val,
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
        print("MoCo v2 encoder requires chunk cache (num_chunks in chunk_cache). Update your config:")
        print("  music:")
        print("    chunk_cache:")
        print("      enabled: true")
        print("      directory: (under music.cache_root, e.g. ./chunks)")
        print(f"      num_chunks: {num_chunks}")
        print("      chunk_duration: 30.0")
        return


def fingerprint_songs_stage(config: dict, exhaust: bool = False, workers: Optional[int] = None, all_missing: bool = False):
    """Extract acoustic fingerprints from original audio files.

    Args:
        config: Configuration dictionary with music and fingerprinting settings
        exhaust: If True, process up to daily API limit (500 for free tier)
        workers: Number of parallel workers (default: from config or 4)
        all_missing: If True, process all songs missing fingerprints (no max_songs limit)
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

    # Initialize fingerprint database (under music.cache_root when path is relative)
    print("\n[2/4] Initializing fingerprint database...")
    fp_db_path = get_fingerprint_db_path(config)
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

    # Determine max_songs based on exhaust and all_missing flags
    if all_missing:
        max_songs = None  # No limit: process all missing
        print("\n⚡ ALL MODE: Processing all songs missing fingerprints (no limit)")
    elif exhaust:
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


def backfill_fingerprint_bits_stage(config: dict):
    """Backfill fingerprint DB 'bits' column using the same pipeline as fingerprint_baseline extraction.

    Decode runs inside encoder.forward() (same code path as extraction); we force 256-dim output
    and pack to 32-byte blobs for the DB.
    """
    import copy
    import sqlite3
    import numpy as np
    from tqdm import tqdm

    from ml_skeleton.music.fingerprint_encoder import collate_fingerprint_baseline
    from ml_skeleton.music.fingerprint_db import FingerprintDB

    print("=" * 80)
    print("BACKFILL FINGERPRINT BITS (same pipeline as fingerprint_baseline extraction)")
    print("=" * 80)

    music_config = config.get("music", {})
    encoder_config = config.get("encoder", {})
    fp_config = config.get("fingerprinting", {})

    fp_db_path = get_fingerprint_db_path(config)
    chromaprint_chunk_idx = fp_config.get("chunk_for_fingerprinting", 0)

    # [1/5] Load Clementine (same as train_encoder)
    print("\n[1/5] Loading Clementine database...")
    db = ClementineDB(music_config["database_path"])
    all_songs = db.get_all_songs()
    print(f"  Loaded {len(all_songs)} songs")
    album_to_idx, filename_to_albums = build_album_mapping(all_songs)

    # [2/5] Config: fingerprint_baseline with 256-dim output (no projector) so we can pack to 32 bytes.
    # Order must match working extraction: create_encoder BEFORE create_dataset (chromaprint loads before DB use).
    print("\n[2/5] Creating encoder then dataset (fingerprint_baseline, 256-dim)...")
    config_bf = copy.deepcopy(config)
    config_bf.setdefault("encoder", {})["encoder_type"] = "fingerprint_baseline"
    config_bf.setdefault("fingerprinting", {})["fingerprint_db_path"] = fp_db_path
    bl_cfg = config_bf.setdefault("encoder", {}).setdefault("fingerprint_baseline", {})
    bl_cfg["project_dim"] = None
    bl_cfg["embedding_dim"] = 256

    encoder = create_encoder(config_bf)
    encoder.eval()
    extraction_dataset = create_dataset(
        config=config_bf,
        songs=all_songs,
        album_to_idx=album_to_idx,
        filename_to_albums=filename_to_albums,
        is_training=False,
    )
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    batch_size = encoder_config.get("batch_size", 64)
    # num_workers=0: avoid fork so chromaprint C state is not corrupted (decoder runs in main process)
    num_workers = 0
    all_loader = DataLoader(
        extraction_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fingerprint_baseline,
    )
    print(f"  Dataset: {len(extraction_dataset)} songs, batch_size={batch_size}, num_workers={num_workers} (no fork)")

    # [3/5] Open fingerprint DB for UPDATE only
    print("\n[3/5] Opening fingerprint database...")
    FingerprintDB(fp_db_path)  # ensure bits column exists (migration)
    conn = sqlite3.connect(fp_db_path)
    cur = conn.execute("PRAGMA table_info(fingerprints)")
    columns = [row[1] for row in cur.fetchall()]
    if "bits" not in columns:
        print("  Adding 'bits' column...")
        conn.execute("ALTER TABLE fingerprints ADD COLUMN bits BLOB")
        conn.commit()

    # [4/5] Run decode inside encoder.forward() (exact same path as extraction), pack output to 32 bytes
    print("\n[4/5] Backfilling bits (encoder.forward() = same as extraction)...")
    updated = 0
    failed = 0
    with torch.no_grad():
        for batch in tqdm(all_loader, desc="Backfill"):
            song_ids = batch["song_id"].to(device)
            vecs = encoder(song_ids)  # (B, 256) - decode happens here, same as extraction
            vecs = vecs.cpu().numpy()
            for i in range(vecs.shape[0]):
                sid = int(song_ids[i].item())
                blob = np.packbits((vecs[i] > 0.5).astype(np.uint8)).tobytes()
                if len(blob) != 32:
                    failed += 1
                    continue
                conn.execute(
                    "UPDATE fingerprints SET bits = ? WHERE song_id = ? AND chunk_idx = ?",
                    (blob, sid, chromaprint_chunk_idx),
                )
                updated += 1
            # Commit after each batch so encoder's get_fingerprint() in next batch doesn't get "database is locked"
            conn.commit()

    conn.commit()
    conn.close()

    # [5/5] Done
    print(f"\n[5/5] Done: updated {updated}, failed {failed}, total {updated + failed}")
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

    # Initialize databases (fingerprint DB: same path as encoder/classifier HPO)
    print("\n[2/5] Initializing databases...")
    fp_db_path = get_fingerprint_db_path(config)
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
        choices=['encoder', 'classifier', 'joint-finetune', 'recommend', 'all', 'tune-encoder', 'tune-classifier', 'build-cache', 'clear-chunk-cache', 'fingerprint', 'enrich-metadata', 'backfill-fingerprint-bits', 'init-baseline', 'generate-model-card'],
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
        '--reset-study',
        action='store_true',
        dest='reset_study',
        help='Delete existing Optuna study from storage before running (fresh HPO run); only applies when optuna_storage is set'
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
        '--mlflow-run-id',
        type=str,
        default=None,
        dest='mlflow_run_id',
        help='MLflow run ID (e.g. classifier HPO parent run) to load hyperparameters from instead of --best-params'
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
        '--all',
        action='store_true',
        dest='fingerprint_all',
        help='Fingerprint stage: process all missing songs (no max_songs limit). Skip existing is still applied.'
    )
    parser.add_argument(
        '-N', '--num-runs',
        type=int,
        default=1,
        dest='num_runs',
        help='Number of training runs with different seeds (reports mean/std, saves best model)'
    )
    parser.add_argument(
        '--reps',
        type=int,
        default=None,
        dest='reps',
        help='Classifier HPO only: repetitions per trial with different init seeds; best (min) value and seed reported. Default: 1.'
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
        '--error-playlist-size',
        type=int,
        default=None,
        dest='error_playlist_size',
        help='Max songs per false_positives / false_negatives playlist. '
             'Default: from config (100). Use 0 to disable error playlists.'
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
        default=1000,
        dest='vault_size',
        help='Number of ratings to reserve in vault for A/B testing only (default: 1000). '
             'Vault files are never used for training, only for comparing models.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        dest='overwrite_cache',
        help='With build-cache: overwrite existing chunk files (use when changing num_chunks).'
    )

    args = parser.parse_args()

    # Handle backwards compatibility for --model-version
    if args.model_version and not args.encoder_version:
        args.encoder_version = args.model_version
        print(f"NOTE: --model-version is deprecated, use --encoder-version instead")

    # Load configuration
    config = load_config(args.config)

    # Apply GPU memory limit from config before any CUDA use (avoids OOM, consistent with config)
    gpu_limit_gb = config.get("gpu_memory_limit_gb", 0)
    if gpu_limit_gb and float(gpu_limit_gb) > 0:
        limit_gpu_memory(max_memory_gb=float(gpu_limit_gb))

    # Apply encoder type override if provided
    if args.encoder_type:
        config['encoder']['encoder_type'] = args.encoder_type
        print(f"Encoder type overridden to: {args.encoder_type}")
    elif args.stage == 'tune-encoder' and config.get('encoder', {}).get('encoder_type') == 'fingerprint_baseline':
        # Encoder HPO must train the real encoder (MoCo), not fingerprint_baseline (extraction-only)
        config['encoder']['encoder_type'] = 'moco'
        print("Encoder HPO: using encoder_type=moco (config had fingerprint_baseline; add --encoder-type to override)")

    # Apply best parameters: from MLflow run (classifier only) or from JSON file
    if args.stage == 'classifier' and args.mlflow_run_id:
        tracking_uri = config.get('mlflow', {}).get('tracking_uri', 'http://localhost:5000')
        print(f"\nLoading classifier hyperparameters from MLflow run: {args.mlflow_run_id}")
        best_params, hpo_run_name = load_classifier_params_from_mlflow_run(tracking_uri, args.mlflow_run_id)
        print("Applying parameters to classifier config:")
        for key, value in best_params.items():
            config['classifier'][key] = value
            print(f"  {key} = {value}")
        config['classifier']['hpo_mlflow_run_id'] = args.mlflow_run_id
        config['classifier']['hpo_mlflow_run_name'] = hpo_run_name
        # Fetch HPO best val ROC AUC and best_training_seed from parent run for final-training
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=tracking_uri)
            run = client.get_run(args.mlflow_run_id)
            hpo_auc = (run.data.metrics or {}).get("best_val_roc_auc")
            if hpo_auc is not None:
                config['classifier']['hpo_val_roc_auc'] = float(hpo_auc)
            # Ensure final training uses same seed as HPO best run (param is best_training_seed on parent)
            seed_param = (run.data.params or {}).get("best_training_seed")
            if seed_param is not None:
                try:
                    config['classifier']['training_seed'] = int(float(seed_param))
                except (TypeError, ValueError):
                    config['classifier']['training_seed'] = seed_param
        except Exception:
            pass
        print("")
    elif args.best_params:
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
            if best_params.get('mlflow_parent_run_id'):
                config['classifier']['hpo_mlflow_run_id'] = best_params['mlflow_parent_run_id']
                tracking_uri = config.get('mlflow', {}).get('tracking_uri', 'http://localhost:5000')
                try:
                    config['classifier']['hpo_mlflow_run_name'] = get_mlflow_run_name(tracking_uri, best_params['mlflow_parent_run_id'])
                except Exception:
                    config['classifier']['hpo_mlflow_run_name'] = ""
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
        # Use same seed as winning HPO run: from MLflow (best_training_seed), JSON (best_reps_seed), or config (init_seed)
        clf_cfg = config.get('classifier', {})
        training_seed = None
        if args.best_params or args.mlflow_run_id:
            # Prefer: best_reps_seed (from JSON), then training_seed (from MLflow best_training_seed), then init_seed (config)
            training_seed = clf_cfg.get('best_reps_seed')
            if training_seed is None:
                training_seed = clf_cfg.get('training_seed')
            if training_seed is None:
                training_seed = clf_cfg.get('init_seed')
            if training_seed is not None:
                training_seed = int(training_seed)
                print(f"Using init seed from HPO best run: {training_seed}")
        model_card = train_classifier(
            config,
            final_training=args.final_training,
            classifier_version_override=args.classifier_version,
            num_runs=args.num_runs,
            init_from_prod=not args.random_init,
            vault_size=args.vault_size,
            training_seed=training_seed,
        )
        cleanup_memory()
        print("\nNext step: Run with --stage recommend or --stage joint-finetune (if enabled) to push accuracy further")

    elif args.stage == 'joint-finetune':
        train_joint_finetune(config, verbose=True)
        cleanup_memory()
        print("\nNext step: Run with --stage recommend to generate recommendations")

    elif args.stage == 'recommend':
        generate_recommendations(
            config,
            prod_dir=args.prod_dir,
            low_rating_ratio=args.low_rating_ratio,
            genre_filter=args.genre,
            error_playlist_size=args.error_playlist_size
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

        build_waveform_cache(config, overwrite=args.overwrite_cache)
        cleanup_memory()
        print("\nCache build complete! Training will now have consistent speed.")

    elif args.stage == 'clear-chunk-cache':
        # Remove only chunk cache directory; preserve fingerprint DB (same DB for encoder/classifier/fingerprint)
        music_config = config.get('music', {})
        chunk_config = music_config.get('chunk_cache', {})
        chunk_dir = chunk_config.get('directory') or music_config.get('waveform_cache_dir') or './cache/chunks'
        chunk_dir = os.path.abspath(chunk_dir)
        fp_path = get_fingerprint_db_path(config)
        print("\n" + "=" * 60)
        print("CLEAR CHUNK CACHE (fingerprint DB preserved)")
        print("=" * 60)
        print(f"  Chunk cache dir: {chunk_dir}")
        print(f"  Fingerprint DB (kept): {fp_path}")
        if not os.path.isdir(chunk_dir):
            print("  No chunk cache directory found. Nothing to clear.")
        else:
            import shutil
            n_files = 0
            size_bytes = 0
            for root, _, files in os.walk(chunk_dir):
                for f in files:
                    n_files += 1
                    try:
                        size_bytes += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            size_gb = size_bytes / (1024**3)
            print(f"  Files: {n_files}, size: ~{size_gb:.1f} GB")
            reply = input("  Delete chunk cache only? [y/N] ").strip().lower()
            if reply == 'y':
                shutil.rmtree(chunk_dir, ignore_errors=True)
                print("  Chunk cache cleared. Fingerprint DB untouched.")
            else:
                print("  Skipped.")

    elif args.stage == 'fingerprint':
        # Extract acoustic fingerprints from original files
        fingerprint_songs_stage(config, exhaust=args.exhaust, workers=args.workers, all_missing=getattr(args, 'fingerprint_all', False))
        cleanup_memory()

    elif args.stage == 'enrich-metadata':
        # Enrich metadata using AcoustID/MusicBrainz APIs
        enrich_metadata_stage(config, exhaust=args.exhaust)
        cleanup_memory()

    elif args.stage == 'backfill-fingerprint-bits':
        # Backfill fingerprint DB 'bits' column using same pipeline as fingerprint_baseline extraction
        # (main process, same DataLoader/encoder load order) so chromaprint decoder does not crash
        backfill_fingerprint_bits_stage(config)
        cleanup_memory()

    elif args.stage == 'all':
        # Run complete pipeline: encoder -> classifier -> [joint-finetune if enabled] -> model card
        jf_enabled = config.get("joint_finetune", {}).get("enabled", False)
        print("\n" + "=" * 80)
        print("RUNNING COMPLETE PIPELINE")
        print("=" * 80)
        print("This will run:")
        print("  1. Encoder training (Stage 1)")
        print("  2. Classifier training (Stage 2)")
        if jf_enabled:
            print("  3. Joint fine-tune (encoder + classifier on audio)")
        print("  Model card generation")
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

        # Optional: Joint fine-tune (freeze → train classifier → unfreeze both)
        if jf_enabled:
            train_joint_finetune(config, verbose=True)
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
            artifact_dir=config.get('artifact_dir', './artifacts'),
            tags=config.get('tags', {}),
        )

        # Configure MLflow
        if 'mlflow' in config:
            exp_config.mlflow = MLflowConfig(**config['mlflow'])

        # Configure tuning (optuna_storage for persistence; reset_study for fresh HPO)
        exp_config.tuning = TuningConfig(
            tuner_type=TunerType.OPTUNA if args.tuner == 'optuna' else TunerType.RAY_TUNE,
            n_trials=n_trials,
            timeout=args.timeout,
            sampler=tuning_dict.get('sampler', 'TPESampler'),
            pruner=tuning_dict.get('pruner', 'MedianPruner'),
            optuna_storage=tuning_dict.get('optuna_storage'),
            reset_study=getattr(args, 'reset_study', False),
        )

        # Set encoder search space
        if 'encoder_search_space' in tuning_dict:
            exp_config.tuning.search_space = SearchSpaceConfig(
                parameters=tuning_dict['encoder_search_space']['parameters']
            )

        tracking_uri = exp_config.mlflow.tracking_uri if exp_config.mlflow else "http://localhost:5000"
        print(f"MLflow UI: {tracking_uri}")
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
        to_save = dict(results['best_params'])
        if results.get('mlflow_parent_run_id'):
            to_save['mlflow_parent_run_id'] = results['mlflow_parent_run_id']
        with open(best_params_file, 'w') as f:
            json.dump(to_save, f, indent=2)
        print(f"\nBest parameters saved to: {best_params_file}")

        # Check if HPO best checkpoint exists; write MLflow parent run ID into it for traceability
        hpo_best_checkpoint = checkpoint_dir / 'encoder_hpo_best.pt'
        if hpo_best_checkpoint.exists() and results.get('mlflow_parent_run_id'):
            enc_ckpt = torch.load(hpo_best_checkpoint, map_location='cpu', weights_only=False)
            enc_ckpt['mlflow_parent_run_id'] = results['mlflow_parent_run_id']
            torch.save(enc_ckpt, hpo_best_checkpoint)
            print(f"  MLflow HPO parent run ID saved to checkpoint: {hpo_best_checkpoint}")
        if hpo_best_checkpoint.exists():
            print(f"Best HPO model saved to: {hpo_best_checkpoint}")
            print("\nTo continue training from best HPO model with best parameters:")
            print(f"  python examples/music_recommendation.py --stage encoder --config configs/music_recommendation.yaml \\")
            print(f"      --final-training --best-params {best_params_file} --resume-checkpoint {hpo_best_checkpoint}")
        else:
            print("\nTo run final training with best parameters:")
            print("  python examples/music_recommendation.py --stage encoder --config configs/music_recommendation.yaml --final-training")

        if results.get('mlflow_parent_run_id'):
            tracking_uri = exp_config.mlflow.tracking_uri if exp_config.mlflow else "http://localhost:5000"
            run_id = results['mlflow_parent_run_id']
            print("\n" + "=" * 40)
            print("MLflow best run (HPO parent)")
            print("=" * 40)
            print(f"  Run ID: {run_id}")
            print(f"  View at: {tracking_uri}/#/runs/{run_id}")
            print("=" * 40)

    elif args.stage == 'tune-classifier':
        # Hyperparameter tuning for classifier
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING: CLASSIFIER")
        print("=" * 80)

        # Verify embeddings exist
        embedding_db_path = config['music']['embedding_db_path']
        if not Path(embedding_db_path).exists():
            print(f"ERROR: Embeddings database not found: {embedding_db_path}")
            print("Run encoder training first (or fingerprint-baseline extraction with encoder_type: fingerprint_baseline):")
            print("  python examples/music_recommendation.py --stage encoder --config configs/music_moco.yaml")
            sys.exit(1)

        # Create ExperimentConfig manually (our music config has custom structure)
        from ml_skeleton.core.config import TuningConfig, SearchSpaceConfig, MLflowConfig

        # Extract tuning config
        tuning_dict = config.get('tuning', {})
        n_trials = args.n_trials if args.n_trials else tuning_dict.get('n_trials', 20)

        # Experiment name includes encoder_version so each embedding type gets its own
        # Optuna study (avoids mixing trials when switching e.g. fingerprint_baseline vs moco).
        base_name = config.get('name', 'music_recommendation_classifier')
        encoder_version = config.get('music', {}).get('encoder_version', 'default')
        exp_name = f"{base_name}_classifier_{encoder_version}"

        # Create experiment config
        exp_config = ExperimentConfig(
            name=exp_name,
            framework=config.get('framework', 'pytorch'),
            hyperparameters=config['classifier'].copy(),
            seed=config.get('seed', 42),
            checkpoint_dir=config.get('checkpoint_dir', './checkpoints'),
            artifact_dir=config.get('artifact_dir', './artifacts'),
            tags=config.get('tags', {}),
        )

        # Configure MLflow
        if 'mlflow' in config:
            exp_config.mlflow = MLflowConfig(**config['mlflow'])

        # Configure tuning (optuna_storage so studies persist; study name = exp_name + "_optuna")
        exp_config.tuning = TuningConfig(
            tuner_type=TunerType.OPTUNA if args.tuner == 'optuna' else TunerType.RAY_TUNE,
            n_trials=n_trials,
            timeout=args.timeout,
            sampler=tuning_dict.get('sampler', 'TPESampler'),
            pruner=tuning_dict.get('pruner', 'MedianPruner'),
            optuna_storage=tuning_dict.get('optuna_storage'),
            reset_study=getattr(args, 'reset_study', False),
            parent_run_name_prefix="classifier-tune-hpo",
        )

        # Set classifier search space
        if 'classifier_search_space' in tuning_dict:
            exp_config.tuning.search_space = SearchSpaceConfig(
                parameters=tuning_dict['classifier_search_space']['parameters']
            )

        # Reps per trial (different init seeds); default from config or 1
        reps = args.reps if getattr(args, 'reps', None) is not None else tuning_dict.get('reps', 1)
        reps = max(1, int(reps))

        tracking_uri = exp_config.mlflow.tracking_uri if exp_config.mlflow else "http://localhost:5000"
        print(f"MLflow UI: {tracking_uri}")
        print(f"Tuner: {args.tuner}")
        print(f"Trials: {n_trials}")
        print(f"Reps per trial: {reps} (best min value and init seed reported)")
        print(f"Search space parameters: {list(exp_config.tuning.search_space.parameters.keys())}")
        print("")

        # Create training function (pass n_trials and reps for progress logging and multi-seed)
        train_fn = create_classifier_training_fn(config, n_trials=n_trials, hpo_runs=reps)

        # Run hyperparameter tuning
        results = run_experiment(train_fn, exp_config, tune=True)

        print("\n" + "=" * 80)
        print("CLASSIFIER TUNING COMPLETE")
        print("=" * 80)
        print(f"Best value: {results['best_value']:.6f}")
        # Best init seed (when reps > 1)
        best_trial = results.get('study') and results['study'].best_trial
        best_reps_seed = _trial_best_reps_seed.get(best_trial.number) if best_trial is not None else None
        seed_used = best_reps_seed if best_reps_seed is not None else config.get('seed', 42)
        if best_trial is not None:
            print(f"Best trial number: {best_trial.number} (in MLflow: open parent run → find child with tag trial_number={best_trial.number})")
        print(f"Best run seed: {seed_used} (run name under that trial: seed_{seed_used})")
        print(f"Best parameters:")
        for key, value in results['best_params'].items():
            print(f"  {key}: {value}")

        # Save best parameters, seed, and best trial number (so you can find the run in MLflow: parent -> trial N -> seed_*)
        import json
        best_params_file = Path(config.get('checkpoint_dir', './checkpoints')) / 'best_classifier_params.json'
        best_params_file.parent.mkdir(parents=True, exist_ok=True)
        to_save = dict(results['best_params'])
        to_save['best_reps_seed'] = seed_used
        if best_trial is not None:
            to_save['best_trial_number'] = best_trial.number
        if results.get('mlflow_parent_run_id'):
            to_save['mlflow_parent_run_id'] = results['mlflow_parent_run_id']
        # Save HPO best val ROC AUC for final-training comparison (when HPO optimized val_roc_auc)
        hpo_metric = config.get('classifier', {}).get('hpo_metric', 'val_roc_auc')
        if hpo_metric == 'val_roc_auc':
            to_save['hpo_val_roc_auc'] = 1.0 - results['best_value']
        with open(best_params_file, 'w') as f:
            json.dump(to_save, f, indent=2)
        print(f"\nBest parameters saved to: {best_params_file}")

        # Save MLflow parent run ID into the classifier checkpoint for traceability
        checkpoint_path = Path(config.get('checkpoint_dir', './checkpoints')) / 'classifier_best.pt'
        if results.get('mlflow_parent_run_id') and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            checkpoint['mlflow_parent_run_id'] = results['mlflow_parent_run_id']
            torch.save(checkpoint, checkpoint_path)
            print(f"  MLflow HPO parent run ID saved to checkpoint: {checkpoint_path}")

        print("\nUpdate your config file with these parameters and run:")
        print("  python examples/music_recommendation.py --stage classifier --config configs/music_recommendation.yaml --final-training")

        if results.get('mlflow_parent_run_id'):
            tracking_uri = exp_config.mlflow.tracking_uri if exp_config.mlflow else "http://localhost:5000"
            run_id = results['mlflow_parent_run_id']
            # Log best run's seed and best val ROC AUC to parent for final-training comparison
            from mlflow.tracking import MlflowClient
            try:
                client = MlflowClient(tracking_uri=tracking_uri)
                client.log_param(run_id, "best_training_seed", str(seed_used))
                if hpo_metric == 'val_roc_auc':
                    client.log_metric(run_id, "best_val_roc_auc", 1.0 - results['best_value'])
            except Exception as e:
                print(f"  Note: could not log best_training_seed/best_val_roc_auc to MLflow: {e}")
            print("\n" + "=" * 40)
            print("MLflow best run (HPO parent)")
            print("=" * 40)
            print(f"  Run ID: {run_id}")
            print(f"  View at: {tracking_uri}/#/runs/{run_id}")
            print("=" * 40)

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
            dropout=0.0,
            use_batch_norm=classifier_config.get('use_batch_norm', False),
            use_residual=classifier_config.get('use_residual', False),
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
            'use_batch_norm': classifier_config.get('use_batch_norm', False),
            'use_residual': classifier_config.get('use_residual', False),
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
                'train_prevalence': manifest_data.get('metadata', {}).get('train_prevalence'),
                'val_prevalence': manifest_data.get('metadata', {}).get('val_prevalence'),
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
