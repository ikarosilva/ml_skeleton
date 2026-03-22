"""Classifier training orchestration.

Handles Stage 2: Train rating classifier on pre-extracted embeddings.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm
import time
import numpy as np

from ..utils.early_stopping import EarlyStopping
from ..utils.gpu import GPUMonitor
from ..music.losses import BinaryRatingLoss


def get_encoder_version_from_checkpoint(checkpoint_path: str) -> str:
    """Read encoder version from an encoder checkpoint file.

    Args:
        checkpoint_path: Path to encoder checkpoint (.pt file)

    Returns:
        Encoder version string (e.g., "v1", "v2")
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("encoder_version", checkpoint.get("model_version", "unknown"))


def get_classifier_versions_from_checkpoint(checkpoint_path: str) -> tuple[str, str]:
    """Read version info from a classifier checkpoint file.

    Args:
        checkpoint_path: Path to classifier checkpoint (.pt file)

    Returns:
        Tuple of (classifier_version, encoder_version)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classifier_version = checkpoint.get("classifier_version", "unknown")
    encoder_version = checkpoint.get("encoder_version", "unknown")
    return classifier_version, encoder_version


def validate_model_compatibility(
    encoder_checkpoint: str,
    classifier_checkpoint: str
) -> None:
    """Validate that classifier was trained with the current encoder version.

    Args:
        encoder_checkpoint: Path to encoder checkpoint
        classifier_checkpoint: Path to classifier checkpoint

    Raises:
        ValueError: If encoder versions don't match
    """
    encoder_version = get_encoder_version_from_checkpoint(encoder_checkpoint)
    classifier_version, classifier_encoder_version = get_classifier_versions_from_checkpoint(
        classifier_checkpoint
    )

    if classifier_encoder_version != encoder_version:
        raise ValueError(
            f"\n{'='*60}\n"
            f"MODEL VERSION MISMATCH - DEPLOYMENT BLOCKED\n"
            f"{'='*60}\n"
            f"Current encoder version: {encoder_version}\n"
            f"Classifier trained with encoder version: {classifier_encoder_version}\n"
            f"Classifier version: {classifier_version}\n"
            f"\n"
            f"The classifier must be retrained with the new encoder.\n"
            f"Run: ./run_music_pipeline.sh classifier\n"
            f"{'='*60}"
        )

    print(f"Model compatibility validated:")
    print(f"  Encoder version: {encoder_version}")
    print(f"  Classifier version: {classifier_version}")
    print(f"  Classifier trained with encoder: {classifier_encoder_version} ✓")


class ClassifierTrainer:
    """Trainer for rating classifier models.

    Trains on pre-extracted embeddings from Stage 1.
    Predicts continuous ratings in [0, 1] range.
    Supports MLflow metric logging for learning curves.

    Version Compatibility:
        The classifier stores which encoder_version it was trained with.
        During deployment, the system validates that the classifier's
        encoder_version matches the current encoder to prevent mismatches.

    Args:
        classifier: Rating classifier model (conforms to RatingClassifier protocol)
        device: Device to train on ('cuda' or 'cpu')
        loss_fn: Loss function (typically MSE)
        optimizer: PyTorch optimizer
        tracker: Optional MLflow tracker (ExplrTracker) for logging metrics
        encoder_version: Version of encoder used to create embeddings (for compatibility)
        classifier_version: Version of this classifier
    """

    def __init__(
        self,
        classifier: nn.Module,
        device: str,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        tracker: Optional[Any] = None,
        encoder_version: str = "v1",
        classifier_version: str = "v1",
        classification_mode: str = "regression",
        genre_centroids: Optional[np.ndarray] = None,
        chunk_aggregation: str = "mean",
        clip_grad: bool = False,
        clip_grad_norm: float = 1.0,
        training_label_noise: float = 0.0,
        hpo_mlflow_run_id: Optional[str] = None,
        hpo_mlflow_run_name: Optional[str] = None,
        binary_positive_threshold: float = 4.0,
    ):
        self.classifier = classifier.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.genre_centroids = genre_centroids  # (NUM_GENRES, D), saved in checkpoint when use_genre
        self.tracker = tracker  # MLflow tracker for logging learning curves
        self.classification_mode = classification_mode
        self.chunk_aggregation = chunk_aggregation  # "mean" or "max" over chunk predictions per song
        self.clip_grad = clip_grad
        self.clip_grad_norm = clip_grad_norm
        # Label smoothing for training only (reduces bias toward positives): 0 = none; 10 = 10% (0.1)
        if training_label_noise >= 1.0:
            self._label_noise_epsilon = float(training_label_noise) / 100.0
        else:
            self._label_noise_epsilon = float(training_label_noise)

        # Version tracking for compatibility validation
        self.encoder_version = encoder_version  # Encoder version this classifier was trained with
        self.classifier_version = classifier_version  # This classifier's version
        self.hpo_mlflow_run_id = hpo_mlflow_run_id  # HPO run params were loaded from (traceability)
        self.hpo_mlflow_run_name = hpo_mlflow_run_name  # HPO parent run display name
        # P@5: top-5 by pred score; count as hit if true rating (1–5) >= this threshold
        self._p5_pos_rating_norm = float(binary_positive_threshold) / 5.0

        # GPU monitoring (samples utilization during training)
        self.gpu_monitor = GPUMonitor() if device == "cuda" else None

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_mae = float('inf')  # Also used for accuracy in binary mode
        self.best_accuracy = 0.0
        self.best_recall = 0.0
        self.best_precision = 0.0
        self.best_roc_auc = 0.0
        self.roc_auc_at_best_checkpoint: Optional[float] = None  # ROC AUC of the saved best model (early stopping)
        self.precision_at_5_at_best_checkpoint: Optional[float] = None
        self.precision_at_20_at_best_checkpoint: Optional[float] = None
        self.best_epoch_saved: Optional[int] = None  # Epoch (1-based) when best checkpoint was saved
        self.best_correlation = 0.0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],  # In binary mode, this stores (1 - accuracy)
            "val_accuracy": [],
            "val_precision": [],  # Binary only: PPV = TP/(TP+FP) per epoch
            "val_recall": [],  # Binary only: recall per epoch
            "val_f1": [],  # Binary only: F1 per epoch
            "val_roc_auc": [],  # Binary only: ROC AUC (probs vs binary labels)
            "val_precision_at_5": [],
            "val_precision_at_20": [],
            "val_rating_mse": [],  # Binary: MSE(pred_prob, normalized 1-5 rating)
            "val_rating_corr": [],  # Binary: correlation(pred_prob, normalized 1-5 rating)
        }

    def _aggregate_chunk_predictions(self, pred_chunks: torch.Tensor, B: int, C: int) -> torch.Tensor:
        """Aggregate per-chunk predictions to one per song. pred_chunks shape (B*C,) or (B*C, 1)."""
        view = pred_chunks.view(B, C)
        if self.chunk_aggregation == "max":
            return view.max(dim=1)[0]
        return view.mean(dim=1)

    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch.

        Args:
            train_loader: Training data loader (EmbeddingDataset)

        Returns:
            Dictionary with training metrics
        """
        self.classifier.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in pbar:
            # Move data to device
            embeddings = batch["embedding"].to(self.device)
            ratings = batch["rating"].to(self.device)
            # Apply label smoothing during training only (soften 0/1 to reduce positive bias)
            if self._label_noise_epsilon > 0 and self.classification_mode == "binary":
                eps = self._label_noise_epsilon
                # Hard 0 -> eps, hard 1 -> 1-eps, middle (0.5) unchanged
                ratings = torch.where(
                    ratings <= 0.01,
                    torch.full_like(ratings, eps, device=ratings.device),
                    torch.where(ratings >= 0.99, torch.full_like(ratings, 1.0 - eps, device=ratings.device), ratings),
                )
            genre = batch.get("genre")
            if genre is not None:
                genre = genre.to(self.device)

            # Forward pass: (B, C, D) -> one rating per song (attention classifier) or per-chunk then aggregate
            if embeddings.dim() == 3 and getattr(self.classifier, "handles_chunk_sequence", False):
                predictions = self.classifier(embeddings, genre)
            elif embeddings.dim() == 3:
                B, C, D = embeddings.shape
                emb_flat = embeddings.view(B * C, D)
                if genre is not None:
                    genre_flat = genre.unsqueeze(1).expand(-1, C, -1).reshape(B * C, genre.size(-1))
                    pred_chunks = self.classifier(emb_flat, genre_flat)
                else:
                    pred_chunks = self.classifier(emb_flat)
                predictions = self._aggregate_chunk_predictions(pred_chunks, B, C)
            else:
                predictions = self.classifier(embeddings, genre) if genre is not None else self.classifier(embeddings)

            # Compute loss (predictions are already (batch_size,))
            if isinstance(self.loss_fn, BinaryRatingLoss) and "is_middle" in batch:
                loss = self.loss_fn(predictions, ratings, is_middle=batch["is_middle"].to(self.device))
            else:
                loss = self.loss_fn(predictions, ratings)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Sample GPU utilization every 10 batches
            if self.gpu_monitor and num_batches % 10 == 0:
                self.gpu_monitor.sample()

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        self.history["train_loss"].append(avg_loss)

        # Get GPU stats for this epoch
        gpu_stats = {}
        if self.gpu_monitor:
            gpu_stats = self.gpu_monitor.get_stats()
            self.gpu_monitor.reset()

        return {
            "loss": avg_loss,
            "num_batches": num_batches,
            "gpu_stats": gpu_stats,
        }

    def validate(self, val_loader: DataLoader) -> dict:
        """Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics (loss, MAE/accuracy)
        """
        self.classifier.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        all_predictions = []
        all_targets = []
        all_rating_continuous = []  # Original 1-5 normalized to [0,1], for rating MSE/corr

        # Binary classification metrics
        correct = 0
        total = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                embeddings = batch["embedding"].to(self.device)
                ratings = batch["rating"].to(self.device)
                genre = batch.get("genre")
                if genre is not None:
                    genre = genre.to(self.device)

                # Forward pass: (B, C, D) -> one rating per song (attention) or per-chunk then aggregate
                if embeddings.dim() == 3 and getattr(self.classifier, "handles_chunk_sequence", False):
                    predictions = self.classifier(embeddings, genre)
                elif embeddings.dim() == 3:
                    B, C, D = embeddings.shape
                    emb_flat = embeddings.view(B * C, D)
                    if genre is not None:
                        genre_flat = genre.unsqueeze(1).expand(-1, C, -1).reshape(B * C, genre.size(-1))
                        pred_chunks = self.classifier(emb_flat, genre_flat)
                    else:
                        pred_chunks = self.classifier(emb_flat)
                    predictions = self._aggregate_chunk_predictions(pred_chunks, B, C)
                else:
                    predictions = self.classifier(embeddings, genre) if genre is not None else self.classifier(embeddings)

                # Compute loss (predictions are already (batch_size,))
                # Use unweighted BCE for validation so val_loss does not reward collapse (predict-all-1)
                if isinstance(self.loss_fn, BinaryRatingLoss) and "is_middle" in batch:
                    loss = self.loss_fn(
                        predictions, ratings,
                        is_middle=batch["is_middle"].to(self.device),
                        validation=True,
                    )
                elif isinstance(self.loss_fn, BinaryRatingLoss):
                    loss = self.loss_fn(predictions, ratings, validation=True)
                else:
                    loss = self.loss_fn(predictions, ratings)
                total_loss += loss.item()
                num_batches += 1

                if self.classification_mode == "binary":
                    # Binary: apply sigmoid and threshold at 0.5
                    probs = torch.sigmoid(predictions).squeeze()
                    predicted_labels = (probs > 0.5).float()
                    target_labels = ratings.squeeze()
                    # Exclude middle-rated samples from accuracy (they have target 0.5; we only score like/dislike)
                    non_middle = ~batch["is_middle"].to(self.device) if "is_middle" in batch else torch.ones_like(target_labels, dtype=torch.bool)
                    if non_middle.any():
                        correct += ((predicted_labels == target_labels) & non_middle).sum().item()
                        total += non_middle.sum().item()
                        # Precision/Recall only over non-middle
                        true_positives += ((predicted_labels == 1) & (target_labels == 1) & non_middle).sum().item()
                        false_positives += ((predicted_labels == 1) & (target_labels == 0) & non_middle).sum().item()
                        false_negatives += ((predicted_labels == 0) & (target_labels == 1) & non_middle).sum().item()

                    # Store probabilities for analysis
                    all_predictions.extend(probs.cpu().numpy().tolist())
                    all_targets.extend(target_labels.cpu().numpy().tolist())
                    # Continuous rating (1-5 normalized) for rating MSE/correlation
                    rc = batch.get("rating_continuous")
                    if rc is not None:
                        all_rating_continuous.extend(rc.cpu().numpy().tolist())
                else:
                    # Regression: compute MAE
                    mae = torch.abs(predictions - ratings).mean()
                    total_mae += mae.item()

                    # Store for correlation analysis
                    all_predictions.extend(predictions.cpu().numpy().tolist())
                    all_targets.extend(ratings.cpu().numpy().tolist())

        avg_loss = total_loss / num_batches

        if self.classification_mode == "binary":
            # Binary metrics
            accuracy = correct / total if total > 0 else 0.0
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            self.history["val_loss"].append(avg_loss)
            self.history["val_accuracy"].append(accuracy)
            self.history["val_precision"].append(precision)
            self.history["val_recall"].append(recall)
            self.history["val_mae"].append(1.0 - accuracy)  # For compatibility, store error rate
            self.history["val_f1"].append(f1)

            # Precision@K: top-K by pred score; hit = true rating >= positive threshold (same as P@5).
            precision_at_5 = float("nan")
            precision_at_20 = float("nan")
            n_pred = len(all_predictions)
            if n_pred >= 5 and len(all_targets) == n_pred:
                pa = np.asarray(all_predictions, dtype=np.float64)
                thr = self._p5_pos_rating_norm
                if len(all_rating_continuous) == n_pred:
                    hit = np.asarray(all_rating_continuous, dtype=np.float64) >= thr - 1e-9
                else:
                    hit = np.asarray(all_targets, dtype=np.float64) > 0.75
                order = np.argsort(pa)[::-1]
                precision_at_5 = float(hit[order[:5]].astype(np.float64).sum() / 5.0)
                if n_pred >= 20:
                    precision_at_20 = float(hit[order[:20]].astype(np.float64).sum() / 20.0)

            # ROC AUC: strict like/dislike only (exclude middle-rated)
            roc_auc = 0.5
            if all_predictions and all_targets:
                pred_arr = np.array(all_predictions)
                tgt_arr = np.array(all_targets)
                binary_mask = (tgt_arr < 0.25) | (tgt_arr > 0.75)
                if binary_mask.sum() > 0:
                    y_score = pred_arr[binary_mask]
                    y_true_bin = (tgt_arr[binary_mask] > 0.5).astype(np.float64)
                    n_pos = int(y_true_bin.sum())
                    n_neg = len(y_true_bin) - n_pos
                    if n_pos > 0 and n_neg > 0:
                        order = np.argsort(y_score)[::-1]
                        y_sorted = np.take(y_true_bin, order)
                        ranks = np.arange(1, len(y_sorted) + 1, dtype=np.float64)
                        roc_auc = float(
                            (np.sum(ranks * y_sorted) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
                        )
                    if roc_auc > self.best_roc_auc:
                        self.best_roc_auc = roc_auc
            self.history["val_roc_auc"].append(roc_auc)
            self.history["val_precision_at_5"].append(precision_at_5)
            self.history["val_precision_at_20"].append(precision_at_20)

            # Rating MSE/correlation: predicted prob vs original 1-5 (normalized)
            rating_mse = None
            rating_corr = None
            if all_rating_continuous:
                pred_arr = np.array(all_predictions)
                cont_arr = np.array(all_rating_continuous)
                rating_mse = float(np.mean((pred_arr - cont_arr) ** 2))
                self.history["val_rating_mse"].append(rating_mse)
                if pred_arr.std() >= 1e-8 and cont_arr.std() >= 1e-8:
                    rating_corr = float(np.corrcoef(pred_arr, cont_arr)[0, 1])
                else:
                    rating_corr = float("nan")
                self.history["val_rating_corr"].append(rating_corr)

            # Update best
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
            if recall > self.best_recall:
                self.best_recall = recall
            if precision > self.best_precision:
                self.best_precision = precision

            out = {
                "loss": avg_loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "precision_at_5": precision_at_5,
                "precision_at_20": precision_at_20,
                "mae": 1.0 - accuracy,  # Error rate for compatibility
                "num_batches": num_batches
            }
            if rating_mse is not None:
                out["rating_mse"] = rating_mse
            if rating_corr is not None and not np.isnan(rating_corr):
                out["rating_corr"] = rating_corr
            return out
        else:
            # Regression metrics
            avg_mae = total_mae / num_batches
            self.history["val_loss"].append(avg_loss)
            self.history["val_mae"].append(avg_mae)

            # Compute correlation with proper NaN handling
            all_predictions_arr = np.array(all_predictions)
            all_targets_arr = np.array(all_targets)

            pred_std = np.std(all_predictions_arr)
            target_std = np.std(all_targets_arr)

            if pred_std < 1e-8 or target_std < 1e-8:
                correlation = np.nan
                if pred_std < 1e-8:
                    pred_mean = np.mean(all_predictions_arr)
                    print(f"  [Warning] Predictions have zero variance (all ~{pred_mean:.4f})")
            else:
                corr_matrix = np.corrcoef(all_predictions_arr, all_targets_arr)
                correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else np.nan

            return {
                "loss": avg_loss,
                "mae": avg_mae,
                "correlation": correlation,
                "num_batches": num_batches,
                "pred_std": pred_std,
                "target_std": target_std
            }

    def _compute_val_rating_corr_and_f1(
        self,
        val_metrics: dict,
        w_corr: float,
        w_f1: float,
    ) -> Optional[float]:
        """Compute val_rating_corr_and_f1 composite from validation metrics (binary mode).

        Returns None if neither correlation nor F1 is available.
        """
        if self.classification_mode != "binary":
            return None
        rating_corr = val_metrics.get("rating_corr")
        f1 = val_metrics.get("f1")
        total_w = w_corr + w_f1
        if total_w <= 0:
            return None
        w_corr, w_f1 = w_corr / total_w, w_f1 / total_w
        norm_corr = None
        if rating_corr is not None and not (isinstance(rating_corr, float) and np.isnan(rating_corr)):
            norm_corr = (float(rating_corr) + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        if norm_corr is not None and f1 is not None:
            return w_corr * norm_corr + w_f1 * float(f1)
        if norm_corr is not None:
            return norm_corr
        if f1 is not None:
            return float(f1)
        return None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_epochs: int,
        checkpoint_dir: str = "./checkpoints",
        save_best_only: bool = True,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        monitor_metric: Optional[str] = None,
        hpo_metric_corr_weight: float = 0.5,
        hpo_metric_f1_weight: float = 0.5,
        hpo_metric_ppv_weight: float = 0.75,
        hpo_metric_recall_weight: float = 0.25,
    ) -> dict:
        """Full training loop with optional early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            save_best_only: If True, only saves best model
            early_stopping_patience: Number of epochs to wait for improvement before stopping
                                     (None = no early stopping)
            early_stopping_min_delta: Minimum improvement to count as progress
            monitor_metric: If 'val_rating_corr_and_f1', 'val_ppv', or 'val_ppv_recall', early
                            stopping and best checkpoint use that metric (maximize) instead of val loss.
            hpo_metric_corr_weight: Weight for correlation in composite (when monitor_metric set).
            hpo_metric_f1_weight: Weight for F1 in composite (when monitor_metric set).
            hpo_metric_ppv_weight: Weight for PPV in val_ppv_recall (default 0.75).
            hpo_metric_recall_weight: Weight for recall in val_ppv_recall (default 0.25).

        Returns:
            Dictionary with training history
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        use_composite = (
            monitor_metric == "val_rating_corr_and_f1"
            and self.classification_mode == "binary"
            and early_stopping_patience is not None
            and val_loader is not None
        )
        use_f1_monitor = (
            monitor_metric == "val_f1"
            and self.classification_mode == "binary"
            and early_stopping_patience is not None
            and val_loader is not None
        )
        use_roc_auc_monitor = (
            monitor_metric == "val_roc_auc"
            and self.classification_mode == "binary"
            and early_stopping_patience is not None
            and val_loader is not None
        )
        use_ppv_monitor = (
            monitor_metric == "val_ppv"
            and self.classification_mode == "binary"
            and early_stopping_patience is not None
            and val_loader is not None
        )
        total_pr = hpo_metric_ppv_weight + hpo_metric_recall_weight
        if total_pr > 0:
            w_ppv, w_recall = hpo_metric_ppv_weight / total_pr, hpo_metric_recall_weight / total_pr
        else:
            w_ppv, w_recall = 0.75, 0.25
        use_ppv_recall_monitor = (
            monitor_metric == "val_ppv_recall"
            and self.classification_mode == "binary"
            and early_stopping_patience is not None
            and val_loader is not None
        )
        use_max_metric = use_composite or use_f1_monitor or use_roc_auc_monitor or use_ppv_monitor or use_ppv_recall_monitor

        # Initialize early stopping if enabled
        early_stop = None
        if early_stopping_patience is not None and val_loader is not None:
            early_stop = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                mode="max" if use_max_metric else "min",
                verbose=True,
            )
            if use_composite:
                print(
                    f"Early stopping enabled (monitor=val_rating_corr_and_f1): "
                    f"patience={early_stopping_patience}, min_delta={early_stopping_min_delta}"
                )
            elif use_f1_monitor:
                print(
                    f"Early stopping enabled (monitor=val_f1): "
                    f"patience={early_stopping_patience}, min_delta={early_stopping_min_delta}"
                )
            elif use_roc_auc_monitor:
                print(
                    f"Early stopping enabled (monitor=val_roc_auc): "
                    f"patience={early_stopping_patience}, min_delta={early_stopping_min_delta}"
                )
            elif use_ppv_monitor:
                print(
                    f"Early stopping enabled (monitor=val_ppv): "
                    f"patience={early_stopping_patience}, min_delta={early_stopping_min_delta}"
                )
            elif use_ppv_recall_monitor:
                print(
                    f"Early stopping enabled (monitor=val_ppv_recall, PPV weight={w_ppv:.2f}, recall weight={w_recall:.2f}): "
                    f"patience={early_stopping_patience}, min_delta={early_stopping_min_delta}"
                )
            else:
                print(f"Early stopping enabled: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")

        print(f"Training classifier for up to {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Checkpoint dir: {checkpoint_dir}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.validate(val_loader)

            epoch_time = time.time() - start_time

            # Print metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            gpu_stats = train_metrics.get("gpu_stats", {})
            if gpu_stats:
                print(f"  GPU Util: {gpu_stats.get('gpu_util_avg', 0):.1f}% avg "
                      f"(min={gpu_stats.get('gpu_util_min', 0):.0f}%, max={gpu_stats.get('gpu_util_max', 0):.0f}%)")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                if self.classification_mode == "binary":
                    print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                    p5 = val_metrics.get("precision_at_5")
                    p20 = val_metrics.get("precision_at_20")
                    p5_str = f"{p5:.4f}" if p5 is not None and not (isinstance(p5, float) and np.isnan(p5)) else "n/a"
                    p20_str = f"{p20:.4f}" if p20 is not None and not (isinstance(p20, float) and np.isnan(p20)) else "n/a"
                    print(
                        f"  Val Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, "
                        f"F1: {val_metrics['f1']:.4f}, ROC AUC: {val_metrics.get('roc_auc', 0.5):.4f}, "
                        f"P@5: {p5_str}, P@20: {p20_str}"
                    )
                    if "rating_mse" in val_metrics:
                        rc_str = f"{val_metrics['rating_corr']:.4f}" if "rating_corr" in val_metrics and not np.isnan(val_metrics["rating_corr"]) else "nan"
                        print(f"  Val Rating MSE: {val_metrics['rating_mse']:.4f}, Corr(prob,1-5): {rc_str}")
                else:
                    print(f"  Val MAE: {val_metrics['mae']:.4f}")
                    corr_str = f"{val_metrics['correlation']:.4f}" if not np.isnan(val_metrics['correlation']) else "nan"
                    print(f"  Val Correlation: {corr_str}")
                    if epoch < 3:  # Log variance diagnostics for first few epochs
                        print(f"  Pred StdDev: {val_metrics.get('pred_std', 0):.4f}, Target StdDev: {val_metrics.get('target_std', 0):.4f}")

            # Track best metrics
            if val_metrics:
                if self.classification_mode == "binary":
                    # Binary: track accuracy
                    if val_metrics['accuracy'] > self.best_accuracy:
                        self.best_accuracy = val_metrics['accuracy']
                else:
                    # Regression: track MAE and correlation
                    if val_metrics['mae'] < self.best_mae:
                        self.best_mae = val_metrics['mae']
                    # Only track correlation if not NaN
                    if not np.isnan(val_metrics['correlation']) and val_metrics['correlation'] > self.best_correlation:
                        self.best_correlation = val_metrics['correlation']

            # Save checkpoint
            current_loss = val_metrics['loss'] if val_metrics else train_metrics['loss']
            # When monitoring composite, F1, or ROC AUC, use that for early stop and save; otherwise use loss
            if use_composite and val_metrics:
                monitored_value = self._compute_val_rating_corr_and_f1(
                    val_metrics, hpo_metric_corr_weight, hpo_metric_f1_weight
                )
            elif use_f1_monitor and val_metrics:
                monitored_value = val_metrics.get('f1')
            elif use_roc_auc_monitor and val_metrics:
                monitored_value = val_metrics.get('roc_auc')
            elif use_ppv_monitor and val_metrics:
                monitored_value = val_metrics.get('precision')
            elif use_ppv_recall_monitor and val_metrics:
                prec = val_metrics.get('precision', 0.0) or 0.0
                rec = val_metrics.get('recall', 0.0) or 0.0
                monitored_value = w_ppv * prec + w_recall * rec
            else:
                monitored_value = current_loss

            # Minimum composite/F1 to count as "improvement" (avoids saving collapsed model with F1=0)
            MIN_MAX_METRIC_THRESHOLD = 0.01
            # Bounds to allow saving as "best" in binary mode (avoid both collapses)
            MIN_RECALL_FOR_BEST = 0.05    # avoid predict-all-negative
            MIN_PRECISION_FOR_BEST = 0.10  # avoid predict-all-positive (PPV/precision then low)
            current_recall = val_metrics.get("recall", 0.0) if val_metrics and self.classification_mode == "binary" else 1.0
            current_precision = val_metrics.get("precision", 0.0) if val_metrics and self.classification_mode == "binary" else 1.0
            recall_acceptable = current_recall >= MIN_RECALL_FOR_BEST and current_precision >= MIN_PRECISION_FOR_BEST
            composite_acceptable = (
                (use_composite or use_f1_monitor or use_roc_auc_monitor or use_ppv_monitor or use_ppv_recall_monitor)
                and monitored_value is not None
                and not (isinstance(monitored_value, float) and np.isnan(monitored_value))
                and monitored_value > MIN_MAX_METRIC_THRESHOLD
                and recall_acceptable
            )

            # Check early stopping (only when we have a valid monitored value; for composite, only when above threshold)
            if early_stop is not None and monitored_value is not None:
                if not (isinstance(monitored_value, float) and np.isnan(monitored_value)):
                    if use_composite and not composite_acceptable:
                        pass  # Don't update early_stop when composite is 0 or trivial (avoid "best" = epoch 1 with 0.0)
                    else:
                        if early_stop(monitored_value, epoch):
                            # Early stopping triggered
                            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                            if use_composite:
                                print(f"Best val_rating_corr_and_f1: {early_stop.get_best_score():.6f} at epoch {early_stop.get_best_epoch() + 1}")
                            elif use_f1_monitor:
                                print(f"Best val_f1: {early_stop.get_best_score():.6f} at epoch {early_stop.get_best_epoch() + 1}")
                            elif use_roc_auc_monitor:
                                print(f"Best val_roc_auc: {early_stop.get_best_score():.6f} at epoch {early_stop.get_best_epoch() + 1}")
                            elif use_ppv_monitor:
                                print(f"Best val_ppv: {early_stop.get_best_score():.6f} at epoch {early_stop.get_best_epoch() + 1}")
                            elif use_ppv_recall_monitor:
                                print(f"Best val_ppv_recall: {early_stop.get_best_score():.6f} at epoch {early_stop.get_best_epoch() + 1}")
                            else:
                                print(f"Best validation loss: {early_stop.get_best_score():.6f} at epoch {early_stop.get_best_epoch() + 1}")
                            break

            if save_best_only:
                # Save best model: by composite when acceptable, else by loss (fallback when model is collapsed)
                should_save = False
                if (use_composite or use_f1_monitor or use_roc_auc_monitor or use_ppv_monitor or use_ppv_recall_monitor) and composite_acceptable:
                    should_save = early_stop and early_stop.should_save_checkpoint()
                elif not (use_composite or use_f1_monitor or use_roc_auc_monitor or use_ppv_monitor or use_ppv_recall_monitor):
                    should_save = (early_stop and early_stop.should_save_checkpoint()) or (not early_stop and current_loss < self.best_loss)
                if should_save:
                    self.best_loss = current_loss
                    if val_metrics and "roc_auc" in val_metrics:
                        self.roc_auc_at_best_checkpoint = val_metrics["roc_auc"]
                    if val_metrics and "precision_at_5" in val_metrics:
                        p5 = val_metrics["precision_at_5"]
                        if p5 is not None and not (isinstance(p5, float) and np.isnan(p5)):
                            self.precision_at_5_at_best_checkpoint = p5
                    if val_metrics and "precision_at_20" in val_metrics:
                        p20 = val_metrics["precision_at_20"]
                        if p20 is not None and not (isinstance(p20, float) and np.isnan(p20)):
                            self.precision_at_20_at_best_checkpoint = p20
                    self.best_epoch_saved = epoch + 1
                    self.save_checkpoint(
                        checkpoint_dir / "classifier_best.pt",
                        metrics={
                            "loss": current_loss,
                            "mae": val_metrics.get('mae') if val_metrics else None,
                            "correlation": val_metrics.get('correlation') if val_metrics else None,
                            "epoch": epoch
                        }
                    )
                    if use_composite:
                        print(f"  Saved best model (val_rating_corr_and_f1: {monitored_value:.4f})")
                    elif use_f1_monitor:
                        print(f"  Saved best model (val_f1: {monitored_value:.4f})")
                    elif use_roc_auc_monitor:
                        print(f"  Saved best model (val_roc_auc: {monitored_value:.4f})")
                    elif use_ppv_monitor:
                        print(f"  Saved best model (val_ppv: {monitored_value:.4f})")
                    elif use_ppv_recall_monitor:
                        print(f"  Saved best model (val_ppv_recall: {monitored_value:.4f})")
                    else:
                        print(f"  Saved best model (loss: {current_loss:.4f})")
                elif ((use_composite or use_f1_monitor or use_roc_auc_monitor or use_ppv_monitor or use_ppv_recall_monitor) or not early_stop) and current_loss < self.best_loss:
                    # Fallback: save by loss only if recall is acceptable (never save collapsed model as best)
                    if (use_composite or use_f1_monitor or use_roc_auc_monitor or use_ppv_monitor or use_ppv_recall_monitor) and not recall_acceptable:
                        if current_recall < 0.02:
                            print(f"  Skipping save by loss (recall={current_recall:.2%} < 5%, avoid predict-all-negative)")
                        elif current_precision < MIN_PRECISION_FOR_BEST:
                            print(f"  Skipping save by loss (precision={current_precision:.2%} < {MIN_PRECISION_FOR_BEST:.0%}, avoid predict-all-positive)")
                    else:
                        self.best_loss = current_loss
                        if val_metrics and "roc_auc" in val_metrics:
                            self.roc_auc_at_best_checkpoint = val_metrics["roc_auc"]
                        if val_metrics and "precision_at_5" in val_metrics:
                            p5 = val_metrics["precision_at_5"]
                            if p5 is not None and not (isinstance(p5, float) and np.isnan(p5)):
                                self.precision_at_5_at_best_checkpoint = p5
                        if val_metrics and "precision_at_20" in val_metrics:
                            p20 = val_metrics["precision_at_20"]
                            if p20 is not None and not (isinstance(p20, float) and np.isnan(p20)):
                                self.precision_at_20_at_best_checkpoint = p20
                        self.best_epoch_saved = epoch + 1
                        self.save_checkpoint(
                            checkpoint_dir / "classifier_best.pt",
                            metrics={
                                "loss": current_loss,
                                "mae": val_metrics.get('mae') if val_metrics else None,
                                "correlation": val_metrics.get('correlation') if val_metrics else None,
                                "epoch": epoch
                            }
                        )
                        if use_composite:
                            print(f"  Saved best model by loss (val_rating_corr_and_f1 still below threshold, loss: {current_loss:.4f})")
                        elif use_f1_monitor:
                            print(f"  Saved best model by loss (val_f1 still below threshold, loss: {current_loss:.4f})")
                        elif use_roc_auc_monitor:
                            print(f"  Saved best model by loss (val_roc_auc still below threshold, loss: {current_loss:.4f})")
                        elif use_ppv_monitor:
                            print(f"  Saved best model by loss (val_ppv still below threshold, loss: {current_loss:.4f})")
                        elif use_ppv_recall_monitor:
                            print(f"  Saved best model by loss (val_ppv_recall still below threshold, loss: {current_loss:.4f})")
                        else:
                            print(f"  Saved best model (loss: {current_loss:.4f})")
            else:
                self.save_checkpoint(
                    checkpoint_dir / f"classifier_epoch_{epoch + 1}.pt",
                    metrics={"loss": current_loss, "epoch": epoch}
                )

        # Save final model
        self.save_checkpoint(
            checkpoint_dir / "classifier_final.pt",
            metrics={"epoch": num_epochs}
        )

        # ROC AUC and epoch of the saved best model (early-stopping checkpoint); fallback to max if never set
        if self.roc_auc_at_best_checkpoint is not None:
            self.history["roc_auc"] = self.roc_auc_at_best_checkpoint
        elif self.history.get("val_roc_auc"):
            self.history["roc_auc"] = max(self.history["val_roc_auc"])
        else:
            self.history["roc_auc"] = None
        if self.precision_at_5_at_best_checkpoint is not None:
            self.history["val_precision_at_5_saved"] = self.precision_at_5_at_best_checkpoint
        else:
            p5_hist = [
                x for x in self.history.get("val_precision_at_5", [])
                if x is not None and not (isinstance(x, float) and np.isnan(x))
            ]
            self.history["val_precision_at_5_saved"] = max(p5_hist) if p5_hist else None
        if self.precision_at_20_at_best_checkpoint is not None:
            self.history["val_precision_at_20_saved"] = self.precision_at_20_at_best_checkpoint
        else:
            p20_hist = [
                x for x in self.history.get("val_precision_at_20", [])
                if x is not None and not (isinstance(x, float) and np.isnan(x))
            ]
            self.history["val_precision_at_20_saved"] = max(p20_hist) if p20_hist else None
        if self.best_epoch_saved is not None:
            self.history["best_epoch"] = self.best_epoch_saved

        # Log final summary metrics to MLflow (unified names; use tag "stage" = "classifier" to identify)
        if self.tracker is not None:
            self.tracker.set_tag("stage", "classifier")
            epochs_completed = len(self.history['train_loss'])
            self.tracker.log_metric('epochs_completed', epochs_completed)
            self.tracker.log_metric('val_loss', self.best_loss)
            if self.classification_mode == "binary":
                self.tracker.log_metric('val_accuracy', self.best_accuracy)
                self.tracker.log_metric('val_precision', self.best_precision)
                self.tracker.log_metric('val_ppv', self.best_precision)  # PPV = precision
                self.tracker.log_metric('val_recall', self.best_recall)
                # Log ROC AUC of the saved checkpoint (early stopping)
                roc_auc_saved = self.roc_auc_at_best_checkpoint
                if roc_auc_saved is None and self.history.get("val_roc_auc"):
                    roc_auc_saved = max(self.history["val_roc_auc"])
                if roc_auc_saved is not None:
                    self.tracker.log_metric('roc_auc', roc_auc_saved)
                p5_saved = self.precision_at_5_at_best_checkpoint
                if p5_saved is None and self.history.get("val_precision_at_5_saved") is not None:
                    p5_saved = self.history["val_precision_at_5_saved"]
                if p5_saved is not None:
                    self.tracker.log_metric('val_precision_at_5', p5_saved)
                p20_saved = self.precision_at_20_at_best_checkpoint
                if p20_saved is None and self.history.get("val_precision_at_20_saved") is not None:
                    p20_saved = self.history["val_precision_at_20_saved"]
                if p20_saved is not None:
                    self.tracker.log_metric('val_precision_at_20', p20_saved)
            else:
                self.tracker.log_metric('val_mae', self.best_mae)
                self.tracker.log_metric('val_correlation', self.best_correlation)
            if self.history['train_loss']:
                self.tracker.log_metric('final_train_loss', self.history['train_loss'][-1])
            if self.history['val_loss']:
                self.tracker.log_metric('final_val_loss', self.history['val_loss'][-1])

        return self.history

    def save_checkpoint(self, path: Path, metrics: dict = None):
        """Save model checkpoint.

        Includes version information for compatibility validation:
        - encoder_version: The encoder version this classifier was trained with
        - classifier_version: This classifier's version

        Args:
            path: Path to save checkpoint
            metrics: Optional metrics to save with checkpoint
        """
        checkpoint = {
            "model_state_dict": self.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "history": self.history,
            "encoder_version": self.encoder_version,  # Required for compatibility check
            "classifier_version": self.classifier_version,
            "chunk_aggregation": self.chunk_aggregation,
            "hpo_mlflow_run_id": getattr(
                self, "hpo_mlflow_run_id", None
            ),  # HPO run params came from (traceability)
            "hpo_mlflow_run_name": getattr(
                self, "hpo_mlflow_run_name", None
            ),  # HPO parent run display name
            "use_genre": getattr(self.classifier, "use_genre", False),
            "embedding_dim": getattr(self.classifier, "embedding_dim", 0),
            "hidden_dims": getattr(self.classifier, "hidden_dims", None),
            "use_batch_norm": getattr(self.classifier, "use_batch_norm", False),
            "use_residual": getattr(self.classifier, "use_residual", False),
            "classifier_type": type(self.classifier).__name__,
        }
        if checkpoint["classifier_type"] == "AttentionRatingClassifier":
            checkpoint["d_model"] = getattr(self.classifier, "d_model", 512)
            checkpoint["num_heads"] = getattr(self.classifier, "num_heads", 4)
            pos_embed = getattr(self.classifier, "pos_embed", None)
            checkpoint["max_chunks"] = pos_embed.shape[1] if pos_embed is not None else 16
            checkpoint["use_pos_encoding"] = getattr(self.classifier, "use_pos_encoding", True)

        if metrics:
            checkpoint["metrics"] = metrics
        if self.genre_centroids is not None:
            checkpoint["genre_centroids"] = self.genre_centroids

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path, validate_encoder_version: str = None):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
            validate_encoder_version: If provided, validates that the checkpoint's
                                      encoder_version matches. Raises ValueError on mismatch.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Extract version info
        checkpoint_encoder_version = checkpoint.get("encoder_version", "unknown")
        checkpoint_classifier_version = checkpoint.get("classifier_version", "unknown")

        # Validate encoder version compatibility if requested
        if validate_encoder_version is not None:
            if checkpoint_encoder_version != validate_encoder_version:
                raise ValueError(
                    f"Classifier/Encoder version mismatch!\n"
                    f"  Classifier was trained with encoder version: {checkpoint_encoder_version}\n"
                    f"  Current encoder version: {validate_encoder_version}\n"
                    f"  You must retrain the classifier with the new encoder.\n"
                    f"  Run: ./run_music_pipeline.sh classifier"
                )

        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.history = checkpoint.get("history", {
            "train_loss": [],
            "val_loss": [],
            "val_mae": []
        })
        self.encoder_version = checkpoint_encoder_version
        self.classifier_version = checkpoint_classifier_version
        self.chunk_aggregation = checkpoint.get("chunk_aggregation", "mean")

        print(f"Loaded classifier checkpoint from epoch {self.current_epoch}")
        print(f"  Classifier version: {self.classifier_version}")
        print(f"  Trained with encoder version: {self.encoder_version}")

    def predict(
        self,
        data_loader: DataLoader
    ) -> tuple[list[float], list[str]]:
        """Generate predictions for all songs.

        Args:
            data_loader: Data loader with embeddings

        Returns:
            predictions: List of predicted ratings/probabilities
            filenames: List of corresponding filenames
        """
        self.classifier.eval()
        predictions = []
        filenames = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                embeddings = batch["embedding"].to(self.device)
                batch_filenames = batch["filename"]
                genre = batch.get("genre")
                if genre is not None:
                    genre = genre.to(self.device)

                # Predict: (B, C, D) -> one rating per song (attention) or per-chunk then aggregate
                if embeddings.dim() == 3 and getattr(self.classifier, "handles_chunk_sequence", False):
                    preds = self.classifier(embeddings, genre)
                elif embeddings.dim() == 3:
                    B, C, D = embeddings.shape
                    emb_flat = embeddings.view(B * C, D)
                    if genre is not None:
                        genre_flat = genre.unsqueeze(1).expand(-1, C, -1).reshape(B * C, genre.size(-1))
                        pred_chunks = self.classifier(emb_flat, genre_flat)
                    else:
                        pred_chunks = self.classifier(emb_flat)
                    preds = self._aggregate_chunk_predictions(pred_chunks, B, C)
                else:
                    preds = self.classifier(embeddings, genre) if genre is not None else self.classifier(embeddings)

                # For binary mode, apply sigmoid to get probabilities
                if self.classification_mode == "binary":
                    preds = torch.sigmoid(preds)

                # Store results
                predictions.extend(preds.cpu().numpy().tolist())
                filenames.extend(batch_filenames)

        return predictions, filenames

    def evaluate(self, data_loader: DataLoader) -> dict:
        """Evaluate model on a dataset.

        Args:
            data_loader: Data loader with embeddings and ratings

        Returns:
            Dictionary with evaluation metrics
        """
        self.classifier.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                embeddings = batch["embedding"].to(self.device)
                ratings = batch["rating"].to(self.device)
                genre = batch.get("genre")
                if genre is not None:
                    genre = genre.to(self.device)

                # Predict: (B, C, D) -> one per song (attention) or per-chunk then aggregate
                if embeddings.dim() == 3 and getattr(self.classifier, "handles_chunk_sequence", False):
                    predictions = self.classifier(embeddings, genre)
                elif embeddings.dim() == 3:
                    B, C, D = embeddings.shape
                    emb_flat = embeddings.view(B * C, D)
                    if genre is not None:
                        genre_flat = genre.unsqueeze(1).expand(-1, C, -1).reshape(B * C, genre.size(-1))
                        pred_chunks = self.classifier(emb_flat, genre_flat)
                    else:
                        pred_chunks = self.classifier(emb_flat)
                    predictions = self._aggregate_chunk_predictions(pred_chunks, B, C)
                else:
                    predictions = self.classifier(embeddings, genre) if genre is not None else self.classifier(embeddings)

                # Compute loss
                loss = self.loss_fn(predictions, ratings)
                total_loss += loss.item()
                num_batches += 1

                # Store for metrics
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_targets.extend(ratings.cpu().numpy().tolist())

        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        mse = np.mean((all_predictions - all_targets) ** 2)
        mae = np.mean(np.abs(all_predictions - all_targets))

        # Compute correlation with proper NaN handling
        pred_std = np.std(all_predictions)
        target_std = np.std(all_targets)

        if pred_std < 1e-8 or target_std < 1e-8:
            correlation = np.nan
        else:
            corr_matrix = np.corrcoef(all_predictions, all_targets)
            correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else np.nan

        # RMSE
        rmse = np.sqrt(mse)

        return {
            "loss": total_loss / num_batches,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation,
            "num_samples": len(all_predictions),
            "pred_std": pred_std,
            "target_std": target_std
        }
