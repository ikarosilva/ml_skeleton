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
        classifier_version: str = "v1"
    ):
        self.classifier = classifier.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.tracker = tracker  # MLflow tracker for logging learning curves

        # Version tracking for compatibility validation
        self.encoder_version = encoder_version  # Encoder version this classifier was trained with
        self.classifier_version = classifier_version  # This classifier's version

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_mae = float('inf')
        self.best_correlation = 0.0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": []
        }

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

            # Forward pass
            predictions = self.classifier(embeddings)

            # Compute loss
            loss = self.loss_fn(predictions.squeeze(), ratings)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        self.history["train_loss"].append(avg_loss)

        return {
            "loss": avg_loss,
            "num_batches": num_batches
        }

    def validate(self, val_loader: DataLoader) -> dict:
        """Validate model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics (loss, MAE)
        """
        self.classifier.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                embeddings = batch["embedding"].to(self.device)
                ratings = batch["rating"].to(self.device)

                # Forward pass
                predictions = self.classifier(embeddings)

                # Compute loss
                loss = self.loss_fn(predictions.squeeze(), ratings)

                # Compute MAE
                mae = torch.abs(predictions.squeeze() - ratings).mean()

                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1

                # Store for correlation analysis
                all_predictions.extend(predictions.squeeze().cpu().numpy().tolist())
                all_targets.extend(ratings.cpu().numpy().tolist())

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches

        self.history["val_loss"].append(avg_loss)
        self.history["val_mae"].append(avg_mae)

        # Compute correlation
        correlation = np.corrcoef(all_predictions, all_targets)[0, 1]

        return {
            "loss": avg_loss,
            "mae": avg_mae,
            "correlation": correlation,
            "num_batches": num_batches
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_epochs: int,
        checkpoint_dir: str = "./checkpoints",
        save_best_only: bool = True,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0
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

        Returns:
            Dictionary with training history
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize early stopping if enabled
        early_stop = None
        if early_stopping_patience is not None and val_loader is not None:
            early_stop = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                mode='min',
                verbose=True
            )
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
            if val_metrics:
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
                print(f"  Val MAE: {val_metrics['mae']:.4f}")
                print(f"  Val Correlation: {val_metrics['correlation']:.4f}")

            # Log metrics to MLflow for learning curves
            if self.tracker is not None:
                self.tracker.log_metric('classifier/train_loss', train_metrics['loss'], step=epoch)
                if val_metrics:
                    self.tracker.log_metric('classifier/val_loss', val_metrics['loss'], step=epoch)
                    self.tracker.log_metric('classifier/val_mae', val_metrics['mae'], step=epoch)
                    self.tracker.log_metric('classifier/val_correlation', val_metrics['correlation'], step=epoch)
                self.tracker.log_metric('classifier/epoch_time', epoch_time, step=epoch)

            # Track best metrics
            if val_metrics:
                if val_metrics['mae'] < self.best_mae:
                    self.best_mae = val_metrics['mae']
                if val_metrics['correlation'] > self.best_correlation:
                    self.best_correlation = val_metrics['correlation']

            # Save checkpoint
            current_loss = val_metrics['loss'] if val_metrics else train_metrics['loss']

            # Check early stopping
            if early_stop is not None:
                if early_stop(current_loss, epoch):
                    # Early stopping triggered
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    print(f"Best validation loss: {early_stop.get_best_score():.6f} at epoch {early_stop.get_best_epoch() + 1}")
                    break

            if save_best_only:
                # Save only if this is the best model (either by early stopping or manual check)
                if early_stop and early_stop.should_save_checkpoint():
                    self.best_loss = current_loss
                    self.save_checkpoint(
                        checkpoint_dir / "classifier_best.pt",
                        metrics={
                            "loss": current_loss,
                            "mae": val_metrics.get('mae') if val_metrics else None,
                            "correlation": val_metrics.get('correlation') if val_metrics else None,
                            "epoch": epoch
                        }
                    )
                    print(f"  Saved best model (loss: {current_loss:.4f})")
                elif not early_stop and current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.save_checkpoint(
                        checkpoint_dir / "classifier_best.pt",
                        metrics={
                            "loss": current_loss,
                            "mae": val_metrics.get('mae') if val_metrics else None,
                            "correlation": val_metrics.get('correlation') if val_metrics else None,
                            "epoch": epoch
                        }
                    )
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

        # Log final summary metrics to MLflow
        if self.tracker is not None:
            epochs_completed = len(self.history['train_loss'])
            self.tracker.log_metric('classifier/epochs_completed', epochs_completed)
            self.tracker.log_metric('classifier/best_val_loss', self.best_loss)
            self.tracker.log_metric('classifier/best_val_mae', self.best_mae)
            self.tracker.log_metric('classifier/best_val_correlation', self.best_correlation)
            if self.history['train_loss']:
                self.tracker.log_metric('classifier/final_train_loss', self.history['train_loss'][-1])
            if self.history['val_loss']:
                self.tracker.log_metric('classifier/final_val_loss', self.history['val_loss'][-1])

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
            "classifier_version": self.classifier_version
        }

        if metrics:
            checkpoint["metrics"] = metrics

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
            predictions: List of predicted ratings
            filenames: List of corresponding filenames
        """
        self.classifier.eval()
        predictions = []
        filenames = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                embeddings = batch["embedding"].to(self.device)
                batch_filenames = batch["filename"]

                # Predict
                preds = self.classifier(embeddings)

                # Store results
                predictions.extend(preds.squeeze().cpu().numpy().tolist())
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

                # Predict
                predictions = self.classifier(embeddings)

                # Compute loss
                loss = self.loss_fn(predictions.squeeze(), ratings)
                total_loss += loss.item()
                num_batches += 1

                # Store for metrics
                all_predictions.extend(predictions.squeeze().cpu().numpy().tolist())
                all_targets.extend(ratings.cpu().numpy().tolist())

        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        mse = np.mean((all_predictions - all_targets) ** 2)
        mae = np.mean(np.abs(all_predictions - all_targets))
        correlation = np.corrcoef(all_predictions, all_targets)[0, 1]

        # RMSE
        rmse = np.sqrt(mse)

        return {
            "loss": total_loss / num_batches,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "correlation": correlation,
            "num_samples": len(all_predictions)
        }
