"""Classifier training orchestration.

Handles Stage 2: Train rating classifier on pre-extracted embeddings.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import time
import numpy as np


class ClassifierTrainer:
    """Trainer for rating classifier models.

    Trains on pre-extracted embeddings from Stage 1.
    Predicts continuous ratings in [0, 1] range.

    Args:
        classifier: Rating classifier model (conforms to RatingClassifier protocol)
        device: Device to train on ('cuda' or 'cpu')
        loss_fn: Loss function (typically MSE)
        optimizer: PyTorch optimizer
    """

    def __init__(
        self,
        classifier: nn.Module,
        device: str,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer
    ):
        self.classifier = classifier.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
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
        save_best_only: bool = True
    ) -> dict:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            save_best_only: If True, only saves best model

        Returns:
            Dictionary with training history
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Training classifier for {num_epochs} epochs")
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

            # Save checkpoint
            current_loss = val_metrics['loss'] if val_metrics else train_metrics['loss']

            if save_best_only:
                if current_loss < self.best_loss:
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

        return self.history

    def save_checkpoint(self, path: Path, metrics: dict = None):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            metrics: Optional metrics to save with checkpoint
        """
        checkpoint = {
            "model_state_dict": self.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "history": self.history
        }

        if metrics:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.history = checkpoint.get("history", {
            "train_loss": [],
            "val_loss": [],
            "val_mae": []
        })

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

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
