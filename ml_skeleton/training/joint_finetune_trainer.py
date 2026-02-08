"""Joint fine-tune trainer: unfreeze encoder + classifier and train on audio → rating.

Used after Stage 1 (encoder) and Stage 2 (classifier): load both models,
unfreeze all parameters, and train on (audio, rating) with two learning rates
(typically lower for encoder, higher for classifier) to push accuracy further.
"""

import time
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.early_stopping import EarlyStopping
from ..music.losses import BinaryRatingLoss


def _encoder_embedding(encoder: nn.Module, audio: torch.Tensor) -> torch.Tensor:
    """Get embedding from encoder; supports dict output (e.g. MoCo) or tensor."""
    out = encoder(audio)
    if isinstance(out, dict):
        return out["embedding"]
    return out


class JointFinetuneTrainer:
    """Trainer for joint encoder + classifier fine-tuning on audio → rating.

    Expects data loader batches with:
    - "audio": (B, T) waveform
    - "rating": (B,) target
    - "filename": list of str (for optional genre lookup)

    Optional: genre_dict mapping filename -> (7,) numpy for use_genre classifier.
    """

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        device: str,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        encoder_version: str = "v1",
        classifier_version: str = "v1",
        classification_mode: str = "regression",
        use_genre: bool = False,
        genre_dict: Optional[dict] = None,
        genre_centroids: Optional[np.ndarray] = None,
        tracker: Optional[Any] = None,
    ):
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.encoder_version = encoder_version
        self.classifier_version = classifier_version
        self.classification_mode = classification_mode
        self.use_genre = use_genre
        self.genre_dict = genre_dict or {}
        self.genre_centroids = genre_centroids
        self.tracker = tracker

        self.current_epoch = 0
        self.best_loss = float("inf")
        self.best_mae = float("inf")
        self.best_accuracy = 0.0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
            "val_accuracy": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> dict:
        self.encoder.train()
        self.classifier.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in pbar:
            audio = batch["audio"].to(self.device)
            ratings = batch["rating"].to(self.device)
            filenames = batch["filename"]

            # Embedding from encoder (gradients flow)
            emb = _encoder_embedding(self.encoder, audio)

            if self.use_genre and self.genre_dict:
                genre_list = [
                    self.genre_dict.get(f, np.zeros(7, dtype=np.float32))
                    for f in filenames
                ]
                genre = torch.tensor(
                    np.stack(genre_list), device=self.device, dtype=torch.float32
                )
                predictions = self.classifier(emb, genre)
            else:
                predictions = self.classifier(emb)

            loss = self.loss_fn(predictions.squeeze(-1) if predictions.dim() > 1 else predictions, ratings)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        self.history["train_loss"].append(avg_loss)
        return {"loss": avg_loss, "num_batches": num_batches}

    def validate(self, val_loader: DataLoader) -> dict:
        self.encoder.eval()
        self.classifier.eval()
        total_loss = 0.0
        total_mae = 0.0
        correct = 0
        total = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                audio = batch["audio"].to(self.device)
                ratings = batch["rating"].to(self.device)
                filenames = batch["filename"]

                emb = _encoder_embedding(self.encoder, audio)
                if self.use_genre and self.genre_dict:
                    genre_list = [
                        self.genre_dict.get(f, np.zeros(7, dtype=np.float32))
                        for f in filenames
                    ]
                    genre = torch.tensor(
                        np.stack(genre_list), device=self.device, dtype=torch.float32
                    )
                    predictions = self.classifier(emb, genre)
                else:
                    predictions = self.classifier(emb)

                if predictions.dim() > 1:
                    predictions = predictions.squeeze(-1)
                # Unweighted BCE for validation so val_loss does not reward collapse
                if isinstance(self.loss_fn, BinaryRatingLoss):
                    loss = self.loss_fn(predictions, ratings, validation=True)
                else:
                    loss = self.loss_fn(predictions, ratings)
                total_loss += loss.item()
                num_batches += 1

                if self.classification_mode == "binary":
                    probs = torch.sigmoid(predictions)
                    pred_labels = (probs > 0.5).float()
                    target_labels = ratings
                    correct += (pred_labels == target_labels).sum().item()
                    total += target_labels.size(0)
                else:
                    total_mae += torch.abs(predictions - ratings).mean().item()

        avg_loss = total_loss / num_batches
        self.history["val_loss"].append(avg_loss)
        if self.classification_mode == "binary":
            accuracy = correct / total if total > 0 else 0.0
            self.history["val_accuracy"].append(accuracy)
            return {
                "loss": avg_loss,
                "accuracy": accuracy,
                "mae": 1.0 - accuracy,
                "num_batches": num_batches,
            }
        else:
            avg_mae = total_mae / num_batches
            self.history["val_mae"].append(avg_mae)
            return {"loss": avg_loss, "mae": avg_mae, "num_batches": num_batches}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_epochs: int,
        checkpoint_dir: str,
        save_best_only: bool = True,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        verbose: bool = True,
    ) -> dict:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        early_stop = None
        if early_stopping_patience is not None and val_loader is not None:
            early_stop = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                mode="min",
                verbose=verbose,
            )
            if verbose:
                print(
                    f"Early stopping: patience={early_stopping_patience}, "
                    f"min_delta={early_stopping_min_delta}"
                )

        if verbose:
            print(f"Joint fine-tune for up to {num_epochs} epochs")
            print(f"Device: {self.device}")
            print(f"Checkpoint dir: {checkpoint_dir}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.validate(val_loader)

            epoch_time = time.time() - start_time
            if verbose:
                print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                if val_metrics:
                    print(f"  Val Loss: {val_metrics['loss']:.4f}")
                    if self.classification_mode == "binary":
                        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                    else:
                        print(f"  Val MAE: {val_metrics['mae']:.4f}")

            current_loss = val_metrics["loss"] if val_metrics else train_metrics["loss"]
            if val_metrics:
                if self.classification_mode == "binary":
                    if val_metrics["accuracy"] > self.best_accuracy:
                        self.best_accuracy = val_metrics["accuracy"]
                else:
                    if val_metrics["mae"] < self.best_mae:
                        self.best_mae = val_metrics["mae"]
            improved = current_loss < self.best_loss
            if improved:
                self.best_loss = current_loss

            if early_stop is not None and early_stop(current_loss, epoch):
                if verbose:
                    print(
                        f"\nEarly stopping at epoch {epoch + 1}; "
                        f"best val loss: {early_stop.get_best_score():.6f}"
                    )
                break

            if save_best_only and val_metrics:
                if early_stop and early_stop.should_save_checkpoint():
                    self._save_checkpoints(checkpoint_dir, val_metrics, epoch)
                    if verbose:
                        print(f"  Saved best model (loss: {current_loss:.4f})")
                elif not early_stop and improved:
                    self._save_checkpoints(checkpoint_dir, val_metrics, epoch)
                    if verbose:
                        print(f"  Saved best model (loss: {current_loss:.4f})")

        return self.history

    def _save_checkpoints(
        self, checkpoint_dir: Path, val_metrics: dict, epoch: int
    ) -> None:
        """Save encoder and classifier checkpoints for deployment."""
        encoder_path = checkpoint_dir / "encoder_best.pt"
        classifier_path = checkpoint_dir / "classifier_best.pt"

        # Encoder: same format as EncoderTrainer (model_state_dict, etc.)
        encoder_ckpt = {
            "model_state_dict": self.encoder.state_dict(),
            "encoder_version": self.encoder_version,
            "model_version": self.encoder_version,
        }
        if hasattr(self.encoder, "sample_rate"):
            encoder_ckpt["sample_rate"] = self.encoder.sample_rate
        torch.save(encoder_ckpt, encoder_path)

        # Classifier: same format as ClassifierTrainer.save_checkpoint
        classifier_state = self.classifier.state_dict()
        embedding_dim = getattr(self.classifier, "embedding_dim", None)
        if embedding_dim is None and "mlp.0.weight" in classifier_state:
            # input dim may be embedding_dim + NUM_GENRES when use_genre
            embedding_dim = int(classifier_state["mlp.0.weight"].shape[1])
            if self.use_genre:
                from ..music.genre_mapper import NUM_GENRES
                embedding_dim -= NUM_GENRES
        hidden_dims = getattr(self.classifier, "hidden_dims", [])
        if not hidden_dims and "mlp.0.weight" in classifier_state:
            # Infer from state dict (Linear, ReLU, Dropout per block)
            idx = 0
            while f"mlp.{idx}.weight" in classifier_state:
                out = int(classifier_state[f"mlp.{idx}.weight"].shape[0])
                if out > 1:
                    hidden_dims.append(out)
                idx += 3
        classifier_ckpt = {
            "model_state_dict": classifier_state,
            "encoder_version": self.encoder_version,
            "classifier_version": self.classifier_version,
            "embedding_dim": embedding_dim,
            "hidden_dims": hidden_dims,
            "epoch": epoch,
            "use_genre": self.use_genre,
        }
        if self.use_genre and self.genre_centroids is not None:
            classifier_ckpt["genre_centroids"] = self.genre_centroids
        torch.save(classifier_ckpt, classifier_path)
