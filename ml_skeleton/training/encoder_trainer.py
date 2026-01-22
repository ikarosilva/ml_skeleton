"""Encoder training orchestration.

Handles Stage 1: Train audio encoder to extract embeddings.
Supports both supervised and self-supervised training modes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import time

from ..music.embedding_store import EmbeddingStore
from ..music.audio_loader import load_audio_file


class EncoderTrainer:
    """Trainer for audio encoder models.

    Supports:
    - Supervised training with rating labels
    - Self-supervised training with contrastive loss
    - Multi-task training with album classification
    - Embedding extraction and storage

    Args:
        encoder: Audio encoder model (conforms to AudioEncoder protocol)
        device: Device to train on ('cuda' or 'cpu')
        loss_fn: Loss function module
        optimizer: PyTorch optimizer
        embedding_store: EmbeddingStore for saving embeddings
        model_version: Version identifier for embeddings
    """

    def __init__(
        self,
        encoder: nn.Module,
        device: str,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        embedding_store: Optional[EmbeddingStore] = None,
        model_version: str = "default"
    ):
        self.encoder = encoder.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.embedding_store = embedding_store
        self.model_version = model_version

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {
            "train_loss": [],
            "val_loss": []
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        use_multi_task: bool = False
    ) -> dict:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            use_multi_task: If True, expects album labels and uses multi-task loss

        Returns:
            Dictionary with training metrics
        """
        self.encoder.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in pbar:
            # Move data to device
            audio = batch["audio"].to(self.device)
            ratings = batch["rating"].to(self.device)

            # Forward pass
            if use_multi_task:
                # Multi-task mode: predict ratings and albums
                embeddings, album_logits = self.encoder(
                    audio,
                    return_album_logits=True
                )
                album_labels = batch["albums"]  # List of lists

                # Compute multi-task loss
                loss, loss_dict = self.loss_fn(
                    embeddings,
                    ratings,
                    album_logits,
                    album_labels
                )
            else:
                # Simple mode: just embeddings
                embeddings = self.encoder(audio)
                loss = self.loss_fn(embeddings, ratings)
                loss_dict = {"total": loss.item()}

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

    def validate(
        self,
        val_loader: DataLoader,
        use_multi_task: bool = False
    ) -> dict:
        """Validate model.

        Args:
            val_loader: Validation data loader
            use_multi_task: If True, uses multi-task loss

        Returns:
            Dictionary with validation metrics
        """
        self.encoder.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                audio = batch["audio"].to(self.device)
                ratings = batch["rating"].to(self.device)

                if use_multi_task:
                    embeddings, album_logits = self.encoder(
                        audio,
                        return_album_logits=True
                    )
                    album_labels = batch["albums"]
                    loss, _ = self.loss_fn(
                        embeddings,
                        ratings,
                        album_logits,
                        album_labels
                    )
                else:
                    embeddings = self.encoder(audio)
                    loss = self.loss_fn(embeddings, ratings)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        self.history["val_loss"].append(avg_loss)

        return {
            "loss": avg_loss,
            "num_batches": num_batches
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_epochs: int,
        checkpoint_dir: str = "./checkpoints",
        use_multi_task: bool = False,
        save_best_only: bool = True
    ) -> dict:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            use_multi_task: If True, uses multi-task loss
            save_best_only: If True, only saves best model

        Returns:
            Dictionary with training history
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Training encoder for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Checkpoint dir: {checkpoint_dir}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, use_multi_task)

            # Validate
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.validate(val_loader, use_multi_task)

            epoch_time = time.time() - start_time

            # Print metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['loss']:.4f}")

            # Save checkpoint
            current_loss = val_metrics['loss'] if val_metrics else train_metrics['loss']

            if save_best_only:
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.save_checkpoint(
                        checkpoint_dir / "encoder_best.pt",
                        metrics={"loss": current_loss, "epoch": epoch}
                    )
                    print(f"  Saved best model (loss: {current_loss:.4f})")
            else:
                self.save_checkpoint(
                    checkpoint_dir / f"encoder_epoch_{epoch + 1}.pt",
                    metrics={"loss": current_loss, "epoch": epoch}
                )

        # Save final model
        self.save_checkpoint(
            checkpoint_dir / "encoder_final.pt",
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
            "model_state_dict": self.encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "history": self.history,
            "model_version": self.model_version
        }

        if metrics:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
        self.model_version = checkpoint.get("model_version", "default")

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def extract_embeddings(
        self,
        data_loader: DataLoader,
        save_to_store: bool = True
    ) -> dict[str, torch.Tensor]:
        """Extract embeddings for all songs in dataset.

        Args:
            data_loader: Data loader with songs
            save_to_store: If True, save to embedding store

        Returns:
            Dictionary mapping filename -> embedding tensor
        """
        self.encoder.eval()
        embeddings_dict = {}

        print(f"Extracting embeddings (model_version: {self.model_version})")

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting embeddings"):
                audio = batch["audio"].to(self.device)
                filenames = batch["filename"]

                # Get embeddings
                if hasattr(self.encoder, 'base_encoder'):
                    # Multi-task encoder
                    embeddings, _ = self.encoder(audio, return_album_logits=False)
                else:
                    # Simple encoder
                    embeddings = self.encoder(audio)

                # Store embeddings
                embeddings_cpu = embeddings.cpu()
                for i, filename in enumerate(filenames):
                    embeddings_dict[filename] = embeddings_cpu[i]

                    # Save to store if requested
                    if save_to_store and self.embedding_store:
                        self.embedding_store.store_embedding(
                            filename,
                            embeddings_cpu[i].numpy(),
                            self.model_version
                        )

        print(f"Extracted {len(embeddings_dict)} embeddings")

        return embeddings_dict

    def extract_embeddings_batch(
        self,
        songs: list,
        batch_size: int = 64,
        sample_rate: int = 22050,
        duration: float = 30.0
    ):
        """Extract embeddings in batches without DataLoader.

        Useful for quick extraction when you have a list of songs.

        Args:
            songs: List of Song objects
            batch_size: Batch size for extraction
            sample_rate: Audio sample rate
            duration: Audio duration
        """
        self.encoder.eval()
        embeddings_list = []

        print(f"Extracting embeddings for {len(songs)} songs")

        with torch.no_grad():
            for i in tqdm(range(0, len(songs), batch_size)):
                batch_songs = songs[i:i + batch_size]

                # Load audio for batch
                audio_tensors = []
                valid_filenames = []

                for song in batch_songs:
                    audio = load_audio_file(
                        song.filename,
                        sample_rate=sample_rate,
                        duration=duration,
                        center_crop=True
                    )

                    if audio is not None:
                        audio_tensors.append(audio)
                        valid_filenames.append(song.filename)

                if len(audio_tensors) == 0:
                    continue

                # Stack and move to device
                audio_batch = torch.stack(audio_tensors).to(self.device)

                # Extract embeddings
                if hasattr(self.encoder, 'base_encoder'):
                    embeddings, _ = self.encoder(audio_batch, return_album_logits=False)
                else:
                    embeddings = self.encoder(audio_batch)

                # Store embeddings
                embeddings_cpu = embeddings.cpu().numpy()
                for filename, embedding in zip(valid_filenames, embeddings_cpu):
                    if self.embedding_store:
                        self.embedding_store.store_embedding(
                            filename,
                            embedding,
                            self.model_version
                        )
                    embeddings_list.append((filename, embedding))

        print(f"Extracted {len(embeddings_list)} embeddings")
        return embeddings_list
