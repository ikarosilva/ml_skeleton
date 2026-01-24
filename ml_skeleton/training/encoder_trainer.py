"""Encoder training orchestration.

Handles Stage 1: Train audio encoder to extract embeddings.
Supports both supervised and self-supervised training modes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm
import time

from ..music.embedding_store import EmbeddingStore
from ..music.audio_loader import load_audio_file
from ..utils.early_stopping import EarlyStopping
from ..utils.gpu import GPUMonitor


class EncoderTrainer:
    """Trainer for audio encoder models.

    Supports:
    - Supervised training with rating labels
    - Self-supervised training with contrastive loss
    - Multi-task training with album classification
    - Embedding extraction and storage
    - MLflow metric logging for learning curves

    Args:
        encoder: Audio encoder model (conforms to AudioEncoder protocol)
        device: Device to train on ('cuda' or 'cpu')
        loss_fn: Loss function module
        optimizer: PyTorch optimizer
        embedding_store: EmbeddingStore for saving embeddings
        model_version: Version identifier for embeddings
        tracker: Optional MLflow tracker (ExplrTracker) for logging metrics
    """

    def __init__(
        self,
        encoder: nn.Module,
        device: str,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        embedding_store: Optional[EmbeddingStore] = None,
        model_version: str = "default",
        tracker: Optional[Any] = None
    ):
        self.encoder = encoder.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.embedding_store = embedding_store
        self.model_version = model_version
        self.tracker = tracker  # MLflow tracker for logging learning curves

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {
            "train_loss": [],
            "val_loss": []
        }

        # GPU monitoring (samples utilization during training)
        self.gpu_monitor = GPUMonitor() if device == "cuda" else None

    def train_epoch(
        self,
        train_loader: DataLoader,
        use_multi_task: bool = False,
        use_augmentation: bool = False,
        use_simsiam: bool = False
    ) -> dict:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            use_multi_task: If True, expects album labels and uses multi-task loss
            use_augmentation: If True, expects dual audio views for contrastive learning
            use_simsiam: If True, uses SimSiam training loop with spectrogram views

        Returns:
            Dictionary with training metrics
        """
        self.encoder.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in pbar:
            # SimSiam mode (spectrogram-based dual views)
            if use_simsiam and "view1" in batch:
                view1 = batch["view1"].to(self.device)
                view2 = batch["view2"].to(self.device)

                # Forward through SimSiam encoder
                p1, p2, z1, z2 = self.encoder.forward_simsiam(view1, view2)

                # SimSiam loss
                loss = self.loss_fn(p1, p2, z1, z2)

                loss_dict = {"total": loss.item()}

            # Check for augmentation mode (dual audio views - waveform based)
            elif use_augmentation and "audio_view1" in batch:
                # Augmentation mode: two views per song
                audio_view1 = batch["audio_view1"].to(self.device)
                audio_view2 = batch["audio_view2"].to(self.device)

                # Get embeddings for both views
                embeddings1 = self.encoder(audio_view1)
                embeddings2 = self.encoder(audio_view2)

                # Concatenate: (2*batch_size, embedding_dim)
                # NTXentLoss expects this format
                embeddings = torch.cat([embeddings1, embeddings2], dim=0)

                # Use NTXentLoss for augmentation-based contrastive learning
                from ml_skeleton.music.losses import NTXentLoss
                if isinstance(self.loss_fn, NTXentLoss):
                    loss = self.loss_fn(embeddings)
                else:
                    # Fallback: if not NTXentLoss, we can still use metadata contrastive
                    # on the first view only (less optimal but compatible)
                    from ml_skeleton.music.losses import MetadataContrastiveLoss
                    if isinstance(self.loss_fn, MetadataContrastiveLoss):
                        loss = self.loss_fn(
                            embeddings1,
                            batch["artist"],
                            batch["album"],
                            batch["year"]
                        )
                    else:
                        ratings = batch["rating"].to(self.device)
                        loss = self.loss_fn(embeddings1, ratings)

                loss_dict = {"total": loss.item()}
            else:
                # Standard mode (single audio)
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

                    # Check if using metadata contrastive loss
                    from ml_skeleton.music.losses import MetadataContrastiveLoss
                    if isinstance(self.loss_fn, MetadataContrastiveLoss):
                        # Pass metadata for contrastive loss (no ratings!)
                        loss = self.loss_fn(
                            embeddings,
                            batch["artist"],
                            batch["album"],
                            batch["year"]
                        )
                    else:
                        # Other losses expect embeddings and ratings
                        loss = self.loss_fn(embeddings, ratings)

                    loss_dict = {"total": loss.item()}

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
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
            "gpu_stats": gpu_stats
        }

    def validate(
        self,
        val_loader: DataLoader,
        use_multi_task: bool = False,
        use_augmentation: bool = False,
        use_simsiam: bool = False
    ) -> dict:
        """Validate model.

        Args:
            val_loader: Validation data loader
            use_multi_task: If True, uses multi-task loss
            use_augmentation: If True, expects dual audio views for contrastive learning
            use_simsiam: If True, uses SimSiam validation loop with spectrogram views

        Returns:
            Dictionary with validation metrics
        """
        self.encoder.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # SimSiam mode (spectrogram-based dual views)
                if use_simsiam and "view1" in batch:
                    view1 = batch["view1"].to(self.device)
                    view2 = batch["view2"].to(self.device)

                    # Forward through SimSiam encoder
                    p1, p2, z1, z2 = self.encoder.forward_simsiam(view1, view2)

                    # SimSiam loss
                    loss = self.loss_fn(p1, p2, z1, z2)

                # Check for augmentation mode (dual audio views)
                elif use_augmentation and "audio_view1" in batch:
                    audio_view1 = batch["audio_view1"].to(self.device)
                    audio_view2 = batch["audio_view2"].to(self.device)

                    # Get embeddings for both views
                    embeddings1 = self.encoder(audio_view1)
                    embeddings2 = self.encoder(audio_view2)

                    # Concatenate for NTXentLoss
                    embeddings = torch.cat([embeddings1, embeddings2], dim=0)

                    from ml_skeleton.music.losses import NTXentLoss
                    if isinstance(self.loss_fn, NTXentLoss):
                        loss = self.loss_fn(embeddings)
                    else:
                        from ml_skeleton.music.losses import MetadataContrastiveLoss
                        if isinstance(self.loss_fn, MetadataContrastiveLoss):
                            loss = self.loss_fn(
                                embeddings1,
                                batch["artist"],
                                batch["album"],
                                batch["year"]
                            )
                        else:
                            ratings = batch["rating"].to(self.device)
                            loss = self.loss_fn(embeddings1, ratings)
                else:
                    # Standard mode
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

                        # Check if using metadata contrastive loss
                        from ml_skeleton.music.losses import MetadataContrastiveLoss
                        if isinstance(self.loss_fn, MetadataContrastiveLoss):
                            # Pass metadata for contrastive loss (no ratings!)
                            loss = self.loss_fn(
                                embeddings,
                                batch["artist"],
                                batch["album"],
                                batch["year"]
                            )
                        else:
                            # Other losses expect embeddings and ratings
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
        use_augmentation: bool = False,
        use_simsiam: bool = False,
        save_best_only: bool = True,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        verbose: bool = True,
        start_epoch: int = 0
    ) -> dict:
        """Full training loop with optional early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train (total, not additional)
            checkpoint_dir: Directory to save checkpoints
            use_multi_task: If True, uses multi-task loss
            use_augmentation: If True, expects dual audio views for contrastive learning
            use_simsiam: If True, uses SimSiam training loop with spectrogram views
            save_best_only: If True, only saves best model
            early_stopping_patience: Number of epochs to wait for improvement before stopping
                                     (None = no early stopping)
            early_stopping_min_delta: Minimum improvement to count as progress
            verbose: If True, print setup info (device, checkpoint dir, etc.)
            start_epoch: Epoch to start training from (for resuming). If 0, uses
                        self.current_epoch if checkpoint was loaded.

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
                verbose=verbose  # Only show early stopping messages if verbose
            )
            if verbose:
                print(f"Early stopping enabled: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")

        # Determine starting epoch (for resuming from checkpoint)
        actual_start = start_epoch if start_epoch > 0 else self.current_epoch
        remaining_epochs = num_epochs - actual_start

        if verbose:
            if actual_start > 0:
                print(f"Resuming encoder training from epoch {actual_start + 1}/{num_epochs}")
                print(f"  {remaining_epochs} epochs remaining")
            else:
                print(f"Training encoder for up to {num_epochs} epochs")
            print(f"Device: {self.device}")
            print(f"Checkpoint dir: {checkpoint_dir}")

        for epoch in range(actual_start, num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, use_multi_task, use_augmentation, use_simsiam)

            # Validate
            val_metrics = None
            if val_loader is not None:
                val_metrics = self.validate(val_loader, use_multi_task, use_augmentation, use_simsiam)

            epoch_time = time.time() - start_time

            # Print metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['loss']:.4f}")

            # Print GPU stats if available
            gpu_stats = train_metrics.get('gpu_stats', {})
            if gpu_stats:
                print(f"  GPU Util: {gpu_stats.get('gpu_util_avg', 0):.1f}% avg "
                      f"(min={gpu_stats.get('gpu_util_min', 0):.0f}%, max={gpu_stats.get('gpu_util_max', 0):.0f}%)")

            # Log metrics to MLflow for learning curves
            if self.tracker is not None:
                self.tracker.log_metric('encoder/train_loss', train_metrics['loss'], step=epoch)
                if val_metrics:
                    self.tracker.log_metric('encoder/val_loss', val_metrics['loss'], step=epoch)
                self.tracker.log_metric('encoder/epoch_time', epoch_time, step=epoch)
                # Log GPU metrics
                if gpu_stats:
                    self.tracker.log_metric('encoder/gpu_util_avg', gpu_stats.get('gpu_util_avg', 0), step=epoch)
                    self.tracker.log_metric('encoder/memory_used_gb', gpu_stats.get('memory_used_avg_gb', 0), step=epoch)

            # Save checkpoint
            current_loss = val_metrics['loss'] if val_metrics else train_metrics['loss']

            # Check early stopping
            if early_stop is not None:
                if early_stop(current_loss, epoch):
                    # Early stopping triggered
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    print(f"Best validation loss: {early_stop.get_best_score():.6f} at epoch {early_stop.get_best_epoch() + 1}")
                    break

            # Versioned checkpoint naming (e.g., encoder_v2_best.pt)
            version_suffix = f"_{self.model_version}" if self.model_version != "default" else ""

            if save_best_only:
                # Save only if this is the best model (either by early stopping or manual check)
                if early_stop and early_stop.should_save_checkpoint():
                    self.best_loss = current_loss
                    self.save_checkpoint(
                        checkpoint_dir / f"encoder{version_suffix}_best.pt",
                        metrics={"loss": current_loss, "epoch": epoch}
                    )
                    # Also save as encoder_best.pt for backwards compatibility
                    self.save_checkpoint(
                        checkpoint_dir / "encoder_best.pt",
                        metrics={"loss": current_loss, "epoch": epoch}
                    )
                    print(f"  Saved best model (loss: {current_loss:.4f})")
                elif not early_stop and current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.save_checkpoint(
                        checkpoint_dir / f"encoder{version_suffix}_best.pt",
                        metrics={"loss": current_loss, "epoch": epoch}
                    )
                    # Also save as encoder_best.pt for backwards compatibility
                    self.save_checkpoint(
                        checkpoint_dir / "encoder_best.pt",
                        metrics={"loss": current_loss, "epoch": epoch}
                    )
                    print(f"  Saved best model (loss: {current_loss:.4f})")
            else:
                self.save_checkpoint(
                    checkpoint_dir / f"encoder{version_suffix}_epoch_{epoch + 1}.pt",
                    metrics={"loss": current_loss, "epoch": epoch}
                )

        # Save final model (versioned and backwards-compatible)
        version_suffix = f"_{self.model_version}" if self.model_version != "default" else ""
        self.save_checkpoint(
            checkpoint_dir / f"encoder{version_suffix}_final.pt",
            metrics={"epoch": num_epochs}
        )
        self.save_checkpoint(
            checkpoint_dir / "encoder_final.pt",
            metrics={"epoch": num_epochs}
        )

        # Log final summary metrics to MLflow
        if self.tracker is not None:
            epochs_completed = len(self.history['train_loss'])
            self.tracker.log_metric('encoder/epochs_completed', epochs_completed)
            self.tracker.log_metric('encoder/best_val_loss', self.best_loss)
            if self.history['train_loss']:
                self.tracker.log_metric('encoder/final_train_loss', self.history['train_loss'][-1])
            if self.history['val_loss']:
                self.tracker.log_metric('encoder/final_val_loss', self.history['val_loss'][-1])

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
            "encoder_version": self.model_version,  # Explicit encoder version
            "model_version": self.model_version,  # Backwards compatibility
            "sample_rate": self.encoder.sample_rate
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

        # Validate sample rate compatibility
        checkpoint_sample_rate = checkpoint.get('sample_rate', 22050)  # legacy default
        current_sample_rate = self.encoder.sample_rate

        if checkpoint_sample_rate != current_sample_rate:
            raise ValueError(
                f"Sample rate mismatch: checkpoint trained with {checkpoint_sample_rate} Hz, "
                f"but current model expects {current_sample_rate} Hz. "
                f"You must retrain the encoder with the new sample rate."
            )

        self.encoder.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
        self.model_version = checkpoint.get("model_version", "default")

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def extract_embeddings(
        self,
        data_loader: DataLoader,
        save_to_store: bool = True,
        use_simsiam: bool = False
    ) -> dict[str, torch.Tensor]:
        """Extract embeddings for all songs in dataset.

        Args:
            data_loader: Data loader with songs
            save_to_store: If True, save to embedding store
            use_simsiam: If True, use SimSiam encoder (expects spectrograms)

        Returns:
            Dictionary mapping filename -> embedding tensor
        """
        self.encoder.eval()
        embeddings_dict = {}

        print(f"Extracting embeddings (model_version: {self.model_version})")

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting embeddings"):
                filenames = batch["filename"]

                # Handle different data formats
                if use_simsiam or "view1" in batch:
                    # SimSiam: use view1 for embedding extraction
                    x = batch["view1"].to(self.device)
                    # SimSiam encoder returns projection z
                    embeddings = self.encoder(x)
                elif "audio" in batch:
                    audio = batch["audio"].to(self.device)
                    # Get embeddings based on encoder type
                    if hasattr(self.encoder, 'base_encoder'):
                        # Multi-task encoder
                        embeddings, _ = self.encoder(audio, return_album_logits=False)
                    elif hasattr(self.encoder, 'forward_simsiam'):
                        # SimSiam encoder with audio input
                        embeddings = self.encoder(audio)
                    else:
                        # Simple encoder
                        embeddings = self.encoder(audio)
                else:
                    raise ValueError("Batch must contain 'audio' or 'view1' key")

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
        sample_rate: int = 16000,
        duration: float = 60.0,
        crop_position: str = "end",
        normalize: bool = True
    ):
        """Extract embeddings in batches without DataLoader.

        Useful for quick extraction when you have a list of songs.

        Args:
            songs: List of Song objects
            batch_size: Batch size for extraction
            sample_rate: Audio sample rate
            duration: Audio duration
            crop_position: Where to extract from - "start", "center", or "end"
            normalize: Apply z-normalization
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
                        crop_position=crop_position,
                        normalize=normalize
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
