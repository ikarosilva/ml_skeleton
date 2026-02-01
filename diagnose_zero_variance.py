#!/usr/bin/env python3
"""Diagnostic script to investigate zero variance in classifier predictions.

This script checks:
1. Embedding quality (variance, distribution)
2. Target ratings distribution
3. Classifier initialization
4. Data loading pipeline
5. Loss function behavior
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from ml_skeleton.music.clementine_db import load_all_songs
from ml_skeleton.music.embedding_store import EmbeddingStore
from ml_skeleton.music.dataset import EmbeddingDataset
from ml_skeleton.music.baseline_classifier import SimpleRatingClassifier


def check_embeddings(embedding_store, songs):
    """Check if embeddings have sufficient variance."""
    print("=" * 80)
    print("DIAGNOSTIC 1: Embedding Quality")
    print("=" * 80)

    # Load all embeddings
    embeddings = []
    for song in songs:
        if song.is_rated:
            emb = embedding_store.get_embedding(song.filename)
            if emb is not None:
                embeddings.append(emb)

    if not embeddings:
        print("❌ No embeddings found!")
        return

    embeddings = np.array(embeddings)
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Check variance across all dimensions
    emb_std = np.std(embeddings, axis=0)
    emb_mean = np.mean(embeddings, axis=0)

    print(f"\nEmbedding statistics (across all dimensions):")
    print(f"  Mean variance: {np.mean(emb_std):.6f}")
    print(f"  Min variance: {np.min(emb_std):.6f}")
    print(f"  Max variance: {np.max(emb_std):.6f}")
    print(f"  Dimensions with zero variance: {np.sum(emb_std < 1e-8)}/{len(emb_std)}")

    # Check overall variance
    overall_std = np.std(embeddings)
    print(f"\nOverall embedding std: {overall_std:.6f}")

    # Check if embeddings are normalized
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nEmbedding norms:")
    print(f"  Mean norm: {np.mean(norms):.6f}")
    print(f"  Std norm: {np.std(norms):.6f}")
    print(f"  Min norm: {np.min(norms):.6f}")
    print(f"  Max norm: {np.max(norms):.6f}")

    # Check for NaN or Inf
    print(f"\nData quality:")
    print(f"  NaN values: {np.isnan(embeddings).sum()}")
    print(f"  Inf values: {np.isinf(embeddings).sum()}")

    # Check if all embeddings are identical
    unique_embeddings = np.unique(embeddings, axis=0)
    print(f"  Unique embeddings: {len(unique_embeddings)}/{len(embeddings)}")

    if overall_std < 1e-6:
        print("\n❌ WARNING: Embeddings have very low variance!")
        print("   This will cause the classifier to predict constant values.")
    else:
        print("\n✓ Embeddings have reasonable variance")

    print()
    return embeddings


def check_ratings(songs):
    """Check ratings distribution."""
    print("=" * 80)
    print("DIAGNOSTIC 2: Ratings Distribution")
    print("=" * 80)

    ratings = []
    for song in songs:
        if song.is_rated:
            # Normalize to [0, 1] like the dataset does
            rating = song.rating / 5.0
            ratings.append(rating)

    ratings = np.array(ratings)

    print(f"Total rated songs: {len(ratings)}")
    print(f"\nRating statistics (normalized to [0, 1]):")
    print(f"  Mean: {np.mean(ratings):.4f}")
    print(f"  Std: {np.std(ratings):.4f}")
    print(f"  Min: {np.min(ratings):.4f}")
    print(f"  Max: {np.max(ratings):.4f}")
    print(f"  Median: {np.median(ratings):.4f}")

    # Distribution
    print(f"\nRating distribution:")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(ratings, bins=bins)
    for i in range(len(bins) - 1):
        pct = 100 * hist[i] / len(ratings)
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}]: {hist[i]:5d} ({pct:5.1f}%)")

    # Check if ratings are too uniform
    if np.std(ratings) < 0.05:
        print("\n❌ WARNING: Ratings have very low variance!")
        print("   All songs are rated similarly, making prediction difficult.")
    else:
        print("\n✓ Ratings have reasonable variance")

    print()
    return ratings


def check_classifier_initialization(embedding_dim=512):
    """Check if classifier initializes properly."""
    print("=" * 80)
    print("DIAGNOSTIC 3: Classifier Initialization")
    print("=" * 80)

    classifier = SimpleRatingClassifier(
        embedding_dim=embedding_dim,
        hidden_dims=[1024, 512, 256, 128],
        dropout=0.02
    )

    # Create dummy input
    dummy_input = torch.randn(32, embedding_dim)

    print(f"Classifier architecture:")
    print(classifier)

    # Check forward pass
    classifier.eval()
    with torch.no_grad():
        output = classifier(dummy_input)

    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: torch.Size([32])")

    if output.shape != (32,):
        print(f"❌ ERROR: Output shape is {output.shape}, expected (32,)")
    else:
        print("✓ Output shape is correct")

    print(f"\nOutput statistics (random input):")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")

    # Check if output is always the same (initialization issue)
    outputs = []
    for _ in range(10):
        dummy_input = torch.randn(32, embedding_dim)
        with torch.no_grad():
            output = classifier(dummy_input)
        outputs.append(output.mean().item())

    output_variance = np.std(outputs)
    print(f"\nOutput variance across 10 random batches: {output_variance:.6f}")

    if output_variance < 1e-6:
        print("❌ WARNING: Classifier always produces same output!")
        print("   This indicates an initialization or architecture problem.")
    else:
        print("✓ Classifier produces varied outputs for different inputs")

    print()
    return classifier


def check_dataset_loading(db_path, embedding_db_path):
    """Check if dataset loads correctly."""
    print("=" * 80)
    print("DIAGNOSTIC 4: Dataset Loading")
    print("=" * 80)

    # Load data
    songs = load_all_songs(db_path)
    print(f"Loaded {len(songs)} songs from database")

    embedding_store = EmbeddingStore(embedding_db_path)
    embeddings_dict = embedding_store.get_all_embeddings()

    # Create dataset
    dataset = EmbeddingDataset(
        embeddings=embeddings_dict,
        songs=songs,
        only_rated=True
    )

    print(f"Dataset size: {len(dataset)}")

    # Check a few samples
    print(f"\nSample data (first 5 items):")
    for i in range(min(5, len(dataset))):
        item = dataset[i]
        print(f"  [{i}] Embedding shape: {item['embedding'].shape}, "
              f"Rating: {item['rating'].item():.4f}, "
              f"Filename: {item['filename'][:50]}...")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Check a batch
    batch = next(iter(dataloader))
    print(f"\nBatch data:")
    print(f"  Embeddings shape: {batch['embedding'].shape}")
    print(f"  Ratings shape: {batch['rating'].shape}")
    print(f"  Ratings dtype: {batch['rating'].dtype}")

    # Check if ratings in batch have variance
    rating_std = batch['rating'].std().item()
    print(f"  Ratings std in batch: {rating_std:.4f}")

    if rating_std < 0.01:
        print("❌ WARNING: Ratings in batch have very low variance!")
    else:
        print("✓ Ratings in batch have reasonable variance")

    print()
    return dataset, dataloader


def check_loss_function():
    """Check loss function behavior."""
    print("=" * 80)
    print("DIAGNOSTIC 5: Loss Function")
    print("=" * 80)

    loss_fn = nn.MSELoss()

    # Test with correct shapes
    print("Testing MSE loss with shape (batch_size,):")
    predictions = torch.tensor([0.5, 0.6, 0.7, 0.8])
    targets = torch.tensor([0.4, 0.5, 0.6, 0.7])

    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Targets shape: {targets.shape}")

    loss = loss_fn(predictions, targets)
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Expected: {((predictions - targets) ** 2).mean().item():.6f}")

    # Test with wrong shapes (what was happening before)
    print("\nTesting MSE loss with mismatched shapes:")
    predictions_wrong = torch.tensor([[0.5], [0.6], [0.7], [0.8]])
    targets = torch.tensor([0.4, 0.5, 0.6, 0.7])

    print(f"  Predictions shape: {predictions_wrong.shape}")
    print(f"  Targets shape: {targets.shape}")

    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loss = loss_fn(predictions_wrong.squeeze(), targets)
            if w:
                print(f"  ⚠ Warnings: {len(w)}")
                for warning in w:
                    print(f"    {warning.message}")
            else:
                print(f"  ✓ No warnings with squeeze()")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    print()


def check_training_step(dataset, classifier):
    """Simulate one training step."""
    print("=" * 80)
    print("DIAGNOSTIC 6: Training Step Simulation")
    print("=" * 80)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    batch = next(iter(dataloader))

    embeddings = batch['embedding']
    ratings = batch['rating']

    print(f"Batch shapes:")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Ratings: {ratings.shape}")

    # Forward pass
    classifier.eval()
    with torch.no_grad():
        predictions = classifier(embeddings)

    print(f"\nPredictions:")
    print(f"  Shape: {predictions.shape}")
    print(f"  Mean: {predictions.mean().item():.4f}")
    print(f"  Std: {predictions.std().item():.6f}")
    print(f"  Min: {predictions.min().item():.4f}")
    print(f"  Max: {predictions.max().item():.4f}")

    # Check variance
    pred_std = predictions.std().item()
    if pred_std < 1e-6:
        print("\n❌ CRITICAL: Predictions have ZERO variance!")
        print("   All predictions are identical.")

        # Check weights
        print("\nInvestigating classifier weights:")
        for name, param in classifier.named_parameters():
            if 'weight' in name:
                weight_std = param.std().item()
                print(f"  {name}: std={weight_std:.6f}")
                if weight_std < 1e-6:
                    print(f"    ❌ Zero variance in weights!")
    else:
        print(f"\n✓ Predictions have variance: {pred_std:.6f}")

    # Compute loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(predictions, ratings)
    mae = torch.abs(predictions - ratings).mean()

    print(f"\nLoss metrics:")
    print(f"  MSE Loss: {loss.item():.6f}")
    print(f"  MAE: {mae.item():.4f}")

    print()


def main():
    """Run all diagnostics."""
    print("\n" + "=" * 80)
    print("ZERO VARIANCE DIAGNOSTIC SUITE")
    print("=" * 80)
    print()

    # Configuration
    db_path = "/Music/database/clementine_backup_2026-01.db"
    embedding_db_path = "./embeddings.db"

    # Check if files exist
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return 1

    if not Path(embedding_db_path).exists():
        print(f"❌ Embedding database not found: {embedding_db_path}")
        return 1

    # Run diagnostics
    try:
        # Load data
        songs = load_all_songs(db_path)
        embedding_store = EmbeddingStore(embedding_db_path)

        # 1. Check embeddings
        embeddings = check_embeddings(embedding_store, songs)

        # 2. Check ratings
        ratings = check_ratings(songs)

        # 3. Check classifier initialization
        embedding_dim = embeddings.shape[1] if embeddings is not None else 512
        classifier = check_classifier_initialization(embedding_dim)

        # 4. Check dataset
        dataset, dataloader = check_dataset_loading(db_path, embedding_db_path)

        # 5. Check loss function
        check_loss_function()

        # 6. Simulate training step
        check_training_step(dataset, classifier)

        # Summary
        print("=" * 80)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 80)
        print("\nKey findings will help identify the root cause of zero variance.")
        print("\nNext steps:")
        print("1. Review the diagnostics above")
        print("2. Fix any issues marked with ❌")
        print("3. Re-run training to verify fixes")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n❌ Error during diagnostics: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
