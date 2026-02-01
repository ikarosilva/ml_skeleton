#!/usr/bin/env python3
"""Simple diagnostic to check why classifier predictions have zero variance."""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ml_skeleton.music.clementine_db import load_all_songs
from ml_skeleton.music.embedding_store import EmbeddingStore

# Config
db_path = "/Music/database/clementine_backup_2026-01.db"
embedding_db_path = "./embeddings.db"
encoder_version = "v1"

print("=" * 80)
print("ZERO VARIANCE INVESTIGATION")
print("=" * 80)

# 1. Load songs
print("\n1. Loading songs...")
songs = load_all_songs(db_path)
rated_songs = [s for s in songs if s.is_rated]
print(f"   Total songs: {len(songs)}")
print(f"   Rated songs: {len(rated_songs)}")

# 2. Check ratings distribution
print("\n2. Ratings distribution:")
ratings = [s.rating for s in rated_songs]
print(f"   Raw ratings (0-5 scale):")
print(f"     Mean: {np.mean(ratings):.2f}")
print(f"     Std: {np.std(ratings):.2f}")
print(f"     Min: {np.min(ratings):.0f}")
print(f"     Max: {np.max(ratings):.0f}")

# Check for > 5 ratings
over_5 = [r for r in ratings if r > 5]
if over_5:
    print(f"   ⚠️  WARNING: {len(over_5)} ratings > 5 (max should be 5)")
    print(f"      Max rating found: {max(over_5)}")

# Normalized ratings
ratings_norm = [r / 5.0 for r in ratings]
print(f"\n   Normalized ratings (0-1 scale):")
print(f"     Mean: {np.mean(ratings_norm):.4f}")
print(f"     Std: {np.std(ratings_norm):.4f}")
print(f"     Min: {np.min(ratings_norm):.4f}")
print(f"     Max: {np.max(ratings_norm):.4f}")

# 3. Load embeddings
print(f"\n3. Loading embeddings (version={encoder_version})...")
embedding_store = EmbeddingStore(embedding_db_path)
filenames = [s.filename for s in rated_songs]
embeddings_dict = embedding_store.get_embeddings_batch(filenames, model_version=encoder_version)

print(f"   Embeddings loaded: {len(embeddings_dict)}/{len(filenames)}")

if not embeddings_dict:
    print("   ❌ ERROR: No embeddings found!")
    print(f"      Check that embeddings exist for version '{encoder_version}'")
    sys.exit(1)

# 4. Check embedding quality
print("\n4. Embedding quality:")
emb_list = list(embeddings_dict.values())
emb_array = np.array(emb_list)

print(f"   Shape: {emb_array.shape}")
print(f"   Embedding dim: {emb_array.shape[1]}")

# Overall stats
print(f"\n   Overall statistics:")
print(f"     Mean: {np.mean(emb_array):.6f}")
print(f"     Std: {np.std(emb_array):.6f}")
print(f"     Min: {np.min(emb_array):.6f}")
print(f"     Max: {np.max(emb_array):.6f}")

# Per-dimension variance
dim_stds = np.std(emb_array, axis=0)
print(f"\n   Per-dimension variance:")
print(f"     Mean std: {np.mean(dim_stds):.6f}")
print(f"     Min std: {np.min(dim_stds):.6f}")
print(f"     Max std: {np.max(dim_stds):.6f}")
print(f"     Dims with zero variance: {np.sum(dim_stds < 1e-8)}/{len(dim_stds)}")

# Check if all embeddings are identical
unique_count = len(np.unique(emb_array, axis=0))
print(f"\n   Unique embeddings: {unique_count}/{len(emb_array)}")

if unique_count == 1:
    print("   ❌ CRITICAL: All embeddings are IDENTICAL!")
elif unique_count < len(emb_array) * 0.9:
    print(f"   ⚠️  WARNING: Only {unique_count/len(emb_array)*100:.1f}% unique embeddings")
else:
    print("   ✓ Embeddings appear diverse")

# Check for NaN/Inf
print(f"\n   Data quality:")
print(f"     NaN values: {np.isnan(emb_array).sum()}")
print(f"     Inf values: {np.isinf(emb_array).sum()}")

# 5. Check if embeddings are all zeros
zero_embeddings = np.sum(np.abs(emb_array).sum(axis=1) < 1e-8)
print(f"     Zero embeddings: {zero_embeddings}/{len(emb_array)}")

if zero_embeddings > 0:
    print(f"   ❌ CRITICAL: {zero_embeddings} embeddings are all zeros!")

# 6. Test classifier with real embeddings
print("\n5. Testing classifier with real embeddings...")
from ml_skeleton.music.baseline_classifier import SimpleRatingClassifier

embedding_dim = emb_array.shape[1]
classifier = SimpleRatingClassifier(
    embedding_dim=embedding_dim,
    hidden_dims=[1024, 512, 256, 128],
    dropout=0.02
)

# Take a sample of embeddings
sample_size = min(100, len(emb_array))
sample_embs = torch.from_numpy(emb_array[:sample_size]).float()

classifier.eval()
with torch.no_grad():
    predictions = classifier(sample_embs)

print(f"   Sample size: {sample_size}")
print(f"   Predictions:")
print(f"     Shape: {predictions.shape}")
print(f"     Mean: {predictions.mean().item():.4f}")
print(f"     Std: {predictions.std().item():.6f}")
print(f"     Min: {predictions.min().item():.4f}")
print(f"     Max: {predictions.max().item():.4f}")

if predictions.std().item() < 1e-6:
    print("   ❌ CRITICAL: Classifier produces ZERO variance predictions!")
    print("      Even with diverse inputs, output is constant.")
else:
    print(f"   ✓ Classifier produces varied outputs")

# Summary
print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)

issues = []

if len(embeddings_dict) < len(filenames) * 0.5:
    issues.append(f"Only {len(embeddings_dict)}/{len(filenames)} embeddings loaded")

if np.std(emb_array) < 0.01:
    issues.append("Embeddings have very low variance")

if unique_count == 1:
    issues.append("All embeddings are identical")

if zero_embeddings > 0:
    issues.append(f"{zero_embeddings} embeddings are all zeros")

if predictions.std().item() < 1e-6:
    issues.append("Classifier produces constant output")

if max(ratings) > 5:
    issues.append(f"Some ratings exceed 5 (max: {max(ratings)})")

if issues:
    print("\n❌ ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
else:
    print("\n✓ No obvious issues found with embeddings or ratings")
    print("  The zero variance in training may be due to:")
    print("  - Optimizer settings (learning rate too low/high)")
    print("  - Gradient vanishing/exploding")
    print("  - Bad initialization")
    print("  - Need to check actual training dynamics")

print("=" * 80)
