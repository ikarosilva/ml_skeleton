import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ml_skeleton.music.embedding_store import EmbeddingStore

embedding_db_path = "./embeddings.db"
encoder_version = "v1"

print("Checking encoder embeddings...")
store = EmbeddingStore(embedding_db_path)

# Get statistics
stats = store.get_stats()
print("\nEmbedding store stats:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Check a few specific embeddings
from ml_skeleton.music.clementine_db import load_all_songs
songs = load_all_songs("/Music/database/clementine_backup_2026-03.db")
rated_songs = [s for s in songs if s.is_rated][:10]

print("\nSample embeddings:")
for i, song in enumerate(rated_songs):
    emb = store.get_embedding(song.filename, model_version=encoder_version)
    if emb is not None:
        print(f"  [{i}] {song.filename[:50]}")
        print(f"       Mean: {np.mean(emb):.6f}, Std: {np.std(emb):.6f}")
        print(f"       Min: {np.min(emb):.6f}, Max: {np.max(emb):.6f}")
        print(f"       Non-zero: {np.sum(np.abs(emb) > 1e-8)}/{len(emb)}")
