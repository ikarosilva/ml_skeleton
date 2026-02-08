"""Genre centroids in embedding space for imputing missing genre.

When a song has no genre metadata, we impute a 7-dim multi-hot by:
1. Computing 7 category centroids from songs that have genre.
2. For the song's embedding(s) (e.g. num_chunks per song), each chunk votes for top_k
   closest centroids; any category with >= min_votes is set to 1 (multi-hot).

This allows genre mix-ins (e.g. rock + electronic) when the embedding is
ambiguous across chunks.
"""

import numpy as np
from pathlib import Path
from typing import Optional

from .genre_mapper import (
    GENRE_CATEGORIES,
    NUM_GENRES,
    parse_genre_string,
    genre_to_multilabel,
)
from .clementine_db import Song


def _embedding_per_song(emb: np.ndarray) -> np.ndarray:
    """Return (D,) from either (D,) or (num_chunks, D) by averaging over chunks."""
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim == 2:
        return np.mean(emb, axis=0)
    return emb


def compute_genre_centroids(
    embeddings: dict[str, np.ndarray],
    songs: list[Song],
    encoder_version: str = "",
) -> np.ndarray:
    """Compute one centroid per genre category in embedding space.

    Uses only songs that have genre metadata. Multi-label songs contribute
    to each of their categories. Each song is represented by a single
    embedding (mean over chunks if shape is (4, D)).

    Args:
        embeddings: filename -> embedding array (D,) or (4, D)
        songs: list of Song (must have .filename and .genre)
        encoder_version: optional label for persistence

    Returns:
        centroids: (NUM_GENRES, D) float32
    """
    # Build filename -> song
    filename_to_song = {s.filename: s for s in songs}

    # Per-category lists of embedding vectors
    category_embeddings: list[list[np.ndarray]] = [[] for _ in range(NUM_GENRES)]

    for filename, emb in embeddings.items():
        song = filename_to_song.get(filename)
        if not song or not song.genre:
            continue
        categories = parse_genre_string(song.genre)
        if not categories:
            continue

        vec = _embedding_per_song(emb)
        if vec.size == 0:
            continue

        for cat in categories:
            if cat in GENRE_CATEGORIES:
                idx = GENRE_CATEGORIES.index(cat)
                category_embeddings[idx].append(vec)

    # Infer D from first non-empty category
    D = None
    for lst in category_embeddings:
        if lst:
            D = lst[0].shape[0]
            break
    if D is None:
        raise ValueError("No songs with genre found; cannot compute centroids")

    centroids = np.zeros((NUM_GENRES, D), dtype=np.float32)
    for c in range(NUM_GENRES):
        if category_embeddings[c]:
            arr = np.stack(category_embeddings[c], axis=0)
            centroids[c] = np.mean(arr, axis=0)
    return centroids


def impute_genre_multihot(
    embedding_chunks: np.ndarray,
    centroids: np.ndarray,
    top_k: int = 2,
    min_votes: int = 1,
    distance: str = "cosine",
) -> np.ndarray:
    """Impute a 7-dim multi-hot genre from embedding(s) using centroid distances.

    Each chunk (or single embedding) votes for its top_k closest centroids.
    Any category with >= min_votes gets 1; others 0. Ties in distance are
    broken by category index (lower wins).

    Args:
        embedding_chunks: (4, D) or (D,) — one or more embedding vectors
        centroids: (NUM_GENRES, D)
        top_k: number of closest categories each chunk can vote for (default 2 for mix-ins)
        min_votes: minimum votes to set a category to 1 (default 1)
        distance: "cosine" or "l2"

    Returns:
        multi_hot: (NUM_GENRES,) float32, 0/1
    """
    chunks = np.asarray(embedding_chunks, dtype=np.float32)
    if chunks.ndim == 1:
        chunks = chunks.reshape(1, -1)
    # chunks: (n_chunks, D)
    cent = np.asarray(centroids, dtype=np.float32)
    if cent.ndim != 2 or cent.shape[0] != NUM_GENRES:
        raise ValueError(f"centroids must be ({NUM_GENRES}, D), got {cent.shape}")

    n_chunks, D = chunks.shape
    votes = np.zeros(NUM_GENRES, dtype=np.float32)

    for i in range(n_chunks):
        vec = chunks[i]
        if distance == "cosine":
            # Cosine distance = 1 - cosine_sim; minimize distance = maximize sim
            norm_c = np.linalg.norm(cent, axis=1, keepdims=True)
            norm_c = np.where(norm_c < 1e-9, 1.0, norm_c)
            sim = (cent @ vec) / (norm_c.squeeze() * np.linalg.norm(vec) + 1e-9)
            dist = -sim  # smaller is closer in "distance"
        else:
            # L2
            diff = cent - vec
            dist = np.sqrt(np.sum(diff * diff, axis=1))
        # top_k smallest distance indices
        if top_k >= NUM_GENRES:
            top_indices = np.arange(NUM_GENRES)
        else:
            top_indices = np.argpartition(dist, min(top_k, len(dist)))[:top_k]
            # Sort by distance so tie-break is consistent
            top_indices = top_indices[np.argsort(dist[top_indices])]
        for idx in top_indices:
            votes[idx] += 1.0

    multi_hot = (votes >= min_votes).astype(np.float32)
    # Ensure at least one category (fallback: closest overall)
    if multi_hot.sum() == 0 and n_chunks > 0:
        vec = np.mean(chunks, axis=0)
        if distance == "cosine":
            norm_c = np.linalg.norm(cent, axis=1, keepdims=True)
            norm_c = np.where(norm_c < 1e-9, 1.0, norm_c)
            sim = (cent @ vec) / (norm_c.squeeze() * np.linalg.norm(vec) + 1e-9)
            dist = -sim
        else:
            diff = cent - vec
            dist = np.sqrt(np.sum(diff * diff, axis=1))
        multi_hot[np.argmin(dist)] = 1.0
    return multi_hot


def get_genre_features(
    song: Song,
    embedding: np.ndarray,
    centroids: Optional[np.ndarray],
    top_k: int = 2,
    min_votes: int = 1,
) -> np.ndarray:
    """Return 7-dim genre multi-hot: from metadata if present, else imputed from centroids.

    Args:
        song: Song with optional .genre
        embedding: (D,) or (4, D) for that song
        centroids: (NUM_GENRES, D) or None (if None and genre missing, return zeros)
        top_k: for imputation
        min_votes: for imputation

    Returns:
        (NUM_GENRES,) float32
    """
    if song.genre and parse_genre_string(song.genre):
        return genre_to_multilabel(song.genre).numpy()
    if centroids is not None:
        return impute_genre_multihot(
            embedding_chunks=embedding,
            centroids=centroids,
            top_k=top_k,
            min_votes=min_votes,
        )
    return np.zeros(NUM_GENRES, dtype=np.float32)


def save_centroids(centroids: np.ndarray, path: str | Path) -> None:
    """Save centroids to .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, centroids, allow_pickle=False)


def load_centroids(path: str | Path) -> np.ndarray:
    """Load centroids from .npy file. Returns (NUM_GENRES, D)."""
    arr = np.load(path, allow_pickle=False)
    if arr.shape[0] != NUM_GENRES:
        raise ValueError(f"Expected {NUM_GENRES} rows, got {arr.shape[0]}")
    return arr.astype(np.float32)
