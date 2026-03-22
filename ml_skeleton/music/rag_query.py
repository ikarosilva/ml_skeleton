"""Playlist-anchored retrieval in embedding space (RAG-style query, no LLM).

Given an XSPF playlist, build a centroid + optional top-K PCA subspace from
mean-pooled chunk embeddings, then rank library tracks by cosine similarity
to the centroid (primary), with subspace alignment as tie-breaker.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import unquote
import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_skeleton.music.ab_testing import load_classifier_from_checkpoint
from ml_skeleton.music.clementine_db import ClementineDB, Song
from ml_skeleton.music.dataset import EmbeddingDataset
from ml_skeleton.music.embedding_store import EmbeddingStore
from ml_skeleton.music.losses import BinaryRatingLoss, RatingLoss
from ml_skeleton.music.xspf_playlist import export_to_xspf, parse_xspf_locations
from ml_skeleton.training.classifier_trainer import (
    ClassifierTrainer,
    get_classifier_versions_from_checkpoint,
)


def _decode_fn(fn: str | bytes) -> str:
    if isinstance(fn, bytes):
        return fn.decode("utf-8", errors="replace")
    return str(fn)


def _absolutize_xspf_location(loc: str, xspf_parent: Path) -> str:
    """Resolve XSPF ``<location>`` for matching.

    Relative paths (no ``file:`` scheme, not root-absolute) are joined with the
    directory containing the ``.xspf`` file, which is how many exporters write playlists.
    """
    t = loc.strip()
    if not t:
        return t
    if t.startswith("file:"):
        return t
    if t.startswith("/"):
        return t
    if len(t) >= 2 and t[1] == ":":  # Windows drive
        return t
    if t.startswith("\\\\"):
        return t
    combined = xspf_parent / t
    try:
        return str(combined.resolve(strict=False))
    except (OSError, RuntimeError, TypeError):
        return str(combined)


def _normalize_path_key(path_str: str) -> str:
    """Normalize a filesystem path or file:// URI for lookup."""
    p = path_str.strip()
    if p.startswith("file://"):
        p = p[7:]
    p = unquote(p)
    remap = os.environ.get("MUSIC_PATH_REMAP")
    if remap and ":" in remap:
        old_prefix, new_prefix = remap.split(":", 1)
        if p.startswith(old_prefix):
            p = new_prefix + p[len(old_prefix) :]
    return os.path.normpath(p)


def _path_after_music_folder(norm_path: str) -> str | None:
    """Return lowercase relative path under the first ``Music`` segment (host vs container paths).

    e.g. ``/home/u/Music/Rock/a.mp3`` → ``rock/a.mp3``; ``/Music/Rock/a.mp3`` → ``rock/a.mp3``.
    Used when XSPF was saved on the host but the Clementine DB uses ``/Music/...`` in the container.
    """
    p = norm_path.replace("\\", "/")
    parts = [x for x in p.split("/") if x]
    for i, seg in enumerate(parts):
        if seg.lower() == "music":
            tail = "/".join(parts[i + 1 :])
            return tail.casefold() if tail else None
    return None


def _build_path_lookups(songs: list[Song]) -> tuple[dict[str, str], dict[str, str]]:
    """Primary: normalized absolute path -> DB ``file://`` URI.

    Secondary: unambiguous path-after-``Music`` (lowercase) -> DB URI for cross-environment playlists.
    """
    idx: dict[str, str] = {}
    tail_bucket: dict[str, list[str]] = {}
    for s in songs:
        fn = _decode_fn(s.filename)
        nk = _normalize_path_key(fn)
        idx[nk] = fn
        try:
            idx[str(s.filepath.resolve())] = fn
        except (OSError, RuntimeError):
            pass
        tail = _path_after_music_folder(nk)
        if tail:
            tail_bucket.setdefault(tail, []).append(fn)
    tail_unique = {t: v[0] for t, v in tail_bucket.items() if len(v) == 1}
    return idx, tail_unique


def _resolve_xspf_location_to_db_filename(
    loc: str,
    lookup: dict[str, str],
    tail_lookup: dict[str, str],
) -> str | None:
    k = _normalize_path_key(loc)
    if k in lookup:
        return lookup[k]
    try:
        r = str(Path(k).resolve())
        hit = lookup.get(r)
        if hit is not None:
            return hit
    except (OSError, RuntimeError):
        pass
    tail = _path_after_music_folder(k)
    if tail and tail in tail_lookup:
        return tail_lookup[tail]
    return None


def _sanitize_stem(name: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE).strip("._")
    return s or "playlist"


def _mean_pool_song_embedding(emb: np.ndarray) -> np.ndarray:
    x = np.asarray(emb, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        return x.mean(axis=0).astype(np.float32, copy=False)
    return x.reshape(-1).astype(np.float32, copy=False)


def _pca_subspace_basis(Xc: np.ndarray, k: int) -> np.ndarray:
    """Return P with shape (D, K_eff) orthonormal columns (principal axes).

    Xc is (N, D) already centered. K_eff = min(k, N-1, D).
    """
    n, d = Xc.shape
    if n < 2 or k <= 0:
        return np.zeros((d, 0), dtype=np.float32)
    k_eff = min(int(k), n - 1, d)
    if k_eff <= 0:
        return np.zeros((d, 0), dtype=np.float32)
    # SVD: Xc = U S Vt; rows of Vt are orthonormal directions in R^D
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    # vt shape (min(n,d), D); take first k_eff rows -> (k_eff, D); columns P = vt[:k_eff].T -> (D, k_eff)
    return np.asarray(vt[:k_eff].T, dtype=np.float32)


def _batch_predict_ratings(
    *,
    config: dict,
    model_dir: Path,
    store: EmbeddingStore,
    filenames: list[str],
    filename_to_song: dict[str, Song],
    encoder_version: str,
    num_chunks: int,
) -> dict[str, float]:
    """Run prod classifier on songs that have full-chunk embeddings. Returns filename -> score in [0, 1]."""
    music_config = config["music"]
    classifier_config = config.get("classifier", {})
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    emb = store.get_embeddings_batch_all_chunks(
        filenames, model_version=encoder_version, num_chunks=num_chunks
    )
    songs = [filename_to_song[f] for f in filenames if f in emb and f in filename_to_song]
    if not songs:
        return {}

    ckpt_path = model_dir / "classifier_best.pt"
    if not ckpt_path.is_file():
        raise SystemExit(
            f"Classifier checkpoint required: {ckpt_path}\n"
            "Run classifier training and promote-to-prod."
        )

    _cls_ver, classifier_encoder_version = get_classifier_versions_from_checkpoint(str(ckpt_path))
    if classifier_encoder_version != encoder_version:
        raise SystemExit(
            f"Classifier encoder version {classifier_encoder_version!r} != config {encoder_version!r}. "
            "Fix music.encoder_version or retrain the classifier."
        )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    use_genre = checkpoint.get("use_genre", False)
    genre_centroids = checkpoint.get("genre_centroids")

    _first = np.asarray(next(iter(emb.values())))
    embedding_dim_from_emb = int(_first.shape[-1]) if _first.ndim > 1 else len(_first)
    embedding_dim_ck = checkpoint.get("embedding_dim")
    embedding_dim = (
        int(embedding_dim_ck)
        if embedding_dim_ck is not None and int(embedding_dim_ck) > 0
        else embedding_dim_from_emb
    )

    classifier, _ = load_classifier_from_checkpoint(
        str(ckpt_path), embedding_dim=embedding_dim, device=device
    )

    dataset = EmbeddingDataset(
        embeddings=emb,
        songs=songs,
        only_rated=False,
        use_genre=use_genre,
        genre_centroids=genre_centroids,
        genre_impute_top_k=classifier_config.get("genre_impute_top_k", 2),
        genre_impute_min_votes=classifier_config.get("genre_impute_min_votes", 1),
        classification_mode=classifier_config.get("classification_mode", "regression"),
    )
    if len(dataset) == 0:
        return {}

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    classification_mode = classifier_config.get("classification_mode", "regression")
    if classification_mode == "binary":
        loss_fn = BinaryRatingLoss()
    else:
        loss_fn = RatingLoss()

    chunk_aggregation = checkpoint.get(
        "chunk_aggregation", classifier_config.get("chunk_aggregation", "mean")
    )
    trainer = ClassifierTrainer(
        classifier=classifier,
        device=device,
        loss_fn=loss_fn,
        optimizer=torch.optim.Adam(classifier.parameters()),
        classification_mode=classification_mode,
        encoder_version=encoder_version,
        classifier_version=_cls_ver,
        chunk_aggregation=chunk_aggregation,
    )
    preds, pred_files = trainer.predict(loader)
    return {fn: float(p) for fn, p in zip(pred_files, preds)}


def _training_positive_label(song: Song, classifier_config: dict) -> bool:
    """Same notion of “positive” as EmbeddingDataset in simple binary mode: rating >= binary_positive_threshold.

    Song.rating is 0–5 (Clementine DB layer converts raw storage). Unrated songs are never positive.
    """
    if not song.is_rated:
        return False
    pos_thr = float(classifier_config.get("binary_positive_threshold", 4.0))
    return float(song.rating) >= pos_thr


def _unrated_pred_pool_cap(top_n: int) -> int:
    """How many unrated (pred > 0.5) tracks to keep before embedding re-ranking."""
    return max(100, int(top_n) * 20)


def _filter_candidates_likes_only(
    candidates: list[str],
    filename_to_song: dict[str, Song],
    pred_by_fn: dict[str, float],
    classifier_config: dict,
    *,
    top_n: int,
    unrated_only: bool,
) -> list[str]:
    """Rated: keep only training-positive labels. Unrated: pred > 0.5, top-K by prediction (then cosine)."""
    cap = _unrated_pred_pool_cap(top_n)
    rated_pos: list[str] = []
    unrated_scored: list[tuple[str, float]] = []
    for fn in candidates:
        song = filename_to_song.get(fn)
        if song is None:
            continue
        pred = pred_by_fn.get(fn)
        if song.is_rated:
            if unrated_only:
                continue
            if _training_positive_label(song, classifier_config):
                rated_pos.append(fn)
        else:
            if pred is not None and pred > 0.5:
                unrated_scored.append((fn, pred))

    unrated_scored.sort(key=lambda x: -x[1])
    unrated_pick = [fn for fn, _ in unrated_scored[:cap]]

    if unrated_only:
        pool = unrated_pick
    else:
        # Stable union: rated positives first (order as in library walk), then unrated by pred
        rated_set = set(rated_pos)
        pool = [fn for fn in candidates if fn in rated_set]
        for fn in unrated_pick:
            if fn not in rated_set:
                pool.append(fn)

    return pool


def run_rag_query(
    *,
    config: dict,
    xspf_path: str | Path,
    num_pc: int = 5,
    top_n: int = 50,
    unrated_only: bool = False,
    likes_only: bool = False,
    prod_dir: str | Path | None = None,
) -> Path:
    """Generate ``rag_<stem>.xspf`` with nearest songs in embedding space.

    Args:
        config: Full YAML config (music, recommendations, etc.)
        xspf_path: Input playlist (.xspf)
        num_pc: Number of principal components for subspace tie-break (max ≈ embedding dim).
        top_n: How many songs to write (excluding playlist tracks).
        unrated_only: If True, candidates must be ``not song.is_rated``.
        likes_only: If True, restrict the candidate pool before cosine ranking:
            * **Rated** tracks: keep only those with the same positive label as classifier training
              (``classifier.binary_positive_threshold`` on 0–5 star scale, default ≥4).
            * **Unrated** tracks: classifier prediction > 0.5, sorted by prediction descending,
              keep the top ``max(100, top_n * 20)`` for embedding re-ranking.
            With ``unrated_only``, only the unrated branch applies. Requires ``classifier_best.pt``.
        prod_dir: Directory containing ``embeddings.db`` (default ``prod``).

    Returns:
        Path to written XSPF.
    """
    music_config = config["music"]
    rec_config = config.get("recommendations", {})
    classifier_config = config.get("classifier", {})
    encoder_version = music_config.get("encoder_version", music_config.get("model_version", "v1"))
    num_chunks = music_config.get("chunk_cache", {}).get("num_chunks", 8)

    pdir = Path(prod_dir or "prod")
    emb_path = pdir / "embeddings.db"
    out_dir = Path(rec_config.get("output_dir", "./"))
    out_dir.mkdir(parents=True, exist_ok=True)

    xspf_path = Path(xspf_path)
    stem = _sanitize_stem(xspf_path.stem)
    out_path = out_dir / f"rag_{stem}.xspf"

    print("=" * 60)
    print("RAG QUERY (playlist → embedding neighborhood)")
    print("=" * 60)
    print(f"  Input XSPF: {xspf_path}")
    print(f"  Embeddings DB: {emb_path}")
    print(f"  Encoder version: {encoder_version}, chunks/song: {num_chunks}, mean pool")
    print(f"  PCs (tie-break): {num_pc}, top_n: {top_n}, unrated_only: {unrated_only}, likes_only: {likes_only}")
    if likes_only:
        pos_thr = float(classifier_config.get("binary_positive_threshold", 4.0))
        cap = _unrated_pred_pool_cap(top_n)
        print(
            f"  Likes-only pool: rated with training-positive (rating>={pos_thr}), "
            f"unrated with pred>0.5 top {cap} by score"
        )
    if not emb_path.is_file():
        raise SystemExit(
            f"Embeddings DB not found: {emb_path}\nRun: ./run_music_pipeline.sh promote-to-prod"
        )

    db = ClementineDB(music_config["database_path"])
    all_songs = db.get_all_songs()
    lookup, tail_lookup = _build_path_lookups(all_songs)
    filename_to_song = {_decode_fn(s.filename): s for s in all_songs}

    locs = parse_xspf_locations(xspf_path)
    print(f"  XSPF locations read: {len(locs)}")

    xspf_dir = xspf_path.parent
    playlist_filenames: list[str] = []
    missing_xspf = 0
    for loc in locs:
        resolved_loc = _absolutize_xspf_location(loc, xspf_dir)
        fn = _resolve_xspf_location_to_db_filename(resolved_loc, lookup, tail_lookup)
        if fn is None:
            missing_xspf += 1
            continue
        if fn not in playlist_filenames:
            playlist_filenames.append(fn)

    if missing_xspf:
        print(f"  Skipped (no DB match): {missing_xspf} location(s)")
    if missing_xspf == len(locs) and locs:
        print("  Hint: relative <location> entries are resolved vs the .xspf directory; full paths must match the DB")
        print("  (or share the same path under .../Music/...).")
        ex_res = _absolutize_xspf_location(locs[0], xspf_dir)
        print(f"  Example XSPF location (raw): {locs[0]!r} → resolved: {ex_res!r}")
        ex_fn = _decode_fn(all_songs[0].filename) if all_songs else ""
        print(f"  Example library filename: {ex_fn!r}")
        print("  Or set MUSIC_PATH_REMAP=old_prefix:new_prefix (see Song.filepath / clementine_db).")

    store = EmbeddingStore(str(emb_path))
    emb_pl = store.get_embeddings_batch_all_chunks(
        playlist_filenames, model_version=encoder_version, num_chunks=num_chunks
    )
    missing_emb: list[str] = []
    playlist_vecs: list[np.ndarray] = []
    used_names: list[str] = []
    for fn in playlist_filenames:
        if fn not in emb_pl:
            missing_emb.append(fn)
            continue
        playlist_vecs.append(_mean_pool_song_embedding(emb_pl[fn]))
        used_names.append(fn)

    if missing_emb:
        print(f"  Skipped (no full chunk embeddings): {len(missing_emb)} playlist track(s)")
    if not playlist_vecs:
        raise SystemExit(
            "No playlist tracks with embeddings; cannot build query. "
            "Check encoder_version, num_chunks, embedding DB, and that XSPF paths match the library "
            "(or share the same path under .../Music/...)."
        )

    X = np.stack(playlist_vecs, axis=0).astype(np.float32)
    mu = X.mean(axis=0)
    d = mu.shape[0]
    k_req = max(0, min(int(num_pc), d))
    Xc = X - mu
    P = _pca_subspace_basis(Xc, k_req)
    k_eff = P.shape[1]
    print(f"  Playlist tracks used: {len(used_names)} / {len(playlist_filenames)} (D={d}, K_eff={k_eff})")

    playlist_set = set(used_names)
    all_with_chunks = store.list_filenames_with_all_chunks(encoder_version, num_chunks)
    candidates: list[str] = []
    for fn in all_with_chunks:
        if fn in playlist_set:
            continue
        song = filename_to_song.get(fn)
        if song is None:
            continue
        if unrated_only and song.is_rated:
            continue
        candidates.append(fn)

    if likes_only:
        print("  Running classifier on candidates (likes-only pool)...")
        pred_by_fn = _batch_predict_ratings(
            config=config,
            model_dir=pdir,
            store=store,
            filenames=candidates,
            filename_to_song=filename_to_song,
            encoder_version=encoder_version,
            num_chunks=num_chunks,
        )
        before = len(candidates)
        candidates = _filter_candidates_likes_only(
            candidates,
            filename_to_song,
            pred_by_fn,
            classifier_config,
            top_n=top_n,
            unrated_only=unrated_only,
        )
        n_rated = sum(
            1
            for fn in candidates
            if (s := filename_to_song.get(fn)) is not None and s.is_rated
        )
        print(
            f"  After likes-only filter: {len(candidates)} / {before} candidates "
            f"({n_rated} rated positive, {len(candidates) - n_rated} unrated pred>0.5)"
        )

    if not candidates:
        raise SystemExit(
            "No candidate songs after filters (try without --rag-likes-only, or --rag-unrated-only=false)."
        )

    print(f"  Candidate pool: {len(candidates)} songs")

    emb_c = store.get_embeddings_batch_all_chunks(
        candidates, model_version=encoder_version, num_chunks=num_chunks
    )
    rows: list[tuple[str, np.ndarray]] = []
    for fn in candidates:
        if fn not in emb_c:
            continue
        rows.append((fn, _mean_pool_song_embedding(emb_c[fn])))
    if not rows:
        raise SystemExit("No candidate embeddings retrieved.")

    F = np.stack([r[1] for r in rows], axis=0).astype(np.float32)
    fn_order = [r[0] for r in rows]

    mu_n = np.linalg.norm(mu) + 1e-12
    f_n = np.linalg.norm(F, axis=1) + 1e-12
    cos_mu = (F @ mu) / (f_n * mu_n)

    vc = F - mu
    vc_n = np.linalg.norm(vc, axis=1) + 1e-12
    if k_eff > 0:
        coef = (P.T @ vc.T).astype(np.float32)  # K x M
        proj = (P @ coef).T
        orth = vc - proj
        orth_n = np.linalg.norm(orth, axis=1)
        align = 1.0 - orth_n / vc_n
    else:
        align = np.ones(len(fn_order), dtype=np.float32)

    # Sort: primary cosine desc, secondary align desc
    order = np.lexsort((-align, -cos_mu))
    picked_idx = order[:top_n]

    out_songs: list[Song] = []
    out_scores: list[float] = []
    for i in picked_idx:
        fn = fn_order[int(i)]
        s = filename_to_song.get(fn)
        if s is None:
            continue
        out_songs.append(s)
        out_scores.append(float(cos_mu[int(i)]))

    if not out_songs:
        raise SystemExit("No output songs after ranking.")

    export_to_xspf(
        songs=out_songs,
        predictions=out_scores,
        output_path=out_path,
        playlist_title=f"RAG query ({stem})",
        annotation_prefix="Cosine vs playlist centroid",
        scale_scores_to_five=False,
    )
    print(f"  Wrote: {out_path} ({len(out_songs)} tracks, sorted by cosine ↓)")
    return out_path
