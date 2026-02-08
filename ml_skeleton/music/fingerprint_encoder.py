"""Fingerprint-baseline encoder: produces embeddings from chromaprints only.

Use this as a drop-in encoder for the pipeline to train a classifier on
fingerprint-only "embeddings" (no audio). Enables ablation: compare classifier
trained on chromaprints vs on audio-encoder embeddings.

Pipeline: FingerprintDB -> 256-bit vector -> optional projection -> embedding_dim.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from .fingerprint_db import FingerprintDB
from .chunk_fingerprinter import fingerprint_to_bits, CHROMAPRINT_BITS


class FingerprintEncoder(nn.Module):
    """Dummy encoder that returns chromaprint-derived vectors keyed by song_id.

    Implements the same embedding API as the audio encoder: forward(x) returns
    (B, embedding_dim). For this encoder, x must be a Long tensor of song_ids
    (rowids from Clementine). Lookups are done against FingerprintDB (chunk_idx=0).
    Optionally projects 256 -> project_dim to match the main encoder's dimension.

    Args:
        fp_db_path: Path to fingerprint SQLite database.
        embedding_dim: Output dimension. If None, use CHROMAPRINT_BITS (256).
        project_dim: If set, add Linear(256, project_dim) so output matches
            the main encoder (e.g. 2048) for classifier compatibility.
        chromaprint_chunk_idx: Chunk index in DB (0 = full-file fingerprint).
    """

    def __init__(
        self,
        fp_db_path: str = "./cache/fingerprints.db",
        embedding_dim: Optional[int] = None,
        project_dim: Optional[int] = None,
        chromaprint_chunk_idx: int = 0,
    ):
        super().__init__()
        self.fp_db_path = Path(fp_db_path)
        self.chromaprint_chunk_idx = chromaprint_chunk_idx
        self._fp_db: Optional[FingerprintDB] = None
        self._cache: dict[int, torch.Tensor] = {}

        out_dim = project_dim if project_dim is not None else (embedding_dim or CHROMAPRINT_BITS)
        if project_dim is not None or (embedding_dim is not None and embedding_dim != CHROMAPRINT_BITS):
            self.projector = nn.Linear(CHROMAPRINT_BITS, out_dim)
            self._embedding_dim = out_dim
        else:
            self.projector = None
            self._embedding_dim = CHROMAPRINT_BITS

    def _get_db(self) -> FingerprintDB:
        if self._fp_db is None:
            self._fp_db = FingerprintDB(str(self.fp_db_path))
        return self._fp_db

    def forward(self, song_ids: torch.Tensor) -> torch.Tensor:
        """Return embedding vectors for the given song rowids.

        Args:
            song_ids: Long tensor of shape (B,) or (B, 1) - Clementine song rowids.

        Returns:
            Tensor of shape (B, embedding_dim).
        """
        if song_ids.dim() == 2:
            song_ids = song_ids.squeeze(1)
        device = song_ids.device
        db = self._get_db()
        vectors = []
        for i in range(song_ids.shape[0]):
            sid = int(song_ids[i].item())
            if sid in self._cache:
                vec = self._cache[sid]
            else:
                fp_obj = db.get_fingerprint(sid, self.chromaprint_chunk_idx)
                if fp_obj and fp_obj.fingerprint:
                    bits = fingerprint_to_bits(fp_obj.fingerprint)
                    if bits is not None:
                        vec = torch.from_numpy(bits).float()
                        self._cache[sid] = vec
                    else:
                        vec = torch.zeros(CHROMAPRINT_BITS, dtype=torch.float32)
                else:
                    vec = torch.zeros(CHROMAPRINT_BITS, dtype=torch.float32)
            vectors.append(vec)
        out = torch.stack(vectors, dim=0).to(device)
        if self.projector is not None:
            out = self.projector(out)
        return out

    def get_embedding_dim(self) -> int:
        return self._embedding_dim


class FingerprintBaselineDataset(torch.utils.data.Dataset):
    """Dataset of songs that have a fingerprint in the DB, for extraction only.

    Yields song_id and filename so the pipeline can call the fingerprint encoder
    and store embeddings by filename. No audio is loaded.

    Args:
        songs: List of Song objects (from Clementine).
        fp_db_path: Path to fingerprint database.
        chromaprint_chunk_idx: Chunk index to require (0 = full-file).
    """

    def __init__(
        self,
        songs: list,
        fp_db_path: str = "./cache/fingerprints.db",
        chromaprint_chunk_idx: int = 0,
    ):
        from .fingerprint_db import FingerprintDB
        db = FingerprintDB(fp_db_path)
        self.songs = [
            s for s in songs
            if db.has_fingerprints(s.rowid, num_chunks=1)
        ]
        self.fp_db_path = fp_db_path
        self.chromaprint_chunk_idx = chromaprint_chunk_idx
        if len(self.songs) < len(songs):
            print(f"FingerprintBaselineDataset: {len(self.songs)}/{len(songs)} songs have fingerprints")

    def __len__(self) -> int:
        return len(self.songs)

    def __getitem__(self, idx: int) -> dict:
        song = self.songs[idx]
        return {
            "song_id": song.rowid,
            "filename": song.filename if isinstance(song.filename, str) else song.filename.decode("utf-8"),
        }


def collate_fingerprint_baseline(batch: list) -> dict:
    """Collate batch of FingerprintBaselineDataset items."""
    return {
        "song_id": torch.tensor([b["song_id"] for b in batch], dtype=torch.long),
        "filename": [b["filename"] for b in batch],
    }


class _DummyLoss(nn.Module):
    """Placeholder loss for fingerprint_baseline (extraction-only, no training)."""

    def forward(self, *args, **kwargs) -> dict:
        return {"loss": torch.tensor(0.0, device=next(self.parameters(), torch.tensor(0.0)).device)}
