#!/usr/bin/env python3
"""Worker subprocess: read one batch of (song_id, chunk_idx, fingerprint) from stdin (pickle),
decode each to bits blob, write list of (song_id, chunk_idx, blob) to stdout (pickle).
Used by backfill_fingerprint_bits.py to avoid chromaprint C library heap corruption
when decoding many fingerprints in one process.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_skeleton.music.chunk_fingerprinter import _fingerprint_to_bits_blob


def main():
    batch = pickle.load(sys.stdin.buffer)
    results = []
    for item in batch:
        song_id = item["song_id"]
        chunk_idx = item["chunk_idx"]
        fp = item["fingerprint"]
        if isinstance(fp, bytes):
            fp = fp.decode("utf-8")
        blob = _fingerprint_to_bits_blob(fp)
        results.append((song_id, chunk_idx, blob))
    pickle.dump(results, sys.stdout.buffer)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
