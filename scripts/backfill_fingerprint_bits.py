#!/usr/bin/env python3
"""Backfill the fingerprint DB 'bits' column so training uses precomputed bits (no chromaprint decode).

Run once on an existing DB that has 'fingerprint' but NULL 'bits'. After this, encoder training
can use chromaprint loss without calling the chromaprint C library (avoids malloc crash).

Usage:
  python scripts/backfill_fingerprint_bits.py --db /music-cache/fingerprints.db
  python scripts/backfill_fingerprint_bits.py --db ./cache/fingerprints.db --batch 1000
  # With pipeline (uses config DB path):
  ./run_music_pipeline.sh backfill-fingerprint-bits
  # Optional: limit rows per run if decode crashes (run repeatedly until done):
  ./run_music_pipeline.sh backfill-fingerprint-bits -- --limit 10000
  # Use subprocess batches to avoid chromaprint C library "double free" (default 500 per process):
  python scripts/backfill_fingerprint_bits.py --db /path/to/fingerprints.db --subprocess-batch 500
  # If workers still crash (malloc/double free), use one decode per process (slower but safe):
  python scripts/backfill_fingerprint_bits.py --db /path/to/fingerprints.db --one-per-process
"""

import argparse
import base64
import multiprocessing
import pickle
import sqlite3
import subprocess
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Match extraction process state: load torch before chromaprint (avoids decoder crash in some envs)
import torch  # noqa: E402

from ml_skeleton.music.chunk_fingerprinter import _fingerprint_to_bits_blob, fingerprint_to_bits


def _decode_one_row(item):
    """Decode a single fingerprint in a worker process (import here so chromaprint loads only in child)."""
    from ml_skeleton.music.chunk_fingerprinter import _fingerprint_to_bits_blob

    song_id = item["song_id"]
    chunk_idx = item["chunk_idx"]
    fp = item["fingerprint"]
    if isinstance(fp, bytes):
        fp = fp.decode("utf-8")
    blob = _fingerprint_to_bits_blob(fp)
    return (song_id, chunk_idx, blob)


def _debug_first_fingerprint(conn):
    """Print format and decode result for the first fingerprint in DB (for diagnostics)."""
    cur = conn.execute(
        """SELECT song_id, chunk_idx, fingerprint FROM fingerprints
           WHERE bits IS NULL AND fingerprint IS NOT NULL AND fingerprint != '' LIMIT 1"""
    )
    row = cur.fetchone()
    if not row:
        print("Debug: no row with fingerprint and NULL bits.")
        return
    fp = row["fingerprint"]
    print(f"Debug: first fingerprint: type={type(fp).__name__}, len={len(fp)}")
    print(f"  repr (first 120 chars): {repr(fp[:120])}")
    # Normalize to str for fingerprint_to_bits (DB may return bytes)
    fp_str = fp.decode("utf-8") if isinstance(fp, bytes) else fp
    # Check if it looks like base64
    try:
        b64norm = fp_str.replace("-", "+").replace("_", "/")
        b64decode = base64.standard_b64decode(b64norm)
        print(f"  base64.standard_b64decode: len={len(b64decode)} bytes, first 8 bytes: {b64decode[:8].hex()}")
    except Exception as e:
        print(f"  base64.standard_b64decode failed: {e}")
    # Try fingerprint_to_bits and capture exception
    try:
        bits = fingerprint_to_bits(fp_str)
        print(f"  fingerprint_to_bits: {bits is not None}, shape={getattr(bits, 'shape', None)}")
    except Exception as e:
        import traceback
        print(f"  fingerprint_to_bits raised: {e}")
        traceback.print_exc()
    # Try chromaprint.decode_fingerprint directly with base64=True and base64=False
    import chromaprint
    data = fp_str.encode("utf-8") if isinstance(fp_str, str) else fp
    for base64_arg in (True, False):
        try:
            decoded, algo = chromaprint.decode_fingerprint(data, base64=base64_arg)
            print(f"  chromaprint.decode_fingerprint(base64={base64_arg}): algo={algo}, decoded type={type(decoded).__name__}, len={len(decoded) if decoded is not None else 0}")
            if decoded is not None and len(decoded) > 0:
                print(f"    first element type={type(decoded[0]).__name__}, value={decoded[0]}")
        except Exception as e:
            import traceback
            print(f"  chromaprint.decode_fingerprint(base64={base64_arg}) raised: {e}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Backfill fingerprint bits column (precomputed 32-byte blobs).")
    parser.add_argument("--db", required=True, help="Path to fingerprints.db")
    parser.add_argument("--batch", type=int, default=500, help="Commit every N rows (default 500)")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Only count rows, do not update")
    parser.add_argument("--debug", action="store_true", help="Print format and decode error for first fingerprint, then exit")
    parser.add_argument(
        "--subprocess-batch",
        type=int,
        default=0,
        metavar="N",
        help="Decode in worker subprocess every N rows. Default 0 = all in main process (same as fingerprint_baseline extraction; use if workers crash).",
    )
    parser.add_argument(
        "--one-per-process",
        action="store_true",
        help="Run one decode per process (multiprocessing, maxtasksperchild=1). Use if subprocess batches still crash.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        metavar="J",
        help="Parallel jobs for --one-per-process (default 4).",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: DB not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Ensure bits column exists
    cur = conn.execute("PRAGMA table_info(fingerprints)")
    columns = [row[1] for row in cur.fetchall()]
    if "bits" not in columns:
        print("Adding 'bits' column...")
        conn.execute("ALTER TABLE fingerprints ADD COLUMN bits BLOB")
        conn.commit()

    cur = conn.execute(
        """SELECT song_id, chunk_idx, fingerprint FROM fingerprints
           WHERE bits IS NULL AND fingerprint IS NOT NULL AND fingerprint != ''"""
    )
    rows = cur.fetchall()
    if args.limit:
        rows = rows[: args.limit]
    total = len(rows)
    print(f"Found {total} rows with fingerprint but NULL bits.")

    if args.debug:
        _debug_first_fingerprint(conn)
        conn.close()
        return

    if args.dry_run:
        print("Dry run: no updates.")
        conn.close()
        return

    project_root = Path(__file__).resolve().parents[1]
    worker_script = Path(__file__).resolve().parent / "backfill_fingerprint_bits_worker.py"
    use_one_per_process = args.one_per_process
    use_subprocess = not use_one_per_process and args.subprocess_batch > 0
    if use_one_per_process:
        print(f"Using one decode per process (jobs={args.jobs}, maxtasksperchild=1) to avoid chromaprint C crash.")
    elif use_subprocess:
        print(f"Using subprocess batches (batch size {args.subprocess_batch}) to avoid chromaprint C library crash.")

    updated = 0
    failed = 0
    i = 0
    while i < total:
        if use_one_per_process:
            # Multiprocessing: one decode per process, then process exits (maxtasksperchild=1)
            batch_data = [
                {"song_id": r["song_id"], "chunk_idx": r["chunk_idx"], "fingerprint": r["fingerprint"]}
                for r in rows
            ]
            # Spawn (not fork) so workers don't inherit chromaprint C library state; each child imports it fresh
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=args.jobs, maxtasksperchild=1) as pool:
                for k, (sid, cidx, blob) in enumerate(pool.imap_unordered(_decode_one_row, batch_data, chunksize=1)):
                    if blob is not None:
                        conn.execute(
                            "UPDATE fingerprints SET bits = ? WHERE song_id = ? AND chunk_idx = ?",
                            (blob, sid, cidx),
                        )
                        updated += 1
                    else:
                        failed += 1
                    if (k + 1) % args.batch == 0:
                        conn.commit()
                        print(f"  Progress: {k + 1}/{total} (updated {updated}, failed {failed})")
            conn.commit()
            print(f"  Progress: {total}/{total} (updated {updated}, failed {failed})")
            i = total
            break
        if use_subprocess:
            chunk = rows[i : i + args.subprocess_batch]
            batch_data = [
                {"song_id": r["song_id"], "chunk_idx": r["chunk_idx"], "fingerprint": r["fingerprint"]}
                for r in chunk
            ]
            try:
                proc = subprocess.Popen(
                    [sys.executable, str(worker_script)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    cwd=str(project_root),
                )
                pickle.dump(batch_data, proc.stdin)
                proc.stdin.close()
                results = pickle.load(proc.stdout)
                proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(f"Worker exited with code {proc.returncode}")
            except Exception as e:
                print(f"  Worker failed at row {i + 1}: {e}")
                failed += len(chunk)
                i += len(chunk)
                continue
            for (sid, cidx, blob) in results:
                if blob is not None:
                    conn.execute(
                        "UPDATE fingerprints SET bits = ? WHERE song_id = ? AND chunk_idx = ?",
                        (blob, sid, cidx),
                    )
                    updated += 1
                else:
                    failed += 1
            i += len(chunk)
        else:
            row = rows[i]
            fp = row["fingerprint"]
            if isinstance(fp, bytes):
                fp = fp.decode("utf-8")
            blob = _fingerprint_to_bits_blob(fp)
            if blob is not None:
                conn.execute(
                    "UPDATE fingerprints SET bits = ? WHERE song_id = ? AND chunk_idx = ?",
                    (blob, row["song_id"], row["chunk_idx"]),
                )
                updated += 1
            else:
                failed += 1
            i += 1
        if i % args.batch == 0 or i == total:
            conn.commit()
            print(f"  Progress: {i}/{total} (updated {updated}, failed {failed})")

    conn.commit()
    conn.close()
    print(f"Done: updated {updated}, failed {failed}, total processed {total}.")


if __name__ == "__main__":
    main()
