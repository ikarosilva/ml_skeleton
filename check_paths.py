#!/usr/bin/env python3
"""Check a few file paths from the database."""

from ml_skeleton.music.clementine_db import load_all_songs

db_path = "/Music/database/clementine_backup_2026-01.db"
songs = load_all_songs(db_path)

print("Checking first 10 songs for file existence:\n")
for i, song in enumerate(songs[:10], 1):
    exists = song.filepath.exists()
    status = "✓" if exists else "✗"
    print(f"{i}. {status} {song.artist} - {song.title}")
    print(f"   Path: {song.filepath}")
    print(f"   Raw filename: {song.filename[:100]}...")
    print()

# Show summary
total = len(songs)
existing = sum(1 for s in songs if s.filepath.exists())
print(f"\nSummary:")
print(f"Total songs: {total}")
print(f"Files exist: {existing} ({existing/total*100:.1f}%)")
print(f"Files missing: {total-existing} ({(total-existing)/total*100:.1f}%)")
