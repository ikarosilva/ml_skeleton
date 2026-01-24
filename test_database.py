#!/usr/bin/env python3
"""Quick test script to verify Clementine database reading."""

from ml_skeleton.music.clementine_db import load_all_songs
from pathlib import Path

# Path from config
db_path = "/Music/database/clementine_backup_2026-01.db"

print("=" * 60)
print("Testing Clementine Database Reading")
print("=" * 60)
print(f"\nDatabase path: {db_path}")
print(f"Database exists: {Path(db_path).exists()}\n")

# Load songs
songs = load_all_songs(db_path, min_songs=500)

print("\n" + "=" * 60)
print("Database Statistics")
print("=" * 60)

# Count rated vs unrated
rated_songs = [s for s in songs if s.is_rated]
unrated_songs = [s for s in songs if not s.is_rated]

print(f"Total songs: {len(songs)}")
print(f"Rated songs: {len(rated_songs)} ({len(rated_songs)/len(songs)*100:.1f}%)")
print(f"Unrated songs: {len(unrated_songs)} ({len(unrated_songs)/len(songs)*100:.1f}%)")

if rated_songs:
    ratings = [s.rating for s in rated_songs]
    print(f"\nRating statistics:")
    print(f"  Min rating: {min(ratings):.2f}")
    print(f"  Max rating: {max(ratings):.2f}")
    print(f"  Average rating: {sum(ratings)/len(ratings):.2f}")

# Show sample songs
print("\n" + "=" * 60)
print("Sample Songs (first 5 rated)")
print("=" * 60)
for i, song in enumerate(rated_songs[:5], 1):
    print(f"\n{i}. {song.title}")
    print(f"   Artist: {song.artist}")
    print(f"   Album: {song.album}")
    print(f"   Rating: {song.rating:.2f} / 5.0")
    print(f"   Year: {song.year}")
    print(f"   File: {song.filename[:80]}...")

# Check file existence for first few rated songs
print("\n" + "=" * 60)
print("File Existence Check (first 5 rated songs)")
print("=" * 60)
for i, song in enumerate(rated_songs[:5], 1):
    exists = song.filepath.exists()
    status = "✓ EXISTS" if exists else "✗ MISSING"
    print(f"{i}. {status}: {song.filepath}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
