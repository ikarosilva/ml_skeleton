#!/usr/bin/env python3
"""Check for missing audio files in the Clementine database.

This script identifies songs in the database whose audio files are missing
from the filesystem. Useful for data cleanup and debugging.
"""

import argparse
from pathlib import Path
from ml_skeleton.music.clementine_db import load_songs_from_db


def check_missing_files(db_path: str, output_file: str = None):
    """Check for missing audio files.

    Args:
        db_path: Path to Clementine database
        output_file: Optional path to save list of missing files
    """
    print(f"Loading songs from database: {db_path}")
    songs = load_songs_from_db(db_path)
    print(f"Total songs in database: {len(songs)}")

    missing_songs = []
    existing_songs = []

    print("\nChecking file existence...")
    for song in songs:
        if not Path(song.filename).exists():
            missing_songs.append(song)
        else:
            existing_songs.append(song)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Existing files: {len(existing_songs)} ({len(existing_songs)/len(songs)*100:.1f}%)")
    print(f"Missing files:  {len(missing_songs)} ({len(missing_songs)/len(songs)*100:.1f}%)")

    if missing_songs:
        print(f"\n{'='*60}")
        print("MISSING FILES BREAKDOWN")
        print(f"{'='*60}")

        # Analyze by rating status
        rated_missing = [s for s in missing_songs if s.rating > 0]
        unrated_missing = [s for s in missing_songs if s.rating == 0]

        print(f"Rated songs missing:   {len(rated_missing)}")
        print(f"Unrated songs missing: {len(unrated_missing)}")

        # Show first 20 examples
        print(f"\nFirst 20 missing files:")
        for i, song in enumerate(missing_songs[:20], 1):
            rating_str = f"{song.rating:.1f}" if song.rating > 0 else "unrated"
            print(f"  {i}. [{rating_str}] {song.artist} - {song.title}")
            print(f"     Path: {song.filename}")

        if len(missing_songs) > 20:
            print(f"  ... and {len(missing_songs) - 20} more")

        # Save to file if requested
        if output_file:
            print(f"\nSaving full list to: {output_file}")
            with open(output_file, 'w') as f:
                f.write("Missing Audio Files Report\n")
                f.write("="*80 + "\n\n")
                f.write(f"Total missing: {len(missing_songs)}\n")
                f.write(f"Rated: {len(rated_missing)}, Unrated: {len(unrated_missing)}\n\n")

                f.write("Full List:\n")
                f.write("-"*80 + "\n")
                for song in missing_songs:
                    rating_str = f"{song.rating:.1f}" if song.rating > 0 else "unrated"
                    f.write(f"[{rating_str}] {song.artist} - {song.title}\n")
                    f.write(f"  Path: {song.filename}\n")
                    f.write(f"  Album: {song.album}\n")
                    f.write(f"  Genre: {song.genre}\n\n")
            print(f"✓ Report saved")


def main():
    parser = argparse.ArgumentParser(
        description="Check for missing audio files in Clementine database"
    )
    parser.add_argument(
        "--db",
        default="/Music/database/clementine_backup_2026-03.db",
        help="Path to Clementine database"
    )
    parser.add_argument(
        "--output",
        default="missing_files_report.txt",
        help="Output file for missing files report"
    )

    args = parser.parse_args()
    check_missing_files(args.db, args.output)


if __name__ == "__main__":
    main()
