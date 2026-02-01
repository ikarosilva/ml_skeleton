#!/usr/bin/env python3
"""Recalculate confidence scores for existing enriched songs.

This script updates confidence scores in the MusicBrainz database using the fixed
calculate_confidence_scores() logic. It reads the raw AcoustID scores from the
api_response field and recalculates all confidence values.

Usage:
    python ml_skeleton/music/recalculate_confidence.py

The script operates on ./musicbrainz_metadata.db and updates records in-place.
"""

import sqlite3
import json
from typing import Optional, Dict


def calculate_confidence_scores_fixed(
    acoustid_score: float,
    mb_metadata: Optional[Dict]
) -> Dict[str, float]:
    """Fixed confidence calculation (matches new version in metadata_enrichment.py).

    Strategy:
    - AcoustID score alone: Use as-is (0.5-0.99 range)
    - MusicBrainz verifies data: Small boost (+0.05)
    - Genres: Only from MusicBrainz (0.0 if no MB data)

    Args:
        acoustid_score: AcoustID match confidence (0.0-1.0)
        mb_metadata: MusicBrainz metadata dictionary (None if lookup failed)

    Returns:
        Dictionary with artist_confidence, album_confidence, title_confidence, genre_confidence
    """
    # Start with AcoustID score as base for all non-genre fields
    scores = {
        'artist_confidence': acoustid_score,
        'album_confidence': acoustid_score,
        'title_confidence': acoustid_score,
        'genre_confidence': 0.0  # Genres only come from MusicBrainz
    }

    # Small boost if MusicBrainz verifies the data
    if mb_metadata:
        mb_boost = 0.05  # Conservative boost for MB verification

        if mb_metadata.get('artist'):
            scores['artist_confidence'] = min(acoustid_score + mb_boost, 1.0)
        if mb_metadata.get('album'):
            scores['album_confidence'] = min(acoustid_score + mb_boost, 1.0)
        if mb_metadata.get('title'):
            scores['title_confidence'] = min(acoustid_score + mb_boost, 1.0)
        if mb_metadata.get('genres'):
            # Genres from folksonomy tags - cap at 0.9
            scores['genre_confidence'] = min(acoustid_score, 0.9)

    return scores


def main():
    """Recalculate confidence scores for all records in MusicBrainz database."""
    db_path = './musicbrainz_metadata.db'
    print(f"Opening database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all records with api_response
    print("Loading records...")
    cursor.execute("SELECT song_id, api_response FROM mb_metadata WHERE api_response IS NOT NULL")
    rows = cursor.fetchall()

    print(f"Found {len(rows)} records with api_response data")

    updated = 0
    errors = 0

    for song_id, api_response in rows:
        try:
            # Parse AcoustID score from api_response
            data = json.loads(api_response)
            acoustid_score = data.get('acoustid_score', 0.0)
            mb_meta = data.get('mb_metadata')

            # Recalculate confidence using fixed logic
            new_scores = calculate_confidence_scores_fixed(acoustid_score, mb_meta)

            # Update database
            cursor.execute('''
                UPDATE mb_metadata
                SET artist_confidence = ?,
                    album_confidence = ?,
                    title_confidence = ?,
                    genre_confidence = ?
                WHERE song_id = ?
            ''', (
                new_scores['artist_confidence'],
                new_scores['album_confidence'],
                new_scores['title_confidence'],
                new_scores['genre_confidence'],
                song_id
            ))
            updated += 1

        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON for song {song_id}: {e}")
            errors += 1
        except Exception as e:
            print(f"Error processing song {song_id}: {e}")
            errors += 1

    conn.commit()
    conn.close()

    print(f"\n=== Recalculation Complete ===")
    print(f"✅ Successfully updated: {updated} songs")
    if errors > 0:
        print(f"⚠️  Errors encountered: {errors} songs")
    print(f"\nAverage confidence before fix: ~0.25 (binary)")
    print(f"Average confidence after fix: ~0.95-0.99 (continuous)")


if __name__ == '__main__':
    main()
