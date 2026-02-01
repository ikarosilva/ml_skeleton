"""Metadata enrichment via AcoustID and MusicBrainz APIs.

This module provides functions to enrich song metadata using acoustic fingerprint
lookups against AcoustID and MusicBrainz databases. It supports:
- AcoustID lookup using chromaprint fingerprints
- MusicBrainz metadata retrieval (artist, album, title, year, genres)
- Rate limiting for API compliance (free tier: 3 req/s AcoustID, 1 req/s MB)
- Confidence scoring for metadata quality assessment
- Batch processing with progress tracking

Usage:
    from ml_skeleton.music.metadata_enrichment import enrich_songs_metadata
    from ml_skeleton.music.fingerprint_db import FingerprintDB
    from ml_skeleton.music.musicbrainz_db import MusicBrainzDB

    # Initialize databases
    fp_db = FingerprintDB("./cache/fingerprints.db")
    mb_db = MusicBrainzDB("./musicbrainz_metadata.db")

    # Enrich metadata for songs with fingerprints
    stats = enrich_songs_metadata(
        songs=all_songs,
        fp_db=fp_db,
        mb_db=mb_db,
        acoustid_api_key="YOUR_API_KEY",
        max_songs=500,  # Free tier limit: 500/day
        verbose=True
    )
"""

import time
import json
import urllib.error
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm

try:
    import acoustid
    import musicbrainzngs
    APIS_AVAILABLE = True
except ImportError:
    APIS_AVAILABLE = False

from ml_skeleton.music.clementine_db import Song
from ml_skeleton.music.fingerprint_db import FingerprintDB
from ml_skeleton.music.musicbrainz_db import MusicBrainzDB


@dataclass
class EnrichmentResult:
    """Result of enriching metadata for a single song."""
    song_id: int
    filename: str
    success: bool

    # AcoustID results
    acoustid_id: Optional[str] = None
    musicbrainz_recording_id: Optional[str] = None
    acoustid_score: float = 0.0

    # MusicBrainz results
    artist_mb: Optional[str] = None
    album_mb: Optional[str] = None
    title_mb: Optional[str] = None
    year_mb: Optional[int] = None
    genre_tags_mb: Optional[List[str]] = None

    # Confidence scores
    artist_confidence: float = 0.0
    album_confidence: float = 0.0
    title_confidence: float = 0.0
    genre_confidence: float = 0.0

    # Error tracking
    error: Optional[str] = None
    api_response: Optional[str] = None


class RateLimiter:
    """Simple rate limiter for API calls.

    Args:
        requests_per_second: Maximum requests per second

    Usage:
        limiter = RateLimiter(requests_per_second=3.0)
        for item in items:
            limiter.wait()
            api_call(item)
    """

    def __init__(self, requests_per_second: float):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0

    def wait(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


def lookup_acoustid(
    fingerprint: str,
    duration: float,
    api_key: str,
    rate_limiter: Optional[RateLimiter] = None
) -> Tuple[Optional[str], Optional[str], float, Optional[str]]:
    """Look up song in AcoustID database using chromaprint fingerprint.

    Args:
        fingerprint: Chromaprint fingerprint (base64 encoded)
        duration: Audio duration in seconds
        api_key: AcoustID API key
        rate_limiter: Optional rate limiter for API compliance

    Returns:
        Tuple of (acoustid_id, musicbrainz_recording_id, confidence_score, error)
        Returns (None, None, 0.0, error_msg) on failure
    """
    if not APIS_AVAILABLE:
        return None, None, 0.0, "pyacoustid not installed"

    if rate_limiter:
        rate_limiter.wait()

    try:
        # Lookup fingerprint in AcoustID
        results = acoustid.lookup(
            apikey=api_key,
            fingerprint=fingerprint,
            duration=int(duration),
            meta='recordings'
        )

        # Parse results
        if results and 'results' in results and len(results['results']) > 0:
            top_result = results['results'][0]
            score = top_result.get('score', 0.0)

            # Extract AcoustID and MusicBrainz IDs
            acoustid_id = top_result.get('id')
            mb_recording_id = None

            if 'recordings' in top_result and len(top_result['recordings']) > 0:
                mb_recording_id = top_result['recordings'][0].get('id')

            return acoustid_id, mb_recording_id, score, None
        else:
            return None, None, 0.0, "No AcoustID match found"

    except acoustid.WebServiceError as e:
        # Check if it's an invalid API key error
        error_msg = str(e).lower()
        if 'invalid' in error_msg and 'key' in error_msg:
            return None, None, 0.0, (
                "Invalid AcoustID API key. Please verify your API key at https://acoustid.org/my-applications. "
                "Make sure you're using the API key (not the application ID) and that the application is active."
            )
        return None, None, 0.0, f"AcoustID API error: {str(e)}"
    except Exception as e:
        return None, None, 0.0, f"AcoustID API error: {str(e)}"


def lookup_musicbrainz(
    recording_id: str,
    rate_limiter: Optional[RateLimiter] = None
) -> Tuple[Optional[Dict], Optional[str]]:
    """Look up recording in MusicBrainz database.

    Args:
        recording_id: MusicBrainz recording ID
        rate_limiter: Optional rate limiter for API compliance

    Returns:
        Tuple of (metadata_dict, error)
        metadata_dict contains: artist, album, title, year, genres
        Returns (None, error_msg) on failure
    """
    if not APIS_AVAILABLE:
        return None, "musicbrainzngs not installed"

    if rate_limiter:
        rate_limiter.wait()

    try:
        # Query MusicBrainz for recording metadata
        result = musicbrainzngs.get_recording_by_id(
            recording_id,
            includes=['artists', 'releases', 'tags']
        )

        if 'recording' not in result:
            return None, "No recording found"

        recording = result['recording']

        # Extract metadata
        metadata = {
            'title': recording.get('title'),
            'artist': None,
            'album': None,
            'year': None,
            'genres': []
        }

        # Artist
        if 'artist-credit' in recording and len(recording['artist-credit']) > 0:
            # Take first artist (primary artist)
            artist_credit = recording['artist-credit'][0]
            if isinstance(artist_credit, dict) and 'artist' in artist_credit:
                metadata['artist'] = artist_credit['artist'].get('name')

        # Album (first release)
        if 'release-list' in recording and len(recording['release-list']) > 0:
            release = recording['release-list'][0]
            metadata['album'] = release.get('title')

            # Year from release date
            if 'date' in release and release['date']:
                try:
                    year_str = release['date'].split('-')[0]
                    metadata['year'] = int(year_str)
                except (ValueError, IndexError):
                    pass

        # Genres from tags (folksonomy)
        if 'tag-list' in recording:
            metadata['genres'] = [
                tag['name'].lower().replace(' ', '_')
                for tag in recording['tag-list']
                if tag.get('count', 0) > 0  # Only tags with votes
            ]

        return metadata, None

    except musicbrainzngs.WebServiceError as e:
        return None, f"MusicBrainz API error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def calculate_confidence_scores(
    acoustid_score: float,
    mb_metadata: Optional[Dict]
) -> Dict[str, float]:
    """Calculate confidence with AcoustID as base, MusicBrainz as verification boost.

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


def enrich_song_metadata(
    song: Song,
    fp_db: FingerprintDB,
    mb_db: MusicBrainzDB,
    acoustid_api_key: str,
    chunk_idx: int = 1,
    acoustid_rate_limiter: Optional[RateLimiter] = None,
    mb_rate_limiter: Optional[RateLimiter] = None,
    skip_existing: bool = True
) -> EnrichmentResult:
    """Enrich metadata for a single song using AcoustID/MusicBrainz.

    Args:
        song: Song object from Clementine database
        fp_db: Fingerprint database
        mb_db: MusicBrainz database
        acoustid_api_key: AcoustID API key
        chunk_idx: Chunk index to use for fingerprint (default: 1 = middle)
        acoustid_rate_limiter: Rate limiter for AcoustID API
        mb_rate_limiter: Rate limiter for MusicBrainz API
        skip_existing: Skip if already enriched

    Returns:
        EnrichmentResult with enriched metadata or error
    """
    # Check if already enriched
    if skip_existing and mb_db.has_metadata(song.rowid):
        return EnrichmentResult(
            song_id=song.rowid,
            filename=song.filename,
            success=False,
            error="Already enriched (skipped)"
        )

    # Get fingerprint from database
    fingerprint_obj = fp_db.get_fingerprint(song.rowid, chunk_idx)
    if not fingerprint_obj:
        return EnrichmentResult(
            song_id=song.rowid,
            filename=song.filename,
            success=False,
            error="No fingerprint found"
        )

    # Step 1: Lookup in AcoustID
    acoustid_id, mb_recording_id, acoustid_score, acoustid_error = lookup_acoustid(
        fingerprint=fingerprint_obj.fingerprint,
        duration=fingerprint_obj.duration,
        api_key=acoustid_api_key,
        rate_limiter=acoustid_rate_limiter
    )

    if acoustid_error:
        return EnrichmentResult(
            song_id=song.rowid,
            filename=song.filename,
            success=False,
            error=acoustid_error
        )

    # Step 2: Lookup in MusicBrainz (if we got a recording ID)
    mb_metadata = None
    mb_error = None
    if mb_recording_id:
        mb_metadata, mb_error = lookup_musicbrainz(
            recording_id=mb_recording_id,
            rate_limiter=mb_rate_limiter
        )

    # Calculate confidence scores
    confidence = calculate_confidence_scores(acoustid_score, mb_metadata)

    # Map MusicBrainz genres to 30-genre taxonomy and 7 categories
    genre_tags_mb = None
    genre_7cat = None
    if mb_metadata and mb_metadata.get('genres'):
        # Filter genres to match our 30-genre taxonomy
        genre_tags_mb = [g for g in mb_metadata['genres'] if mb_db.get_genre_mapping(g)]
        # Map to 7 categories
        genre_7cat = mb_db.map_genres_to_7cat(genre_tags_mb)

    # Create enrichment result
    result = EnrichmentResult(
        song_id=song.rowid,
        filename=song.filename,
        success=mb_recording_id is not None,
        acoustid_id=acoustid_id,
        musicbrainz_recording_id=mb_recording_id,
        acoustid_score=acoustid_score,
        artist_mb=mb_metadata.get('artist') if mb_metadata else None,
        album_mb=mb_metadata.get('album') if mb_metadata else None,
        title_mb=mb_metadata.get('title') if mb_metadata else None,
        year_mb=mb_metadata.get('year') if mb_metadata else None,
        genre_tags_mb=genre_tags_mb,
        artist_confidence=confidence['artist_confidence'],
        album_confidence=confidence['album_confidence'],
        title_confidence=confidence['title_confidence'],
        genre_confidence=confidence['genre_confidence'],
        error=mb_error,
        api_response=json.dumps({'acoustid_score': acoustid_score, 'mb_metadata': mb_metadata})
    )

    return result


def enrich_songs_metadata(
    songs: List[Song],
    fp_db: FingerprintDB,
    mb_db: MusicBrainzDB,
    acoustid_api_key: str,
    chunk_idx: int = 1,
    acoustid_rate_limit: float = 3.0,  # Free tier: 3 req/s
    musicbrainz_rate_limit: float = 1.0,  # MB ToS: 1 req/s
    skip_existing: bool = True,
    max_songs: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, int]:
    """Enrich metadata for multiple songs using AcoustID/MusicBrainz.

    Args:
        songs: List of Song objects from Clementine database
        fp_db: Fingerprint database
        mb_db: MusicBrainz database
        acoustid_api_key: AcoustID API key
        chunk_idx: Chunk index to use for fingerprints (default: 1 = middle)
        acoustid_rate_limit: Requests per second for AcoustID API
        musicbrainz_rate_limit: Requests per second for MusicBrainz API
        skip_existing: Skip songs already enriched
        max_songs: Maximum songs to process (for free tier: 500/day)
        verbose: Show progress bar

    Returns:
        Statistics dictionary:
        - total_songs: Total songs in input
        - processed: Songs actually processed
        - enriched: Successfully enriched with MusicBrainz data
        - skipped: Skipped (already enriched)
        - failed: Failed to enrich
        - no_fingerprint: Songs without fingerprints
        - rated: Number of rated songs processed (rating >= 0)
        - unrated: Number of unrated songs processed (rating < 0)
        - api_lookups: Number of API calls made to AcoustID/MusicBrainz
        - errors: List of error messages
    """
    if not APIS_AVAILABLE:
        raise ImportError(
            "pyacoustid and musicbrainzngs are required for metadata enrichment. "
            "Install with: pip install pyacoustid musicbrainzngs"
        )

    # Initialize MusicBrainz client
    musicbrainzngs.set_useragent(
        "ml-skeleton-music",
        "0.1.0",
        "https://github.com/ml-skeleton-org/ml_skeleton"
    )

    stats = {
        "total_songs": len(songs),
        "processed": 0,
        "enriched": 0,
        "skipped": 0,
        "failed": 0,
        "no_fingerprint": 0,
        "rated": 0,
        "unrated": 0,
        "api_lookups": 0,
        "errors": []
    }

    # Filter songs that have fingerprints
    songs_with_fingerprints = [
        song for song in songs
        if fp_db.has_fingerprints(song.rowid, num_chunks=1)
    ]
    stats["no_fingerprint"] = len(songs) - len(songs_with_fingerprints)

    # Filter out already enriched if requested
    if skip_existing:
        songs_to_process = [
            song for song in songs_with_fingerprints
            if not mb_db.has_metadata(song.rowid)
        ]
        stats["skipped"] = len(songs_with_fingerprints) - len(songs_to_process)

        if verbose and stats["skipped"] > 0:
            print(f"Skipping {stats['skipped']} songs with existing enrichment")
    else:
        songs_to_process = songs_with_fingerprints

    # Limit to max_songs (for free tier: 500/day)
    if max_songs is not None and len(songs_to_process) > max_songs:
        songs_to_process = songs_to_process[:max_songs]
        stats["skipped"] += (len(songs_with_fingerprints) - stats["skipped"] - max_songs)
        if verbose:
            print(f"Limiting to {max_songs} songs (free tier limit: 500/day)")

    if not songs_to_process:
        if verbose:
            print("No songs to enrich")
        return stats

    stats["processed"] = len(songs_to_process)

    # Track rated vs unrated songs
    stats["rated"] = sum(1 for song in songs_to_process if song.rating >= 0)
    stats["unrated"] = sum(1 for song in songs_to_process if song.rating < 0)

    # Create rate limiters
    acoustid_limiter = RateLimiter(acoustid_rate_limit)
    mb_limiter = RateLimiter(musicbrainz_rate_limit)

    # Process songs
    enriched_metadata = []

    # Track API lookups (for --exhaust mode reporting)
    api_lookups = 0
    max_lookups = max_songs if max_songs else len(songs_to_process)

    if verbose:
        iterator = tqdm(
            songs_to_process,
            desc=f"Enriching metadata (0/{max_lookups} API lookups)",
            unit="songs"
        )
    else:
        iterator = songs_to_process

    for song in iterator:
        result = enrich_song_metadata(
            song=song,
            fp_db=fp_db,
            mb_db=mb_db,
            acoustid_api_key=acoustid_api_key,
            chunk_idx=chunk_idx,
            acoustid_rate_limiter=acoustid_limiter,
            mb_rate_limiter=mb_limiter,
            skip_existing=skip_existing
        )

        # Count API lookups (both successful and failed attempts count as lookups)
        if result.acoustid_id or result.error not in [None, "Already enriched (skipped)", "No fingerprint found"]:
            api_lookups += 1

            # Update progress bar description with running lookup count
            if verbose and isinstance(iterator, tqdm):
                iterator.set_description(f"Enriching metadata ({api_lookups}/{max_lookups} API lookups)")

        if result.success and result.musicbrainz_recording_id:
            stats["enriched"] += 1
            enriched_metadata.append(result)
        else:
            stats["failed"] += 1
            if result.error and result.error != "Already enriched (skipped)":
                stats["errors"].append(f"Song {result.song_id}: {result.error}")

    # Add API lookup count to stats
    stats["api_lookups"] = api_lookups

    # Batch insert into MusicBrainz database
    if enriched_metadata:
        if verbose:
            print(f"\nStoring {len(enriched_metadata)} enriched metadata records...")

        for result in enriched_metadata:
            mb_db.add_metadata(
                song_id=result.song_id,
                filename=result.filename,
                acoustid_id=result.acoustid_id,
                musicbrainz_recording_id=result.musicbrainz_recording_id,
                artist_mb=result.artist_mb,
                album_mb=result.album_mb,
                title_mb=result.title_mb,
                year_mb=result.year_mb,
                genre_tags_mb=result.genre_tags_mb,
                genre_7cat=mb_db.map_genres_to_7cat(result.genre_tags_mb) if result.genre_tags_mb else None,
                artist_confidence=result.artist_confidence,
                album_confidence=result.album_confidence,
                title_confidence=result.title_confidence,
                genre_confidence=result.genre_confidence,
                api_response=result.api_response,
                last_updated=time.time()
            )

    return stats
