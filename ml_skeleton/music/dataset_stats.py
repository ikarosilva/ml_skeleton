"""Dataset statistics collection for model card generation."""

from typing import Dict, Any, List
from collections import Counter, defaultdict


def collect_preprocessing_stats(
    total_loaded: int,
    excluded_missing: int,
    excluded_duration: int,
    excluded_speech: int,
    excluded_duplicates: int,
    excluded_unknown_metadata: int,
    final_songs: int,
    rated_count: int,
    unrated_count: int
) -> Dict[str, Any]:
    """Collect preprocessing statistics.

    Args:
        total_loaded: Total songs loaded from database
        excluded_missing: Songs excluded (missing files)
        excluded_duration: Songs excluded (>15 minutes)
        excluded_speech: Songs excluded (speech detection)
        excluded_duplicates: Songs excluded (duplicates)
        excluded_unknown_metadata: Songs excluded (all metadata unknown) - encoder only
        final_songs: Final songs after filtering
        rated_count: Number of rated songs
        unrated_count: Number of unrated songs

    Returns:
        Dictionary with preprocessing statistics
    """
    return {
        "total_songs_loaded": total_loaded,
        "songs_excluded_missing": excluded_missing,
        "songs_excluded_duration": excluded_duration,
        "songs_excluded_speech": excluded_speech,
        "songs_excluded_duplicates": excluded_duplicates,
        "songs_excluded_unknown_metadata": excluded_unknown_metadata,
        "final_songs": final_songs,
        "rated_songs": rated_count,
        "unrated_songs": unrated_count
    }


def collect_dataset_stats(songs: List, only_rated: bool = False) -> Dict[str, Any]:
    """Collect dataset statistics for model card.

    Args:
        songs: List of Song objects
        only_rated: If True, only include rated songs in statistics

    Returns:
        Dictionary with dataset statistics:
        - total_songs: Total number of songs
        - total_artists: Number of unique artists
        - total_albums: Number of unique albums
        - year_distribution: Dict mapping 5-year periods to counts
        - rating_distribution: Dict mapping rating bins to counts (if rated)
    """
    if only_rated:
        songs = [s for s in songs if s.is_rated]

    total_songs = len(songs)

    # Collect unique artists and albums
    artists = set()
    albums = set()
    years = []
    ratings = []

    for song in songs:
        artists.add(song.artist)
        albums.add(f"{song.artist}|||{song.album}")  # Use album key format
        if song.year and song.year > 0:
            years.append(song.year)
        if only_rated and song.is_rated:
            ratings.append(song.rating)

    # Year distribution (5-year periods)
    year_distribution = {}
    if years:
        min_year = min(years)
        max_year = max(years)

        # Create 5-year bins
        for year in range((min_year // 5) * 5, max_year + 5, 5):
            period = f"{year}-{year+4}"
            count = sum(1 for y in years if year <= y < year + 5)
            if count > 0:
                year_distribution[period] = count

    # Rating distribution (for rated songs only)
    rating_distribution = {}
    if ratings:
        # Bin ratings: 0-1, 1-2, 2-3, 3-4, 4-5
        for rating in ratings:
            bin_idx = int(rating)  # 0-5 -> 0-4 bins
            if bin_idx >= 5:
                bin_idx = 4
            bin_label = f"{bin_idx}-{bin_idx+1}"
            rating_distribution[bin_label] = rating_distribution.get(bin_label, 0) + 1

    return {
        "total_songs": total_songs,
        "total_artists": len(artists),
        "total_albums": len(albums),
        "year_distribution": year_distribution,
        "rating_distribution": rating_distribution if ratings else {},
        "genre_distribution": {}  # Placeholder - Clementine may not have genre info
    }


def collect_training_stats(
    trainer,
    training_time_seconds: float,
    dataset_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """Collect training statistics from trainer.

    Args:
        trainer: EncoderTrainer or ClassifierTrainer instance
        training_time_seconds: Total training time in seconds
        dataset_stats: Dataset statistics from collect_dataset_stats

    Returns:
        Dictionary with training statistics
    """
    history = trainer.history

    # Get best epoch info
    val_losses = history.get('val_loss', [])
    if val_losses:
        best_val_loss = min(val_losses)
        best_epoch = val_losses.index(best_val_loss) + 1
    else:
        best_val_loss = None
        best_epoch = None

    # Get final losses
    train_losses = history.get('train_loss', [])
    final_train_loss = train_losses[-1] if train_losses else None
    final_val_loss = val_losses[-1] if val_losses else None

    stats = {
        "epochs_run": len(train_losses),
        "training_time_seconds": training_time_seconds,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }

    # Merge dataset stats
    stats.update(dataset_stats)

    return stats
