"""XSPF playlist generation for human-in-the-loop reinforcement learning.

Generates playlists compatible with Clementine music player for human rating.
"""

import html
from pathlib import Path
from urllib.parse import unquote
from typing import Optional

from .clementine_db import Song


def export_to_xspf(
    songs: list[Song],
    predictions: list[float],
    output_path: Path,
    playlist_title: str = "Music Recommendations",
    annotation_prefix: str = "Predicted rating"
) -> None:
    """Export songs with predictions to XSPF playlist (Clementine-compatible).

    Args:
        songs: List of Song objects
        predictions: List of predicted ratings (same length as songs, in [0, 1])
        output_path: Path to output XSPF file
        playlist_title: Title for the playlist
        annotation_prefix: Prefix for annotation text (e.g., "Predicted rating")
    """
    if len(songs) != len(predictions):
        raise ValueError(f"Songs ({len(songs)}) and predictions ({len(predictions)}) must have same length")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<playlist version="1" xmlns="http://xspf.org/ns/0/">\n')
        f.write(f'  <title>{html.escape(playlist_title)}</title>\n')
        f.write('  <trackList>\n')

        for song, prediction in zip(songs, predictions):
            # Handle filename (might be bytes or have file:// prefix)
            filename = song.filename
            if isinstance(filename, bytes):
                filename = filename.decode("utf-8", errors="replace")

            # Extract location path
            location = filename
            if location.startswith("file://"):
                location = location[7:]
            location = unquote(location)

            # Escape XML special characters
            title = html.escape(song.title)
            artist = html.escape(song.artist)
            album = html.escape(song.album)
            location_escaped = html.escape(location)

            # Create annotation with predicted rating (scaled to 0-5 for Clementine)
            rating_5_scale = prediction * 5.0
            annotation = f"{annotation_prefix}: {rating_5_scale:.2f}/5.00"
            annotation_escaped = html.escape(annotation)

            f.write('    <track>\n')
            f.write(f'      <location>{location_escaped}</location>\n')
            f.write(f'      <title>{title}</title>\n')
            if artist:
                f.write(f'      <creator>{artist}</creator>\n')
            if album:
                f.write(f'      <album>{album}</album>\n')
            f.write(f'      <annotation>{annotation_escaped}</annotation>\n')
            f.write('    </track>\n')

        f.write('  </trackList>\n')
        f.write('</playlist>\n')

    print(f"Exported {len(songs)} songs to XSPF playlist: {output_path}")


def generate_uncertainty_playlist(
    songs: list[Song],
    predictions: list[float],
    uncertainties: list[float],
    output_path: Path,
    top_n: int = 50,
    playlist_title: str = "High Uncertainty - Please Rate"
) -> None:
    """Generate playlist with most uncertain predictions for human rating.

    Selects songs where the model is most uncertain (high variance/entropy)
    to maximize information gain from human ratings.

    Args:
        songs: List of Song objects
        predictions: List of predicted ratings in [0, 1]
        uncertainties: List of uncertainty scores (higher = more uncertain)
        output_path: Path to output XSPF file
        top_n: Number of songs to include in playlist
        playlist_title: Title for the playlist
    """
    if len(songs) != len(predictions) or len(songs) != len(uncertainties):
        raise ValueError("Songs, predictions, and uncertainties must have same length")

    # Sort by uncertainty (descending)
    sorted_indices = sorted(
        range(len(uncertainties)),
        key=lambda i: uncertainties[i],
        reverse=True
    )

    # Select top-N most uncertain
    top_indices = sorted_indices[:top_n]

    selected_songs = [songs[i] for i in top_indices]
    selected_predictions = [predictions[i] for i in top_indices]
    selected_uncertainties = [uncertainties[i] for i in top_indices]

    # Export with uncertainty annotation
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<playlist version="1" xmlns="http://xspf.org/ns/0/">\n')
        f.write(f'  <title>{html.escape(playlist_title)}</title>\n')
        f.write('  <trackList>\n')

        for song, pred, unc in zip(selected_songs, selected_predictions, selected_uncertainties):
            # Handle filename
            filename = song.filename
            if isinstance(filename, bytes):
                filename = filename.decode("utf-8", errors="replace")

            location = filename
            if location.startswith("file://"):
                location = location[7:]
            location = unquote(location)

            # Escape XML
            title = html.escape(song.title)
            artist = html.escape(song.artist)
            album = html.escape(song.album)
            location_escaped = html.escape(location)

            # Annotation with prediction and uncertainty
            rating_5_scale = pred * 5.0
            annotation = f"Predicted: {rating_5_scale:.2f}/5.00 (Uncertainty: {unc:.3f})"
            annotation_escaped = html.escape(annotation)

            f.write('    <track>\n')
            f.write(f'      <location>{location_escaped}</location>\n')
            f.write(f'      <title>{title}</title>\n')
            if artist:
                f.write(f'      <creator>{artist}</creator>\n')
            if album:
                f.write(f'      <album>{album}</album>\n')
            f.write(f'      <annotation>{annotation_escaped}</annotation>\n')
            f.write('    </track>\n')

        f.write('  </trackList>\n')
        f.write('</playlist>\n')

    print(f"Exported {len(selected_songs)} high-uncertainty songs to: {output_path}")
    if selected_uncertainties:
        print(f"Average uncertainty: {sum(selected_uncertainties) / len(selected_uncertainties):.3f}")
    else:
        print("No songs with uncertainties available")


def generate_best_predictions_playlist(
    songs: list[Song],
    predictions: list[float],
    output_path: Path,
    top_n: int = 50,
    playlist_title: str = "Top Predictions - Validate Quality"
) -> None:
    """Generate playlist with highest predicted ratings for validation.

    Selects songs with highest predicted ratings to validate model quality
    and ensure recommendations are good.

    Args:
        songs: List of Song objects
        predictions: List of predicted ratings in [0, 1]
        output_path: Path to output XSPF file
        top_n: Number of songs to include in playlist
        playlist_title: Title for the playlist
    """
    if len(songs) != len(predictions):
        raise ValueError("Songs and predictions must have same length")

    # Sort by prediction (descending)
    sorted_indices = sorted(
        range(len(predictions)),
        key=lambda i: predictions[i],
        reverse=True
    )

    # Select top-N highest predictions
    top_indices = sorted_indices[:top_n]

    selected_songs = [songs[i] for i in top_indices]
    selected_predictions = [predictions[i] for i in top_indices]

    # Export
    export_to_xspf(
        songs=selected_songs,
        predictions=selected_predictions,
        output_path=output_path,
        playlist_title=playlist_title,
        annotation_prefix="Predicted rating"
    )

    if selected_predictions:
        print(f"Average predicted rating: {sum(selected_predictions) / len(selected_predictions):.3f}")
    else:
        print("No songs with predictions available")


def compute_prediction_uncertainty(
    predictions: list[float],
    method: str = "distance_from_middle"
) -> list[float]:
    """Compute uncertainty scores for predictions.

    For a simple classifier without explicit uncertainty quantification,
    we approximate uncertainty using distance from decision boundary.

    Args:
        predictions: List of predicted ratings in [0, 1]
        method: Uncertainty estimation method:
            - "distance_from_middle": Distance from 0.5 (middle rating)
            - "entropy": Entropy-based (treats as binary: like vs dislike)

    Returns:
        List of uncertainty scores (higher = more uncertain)
    """
    import math

    uncertainties = []

    for pred in predictions:
        if method == "distance_from_middle":
            # Songs near 0.5 are most uncertain (neutral)
            # Songs near 0 or 1 are least uncertain (strong preference)
            uncertainty = 1.0 - abs(pred - 0.5) * 2.0
            uncertainties.append(uncertainty)

        elif method == "entropy":
            # Treat as binary classification: like (>0.5) vs dislike (<=0.5)
            # Entropy is highest at p=0.5
            p = max(0.001, min(0.999, pred))  # Clip to avoid log(0)
            entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
            # Normalize to [0, 1] (max entropy = 1.0)
            normalized_entropy = entropy
            uncertainties.append(normalized_entropy)

        else:
            raise ValueError(f"Unknown uncertainty method: {method}")

    return uncertainties


def generate_human_feedback_playlists(
    songs: list[Song],
    predictions: list[float],
    output_dir: Path,
    top_n_uncertain: int = 100,
    top_n_best: int = 50,
    uncertainty_method: str = "distance_from_middle"
) -> dict:
    """Generate both uncertainty and best-predictions playlists.

    This is the main function for human-in-the-loop reinforcement learning.

    Args:
        songs: List of unrated Song objects
        predictions: List of predicted ratings in [0, 1]
        output_dir: Directory to save playlists
        top_n_uncertain: Number of uncertain songs for human rating
        top_n_best: Number of best predictions for validation
        uncertainty_method: Method for computing uncertainty

    Returns:
        Dictionary with statistics about generated playlists
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute uncertainties
    print("\nComputing prediction uncertainties...")
    uncertainties = compute_prediction_uncertainty(predictions, method=uncertainty_method)

    # Generate uncertainty playlist (for maximum learning)
    print(f"\nGenerating high-uncertainty playlist ({top_n_uncertain} songs)...")
    uncertainty_path = output_dir / "recommender_help.xspf"
    generate_uncertainty_playlist(
        songs=songs,
        predictions=predictions,
        uncertainties=uncertainties,
        output_path=uncertainty_path,
        top_n=top_n_uncertain,
        playlist_title="High Uncertainty - Help Train Model"
    )

    # Generate best predictions playlist (for validation)
    print(f"\nGenerating best-predictions playlist ({top_n_best} songs)...")
    best_path = output_dir / "recommender_best.xspf"
    generate_best_predictions_playlist(
        songs=songs,
        predictions=predictions,
        output_path=best_path,
        top_n=top_n_best,
        playlist_title="Top Predictions - Validate Quality"
    )

    # Compute statistics
    stats = {
        "total_songs": len(songs),
        "uncertainty_playlist_size": top_n_uncertain,
        "best_predictions_playlist_size": top_n_best,
        "uncertainty_playlist_path": str(uncertainty_path),
        "best_predictions_playlist_path": str(best_path),
        "avg_uncertainty": sum(uncertainties) / len(uncertainties) if uncertainties else 0.0,
        "max_uncertainty": max(uncertainties) if uncertainties else 0.0,
        "avg_prediction": sum(predictions) / len(predictions) if predictions else 0.0,
        "max_prediction": max(predictions) if predictions else 0.0,
        "min_prediction": min(predictions) if predictions else 0.0
    }

    print("\n" + "=" * 60)
    print("HUMAN FEEDBACK PLAYLISTS GENERATED")
    print("=" * 60)
    print(f"Uncertainty playlist: {uncertainty_path}")
    print(f"Best predictions playlist: {best_path}")
    print(f"\nNext steps:")
    print(f"1. Open playlists in Clementine music player")
    print(f"2. Listen and rate songs in your library")
    print(f"3. Run training again with updated ratings")
    print(f"4. Repeat for continuous improvement!")

    return stats
