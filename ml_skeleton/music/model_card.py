"""Model card generation for HuggingFace compatibility.

Generates comprehensive model cards documenting:
- Model architecture and training procedure
- Dataset statistics and preprocessing
- Performance metrics and limitations
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json


class ModelCardGenerator:
    """Generate HuggingFace-compatible model cards for music recommendation models."""

    def __init__(self):
        self.encoder_stats = {}
        self.classifier_stats = {}
        self.preprocessing_stats = {}
        self.config = {}

    def set_config(self, config: Dict[str, Any]):
        """Store configuration for model card generation."""
        self.config = config

    def set_preprocessing_stats(self, stats: Dict[str, Any]):
        """Set preprocessing statistics.

        Args:
            stats: Dictionary with keys:
                - total_songs_loaded: Total songs from database
                - songs_excluded_missing: Songs with missing audio files
                - songs_excluded_duration: Songs >15 minutes
                - songs_excluded_speech: Songs filtered by speech detection
                - songs_excluded_duplicates: Duplicate songs removed
                - songs_excluded_unknown_metadata: Songs with all-unknown metadata (encoder only)
                - final_songs: Final number of songs after filtering
                - rated_songs: Number of rated songs
                - unrated_songs: Number of unrated songs
        """
        self.preprocessing_stats = stats

    def set_encoder_stats(self, stats: Dict[str, Any]):
        """Set encoder training statistics.

        Args:
            stats: Dictionary with keys:
                - epochs_run: Total epochs completed
                - training_time_seconds: Total training time
                - final_train_loss: Final training loss
                - final_val_loss: Final validation loss
                - best_val_loss: Best validation loss achieved
                - best_epoch: Epoch with best validation loss
                - total_songs: Total songs in training set
                - total_artists: Number of unique artists
                - total_albums: Number of unique albums
                - year_distribution: Dict mapping 5-year periods to counts
                - genre_distribution: Dict mapping genres to counts (optional)
        """
        self.encoder_stats = stats

    def set_classifier_stats(self, stats: Dict[str, Any]):
        """Set classifier training statistics.

        Args:
            stats: Dictionary with keys:
                - epochs_run: Total epochs completed
                - training_time_seconds: Total training time
                - final_train_loss: Final training loss
                - final_val_loss: Final validation loss
                - best_val_loss: Best validation loss achieved
                - best_epoch: Epoch with best validation loss
                - total_rated_songs: Total rated songs
                - total_artists: Number of unique artists (rated songs only)
                - total_albums: Number of unique albums (rated songs only)
                - rating_distribution: Dict mapping rating bins to counts
                - genre_distribution: Dict mapping genres to counts (optional)
        """
        self.classifier_stats = stats

    def generate(self, output_path: Path):
        """Generate and save model card to file.

        Args:
            output_path: Path to save model card (e.g., MODEL_CARD.md)
        """
        card = self._build_card()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(card)

        print(f"\n✓ Model card saved to: {output_path}")

    def _build_card(self) -> str:
        """Build the complete model card as markdown."""
        sections = [
            self._header(),
            self._model_description(),
            self._intended_use(),
            self._model_architecture(),
            self._training_procedure(),
            self._dataset_information(),
            self._preprocessing(),
            self._encoder_training(),
            self._classifier_training(),
            self._performance_metrics(),
            self._limitations(),
            self._usage_example(),
            self._citation(),
            self._metadata()
        ]

        return "\n\n".join(sections)

    def _header(self) -> str:
        """Generate model card header."""
        return f"""---
library_name: pytorch
tags:
- music-recommendation
- audio-classification
- contrastive-learning
- clementine
license: mit
---

# Music Recommendation Model

**Model Card Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    def _model_description(self) -> str:
        """Generate model description section."""
        return """## Model Description

This is a two-stage music recommendation system trained on personal music ratings from Clementine Music Player.

**Stage 1 - Audio Encoder:**
- Maps 30-second audio clips to 512-dimensional embeddings
- Trained using metadata-based contrastive learning (artist, album, year)
- **Does NOT use rating information** - learns pure audio features
- Processes audio at 16kHz sample rate with center-crop extraction

**Stage 2 - Rating Classifier:**
- Predicts music ratings (0-5 stars) from audio embeddings
- Trained on user's historical ratings
- Simple MLP architecture for fast inference
"""

    def _intended_use(self) -> str:
        """Generate intended use section."""
        return """## Intended Use

**Primary Use:** Generate personalized music recommendations based on learned preferences from Clementine ratings.

**Applications:**
- Recommend unrated songs from your library
- Discover similar music based on audio features
- Generate playlists tailored to your taste

**Out of Scope:**
- Commercial recommendation systems (trained on personal data)
- Cross-user recommendations (model is user-specific)
- Real-time streaming recommendations (designed for batch processing)
"""

    def _model_architecture(self) -> str:
        """Generate architecture section."""
        encoder_config = self.config.get('encoder', {})
        classifier_config = self.config.get('classifier', {})

        return f"""## Model Architecture

### Audio Encoder
- **Architecture:** Simple 1D CNN
- **Input:** 30-second mono audio at 16kHz (480,000 samples)
- **Output:** {encoder_config.get('embedding_dim', 512)}-dimensional embedding
- **Base channels:** {encoder_config.get('base_channels', 32)}

### Rating Classifier
- **Architecture:** Multi-layer Perceptron (MLP)
- **Input:** {encoder_config.get('embedding_dim', 512)}-dimensional embedding
- **Hidden layers:** {classifier_config.get('hidden_dims', [256, 128])}
- **Output:** Single value (predicted rating 0-5)
- **Dropout:** {classifier_config.get('dropout', 0.3)}
- **Batch normalization:** {classifier_config.get('use_batch_norm', False)}
"""

    def _training_procedure(self) -> str:
        """Generate training procedure section."""
        encoder_config = self.config.get('encoder', {})
        classifier_config = self.config.get('classifier', {})
        music_config = self.config.get('music', {})

        return f"""## Training Procedure

### Encoder Training
- **Loss function:** Metadata Contrastive Loss (NO rating information)
- **Positive pairs:** Same artist OR same album OR similar year (±{encoder_config.get('year_threshold', 5)} years)
- **Temperature:** {encoder_config.get('contrastive_temperature', 0.5)}
- **Optimizer:** {encoder_config.get('optimizer', 'adam').upper()}
- **Learning rate:** {encoder_config.get('learning_rate', 0.001)}
- **Weight decay:** {encoder_config.get('weight_decay', 0.0001)}
- **Batch size:** {encoder_config.get('batch_size', 32)}
- **Scheduler:** {encoder_config.get('scheduler', 'reduce_on_plateau')}

### Classifier Training
- **Loss function:** MSE (Mean Squared Error)
- **Optimizer:** {classifier_config.get('optimizer', 'adam').upper()}
- **Learning rate:** {classifier_config.get('learning_rate', 0.0001)}
- **Weight decay:** {classifier_config.get('weight_decay', 0.00001)}
- **Batch size:** {classifier_config.get('batch_size', 256)}
- **Scheduler:** {classifier_config.get('scheduler', 'reduce_on_plateau')}

### Audio Processing
- **Sample rate:** {music_config.get('sample_rate', 16000)} Hz (downsampled from source)
- **Duration:** {music_config.get('audio_duration', 60.0)} seconds
- **Extraction:** {music_config.get('crop_position', 'end').title()} crop (from {music_config.get('crop_position', 'end')} of song)
- **Normalization:** {'Z-normalization (zero mean, unit variance)' if music_config.get('normalize', True) else 'None'}
- **Max duration filter:** {music_config.get('max_duration', 900.0)} seconds
- **Workers:** {music_config.get('dataloader_workers', 4)} parallel DataLoader workers
- **Prefetch factor:** 4 batches per worker
"""

    def _dataset_information(self) -> str:
        """Generate dataset information section."""
        return f"""## Dataset Information

**Source:** Personal Clementine Music Player database

**Database path:** `{self.config.get('music', {}).get('database_path', 'N/A')}`

**Total songs loaded:** {self.preprocessing_stats.get('total_songs_loaded', 'N/A')}

**Rating scale:** 0-5 stars (Clementine's 5-star rating system)
"""

    def _preprocessing(self) -> str:
        """Generate preprocessing section."""
        stats = self.preprocessing_stats

        total = stats.get('total_songs_loaded', 0)
        excluded_missing = stats.get('songs_excluded_missing', 0)
        excluded_duration = stats.get('songs_excluded_duration', 0)
        excluded_speech = stats.get('songs_excluded_speech', 0)
        excluded_duplicates = stats.get('songs_excluded_duplicates', 0)
        excluded_unknown_metadata = stats.get('songs_excluded_unknown_metadata', 0)
        final = stats.get('final_songs', 0)

        return f"""## Preprocessing

### Filtering Criteria

Songs were excluded based on the following criteria:

| Filter Criterion | Songs Excluded | Reason |
|-----------------|----------------|---------|
| Missing audio files | {excluded_missing:,} | File not found on disk |
| Duration >15 minutes | {excluded_duration:,} | Filters live albums, DJ sets, podcasts |
| Speech detection | {excluded_speech:,} | High speech probability (>50%) |
| Duplicates | {excluded_duplicates:,} | Duplicate audio fingerprints |
| Unknown metadata | {excluded_unknown_metadata:,} | All metadata (artist, album, title) unknown or placeholder |
| **Total excluded** | **{total - final:,}** | - |

### Final Dataset

- **Songs after filtering:** {final:,}
- **Rated songs:** {stats.get('rated_songs', 'N/A'):,}
- **Unrated songs:** {stats.get('unrated_songs', 'N/A'):,}
- **Retention rate:** {(final / total * 100) if total > 0 else 0:.1f}%

**Note:** Unknown metadata filtering only applies to encoder training (metadata-based contrastive loss). Classifier training uses all rated songs regardless of metadata quality.
"""

    def _encoder_training(self) -> str:
        """Generate encoder training results section."""
        stats = self.encoder_stats
        if not stats:
            return "## Encoder Training\n\n*Statistics not available*"

        training_time_hours = stats.get('training_time_seconds', 0) / 3600
        year_dist = stats.get('year_distribution', {})
        genre_dist = stats.get('genre_distribution', {})

        sections = [
            "## Encoder Training Results",
            "",
            "### Training Metrics",
            "",
            f"- **Epochs run:** {stats.get('epochs_run', 'N/A')}",
            f"- **Training time:** {training_time_hours:.2f} hours",
            f"- **Final training loss:** {stats.get('final_train_loss', 'N/A'):.6f}",
            f"- **Final validation loss:** {stats.get('final_val_loss', 'N/A'):.6f}",
            f"- **Best validation loss:** {stats.get('best_val_loss', 'N/A'):.6f} (epoch {stats.get('best_epoch', 'N/A')})",
            "",
            "### Training Data Statistics",
            "",
            f"- **Total songs:** {stats.get('total_songs', 'N/A'):,}",
            f"- **Unique artists:** {stats.get('total_artists', 'N/A'):,}",
            f"- **Unique albums:** {stats.get('total_albums', 'N/A'):,}",
        ]

        # Year distribution
        if year_dist:
            sections.extend([
                "",
                "### Year Distribution (5-year periods)",
                "",
                "| Period | Count | Percentage |",
                "|--------|-------|------------|"
            ])
            total = sum(year_dist.values())
            for period in sorted(year_dist.keys()):
                count = year_dist[period]
                pct = (count / total * 100) if total > 0 else 0
                sections.append(f"| {period} | {count:,} | {pct:.1f}% |")

        # Genre distribution
        if genre_dist:
            sections.extend([
                "",
                "### Genre Distribution (Top 10)",
                "",
                "| Genre | Count | Percentage |",
                "|-------|-------|------------|"
            ])
            total = sum(genre_dist.values())
            sorted_genres = sorted(genre_dist.items(), key=lambda x: x[1], reverse=True)[:10]
            for genre, count in sorted_genres:
                pct = (count / total * 100) if total > 0 else 0
                sections.append(f"| {genre} | {count:,} | {pct:.1f}% |")

        return "\n".join(sections)

    def _classifier_training(self) -> str:
        """Generate classifier training results section."""
        stats = self.classifier_stats
        if not stats:
            return "## Classifier Training\n\n*Statistics not available*"

        training_time_hours = stats.get('training_time_seconds', 0) / 3600
        rating_dist = stats.get('rating_distribution', {})
        genre_dist = stats.get('genre_distribution', {})

        sections = [
            "## Classifier Training Results",
            "",
            "### Training Metrics",
            "",
            f"- **Epochs run:** {stats.get('epochs_run', 'N/A')}",
            f"- **Training time:** {training_time_hours:.2f} hours",
            f"- **Final training loss (MSE):** {stats.get('final_train_loss', 'N/A'):.6f}",
            f"- **Final validation loss (MSE):** {stats.get('final_val_loss', 'N/A'):.6f}",
            f"- **Best validation loss:** {stats.get('best_val_loss', 'N/A'):.6f} (epoch {stats.get('best_epoch', 'N/A')})",
            "",
            "### Training Data Statistics (Rated Songs Only)",
            "",
            f"- **Total rated songs:** {stats.get('total_rated_songs', 'N/A'):,}",
            f"- **Unique artists:** {stats.get('total_artists', 'N/A'):,}",
            f"- **Unique albums:** {stats.get('total_albums', 'N/A'):,}",
        ]

        # Rating distribution
        if rating_dist:
            sections.extend([
                "",
                "### Rating Distribution",
                "",
                "| Stars | Count | Percentage |",
                "|-------|-------|------------|"
            ])
            total = sum(rating_dist.values())
            for rating in sorted(rating_dist.keys()):
                count = rating_dist[rating]
                pct = (count / total * 100) if total > 0 else 0
                sections.append(f"| {rating} ⭐ | {count:,} | {pct:.1f}% |")

        # Genre distribution
        if genre_dist:
            sections.extend([
                "",
                "### Genre Distribution (Rated Songs, Top 10)",
                "",
                "| Genre | Count | Percentage |",
                "|-------|-------|------------|"
            ])
            total = sum(genre_dist.values())
            sorted_genres = sorted(genre_dist.items(), key=lambda x: x[1], reverse=True)[:10]
            for genre, count in sorted_genres:
                pct = (count / total * 100) if total > 0 else 0
                sections.append(f"| {genre} | {count:,} | {pct:.1f}% |")

        return "\n".join(sections)

    def _performance_metrics(self) -> str:
        """Generate performance metrics section."""
        return """## Performance Metrics

### Evaluation

See training sections above for validation losses.

**Note:** Model is trained on personal preferences, so traditional metrics (accuracy, F1) are less meaningful than user satisfaction with recommendations.
"""

    def _limitations(self) -> str:
        """Generate limitations section."""
        return """## Limitations & Biases

### Known Limitations

1. **Personal model** - Trained on one user's preferences, not generalizable
2. **Cold start** - Cannot handle completely new genres/artists not in training data
3. **Audio quality** - Performance depends on audio file quality
4. **Duration bias** - Only uses 30-second center clip, may miss intros/outros
5. **Metadata dependency** - Encoder relies on accurate artist/album/year metadata

### Potential Biases

1. **Temporal bias** - May favor music from certain time periods in training data
2. **Genre bias** - Performance varies by genre representation in training set
3. **Rating bias** - User's rating patterns may not be temporally consistent
4. **Selection bias** - Only songs in Clementine library (user's existing collection)

### Ethical Considerations

- **Privacy:** Model trained on personal music preferences - do not share publicly
- **Copyright:** Model does not contain or distribute copyrighted audio
- **Fairness:** Model reflects individual preferences, not universal music quality
"""

    def _usage_example(self) -> str:
        """Generate usage example section."""
        return """## Usage Example

```python
import torch
from ml_skeleton.music.simsiam_encoder import SimSiamEncoder
from ml_skeleton.music.baseline_classifier import SimpleRatingClassifier
from ml_skeleton.music.audio_loader import load_audio_file

# Load models
encoder = SimSiamEncoder(sample_rate=16000, embedding_dim=512)
encoder.load_state_dict(torch.load('checkpoints/encoder_best.pt'))
encoder.eval()

classifier = SimpleRatingClassifier(embedding_dim=512)
classifier.load_state_dict(torch.load('checkpoints/classifier_best.pt'))
classifier.eval()

# Load audio (60 seconds from end at 16kHz, with z-normalization)
audio = load_audio_file(
    'path/to/song.mp3',
    sample_rate=16000,
    duration=60.0,
    crop_position='end',
    normalize=True
)

# Get prediction
with torch.no_grad():
    embedding = encoder(audio.unsqueeze(0))
    rating = classifier(embedding).item()

print(f"Predicted rating: {rating:.2f} / 5.0")
```

## Recommendation Generation

```bash
# Generate top-100 recommendations
python examples/music_recommendation.py --stage recommend --config configs/music_recommendation.yaml

# Output: recommendations.txt (sorted by predicted rating)
# Output: recommender_best.xspf (top predictions playlist)
# Output: recommender_help.xspf (uncertain predictions for labeling)
```
"""

    def _citation(self) -> str:
        """Generate citation section."""
        return """## Citation

If you use this model or training pipeline, please cite:

```bibtex
@misc{music_recommendation_model,
  title={Personal Music Recommendation Model},
  author={ml_skeleton framework},
  year={2026},
  url={https://github.com/your-repo/ml_skeleton}
}
```
"""

    def _metadata(self) -> str:
        """Generate metadata section."""
        return """## Model Card Authors

Generated automatically by ml_skeleton training pipeline.

## Model Card Contact

For questions or issues, please open an issue in the repository.
"""
