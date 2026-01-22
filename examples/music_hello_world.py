"""
Music Recommendation "Hello World" Example

This is a minimal stub implementation that demonstrates:
1. Basic AudioEncoder structure (processes random audio)
2. Basic RatingClassifier structure (simple linear model)
3. How to integrate with ml_skeleton framework

This is intentionally simple so you can quickly customize it for your needs.

IMPORTANT - AUDIO FILE SAFETY:
    ✅ All operations on your music files are READ-ONLY
    ✅ Your MP3/FLAC files will NEVER be modified
    ✅ Clementine database is accessed READ-ONLY
    ✅ New data (embeddings, models) goes to separate files

    See examples/AUDIO_FILES_README.md for complete documentation.

Usage:
    # Test the stub models locally
    python examples/music_hello_world.py

    # Once you're ready to train (after implementing full framework):
    # mlskel run configs/music_example.yaml --train-fn examples.music_hello_world:train_encoder_stage
"""

import torch
import torch.nn as nn


# =============================================================================
# ENCODER STUB - Replace this with your actual audio encoder
# =============================================================================

class HelloWorldEncoder(nn.Module):
    """
    Minimal encoder stub - processes audio and outputs embeddings.

    Replace this with your actual architecture (ResNet, Transformer, etc.)
    This version just uses a simple 1D convolution for demonstration.
    """

    def __init__(self, embedding_dim: int = 128, sample_rate: int = 16000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sample_rate = sample_rate

        # Simple 1D CNN - REPLACE WITH YOUR ARCHITECTURE
        self.features = nn.Sequential(
            # Input: (batch, 1, num_samples)
            nn.Conv1d(1, 32, kernel_size=1024, stride=512),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Output: (batch, 64, 1)
        )

        # Project to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to embeddings.

        Args:
            audio: Raw waveform tensor, shape (batch_size, num_samples)
                   Expected: 30 seconds at 16000 Hz = 480,000 samples

        Returns:
            embeddings: shape (batch_size, embedding_dim)
        """
        # Add channel dimension: (B, num_samples) -> (B, 1, num_samples)
        x = audio.unsqueeze(1)

        # Extract features
        x = self.features(x)  # (B, 64, 1)
        x = x.squeeze(-1)     # (B, 64)

        # Project to embedding
        embeddings = self.projection(x)  # (B, embedding_dim)

        return embeddings

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension Z."""
        return self.embedding_dim


# =============================================================================
# CLASSIFIER STUB - Replace this with your actual rating classifier
# =============================================================================

class HelloWorldClassifier(nn.Module):
    """
    Minimal classifier stub - predicts ratings from embeddings.

    Replace this with your actual architecture.
    This version is just a simple 2-layer MLP for demonstration.
    """

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64):
        super().__init__()

        # Simple MLP - REPLACE WITH YOUR ARCHITECTURE
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output in range [0, 1]
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict ratings from embeddings.

        Args:
            embeddings: shape (batch_size, embedding_dim)

        Returns:
            ratings: shape (batch_size, 1), values in [0, 1]
                    0.0 = unrated/terrible, 1.0 = 5 stars
        """
        ratings = self.network(embeddings)
        return ratings


# =============================================================================
# TEST CODE - Verify the models work correctly
# =============================================================================

def test_hello_world_models():
    """Test the stub models with random data."""
    print("=" * 70)
    print("Testing Hello World Encoder and Classifier Stubs")
    print("=" * 70)

    # Configuration
    batch_size = 4
    sample_rate = 16000
    duration = 30.0  # seconds
    num_samples = int(sample_rate * duration)
    embedding_dim = 128

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration} seconds")
    print(f"  Samples per audio: {num_samples:,}")
    print(f"  Embedding dimension: {embedding_dim}")

    # Create models
    encoder = HelloWorldEncoder(embedding_dim=embedding_dim, sample_rate=sample_rate)
    classifier = HelloWorldClassifier(embedding_dim=embedding_dim)

    print(f"\n{'='*70}")
    print("Encoder Architecture:")
    print(f"{'='*70}")
    print(encoder)

    print(f"\n{'='*70}")
    print("Classifier Architecture:")
    print(f"{'='*70}")
    print(classifier)

    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    classifier_params = sum(p.numel() for p in classifier.parameters())
    print(f"\nParameter counts:")
    print(f"  Encoder: {encoder_params:,} parameters")
    print(f"  Classifier: {classifier_params:,} parameters")
    print(f"  Total: {encoder_params + classifier_params:,} parameters")

    # Test encoder
    print(f"\n{'='*70}")
    print("Testing Encoder")
    print(f"{'='*70}")

    # Create fake audio (random noise)
    fake_audio = torch.randn(batch_size, num_samples)
    print(f"Input audio shape: {fake_audio.shape}")

    # Encode
    with torch.no_grad():
        embeddings = encoder(fake_audio)

    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Embedding statistics:")
    print(f"  Mean: {embeddings.mean().item():.4f}")
    print(f"  Std: {embeddings.std().item():.4f}")
    print(f"  Min: {embeddings.min().item():.4f}")
    print(f"  Max: {embeddings.max().item():.4f}")

    # Verify embedding dimension
    assert embeddings.shape == (batch_size, embedding_dim), \
        f"Expected shape ({batch_size}, {embedding_dim}), got {embeddings.shape}"
    print("✓ Encoder output shape is correct")

    # Test classifier
    print(f"\n{'='*70}")
    print("Testing Classifier")
    print(f"{'='*70}")

    print(f"Input embeddings shape: {embeddings.shape}")

    # Predict ratings
    with torch.no_grad():
        ratings = classifier(embeddings)

    print(f"Output ratings shape: {ratings.shape}")
    print(f"Ratings:")
    for i, rating in enumerate(ratings):
        print(f"  Sample {i+1}: {rating.item():.4f}")

    # Verify ratings are in [0, 1]
    assert ratings.shape == (batch_size, 1), \
        f"Expected shape ({batch_size}, 1), got {ratings.shape}"
    assert torch.all((ratings >= 0) & (ratings <= 1)), \
        "Ratings must be in range [0, 1]"
    print("✓ Classifier output shape and range are correct")

    # Test end-to-end
    print(f"\n{'='*70}")
    print("Testing End-to-End Pipeline")
    print(f"{'='*70}")

    # Create new random audio
    test_audio = torch.randn(2, num_samples)
    print(f"Input: {test_audio.shape}")

    with torch.no_grad():
        # Encode
        test_embeddings = encoder(test_audio)
        print(f"Embeddings: {test_embeddings.shape}")

        # Classify
        test_ratings = classifier(test_embeddings)
        print(f"Ratings: {test_ratings.shape}")
        print(f"  Song 1 predicted rating: {test_ratings[0].item():.4f}")
        print(f"  Song 2 predicted rating: {test_ratings[1].item():.4f}")

    print("\n" + "="*70)
    print("✓ All tests passed! Models are working correctly.")
    print("="*70)
    print("\nNext steps:")
    print("1. Replace HelloWorldEncoder with your actual encoder architecture")
    print("2. Replace HelloWorldClassifier with your actual classifier architecture")
    print("3. Implement the training functions below")
    print("4. Test with real audio data from your Clementine database")
    print("="*70)


# =============================================================================
# TRAINING AND PIPELINE STUBS
# =============================================================================
from types import SimpleNamespace
from ml_skeleton.music.clementine_db import load_all_songs
from ml_skeleton.music.speech_detector import SpeechDetector
from ml_skeleton.music.dataset import MusicDataset

def run_speech_detection_pipeline(config):
    """
    Demonstrates the speech detection and filtering pipeline.
    
    This function:
    1. Loads songs from the (placeholder) Clementine database.
    2. Runs the speech detector to get speech probabilities for each song.
    3. Creates a MusicDataset, which filters the songs based on the threshold.
    """
    print("=" * 70)
    print("Running Speech Detection and Filtering Pipeline")
    print("=" * 70)

    # 1. Load songs from Clementine database
    all_songs = load_all_songs(config.music['database_path'])
    print(f"\nLoaded {len(all_songs)} total songs from the database.")

    # 2. Run speech detection (if enabled)
    speech_results = {}
    if config.music['speech_detection']['enabled']:
        print("\n--- Starting Speech Detection ---")
        detector = SpeechDetector(config.music['speech_detection']['cache_path'])
        results = detector.detect_speech_in_songs(all_songs)
        speech_results = {res.filename: res.speech_probability for res in results}
        print("--- Speech Detection Finished ---")

        # Print results for clarity
        print("\nSpeech detection probabilities:")
        for song in all_songs:
            prob = speech_results.get(song.filename, "N/A")
            if isinstance(prob, float):
                print(f"  - {song.title:<25}: {prob:.2f}")
            else:
                print(f"  - {song.title:<25}: {prob}")

    # 3. Create Dataset with filtering
    print("\n--- Creating and Filtering Dataset ---")
    dataset = MusicDataset(
        songs=all_songs,
        speech_results=speech_results,
        speech_threshold=config.music['speech_detection']['speech_threshold']
    )

    print(f"\nOriginal song count: {len(all_songs)}")
    print(f"Dataset size after filtering: {len(dataset)}")
    print("\n" + "="*70)
    print("✓ Pipeline finished successfully.")
    print("="*70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main function to run the demonstration pipeline.
    """
    # Create a mock config for demonstration purposes.
    # This would normally be loaded from a YAML file.
    mock_config = SimpleNamespace(
        music={
            'database_path': '/path/to/your/clementine.db',
            'speech_detection': {
                'enabled': True,
                'cache_path': "./speech_cache.db",
                'speech_threshold': 0.5,
            }
        }
    )
    
    run_speech_detection_pipeline(mock_config)


if __name__ == "__main__":
    main()
