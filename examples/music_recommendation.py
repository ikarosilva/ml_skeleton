"""Music Recommendation System - Complete Example.

This example demonstrates the full two-phase training pipeline:
1. Stage 1: Train audio encoder (audio -> embeddings)
2. Stage 2: Train rating classifier (embeddings -> ratings)
3. Generate recommendations

Usage:
    # Stage 1: Train encoder
    python examples/music_recommendation.py --stage encoder --config configs/music_recommendation.yaml

    # Stage 2: Train classifier
    python examples/music_recommendation.py --stage classifier --config configs/music_recommendation.yaml

    # Generate recommendations
    python examples/music_recommendation.py --stage recommend --config configs/music_recommendation.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_skeleton.music.clementine_db import ClementineDB
from ml_skeleton.music.dataset import MusicDataset, EmbeddingDataset
from ml_skeleton.music.embedding_store import EmbeddingStore
from ml_skeleton.music.losses import (
    RatingLoss,
    MultiTaskLoss,
    NTXentLoss,
    SupervisedContrastiveLoss,
    build_album_mapping
)
from ml_skeleton.music.baseline_encoder import SimpleAudioEncoder, MultiTaskEncoder
from ml_skeleton.music.baseline_classifier import SimpleRatingClassifier
from ml_skeleton.music.xspf_playlist import generate_human_feedback_playlists
from ml_skeleton.training.encoder_trainer import EncoderTrainer
from ml_skeleton.training.classifier_trainer import ClassifierTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_encoder(config: dict):
    """Stage 1: Train audio encoder.

    Args:
        config: Configuration dictionary
    """
    print("=" * 80)
    print("STAGE 1: ENCODER TRAINING")
    print("=" * 80)

    # Load configuration
    music_config = config['music']
    encoder_config = config['encoder']
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Connect to Clementine database
    print("\n[1/7] Loading Clementine database...")
    db = ClementineDB(music_config['database_path'])
    all_songs = db.get_all_songs()
    print(f"  Found {len(all_songs)} songs in database")

    # Filter rated songs
    if music_config.get('only_rated', True):
        all_songs = [s for s in all_songs if s.is_rated]
        print(f"  Filtered to {len(all_songs)} rated songs")

    # Build album mappings
    print("\n[2/7] Building album mappings...")
    album_to_idx, filename_to_albums = build_album_mapping(all_songs)
    print(f"  Found {len(album_to_idx)} unique albums")

    # Create datasets
    print("\n[3/7] Creating datasets...")
    full_dataset = MusicDataset(
        songs=all_songs,
        album_to_idx=album_to_idx,
        filename_to_albums=filename_to_albums,
        sample_rate=music_config['sample_rate'],
        duration=music_config['audio_duration'],
        center_crop=music_config['center_crop'],
        only_rated=music_config.get('only_rated', True)
    )

    # Train/val split
    train_split = music_config.get('train_split', 0.8)
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )

    print(f"  Train: {len(train_dataset)} songs")
    print(f"  Val: {len(val_dataset)} songs")

    # Create data loaders
    num_workers = music_config.get('dataloader_workers', 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=encoder_config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=encoder_config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Create model
    print("\n[4/7] Creating encoder model...")
    base_encoder = SimpleAudioEncoder(
        sample_rate=music_config['sample_rate'],
        duration=music_config['audio_duration'],
        embedding_dim=encoder_config['embedding_dim'],
        base_channels=encoder_config['base_channels']
    )

    # Optionally wrap with multi-task encoder
    use_multi_task = music_config.get('use_multi_task', False)
    if use_multi_task:
        encoder = MultiTaskEncoder(base_encoder, num_albums=len(album_to_idx))
        print(f"  Using MultiTaskEncoder with {len(album_to_idx)} albums")
    else:
        encoder = base_encoder
        print(f"  Using SimpleAudioEncoder")

    print(f"  Embedding dim: {encoder_config['embedding_dim']}")
    print(f"  Base channels: {encoder_config['base_channels']}")

    # Create loss function
    print("\n[5/7] Creating loss function...")
    loss_type = encoder_config.get('loss_type', 'supervised_contrastive')

    if use_multi_task:
        loss_fn = MultiTaskLoss(
            rating_weight=1.0,
            album_weight=music_config.get('album_weight', 0.5)
        )
        print(f"  Using MultiTaskLoss")
    elif loss_type == 'supervised_contrastive':
        loss_fn = SupervisedContrastiveLoss(
            temperature=encoder_config.get('contrastive_temperature', 0.5),
            rating_threshold=encoder_config.get('rating_threshold', 0.2)
        )
        print(f"  Using SupervisedContrastiveLoss")
    elif loss_type == 'contrastive':
        loss_fn = NTXentLoss(
            temperature=encoder_config.get('contrastive_temperature', 0.5)
        )
        print(f"  Using NTXentLoss")
    else:
        loss_fn = nn.MSELoss()
        print(f"  Using MSELoss")

    # Create optimizer
    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=encoder_config['learning_rate'],
        weight_decay=encoder_config.get('weight_decay', 0.0)
    )

    # Create embedding store
    embedding_store = EmbeddingStore(music_config['embedding_db_path'])

    # Create trainer
    print("\n[6/7] Creating trainer...")
    trainer = EncoderTrainer(
        encoder=encoder,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        embedding_store=embedding_store,
        model_version=music_config['model_version']
    )

    # Train
    print("\n[7/7] Training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=encoder_config['epochs'],
        checkpoint_dir=config['checkpoint_dir'],
        use_multi_task=use_multi_task,
        save_best_only=True
    )

    print("\n" + "=" * 80)
    print("ENCODER TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best val loss: {min(history['val_loss']):.4f}")
    print(f"Checkpoint saved to: {config['checkpoint_dir']}/encoder_best.pt")

    # Extract embeddings
    print("\n" + "=" * 80)
    print("EXTRACTING EMBEDDINGS")
    print("=" * 80)

    # Load best model
    best_checkpoint = Path(config['checkpoint_dir']) / "encoder_best.pt"
    trainer.load_checkpoint(best_checkpoint)

    # Extract embeddings for all songs
    all_loader = DataLoader(
        full_dataset,
        batch_size=encoder_config['batch_size'],
        shuffle=False,
        num_workers=num_workers
    )

    embeddings = trainer.extract_embeddings(
        all_loader,
        save_to_store=True
    )

    print(f"\nExtracted {len(embeddings)} embeddings")
    print(f"Saved to: {music_config['embedding_db_path']}")

    # Print embedding store stats
    stats = embedding_store.get_stats()
    print(f"\nEmbedding Store Stats:")
    print(f"  Total embeddings: {stats['total_embeddings']}")
    print(f"  Unique songs: {stats['unique_songs']}")
    print(f"  Model versions: {stats['model_versions']}")
    print(f"  DB size: {stats['db_size_mb']:.2f} MB")


def train_classifier(config: dict):
    """Stage 2: Train rating classifier.

    Args:
        config: Configuration dictionary
    """
    print("=" * 80)
    print("STAGE 2: CLASSIFIER TRAINING")
    print("=" * 80)

    # Load configuration
    music_config = config['music']
    classifier_config = config['classifier']
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Connect to database
    print("\n[1/6] Loading Clementine database...")
    db = ClementineDB(music_config['database_path'])
    all_songs = db.get_all_songs()

    # Filter rated songs
    if music_config.get('only_rated', True):
        all_songs = [s for s in all_songs if s.is_rated]

    print(f"  Found {len(all_songs)} rated songs")

    # Load embeddings
    print("\n[2/6] Loading embeddings...")
    embedding_store = EmbeddingStore(music_config['embedding_db_path'])

    # Get embeddings for all songs
    filenames = [s.filename for s in all_songs]
    embeddings_dict = embedding_store.get_embeddings_batch(
        filenames,
        model_version=music_config['model_version']
    )

    print(f"  Loaded {len(embeddings_dict)} embeddings")

    # Check embedding dimension
    first_embedding = next(iter(embeddings_dict.values()))
    embedding_dim = len(first_embedding)
    print(f"  Embedding dimension: {embedding_dim}")

    # Create dataset
    print("\n[3/6] Creating dataset...")
    full_dataset = EmbeddingDataset(
        embeddings=embeddings_dict,
        songs=all_songs,
        only_rated=True
    )

    # Train/val split
    train_split = music_config.get('train_split', 0.8)
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )

    print(f"  Train: {len(train_dataset)} songs")
    print(f"  Val: {len(val_dataset)} songs")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=classifier_config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=classifier_config['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Create model
    print("\n[4/6] Creating classifier model...")
    classifier = SimpleRatingClassifier(
        embedding_dim=embedding_dim,
        hidden_dims=classifier_config['hidden_dims'],
        dropout=classifier_config['dropout']
    )

    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Hidden dims: {classifier_config['hidden_dims']}")
    print(f"  Dropout: {classifier_config['dropout']}")

    # Create loss and optimizer
    loss_fn = RatingLoss()
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=classifier_config['learning_rate'],
        weight_decay=classifier_config.get('weight_decay', 0.0)
    )

    # Create trainer
    print("\n[5/6] Creating trainer...")
    trainer = ClassifierTrainer(
        classifier=classifier,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer
    )

    # Train
    print("\n[6/6] Training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=classifier_config['epochs'],
        checkpoint_dir=config['checkpoint_dir'],
        save_best_only=True
    )

    print("\n" + "=" * 80)
    print("CLASSIFIER TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best val loss: {min(history['val_loss']):.4f}")
    print(f"Best val MAE: {min(history['val_mae']):.4f}")
    print(f"Checkpoint saved to: {config['checkpoint_dir']}/classifier_best.pt")


def generate_recommendations(config: dict):
    """Generate recommendations for unrated songs.

    Args:
        config: Configuration dictionary
    """
    print("=" * 80)
    print("GENERATING RECOMMENDATIONS")
    print("=" * 80)

    # Load configuration
    music_config = config['music']
    rec_config = config.get('recommendations', {})
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    # Connect to database
    print("\n[1/5] Loading Clementine database...")
    db = ClementineDB(music_config['database_path'])
    all_songs = db.get_all_songs()

    # Get unrated songs
    unrated_songs = [s for s in all_songs if not s.is_rated]
    print(f"  Found {len(unrated_songs)} unrated songs")

    if len(unrated_songs) == 0:
        print("  No unrated songs to recommend!")
        return

    # Load embeddings
    print("\n[2/5] Loading embeddings...")
    embedding_store = EmbeddingStore(music_config['embedding_db_path'])

    filenames = [s.filename for s in unrated_songs]
    embeddings_dict = embedding_store.get_embeddings_batch(
        filenames,
        model_version=music_config['model_version']
    )

    print(f"  Loaded {len(embeddings_dict)} embeddings")

    if len(embeddings_dict) == 0:
        print("  No embeddings found! Run encoder training first.")
        return

    # Create dataset
    dataset = EmbeddingDataset(
        embeddings=embeddings_dict,
        songs=unrated_songs,
        only_rated=False
    )

    data_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4
    )

    # Load classifier
    print("\n[3/5] Loading classifier...")
    embedding_dim = len(next(iter(embeddings_dict.values())))

    classifier = SimpleRatingClassifier(
        embedding_dim=embedding_dim,
        hidden_dims=config['classifier']['hidden_dims'],
        dropout=config['classifier']['dropout']
    )

    checkpoint_path = Path(config['checkpoint_dir']) / "classifier_best.pt"
    if not checkpoint_path.exists():
        print(f"  Classifier checkpoint not found: {checkpoint_path}")
        print("  Run classifier training first!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.to(device)

    # Generate predictions
    print("\n[4/5] Generating predictions...")
    loss_fn = RatingLoss()
    trainer = ClassifierTrainer(
        classifier=classifier,
        device=device,
        loss_fn=loss_fn,
        optimizer=torch.optim.Adam(classifier.parameters())  # Dummy optimizer
    )

    predictions, pred_filenames = trainer.predict(data_loader)

    # Sort by predicted rating
    results = list(zip(predictions, pred_filenames))
    results.sort(reverse=True)  # Highest ratings first

    # Apply threshold
    min_threshold = rec_config.get('min_rating_threshold', 0.0)
    results = [(r, f) for r, f in results if r >= min_threshold]

    # Get top-N
    top_n = rec_config.get('top_n', 100)
    results = results[:top_n]

    print(f"  Generated {len(results)} recommendations")

    # Save recommendations
    print("\n[5/5] Saving recommendations...")
    output_path = Path(rec_config.get('output_path', './recommendations.txt'))

    with open(output_path, 'w') as f:
        f.write(f"Top {len(results)} Recommendations\n")
        f.write("=" * 80 + "\n\n")

        for i, (rating, filename) in enumerate(results, 1):
            # Find song metadata
            song = next((s for s in unrated_songs if s.filename == filename), None)
            if song:
                f.write(f"{i}. [{rating:.3f}] {song.artist} - {song.title}\n")
                f.write(f"   Album: {song.album} ({song.year})\n")
                f.write(f"   Path: {filename}\n\n")

    print(f"  Saved to: {output_path}")

    # Print top 10
    print("\nTop 10 Recommendations:")
    for i, (rating, filename) in enumerate(results[:10], 1):
        song = next((s for s in unrated_songs if s.filename == filename), None)
        if song:
            print(f"  {i}. [{rating:.3f}] {song.artist} - {song.title}")

    # Generate human feedback playlists (for reinforcement learning loop)
    print("\n" + "=" * 80)
    print("GENERATING HUMAN FEEDBACK PLAYLISTS")
    print("=" * 80)

    # Prepare full list of songs and predictions for playlist generation
    full_predictions = [r for r, _ in results]
    full_songs = []
    for _, filename in results:
        song = next((s for s in unrated_songs if s.filename == filename), None)
        if song:
            full_songs.append(song)

    # Generate both uncertainty and best-predictions playlists
    top_n_uncertain = rec_config.get('human_feedback_uncertain', 100)
    top_n_best = rec_config.get('human_feedback_best', 50)

    playlist_stats = generate_human_feedback_playlists(
        songs=full_songs,
        predictions=full_predictions,
        output_dir=Path('./'),
        top_n_uncertain=top_n_uncertain,
        top_n_best=top_n_best,
        uncertainty_method="distance_from_middle"
    )

    print("\n" + "=" * 80)
    print("RECOMMENDATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {output_path} (text recommendations)")
    print(f"  - recommender_help.xspf (high uncertainty - maximize learning)")
    print(f"  - recommender_best.xspf (top predictions - validate quality)")
    print(f"\nNext steps for human-in-the-loop training:")
    print(f"  1. Open XSPF playlists in Clementine")
    print(f"  2. Listen and rate songs")
    print(f"  3. Re-run training with updated ratings")
    print(f"  4. Repeat for continuous improvement!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Music Recommendation System"
    )
    parser.add_argument(
        '--stage',
        type=str,
        required=True,
        choices=['encoder', 'classifier', 'recommend'],
        help='Training stage or recommendation generation'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/music_recommendation.yaml',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Run stage
    if args.stage == 'encoder':
        train_encoder(config)
    elif args.stage == 'classifier':
        train_classifier(config)
    elif args.stage == 'recommend':
        generate_recommendations(config)


if __name__ == '__main__':
    main()
