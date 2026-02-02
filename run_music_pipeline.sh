#!/bin/bash
# Music Recommendation Pipeline Runner (MoCo v2 + Genre BCE)
#
# Usage:
#   ./run_music_pipeline.sh all              # Run all 3 stages
#   ./run_music_pipeline.sh encoder          # Run Stage 1 only (MoCo v2 + Genre)
#   ./run_music_pipeline.sh classifier       # Run Stage 2 only
#   ./run_music_pipeline.sh recommend        # Run Stage 3 only
#   ./run_music_pipeline.sh quick            # Quick test (5 epochs, 500 rated songs)
#   ./run_music_pipeline.sh hpo              # Full hyperparameter optimization pipeline
#   ./run_music_pipeline.sh build-cache      # Build 4-chunk waveform cache (~30GB)
#   ./run_music_pipeline.sh clcear-cache      # Delete waveform cache (prompts for confirmation)
#
# Architecture:
#   Audio → 16kHz .npy cache (4 chunks/song) → nnAudio CQT → ResNet-50 2D → 2048-dim
#   ├── MoCo v2 contrastive head (queue=4096, τ=0.07)
#   └── Genre BCE head (7 categories)
#
# VERSION COMPATIBILITY RULES:
#   - Encoder and Classifier have SEPARATE versions
#   - Classifier stores which encoder version it was trained with
#   - Classifier can be updated independently IF encoder hasn't changed
#   - If Encoder is updated, Classifier MUST be retrained
#   - Deployment (recommend) FAILS if versions don't match
#
# Resume/Incremental Training (train v2 from v1):
#   # Update encoder to v2 (requires retraining classifier)
#   ./run_music_pipeline.sh encoder --resume-checkpoint checkpoints/encoder_best.pt --encoder-version v2
#   ./run_music_pipeline.sh classifier --classifier-version v1  # New classifier for encoder v2
#
#   # Update classifier only (encoder unchanged)
#   ./run_music_pipeline.sh classifier --classifier-version v2
#
# Environment variables:
#   CONFIG=/path/to/config.yaml              # Override config file
#   CLEMENTINE_DB_PATH=/path/to/db           # Override database path
#   HPO_ENCODER_TRIALS=30                    # Number of encoder HPO trials
#   HPO_CLASSIFIER_TRIALS=20                 # Number of classifier HPO trials
#   RESUME_CHECKPOINT=/path/to/checkpoint    # Resume from previous training
#   ENCODER_VERSION=v2                       # Encoder version for embeddings
#   CLASSIFIER_VERSION=v2                    # Classifier version

set -e  # Exit on error

CONFIG="${CONFIG:-configs/music_moco.yaml}"
SCRIPT="examples/music_recommendation.py"
HPO_ENCODER_TRIALS="${HPO_ENCODER_TRIALS:-30}"
HPO_CLASSIFIER_TRIALS="${HPO_CLASSIFIER_TRIALS:-100}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"  # Optional: path to checkpoint to resume from
ENCODER_VERSION="${ENCODER_VERSION:-}"  # Optional: encoder version for embeddings (e.g., "v2")
CLASSIFIER_VERSION="${CLASSIFIER_VERSION:-}"  # Optional: classifier version (e.g., "v2")

# Set minimum rated songs for placeholder database
export MIN_RATED_SONGS="${MIN_RATED_SONGS:-500}"

# Path remapping for audio files
export MUSIC_PATH_REMAP="${MUSIC_PATH_REMAP:-/home/ikaro/Music:/Music}"

# Allow database path override
if [ -n "$CLEMENTINE_DB_PATH" ]; then
    export CLEMENTINE_DB_PATH
fi

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }

check_prerequisites() {
    print_header "Checking Prerequisites"
    if ! command -v python &> /dev/null; then
        print_error "Python not found"
        exit 1
    fi
    print_success "Python found: $(python --version)"

    if [ ! -f "$SCRIPT" ]; then
        print_error "Script not found: $SCRIPT"
        exit 1
    fi
    print_success "Script found: $SCRIPT"

    if [ ! -f "$CONFIG" ]; then
        print_error "Config not found: $CONFIG"
        exit 1
    fi
    print_success "Config found: $CONFIG"

    # Check for nnAudio
    if python -c "import nnAudio" 2>/dev/null; then
        print_success "nnAudio available"
    else
        print_warning "nnAudio not installed. Run: pip install nnAudio"
    fi
    echo ""
}

run_encoder() {
    local resume_arg=""
    local version_arg=""

    # Add resume checkpoint if specified
    if [ -n "$RESUME_CHECKPOINT" ]; then
        resume_arg="--resume-checkpoint $RESUME_CHECKPOINT"
    fi

    # Add encoder version if specified
    if [ -n "$ENCODER_VERSION" ]; then
        version_arg="--encoder-version $ENCODER_VERSION"
    fi

    # Print header with relevant info
    local header="Stage 1: Training MoCo v2 + Genre BCE Encoder"
    [ -n "$RESUME_CHECKPOINT" ] && header="$header [RESUMING from $RESUME_CHECKPOINT]"
    [ -n "$ENCODER_VERSION" ] && header="$header [encoder: $ENCODER_VERSION]"
    print_header "$header"

    echo "Architecture: Audio → CQT → ResNet-50 2D → 2048-dim"
    echo "Loss: 0.6×MoCo(NT-Xent) + 0.4×Genre_BCE"
    echo ""

    # Track training time
    local start_time=$(date +%s)
    python "$SCRIPT" --stage encoder --config "$CONFIG" $resume_arg $version_arg "$@"
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local minutes=$((elapsed / 60))

    # Format time message
    if [ $minutes -ge 60 ]; then
        local hours_decimal=$(awk "BEGIN {printf \"%.1f\", $elapsed/3600}")
        print_success "Encoder training complete in ${hours_decimal} hours!"
    else
        print_success "Encoder training complete in ${minutes} minutes!"
    fi
    echo ""
}

run_classifier() {
    local classifier_version_arg=""
    local is_final_training=false

    # Add classifier version if specified
    if [ -n "$CLASSIFIER_VERSION" ]; then
        classifier_version_arg="--classifier-version $CLASSIFIER_VERSION"
    fi

    # Check if --final-training is in arguments
    for arg in "$@"; do
        if [ "$arg" = "--final-training" ]; then
            is_final_training=true
            break
        fi
    done

    # Print header with relevant info
    local header="Stage 2: Training Rating Classifier"
    [ -n "$CLASSIFIER_VERSION" ] && header="$header [classifier: $CLASSIFIER_VERSION]"
    print_header "$header"

    # Track training time
    local start_time=$(date +%s)
    python "$SCRIPT" --stage classifier --config "$CONFIG" $classifier_version_arg "$@"
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local minutes=$((elapsed / 60))

    # Format time message
    if [ $minutes -ge 60 ]; then
        local hours_decimal=$(awk "BEGIN {printf \"%.1f\", $elapsed/3600}")
        print_success "Classifier training complete in ${hours_decimal} hours!"
    else
        print_success "Classifier training complete in ${minutes} minutes!"
    fi
    echo ""

    # Show A/B history after final training (if prod history exists)
    if [ "$is_final_training" = true ]; then
        display_ab_history
    fi
}

run_recommend() {
    local use_prod=false
    local extra_args=()

    # Parse arguments
    for arg in "$@"; do
        if [ "$arg" = "--prod" ]; then
            use_prod=true
        else
            # Collect other arguments to pass through (e.g., --low-rating-ratio)
            extra_args+=("$arg")
        fi
    done

    if [ "$use_prod" = true ]; then
        print_header "Stage 3: Generating Recommendations (PRODUCTION)"
        if [ ! -d "prod" ] || [ ! -f "prod/classifier_best.pt" ]; then
            print_error "Production models not found in prod/"
            echo "Run '$0 promote-to-prod' first to deploy models"
            exit 1
        fi
        python "$SCRIPT" --stage recommend --config "$CONFIG" --prod-dir prod "${extra_args[@]}"
    else
        print_header "Stage 3: Generating Recommendations"
        python "$SCRIPT" --stage recommend --config "$CONFIG" "${extra_args[@]}"
    fi
    print_success "Recommendations generated!"
    echo ""
}

run_promote_to_prod() {
    print_header "Promoting Models to Production"

    # Create prod directory if it doesn't exist
    mkdir -p prod
    mkdir -p prod/history

    # Check if source files exist
    if [ ! -f "checkpoints/encoder_best.pt" ]; then
        print_error "Encoder checkpoint not found: checkpoints/encoder_best.pt"
        exit 1
    fi
    if [ ! -f "checkpoints/classifier_best.pt" ]; then
        print_error "Classifier checkpoint not found: checkpoints/classifier_best.pt"
        exit 1
    fi
    if [ ! -f "embeddings.db" ]; then
        print_error "Embeddings database not found: embeddings.db"
        exit 1
    fi

    # Archive existing model card before overwriting (for tracking A/B progress)
    if [ -f "prod/MODEL_CARD.md" ]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        echo "Archiving previous model card..."
        cp prod/MODEL_CARD.md "prod/history/MODEL_CARD_${TIMESTAMP}.md"
        if [ -f "prod/model_card.json" ]; then
            cp prod/model_card.json "prod/history/model_card_${TIMESTAMP}.json"
        fi
        if [ -f "prod/training_manifest.json" ]; then
            cp prod/training_manifest.json "prod/history/training_manifest_${TIMESTAMP}.json"
        fi
        print_success "Previous model card archived to prod/history/"
    fi

    # Copy models to prod
    echo "Copying models to prod/..."
    cp checkpoints/encoder_best.pt prod/
    cp checkpoints/classifier_best.pt prod/
    cp embeddings.db prod/

    # Copy model card if exists
    if [ -f "checkpoints/model_card.json" ]; then
        cp checkpoints/model_card.json prod/
    fi

    # Copy best params if exist
    if [ -f "checkpoints/best_encoder_params.json" ]; then
        cp checkpoints/best_encoder_params.json prod/
    fi
    if [ -f "checkpoints/best_classifier_params.json" ]; then
        cp checkpoints/best_classifier_params.json prod/
    fi

    # Copy training manifest (tracks which files were used for training)
    if [ -f "checkpoints/training_manifest.json" ]; then
        cp checkpoints/training_manifest.json prod/
        print_success "Training manifest copied (for A/B testing)"
    fi

    # Generate model card with A/B test results
    echo ""
    python "$SCRIPT" --stage generate-model-card --config "$CONFIG"

    print_success "Models promoted to production!"
    echo ""
    echo "Production files:"
    ls -lh prod/
    echo ""
    echo "Model card: prod/MODEL_CARD.md"
    if [ -d "prod/history" ] && [ "$(ls -A prod/history 2>/dev/null)" ]; then
        echo ""
        echo "Historical model cards: $(ls prod/history/MODEL_CARD_*.md 2>/dev/null | wc -l) archived"
        echo "  View history: ls -la prod/history/"
    fi
    echo ""
    echo "Now run: $0 recommend --prod"
}

run_sync_db() {
    print_header "Syncing Database (Refreshing Ratings)"

    # The Clementine database is always read fresh, so this command
    # just verifies the database is accessible and shows stats
    echo "Database path: ${CLEMENTINE_DB_PATH:-/Music/database/clementine_backup_2026-01.db}"
    echo ""

    python -c "
from ml_skeleton.music.clementine_db import ClementineDB
import os

db_path = os.environ.get('CLEMENTINE_DB_PATH', '/Music/database/clementine_backup_2026-01.db')
db = ClementineDB(db_path)
songs = db.get_all_songs()

rated = [s for s in songs if s.rating is not None and s.rating > 0]
unrated = [s for s in songs if s.rating is None or s.rating <= 0]

# Rating distribution
from collections import Counter
rating_dist = Counter(int(s.rating) for s in rated if s.rating)

print(f'Total songs: {len(songs)}')
print(f'Rated songs: {len(rated)}')
print(f'Unrated songs: {len(unrated)}')
print('')
print('Rating distribution:')
for r in sorted(rating_dist.keys()):
    print(f'  {r} stars: {rating_dist[r]} songs')
"

    print_success "Database sync complete!"
    echo ""
    echo "NOTE: The Clementine database is always read fresh."
    echo "To incorporate new ratings into the model:"
    echo "  1. Rate songs in Clementine (or via recommender_help.xspf)"
    echo "  2. Re-run classifier training: $0 classifier --final-training"
    echo "  3. Promote to prod: $0 promote-to-prod"
}

run_build_cache() {
    print_header "Building 4-Chunk Waveform Cache"
    echo "Pre-populating cache for fast training..."
    echo "  - 4 chunks per song (evenly spaced)"
    echo "  - 30 seconds per chunk at 16kHz"
    echo "  - Estimated size: ~30GB for 60K songs"
    echo ""
    python "$SCRIPT" --stage build-cache --config "$CONFIG"
    print_success "Cache build complete!"
    echo ""
}

run_fingerprint() {
    print_header "Extracting Acoustic Fingerprints"
    echo "Generating chromaprint fingerprints from original audio files..."
    echo "  - Fingerprints FULL songs (required for AcoustID matching)"
    echo "  - Stores in fingerprint database (./cache/fingerprints.db)"
    echo "  - Enables duplicate detection and metadata enrichment"
    echo ""
    python "$SCRIPT" --stage fingerprint --config "$CONFIG" "$@"
    print_success "Fingerprinting complete!"
    echo ""
}

run_fingerprint_stats() {
    print_header "Fingerprint Database Statistics"
    python -c "
from ml_skeleton.music.fingerprint_db import FingerprintDB
from pathlib import Path

fp_db_path = './cache/fingerprints.db'
if Path(fp_db_path).exists():
    db = FingerprintDB(fp_db_path)
    stats = db.get_stats()
    print(f'  Total fingerprints: {stats[\"total_fingerprints\"]}')
    print(f'  Unique songs: {stats[\"unique_songs\"]}')
    print(f'  Complete fingerprints: {stats[\"songs_with_complete_fingerprints\"]}')
    print(f'  Canonical songs: {stats[\"canonical_songs\"]}')
    print(f'  Duplicate songs: {stats[\"duplicate_songs\"]}')
    print(f'  Duplicate groups: {stats[\"duplicate_groups\"]}')
    print(f'  DB size: {stats[\"db_size_mb\"]} MB')
else:
    print('  No fingerprint database found')
    print('  Run: ./run_music_pipeline.sh fingerprint')
"
    echo ""
}

run_enrich_metadata() {
    print_header "Enriching Metadata via AcoustID/MusicBrainz"
    echo "Querying external APIs for high-confidence metadata..."
    echo "  - Uses fingerprints to look up songs in AcoustID"
    echo "  - Fetches artist, album, genre from MusicBrainz"
    echo "  - Stores with confidence scores in separate database"
    echo ""
    echo "IMPORTANT: Requires ACOUSTID_API_KEY environment variable"
    echo "  Free tier: 500 lookups/day @ 3 req/sec"
    echo "  Register at: https://acoustid.org/new-application"
    echo ""
    python "$SCRIPT" --stage enrich-metadata --config "$CONFIG" "$@"
    print_success "Metadata enrichment complete!"
    echo ""
}

run_musicbrainz_stats() {
    print_header "MusicBrainz Database Statistics"
    python -c "
from ml_skeleton.music.musicbrainz_db import MusicBrainzDB
from pathlib import Path

mb_db_path = './musicbrainz_metadata.db'
if Path(mb_db_path).exists():
    db = MusicBrainzDB(mb_db_path)
    stats = db.get_stats()
    print(f'  Total enriched songs: {stats[\"total_songs\"]}')
    print(f'  With AcoustID: {stats[\"with_acoustid\"]}')
    print(f'  With MusicBrainz ID: {stats[\"with_musicbrainz\"]}')
    print(f'  High confidence artist: {stats[\"high_confidence_artist\"]}')
    print(f'  High confidence album: {stats[\"high_confidence_album\"]}')
    print(f'  Avg artist confidence: {stats[\"avg_artist_confidence\"]:.3f}')
    print(f'  Avg album confidence: {stats[\"avg_album_confidence\"]:.3f}')
    print(f'  DB size: {stats[\"db_size_mb\"]} MB')
else:
    print('  No MusicBrainz database found')
    print('  Run: ./run_music_pipeline.sh enrich-metadata')
"
    echo ""
}

run_fingerprint_and_enrich() {
    print_header "Complete Fingerprinting + Enrichment Pipeline"
    echo "This will:"
    echo "  1. Extract chromaprint fingerprints from cached chunks"
    echo "  2. Enrich metadata via AcoustID/MusicBrainz APIs"
    echo "  3. Display statistics"
    echo ""

    # Step 1: Fingerprint
    run_fingerprint "${EXTRA_ARGS[@]}"

    # Step 2: Enrich (if API key is set)
    if [ -n "$ACOUSTID_API_KEY" ]; then
        run_enrich_metadata "${EXTRA_ARGS[@]}"
    else
        print_warning "Skipping metadata enrichment: ACOUSTID_API_KEY not set"
        echo "  Register for free API key at: https://acoustid.org/"
        echo "  Then set it with: export ACOUSTID_API_KEY=your_key_here"
        echo "  Free tier: 500 lookups/day (perfect for 10-song testing)"
        echo "  Or run enrichment separately: ./run_music_pipeline.sh enrich-metadata"
    fi

    # Step 3: Display stats
    echo ""
    run_fingerprint_stats
    run_musicbrainz_stats

    print_success "Fingerprinting pipeline complete!"
}

# HPO functions
run_encoder_hpo() {
    local n_trials="${1:-$HPO_ENCODER_TRIALS}"

    print_header "HPO Step 1: Encoder Hyperparameter Tuning"
    echo "Running Optuna with $n_trials trials (may take hours)..."
    echo ""

    # Backup config
    cp "$CONFIG" "${CONFIG}.hpo_backup"
    print_success "Config backup: ${CONFIG}.hpo_backup"

    # Run tuning
    HPO_LOG="/tmp/encoder_hpo.log"
    python "$SCRIPT" --stage tune-encoder --config "$CONFIG" \
        --n-trials "$n_trials" 2>&1 | tee "$HPO_LOG"

    # Extract best params
    echo ""
    print_header "Best Encoder Parameters"
    grep -A 10 "Best parameters:" "$HPO_LOG" | grep ":" | head -4
    echo ""
    print_success "Encoder HPO complete! Review parameters above and update config manually."
    echo ""
}

run_classifier_hpo() {
    local n_trials="${1:-$HPO_CLASSIFIER_TRIALS}"

    print_header "Classifier Hyperparameter Tuning"
    echo "Running Optuna with $n_trials trials..."
    echo ""

    # Run tuning
    HPO_LOG="/tmp/classifier_hpo.log"
    python "$SCRIPT" --stage tune-classifier --config "$CONFIG" \
        --n-trials "$n_trials" 2>&1 | tee "$HPO_LOG"

    # Extract best params
    echo ""
    print_header "Best Classifier Parameters"
    grep -A 10 "Best parameters:" "$HPO_LOG" | grep ":" | head -4
    echo ""
    print_success "Classifier HPO complete! Review parameters above and update config manually."
    echo ""
}

display_model_card() {
    print_header "Model Card"
    MODEL_CARD="checkpoints/MODEL_CARD.md"
    if [ -f "$MODEL_CARD" ]; then
        cat "$MODEL_CARD"
        print_success "Full model card: $MODEL_CARD"
    else
        print_warning "Model card not found"
    fi
    echo ""
}

display_ab_history() {
    # Show A/B test history from current training and prod history
    # Called automatically after --final-training and via 'ab-history' command

    # Check if there's any history to show
    local has_history=false
    if [ -d "prod/history" ] && [ -n "$(ls -A prod/history/model_card_*.json 2>/dev/null)" ]; then
        has_history=true
    fi

    # Check if current training has A/B results
    local has_current=false
    if [ -f "checkpoints/training_manifest.json" ]; then
        if grep -q '"ab_test_result"' checkpoints/training_manifest.json 2>/dev/null; then
            has_current=true
        fi
    fi

    if [ "$has_history" = false ] && [ "$has_current" = false ]; then
        return  # Nothing to show
    fi

    print_header "A/B Test History"
    python -c "
import json
from pathlib import Path
from datetime import datetime

results = []

# Parse all historical model cards from prod/history
history_dir = Path('prod/history')
if history_dir.exists():
    for f in sorted(history_dir.glob('model_card_*.json')):
        try:
            with open(f) as fp:
                data = json.load(fp)
            ts_str = f.stem.replace('model_card_', '')
            try:
                ts = datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
                date_str = ts.strftime('%Y-%m-%d %H:%M')
            except:
                date_str = ts_str
            classifier_stats = data.get('classifier_stats', {})
            ab = classifier_stats.get('metadata', {}).get('ab_test_result', {})
            if ab:
                results.append({
                    'date': date_str,
                    'train_size': classifier_stats.get('train_size', '-'),
                    'val_size': classifier_stats.get('val_size', '-'),
                    'n_samples': ab.get('n_samples', 0),
                    'improvement': ab.get('improvement', 0),
                    'p_value': ab.get('p_value', 1.0),
                    'significant': ab.get('significant', False),
                    'label': ''
                })
        except:
            pass

# Check current prod model card
if Path('prod/model_card.json').exists():
    try:
        with open('prod/model_card.json') as fp:
            data = json.load(fp)
        classifier_stats = data.get('classifier_stats', {})
        ab = classifier_stats.get('metadata', {}).get('ab_test_result', {})
        if ab:
            results.append({
                'date': 'PROD',
                'train_size': classifier_stats.get('train_size', '-'),
                'val_size': classifier_stats.get('val_size', '-'),
                'n_samples': ab.get('n_samples', 0),
                'improvement': ab.get('improvement', 0),
                'p_value': ab.get('p_value', 1.0),
                'significant': ab.get('significant', False),
                'label': '← current prod'
            })
    except:
        pass

# Check current training manifest (just trained model)
if Path('checkpoints/training_manifest.json').exists():
    try:
        with open('checkpoints/training_manifest.json') as fp:
            data = json.load(fp)
        metadata = data.get('metadata', {})
        ab = metadata.get('ab_test_result', {})
        if ab:
            results.append({
                'date': 'NEW',
                'train_size': metadata.get('train_size', '-'),
                'val_size': metadata.get('val_size', '-'),
                'n_samples': ab.get('n_samples', 0),
                'improvement': ab.get('improvement', 0),
                'p_value': ab.get('p_value', 1.0),
                'significant': ab.get('significant', False),
                'label': '← just trained'
            })
    except:
        pass

if results:
    print(f\"{'Date':<20} {'Train':>6} {'Val':>6} {'Vault':>6} {'Δ Accuracy':>11} {'p-value':>9} {'Sig?':>5}  Notes\")
    print('-' * 85)
    for r in results:
        sig = '✓' if r['significant'] else ''
        imp = f\"{r['improvement']:+.4f}\" if r['improvement'] != 0 else ' 0.0000'
        train_str = str(r['train_size']) if r['train_size'] != '-' else '-'
        val_str = str(r['val_size']) if r['val_size'] != '-' else '-'
        vault_str = str(r['n_samples'])
        print(f\"{r['date']:<20} {train_str:>6} {val_str:>6} {vault_str:>6} {imp:>11} {r['p_value']:>9.4f} {sig:>5}  {r['label']}\")
    print()
    sig_count = sum(1 for r in results if r['significant'])
    print(f'Models in history: {len(results)} | Significant improvements: {sig_count}')
else:
    print('No A/B test results found.')
"
    echo ""
}

run_hpo_pipeline() {
    print_header "FULL HYPERPARAMETER OPTIMIZATION PIPELINE"
    echo "Steps:"
    echo "  1. Tune encoder ($HPO_ENCODER_TRIALS trials × 20 epochs each)"
    echo "  2. AUTOMATED: Train encoder with best parameters (100 epochs)"
    echo "  3. Tune classifier ($HPO_CLASSIFIER_TRIALS trials × 20 epochs each)"
    echo "  4. AUTOMATED: Train classifier with best parameters (20 epochs)"
    echo "  5. Display model card"
    echo "  6. Generate recommendations"
    echo ""
    echo "This is fully automated! Best parameters are applied automatically."
    echo "WARNING: This takes many hours!"
    echo ""

    # Step 1: Encoder HPO
    run_encoder_hpo "$HPO_ENCODER_TRIALS"

    # Step 2: Train encoder with best params - AUTOMATED
    BEST_ENCODER_PARAMS="checkpoints/best_encoder_params.json"
    HPO_BEST_CHECKPOINT="checkpoints/encoder_hpo_best.pt"
    if [ -f "$BEST_ENCODER_PARAMS" ]; then
        print_header "HPO Step 2: Training Encoder with Best Parameters (100 epochs)"
        print_success "Using best parameters from: $BEST_ENCODER_PARAMS"

        # Resume from HPO best checkpoint if available
        if [ -f "$HPO_BEST_CHECKPOINT" ]; then
            print_success "Resuming from HPO best model: $HPO_BEST_CHECKPOINT"
            python "$SCRIPT" --stage encoder --config "$CONFIG" \
                --final-training --best-params "$BEST_ENCODER_PARAMS" \
                --resume-checkpoint "$HPO_BEST_CHECKPOINT"
        else
            python "$SCRIPT" --stage encoder --config "$CONFIG" \
                --final-training --best-params "$BEST_ENCODER_PARAMS"
        fi
        print_success "Encoder training with best params complete!"
        echo ""
    else
        print_error "Best encoder parameters not found: $BEST_ENCODER_PARAMS"
        echo "Running encoder HPO should have created this file."
        exit 1
    fi

    # Step 3: Classifier HPO
    run_classifier_hpo "$HPO_CLASSIFIER_TRIALS"

    # Step 4: Train classifier with best params - AUTOMATED
    BEST_CLASSIFIER_PARAMS="checkpoints/best_classifier_params.json"
    if [ -f "$BEST_CLASSIFIER_PARAMS" ]; then
        print_header "HPO Step 4: Training Classifier with Best Parameters (20 epochs)"
        print_success "Using best parameters from: $BEST_CLASSIFIER_PARAMS"
        python "$SCRIPT" --stage classifier --config "$CONFIG" \
            --final-training --best-params "$BEST_CLASSIFIER_PARAMS"
        print_success "Classifier training with best params complete!"
        echo ""
    else
        print_error "Best classifier parameters not found: $BEST_CLASSIFIER_PARAMS"
        echo "Running classifier HPO should have created this file."
        exit 1
    fi

    # Step 5: Model card
    display_model_card

    # Step 6: Recommendations
    run_recommend

    # Final summary
    print_header "HPO PIPELINE COMPLETE!"
    echo "Results:"
    echo "  ✓ Optimized models in checkpoints/"
    echo "  ✓ Best encoder params: checkpoints/best_encoder_params.json"
    echo "  ✓ Best classifier params: checkpoints/best_classifier_params.json"
    echo "  ✓ Model card: checkpoints/MODEL_CARD.md"
    echo "  ✓ Recommendations: recommendations.txt"
    echo "  ✓ Config backup: ${CONFIG}.hpo_backup"
    echo ""
    echo "Note: Best parameters were automatically applied during training!"
    echo ""
}

run_quick_test() {
    print_header "Quick Test Mode"
    TEMP_CONFIG="/tmp/music_moco_quick.yaml"
    cp "$CONFIG" "$TEMP_CONFIG"
    sed -i 's/epochs: 100/epochs: 5/g' "$TEMP_CONFIG"
    sed -i 's/epochs: 50/epochs: 5/g' "$TEMP_CONFIG"
    sed -i 's/epochs: 20/epochs: 5/g' "$TEMP_CONFIG"
    print_success "Temp config: $TEMP_CONFIG"
    echo ""

    CONFIG="$TEMP_CONFIG" run_encoder
    CONFIG="$TEMP_CONFIG" run_classifier
    CONFIG="$TEMP_CONFIG" run_recommend

    print_success "Quick test complete!"
}

main() {
    # Parse arguments
    STAGE="${1:-all}"
    shift || true

    # Collect extra arguments to pass through
    EXTRA_ARGS=()

    # Parse additional arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --resume-checkpoint)
                RESUME_CHECKPOINT="$2"
                shift 2
                ;;
            --resume-checkpoint=*)
                RESUME_CHECKPOINT="${1#*=}"
                shift
                ;;
            --encoder-version)
                ENCODER_VERSION="$2"
                shift 2
                ;;
            --encoder-version=*)
                ENCODER_VERSION="${1#*=}"
                shift
                ;;
            --classifier-version)
                CLASSIFIER_VERSION="$2"
                shift 2
                ;;
            --classifier-version=*)
                CLASSIFIER_VERSION="${1#*=}"
                shift
                ;;
            --model-version)
                # Backwards compatibility
                ENCODER_VERSION="$2"
                echo "NOTE: --model-version is deprecated, use --encoder-version instead"
                shift 2
                ;;
            --model-version=*)
                ENCODER_VERSION="${1#*=}"
                echo "NOTE: --model-version is deprecated, use --encoder-version instead"
                shift
                ;;
            --exhaust)
                # Process maximum songs for the day (respects API limits)
                EXTRA_ARGS+=("--exhaust")
                shift
                ;;
            *)
                # Collect unknown arguments to pass through to Python script
                EXTRA_ARGS+=("$1")
                shift
                ;;
        esac
    done

    check_prerequisites

    case "$STAGE" in
        all)
            [ "${MIN_RATED_SONGS}" = "500" ] && export MIN_RATED_SONGS=60000
            run_encoder "${EXTRA_ARGS[@]}"
            run_classifier "${EXTRA_ARGS[@]}"
            run_recommend
            display_model_card
            print_header "Pipeline Complete!"
            ;;
        encoder)
            run_encoder "${EXTRA_ARGS[@]}"
            ;;
        classifier)
            run_classifier "${EXTRA_ARGS[@]}"
            ;;
        recommend)
            run_recommend "${EXTRA_ARGS[@]}"
            ;;
        promote-to-prod)
            run_promote_to_prod
            ;;
        sync-db)
            run_sync_db
            ;;
        init-baseline)
            print_header "Creating Random Baseline for A/B Testing"
            python "$SCRIPT" --stage init-baseline --config "$CONFIG" "${EXTRA_ARGS[@]}"
            print_success "Random baseline created in prod/"
            echo ""
            ;;
        quick)
            run_quick_test
            ;;
        build-cache)
            run_build_cache
            ;;
        fingerprint)
            run_fingerprint "${EXTRA_ARGS[@]}"
            ;;
        fingerprint-stats)
            run_fingerprint_stats
            ;;
        enrich-metadata)
            run_enrich_metadata "${EXTRA_ARGS[@]}"
            ;;
        enrich)
            run_fingerprint_and_enrich "${EXTRA_ARGS[@]}"
            ;;
        musicbrainz-stats)
            run_musicbrainz_stats
            ;;
        hpo)
            run_hpo_pipeline
            ;;
        hpo-encoder)
            run_encoder_hpo
            ;;
        hpo-classifier)
            run_classifier_hpo
            ;;
        clear-cache)
            print_header "Clearing Waveform Cache"
            CACHE_DIR="./cache"
            if [ -d "$CACHE_DIR" ]; then
                CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
                CACHE_FILES=$(find "$CACHE_DIR" -name "*.npy" 2>/dev/null | wc -l)
                echo "Cache directory: $CACHE_DIR"
                echo "Size: $CACHE_SIZE ($CACHE_FILES files)"
                read -p "Delete cache? [y/N] " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    rm -rf "$CACHE_DIR"
                    print_success "Cache cleared!"
                else
                    print_warning "Cache not deleted"
                fi
            else
                print_warning "No cache found at $CACHE_DIR"
            fi
            ;;
        model-card)
            display_model_card
            ;;
        ab-history)
            display_ab_history
            ;;
        cache-stats)
            print_header "Cache Statistics"
            python -c "
from ml_skeleton.music.chunk_cache import get_cache_stats
stats = get_cache_stats()
if stats['exists']:
    print(f'  Directory: {stats[\"cache_dir\"]}')
    print(f'  Files: {stats[\"num_files\"]}')
    print(f'  Songs: {stats[\"num_songs\"]}')
    print(f'  Size: {stats[\"size_gb\"]:.2f} GB')
else:
    print('  No cache found')
"
            ;;
        *)
            echo "Usage: $0 {all|encoder|classifier|recommend|quick|hpo|hpo-encoder|hpo-classifier|build-cache|fingerprint|enrich|enrich-metadata|fingerprint-stats|musicbrainz-stats|clear-cache|cache-stats|model-card|ab-history|promote-to-prod|sync-db} [options]"
            echo ""
            echo "Stages:"
            echo "  all                 - Run complete pipeline (encoder + classifier + recommend)"
            echo "  encoder             - Train MoCo v2 + Genre BCE encoder"
            echo "  classifier          - Train rating classifier"
            echo "  recommend           - Generate recommendations"
            echo "  quick               - Quick test (5 epochs, 500 songs)"
            echo "  hpo                 - Full hyperparameter optimization (encoder + classifier)"
            echo "  hpo-encoder         - Hyperparameter optimization for encoder only"
            echo "  hpo-classifier      - Hyperparameter optimization for classifier only"
            echo "  build-cache         - Build 4-chunk waveform cache (~30GB)"
            echo "  fingerprint         - Extract acoustic fingerprints from original files (for AcoustID)"
            echo "  enrich              - Complete pipeline: fingerprint + enrich + stats (recommended)"
            echo "  enrich-metadata     - Enrich metadata via AcoustID/MusicBrainz (requires API key)"
            echo "  fingerprint-stats   - Display fingerprint database statistics"
            echo "  musicbrainz-stats   - Display MusicBrainz database statistics"
            echo "  clear-cache         - Delete waveform cache"
            echo "  cache-stats         - Show cache statistics"
            echo "  model-card          - Display model card"
            echo "  ab-history          - Show A/B test history from archived model cards"
            echo ""
            echo "Production:"
            echo "  promote-to-prod     - Copy best models to prod/ folder (archives previous)"
            echo "  init-baseline       - Create random baseline in prod/ for A/B testing workflow"
            echo "  sync-db             - Check database status and rating counts"
            echo "  recommend --prod    - Generate recommendations using prod models"
            echo "  recommend --prod --low-rating-ratio 0.1  - Include 10% predicted dislikes"
            echo "  recommend --prod --genre rock           - Recommendations for rock songs only"
            echo ""
            echo "Options:"
            echo "  --resume-checkpoint PATH   - Resume training from checkpoint"
            echo "  --encoder-version VERSION  - Encoder version for embeddings (e.g., v2)"
            echo "  --classifier-version VER   - Classifier version (e.g., v2)"
            echo "  --exhaust                  - Process max songs for the day (500 for free tier)"
            echo "  --workers N                - Number of parallel workers for fingerprinting (default: 4)"
            echo "  --low-rating-ratio N       - Include N% predicted dislikes in recommendations (0.0-1.0)"
            echo "  --genre CATEGORY           - Filter recommendations by genre category"
            echo "                               Categories: rock, pop, electronic, hiphop, jazz_classical, country, latin_world"
            echo "  --random-init              - Use random init instead of loading from prod model (default: prod init)"
            echo "  --vault-size N             - Number of ratings to reserve for A/B testing (default: 200)"
            echo ""
            echo "Environment Variables:"
            echo "  HPO_ENCODER_TRIALS=30"
            echo "  HPO_CLASSIFIER_TRIALS=20"
            echo "  RESUME_CHECKPOINT=/path/to   - Resume from checkpoint"
            echo "  ENCODER_VERSION=v2           - Encoder version for embeddings"
            echo "  CLASSIFIER_VERSION=v2        - Classifier version"
            echo "  ACOUSTID_API_KEY=key         - AcoustID API key for metadata enrichment"
            echo ""
            echo "Getting AcoustID API Key (for metadata enrichment):"
            echo "  1. Register free account at https://acoustid.org/"
            echo "  2. Get your API key from account settings/API applications page"
            echo "  3. Export it: export ACOUSTID_API_KEY=your_key_here"
            echo "  4. Free tier: 500 lookups/day (perfect for 10-song testing)"
            echo "  5. Paid tier: \$10/year for unlimited lookups (recommended for full collection)"
            echo ""
            echo "Architecture:"
            echo "  Audio → 16kHz .npy cache (4 chunks/song) → nnAudio CQT → ResNet-50 2D"
            echo "  ├── MoCo v2 head (queue=4096, τ=0.07)"
            echo "  └── Genre BCE head (7 categories)"
            echo ""
            echo "Examples:"
            echo "  $0 build-cache                          # Build cache first (recommended)"
            echo "  $0 all                                  # Run complete pipeline"
            echo "  $0 encoder                              # Train encoder only"
            echo "  HPO_ENCODER_TRIALS=50 $0 hpo            # Run HPO with 50 encoder trials"
            echo "  $0 fingerprint --workers 8              # Fingerprint with 8 parallel workers"
            echo "  ACOUSTID_API_KEY=key $0 enrich          # Fingerprint + enrich + stats (10 songs)"
            echo "  ACOUSTID_API_KEY=key $0 enrich --exhaust --workers 8 # Process 500 songs with 8 workers"
            echo ""
            echo "Version Compatibility Rules:"
            echo "  - Encoder and Classifier have SEPARATE versions"
            echo "  - Classifier stores which encoder version it was trained with"
            echo "  - If Encoder is updated, Classifier MUST be retrained"
            echo "  - Deployment (recommend) FAILS if versions don't match"
            exit 1
            ;;
    esac
}

main "$@"
