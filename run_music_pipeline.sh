#!/bin/bash
# Music Recommendation Pipeline Runner (MoCo v2 + Genre BCE)
# Requires bash (arrays, [[, etc.). Re-exec with bash if invoked via sh.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
#
# Usage:
#   ./run_music_pipeline.sh all              # Run all 3 stages
#   ./run_music_pipeline.sh encoder          # Run Stage 1 only (MoCo v2 + Genre)
#   ./run_music_pipeline.sh classifier       # Run Stage 2 only
#   ./run_music_pipeline.sh recommend        # Run Stage 3 only
#   ./run_music_pipeline.sh rag-query /path/to/playlist.xspf   # prod/embeddings.db → rag_<name>.xspf
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
#   CHUNK_CACHE_DIR=/path/to/chunks          # Chunk cache dir (e.g. /music-cache/chunks in container)
#   HPO_ENCODER_TRIALS=30                    # Number of encoder HPO trials
#   HPO_CLASSIFIER_TRIALS=800               # Number of classifier HPO trials (default 800)
#   RESUME_CHECKPOINT=/path/to/checkpoint    # Resume from previous training
#   ENCODER_VERSION=v2                       # Encoder version for embeddings
#   CLASSIFIER_VERSION=v2                    # Classifier version

set -e  # Exit on error

CONFIG="${CONFIG:-configs/music_moco.yaml}"
SCRIPT="examples/music_recommendation.py"
HPO_ENCODER_TRIALS="${HPO_ENCODER_TRIALS:-30}"
HPO_CLASSIFIER_TRIALS="${HPO_CLASSIFIER_TRIALS:-800}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"  # Optional: path to checkpoint to resume from
ENCODER_VERSION="${ENCODER_VERSION:-}"  # Optional: encoder version for embeddings (e.g., "v2")
CLASSIFIER_VERSION="${CLASSIFIER_VERSION:-}"  # Optional: classifier version (e.g., "v2")

# Set minimum rated songs for placeholder database
export MIN_RATED_SONGS="${MIN_RATED_SONGS:-500}"

# Path remapping for audio files
export MUSIC_PATH_REMAP="${MUSIC_PATH_REMAP:-/home/${USER}/Music:/Music}"

# Allow database path override
if [ -n "$CLEMENTINE_DB_PATH" ]; then
    export CLEMENTINE_DB_PATH
fi
# Chunk cache directory (e.g. /music-cache/chunks when mounted in container)
[ -n "${CHUNK_CACHE_DIR:-}" ] && export CHUNK_CACHE_DIR

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

    echo "Architecture: Audio → CQT → ResNet-50 2D → 4096-dim"
    echo "Loss: MoCo(NT-Xent) only (genre_bce=0)"
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

run_joint_finetune() {
    print_header "Joint Fine-Tune (encoder + classifier on audio)"
    python "$SCRIPT" --stage joint-finetune --config "$CONFIG" "$@"
    print_success "Joint fine-tune complete!"
    echo ""
    display_ab_history
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

run_rag_query() {
    if [ $# -lt 1 ]; then
        print_error "rag-query requires PLAYLIST.xspf as the first argument"
        echo ""
        echo "Usage: $0 rag-query PLAYLIST.xspf [options]"
        echo "  Uses prod/embeddings.db (run promote-to-prod first)."
        echo "  Playlist path: use /Music/... in Docker; ~/Music → /root/Music and is usually wrong."
        echo "  Options: --rag-top-n N  --rag-num-pc N  --rag-unrated-only  --rag-likes-only (no --prod)"
        exit 1
    fi
    local playlist="$1"
    shift
    if [[ "$playlist" == -* ]]; then
        print_error "First argument must be the playlist path, not a flag"
        exit 1
    fi
    # Expand leading ~/ if passed quoted (unquoted ~ is expanded by the shell before we see it)
    if [[ "$playlist" == "~/"* ]]; then
        playlist="${HOME}/${playlist#~/}"
    fi
    # Docker / dev containers: repo often uses /Music mount while HOME is /root → ~/Music → missing
    if [ ! -f "$playlist" ]; then
        local home_music="${HOME%/}/Music"
        if [[ "$playlist" == "$home_music"/* ]]; then
            local alt="/Music/${playlist#"$home_music"/}"
            if [ -f "$alt" ]; then
                print_warning "Playlist not at $playlist — using $alt (container: use /Music/...)"
                playlist="$alt"
            fi
        fi
    fi
    if [ ! -f "$playlist" ]; then
        print_error "Playlist not found: $playlist"
        echo "  Tip: in Docker, libraries are often mounted at /Music/ (not ~/Music)."
        exit 1
    fi

    for arg in "$@"; do
        if [ "$arg" = "--prod" ]; then
            print_error "rag-query does not take --prod (it always uses prod/embeddings.db)"
            exit 1
        fi
    done

    print_header "RAG query (prod embeddings)"
    if [ ! -f "prod/embeddings.db" ]; then
        print_error "prod/embeddings.db not found"
        echo "Run '$0 promote-to-prod' first."
        exit 1
    fi
    python "$SCRIPT" --stage rag-query --config "$CONFIG" "$@" "$playlist" --prod-dir prod
    print_success "RAG query complete (rag_*.xspf under recommendations.output_dir in config)"
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
    echo "Database path: ${CLEMENTINE_DB_PATH:-/Music/database/clementine_backup_2026-03.db}"
    echo ""

    python -c "
from ml_skeleton.music.clementine_db import ClementineDB
import os

db_path = os.environ.get('CLEMENTINE_DB_PATH', '/Music/database/clementine_backup_2026-03.db')
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
    print_header "Building Waveform Chunk Cache"
    echo "Pre-populating cache for fast training (config: music.chunk_cache)..."
    echo "  - num_chunks per song, 30s per chunk at 16kHz; use --overwrite when changing num_chunks"
    echo ""
    python "$SCRIPT" --stage build-cache --config "$CONFIG" "$@"
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
import os
import yaml
from pathlib import Path
from ml_skeleton.music.fingerprint_db import FingerprintDB
from ml_skeleton.music.encoder_factory import get_fingerprint_db_path

# Use same canonical path as encoder/classifier/fingerprint stage
with open('$CONFIG') as f:
    config = yaml.safe_load(f)
fp_db_path = get_fingerprint_db_path(config)
clem_path = os.getenv('CLEMENTINE_DB_PATH') or config.get('music', {}).get('database_path', '')

if Path(fp_db_path).exists():
    db = FingerprintDB(fp_db_path)
    stats = db.get_stats()
    chunk_cfg = config.get('fingerprinting', {}).get('chunk_for_fingerprinting', 0)
    print(f'  Total fingerprints: {stats[\"total_fingerprints\"]}')
    print(f'  Unique songs: {stats[\"unique_songs\"]}')
    print(f'  Complete (all 8 chunks per song): {stats[\"songs_with_complete_fingerprints\"]} (typical: 1 chromaprint per song at one chunk index)')
    by_chunk = stats.get('fingerprints_by_chunk', {})
    if by_chunk:
        chunks = sorted(by_chunk.keys())
        print('  By chunk index: ' + ' '.join([f'chunk_{c}: {by_chunk[c]}' for c in chunks]))
        print(f'  Config chunk_for_fingerprinting: {chunk_cfg} (encoder/MoCo use this chunk)')
        if chunk_cfg not in by_chunk or by_chunk.get(chunk_cfg, 0) == 0:
            print('  -> No fingerprints at config chunk; set chunk_for_fingerprinting to a populated chunk or re-run fingerprint with that chunk.')
    print(f'  Canonical songs: {stats[\"canonical_songs\"]}')
    print(f'  Duplicate songs: {stats[\"duplicate_songs\"]}')
    print(f'  Duplicate groups: {stats[\"duplicate_groups\"]}')
    print(f'  DB size: {stats[\"db_size_mb\"]} MB')
    # Coverage vs Clementine (local only)
    if clem_path and Path(clem_path).exists():
        from ml_skeleton.music.clementine_db import ClementineDB
        clem = ClementineDB(clem_path)
        total = len(clem.get_all_songs())
        with_fp = stats['unique_songs']
        missing = total - with_fp
        pct = 100.0 * with_fp / total if total else 0
        print('')
        print('  Coverage vs Clementine (local):')
        print(f'    Total songs:     {total:,}')
        print(f'    With fingerprint: {with_fp:,}')
        print(f'    Missing:        {missing:,}')
        print(f'    Coverage:       {pct:.1f}%')
else:
    print('  No fingerprint database found')
    print('  Run: ./run_music_pipeline.sh fingerprint')
"
    echo ""
}

run_backfill_fingerprint_bits() {
    print_header "Backfill fingerprint bits column (precomputed 32-byte blobs)"
    echo "Uses same pipeline as fingerprint_baseline extraction (main process, same DataLoader/encoder)."
    echo ""
    python "$SCRIPT" --stage backfill-fingerprint-bits --config "$CONFIG" "$@"
    print_success "Backfill complete!"
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
    local n_trials="$HPO_ENCODER_TRIALS"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -N)
                n_trials="${2:-$HPO_ENCODER_TRIALS}"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done

    print_header "HPO Step 1: Encoder Hyperparameter Tuning"
    echo "Running Optuna with $n_trials trials (may take hours)..."
    echo ""

    # Backup config
    cp "$CONFIG" "${CONFIG}.hpo_backup"
    print_success "Config backup: ${CONFIG}.hpo_backup"

    # Run tuning (pass any remaining args through to Python, e.g. --reset-study)
    # OMP_NUM_THREADS=1 reduces risk of malloc/OpenMP crashes in Docker; override by setting in env before calling
    HPO_LOG="/tmp/encoder_hpo.log"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    python "$SCRIPT" --stage tune-encoder --config "$CONFIG" \
        --n-trials "$n_trials" "$@" 2>&1 | tee "$HPO_LOG"

    # Extract best params
    echo ""
    print_header "Best Encoder Parameters"
    grep -A 10 "Best parameters:" "$HPO_LOG" | grep ":" | head -4
    echo ""
    print_success "Encoder HPO complete! Review parameters above and update config manually."
    echo "Run next:"
    echo "  ./run_music_pipeline.sh encoder --encoder-type moco --final-training --best-params checkpoints/best_encoder_params.json --resume-checkpoint checkpoints/encoder_hpo_best.pt"
    echo ""
}

run_classifier_hpo() {
    local n_trials="$HPO_CLASSIFIER_TRIALS"
    # Parse -N <num> from args (e.g. ./run_music_pipeline.sh hpo-classifier -N 10)
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -N)
                n_trials="${2:-$HPO_CLASSIFIER_TRIALS}"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done

    print_header "Classifier Hyperparameter Tuning"
    echo "Running Optuna with $n_trials trials..."
    echo ""

    # Run tuning (pass any remaining args through to Python)
    HPO_LOG="/tmp/classifier_hpo.log"
    python "$SCRIPT" --stage tune-classifier --config "$CONFIG" \
        --n-trials "$n_trials" "$@" 2>&1 | tee "$HPO_LOG"

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
    local has_ckpt_val_metrics=false
    if [ -f "checkpoints/training_manifest.json" ]; then
        if grep -q '"ab_test_result"' checkpoints/training_manifest.json 2>/dev/null; then
            has_current=true
        fi
        if grep -qE '"val_ppv"|"val_recall"|"val_precision_at_5"|"val_precision_at_20"' checkpoints/training_manifest.json 2>/dev/null; then
            has_ckpt_val_metrics=true
        fi
    fi

    if [ "$has_history" = false ] && [ "$has_current" = false ] && [ "$has_ckpt_val_metrics" = false ]; then
        return  # Nothing to show
    fi

    print_header "A/B Test History"
    python << 'ABHISTEOF'
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
            except Exception:
                date_str = ts_str
            classifier_stats = data.get('classifier_stats', {})
            ab = classifier_stats.get('metadata', {}).get('ab_test_result', {})
            if ab:
                results.append({
                    'date': date_str,
                    'train_size': classifier_stats.get('train_size', '-'),
                    'val_size': classifier_stats.get('val_size', '-'),
                    'train_prevalence': classifier_stats.get('train_prevalence'),
                    'val_prevalence': classifier_stats.get('val_prevalence'),
                    'val_ppv': classifier_stats.get('val_ppv'),
                    'val_recall': classifier_stats.get('val_recall'),
                    'val_precision_at_5': classifier_stats.get('val_precision_at_5'),
                    'val_precision_at_20': classifier_stats.get('val_precision_at_20'),
                    'n_samples': ab.get('n_samples', 0),
                    'improvement': ab.get('improvement', 0),
                    'p_value': ab.get('p_value', 1.0),
                    'significant': ab.get('significant', False),
                    'label': ''
                })
        except Exception:
            pass

def _overlay_val_metrics(stats, meta):
    """Fill val_ppv / val_recall / P@5 from manifest metadata if missing."""
    if not meta:
        return stats
    out = dict(stats)
    for k in ('val_ppv', 'val_recall', 'val_precision_at_5', 'val_precision_at_20', 'train_prevalence', 'val_prevalence'):
        if out.get(k) is None and meta.get(k) is not None:
            out[k] = meta[k]
    return out

# Check current prod model card
if Path('prod/model_card.json').exists():
    try:
        with open('prod/model_card.json') as fp:
            data = json.load(fp)
        classifier_stats = data.get('classifier_stats', {})
        ab = classifier_stats.get('metadata', {}).get('ab_test_result', {})
        prod_meta = {}
        if Path('prod/training_manifest.json').exists():
            with open('prod/training_manifest.json') as fp:
                prod_meta = json.load(fp).get('metadata') or {}
        classifier_stats = _overlay_val_metrics(classifier_stats, prod_meta)
        if ab:
            results.append({
                'date': 'PROD',
                'train_size': classifier_stats.get('train_size', '-'),
                'val_size': classifier_stats.get('val_size', '-'),
                'train_prevalence': classifier_stats.get('train_prevalence'),
                'val_prevalence': classifier_stats.get('val_prevalence'),
                'val_ppv': classifier_stats.get('val_ppv'),
                'val_recall': classifier_stats.get('val_recall'),
                'val_precision_at_5': classifier_stats.get('val_precision_at_5'),
                'val_precision_at_20': classifier_stats.get('val_precision_at_20'),
                'n_samples': ab.get('n_samples', 0),
                'improvement': ab.get('improvement', 0),
                'p_value': ab.get('p_value', 1.0),
                'significant': ab.get('significant', False),
                'label': '← current prod'
            })
    except Exception:
        pass

# Check current training manifest (just trained model)
if Path('checkpoints/training_manifest.json').exists():
    try:
        with open('checkpoints/training_manifest.json') as fp:
            data = json.load(fp)
        metadata = data.get('metadata', {})
        ab = metadata.get('ab_test_result')
        vault_n = len(data.get('vault_files') or [])
        if ab:
            results.append({
                'date': 'NEW',
                'train_size': metadata.get('train_size', '-'),
                'val_size': metadata.get('val_size', '-'),
                'train_prevalence': metadata.get('train_prevalence'),
                'val_prevalence': metadata.get('val_prevalence'),
                'val_ppv': metadata.get('val_ppv'),
                'val_recall': metadata.get('val_recall'),
                'val_precision_at_5': metadata.get('val_precision_at_5'),
                'val_precision_at_20': metadata.get('val_precision_at_20'),
                'n_samples': ab.get('n_samples', 0),
                'improvement': ab.get('improvement', 0),
                'p_value': ab.get('p_value', 1.0),
                'significant': ab.get('significant', False),
                'label': '← just trained'
            })
        elif metadata.get('val_ppv') is not None or metadata.get('val_recall') is not None or metadata.get('val_precision_at_5') is not None or metadata.get('val_precision_at_20') is not None:
            lt = {
                'date': 'last train',
                'train_size': metadata.get('train_size', '-'),
                'val_size': metadata.get('val_size', '-'),
                'train_prevalence': metadata.get('train_prevalence'),
                'val_prevalence': metadata.get('val_prevalence'),
                'val_ppv': metadata.get('val_ppv'),
                'val_recall': metadata.get('val_recall'),
                'val_precision_at_5': metadata.get('val_precision_at_5'),
                'val_precision_at_20': metadata.get('val_precision_at_20'),
                'n_samples': vault_n if vault_n else '-',
                'improvement': None,
                'p_value': None,
                'significant': False,
                'label': '← val (no vault A/B)'
            }
            ck_mc = Path('checkpoints/model_card.json')
            if ck_mc.exists():
                try:
                    with open(ck_mc) as fp:
                        cs = (json.load(fp).get('classifier_stats') or {})
                    for k in ('val_ppv', 'val_recall', 'val_precision_at_5', 'val_precision_at_20'):
                        if lt.get(k) is None and cs.get(k) is not None:
                            lt[k] = cs[k]
                except Exception:
                    pass
            results.append(lt)
    except Exception:
        pass

def fmt_prev(p):
    if p is None:
        return '-'
    try:
        return f'{float(p)*100:.1f}%'
    except Exception:
        return '-'

if results:
    print(f"{'Date':<20} {'Train':>6} {'Val':>6} {'Vault@':>6} {'TrPrev':>7} {'ValPrev':>7} {'PPV':>6} {'Rec':>6} {'P@5':>6} {'P@20':>6} {'Δ Accuracy':>11} {'p-value':>9} {'Sig?':>5}  Notes")
    print('-' * 125)
    for r in results:
        sig = '✓' if r.get('significant') else ''
        if r.get('improvement') is None:
            imp = '     —     '
        elif r['improvement'] != 0:
            imp = f"{r['improvement']:+.4f}"
        else:
            imp = ' 0.0000'
        if r.get('p_value') is None:
            pv = '    —    '
        else:
            pv = f"{r['p_value']:>9.4f}"
        train_str = str(r['train_size']) if r['train_size'] != '-' else '-'
        val_str = str(r['val_size']) if r['val_size'] != '-' else '-'
        vault_str = str(r['n_samples'])
        tr_prev = fmt_prev(r.get('train_prevalence'))
        val_prev = fmt_prev(r.get('val_prevalence'))
        ppv = r.get('val_ppv')
        rec = r.get('val_recall')
        ppv_str = f"{float(ppv):.3f}" if ppv is not None else '-'
        rec_str = f"{float(rec):.3f}" if rec is not None else '-'
        p5 = r.get('val_precision_at_5')
        try:
            p5_str = f"{float(p5):.3f}" if p5 is not None and p5 == p5 else '-'
        except (TypeError, ValueError):
            p5_str = '-'
        p20 = r.get('val_precision_at_20')
        try:
            p20_str = f"{float(p20):.3f}" if p20 is not None and p20 == p20 else '-'
        except (TypeError, ValueError):
            p20_str = '-'
        print(f"{r['date']:<20} {train_str:>6} {val_str:>6} {vault_str:>6} {tr_prev:>7} {val_prev:>7} {ppv_str:>6} {rec_str:>6} {p5_str:>6} {p20_str:>6} {imp:>11} {pv} {sig:>5}  {r['label']}")
    print()
    sig_count = sum(1 for r in results if r.get('significant'))
    print(f'Models in history: {len(results)} | Significant improvements: {sig_count}')
    print("  Vault@ = vault size when that run's A/B test was done. When comparing two models, both are evaluated on the same current vault.")
    if any(r.get('date') == 'PROD' and r.get('val_ppv') is None for r in results):
        print("  PROD PPV/Rec/P@5/P@20: fill by running promote-to-prod (copies checkpoint manifest + regenerates model_card.json with val metrics).")
    if any(r.get('date') == 'last train' for r in results):
        print("  last train = val set metrics for checkpoints/classifier; — in Δ Acc/p = vault A/B not run (e.g. prod embedding dim mismatch).")
        print("  Backfill P@20 in manifest without retrain: python examples/music_recommendation.py --stage refresh-val-precision --config <yaml>")
else:
    print('No A/B test results found.')
ABHISTEOF
    echo ""
}

run_hpo_pipeline() {
    print_header "FULL HYPERPARAMETER OPTIMIZATION PIPELINE"
    local from_step="${HPO_FROM_STEP:-1}"
    echo "Steps:"
    echo "  1. Tune encoder ($HPO_ENCODER_TRIALS trials × 20 epochs each)"
    echo "  2. AUTOMATED: Train encoder with best parameters (100 epochs)"
    echo "  3. Tune classifier ($HPO_CLASSIFIER_TRIALS trials × 20 epochs each)"
    echo "  4. AUTOMATED: Train classifier with best parameters (20 epochs)"
    echo "  5. Display model card"
    echo "  6. Generate recommendations"
    echo ""
    if [ "$from_step" -gt 1 ]; then
        echo "Starting from Step $from_step (--from-step $from_step)."
        echo ""
    else
        echo "This is fully automated! Best parameters are applied automatically."
        echo "WARNING: This takes many hours!"
        echo ""
    fi

    # Step 1: Encoder HPO
    if [ "$from_step" -le 1 ]; then
        run_encoder_hpo -N "$HPO_ENCODER_TRIALS"
    fi

    # Step 2: Train encoder with best params - AUTOMATED
    BEST_ENCODER_PARAMS="checkpoints/best_encoder_params.json"
    HPO_BEST_CHECKPOINT="checkpoints/encoder_hpo_best.pt"
    if [ "$from_step" -le 2 ] && [ -f "$BEST_ENCODER_PARAMS" ]; then
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
    elif [ "$from_step" -le 2 ]; then
        print_error "Best encoder parameters not found: $BEST_ENCODER_PARAMS"
        echo "Running encoder HPO should have created this file (or run: $0 hpo-encoder -N 0 to save from existing study)."
        exit 1
    fi

    # Step 3: Classifier HPO
    if [ "$from_step" -le 3 ]; then
        run_classifier_hpo -N "$HPO_CLASSIFIER_TRIALS"
    fi

    # Step 4: Train classifier with best params - AUTOMATED
    BEST_CLASSIFIER_PARAMS="checkpoints/best_classifier_params.json"
    if [ "$from_step" -le 4 ] && [ -f "$BEST_CLASSIFIER_PARAMS" ]; then
        print_header "HPO Step 4: Training Classifier with Best Parameters (20 epochs)"
        print_success "Using best parameters from: $BEST_CLASSIFIER_PARAMS"
        python "$SCRIPT" --stage classifier --config "$CONFIG" \
            --final-training --best-params "$BEST_CLASSIFIER_PARAMS"
        print_success "Classifier training with best params complete!"
        echo ""
    elif [ "$from_step" -le 4 ]; then
        print_error "Best classifier parameters not found: $BEST_CLASSIFIER_PARAMS"
        echo "Running classifier HPO should have created this file."
        exit 1
    fi

    # Step 5: Model card
    if [ "$from_step" -le 5 ]; then
        display_model_card
    fi

    # Step 6: Recommendations
    if [ "$from_step" -le 6 ]; then
        run_recommend
    fi

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

run_full_hpo_pipeline() {
    # Full pipeline: build-cache → encoder HPO → encoder train → classifier HPO → classifier train → joint-finetune → recommend → promote
    print_header "FULL HPO PIPELINE (all --hpo)"
    echo "Steps: build-cache → hpo-encoder → encoder (best) → hpo-classifier → classifier (best) → joint-finetune → recommend → promote-to-prod"
    echo ""

    run_build_cache
    run_hpo_pipeline
    run_joint_finetune "${EXTRA_ARGS[@]}"
    run_recommend
    run_promote_to_prod

    print_header "FULL HPO PIPELINE COMPLETE!"
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
    ALL_HPO=0

    HPO_FROM_STEP=1  # 1=run all steps; 2=skip encoder HPO, start at "train encoder with best params"; etc.

    # Parse additional arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --from-step)
                HPO_FROM_STEP="${2:-1}"
                shift 2
                ;;
            --from-step=*)
                HPO_FROM_STEP="${1#*=}"
                shift
                ;;
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
            --hpo)
                ALL_HPO=1
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
            if [ "${ALL_HPO:-0}" = "1" ]; then
                run_full_hpo_pipeline
            else
                [ "${MIN_RATED_SONGS}" = "500" ] && export MIN_RATED_SONGS=60000
                run_encoder "${EXTRA_ARGS[@]}"
                run_classifier "${EXTRA_ARGS[@]}"
                run_joint_finetune "${EXTRA_ARGS[@]}"
                run_recommend
                display_model_card
                print_header "Pipeline Complete!"
            fi
            ;;
        encoder)
            run_encoder "${EXTRA_ARGS[@]}"
            ;;
        classifier)
            run_classifier "${EXTRA_ARGS[@]}"
            ;;
        joint-finetune)
            run_joint_finetune "${EXTRA_ARGS[@]}"
            ;;
        recommend)
            run_recommend "${EXTRA_ARGS[@]}"
            ;;
        rag-query)
            run_rag_query "${EXTRA_ARGS[@]}"
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
            run_build_cache "${EXTRA_ARGS[@]}"
            ;;
        fingerprint)
            run_fingerprint "${EXTRA_ARGS[@]}"
            ;;
        fingerprint-stats)
            run_fingerprint_stats
            ;;
        backfill-fingerprint-bits)
            run_backfill_fingerprint_bits "${EXTRA_ARGS[@]}"
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
            run_encoder_hpo "${EXTRA_ARGS[@]}"
            ;;
        hpo-classifier)
            run_classifier_hpo "${EXTRA_ARGS[@]}"
            ;;
        convert-cache-16bit)
            print_header "Convert Chunk Cache to 16-bit (in place, faster than clear + rebuild)"
            python "$SCRIPT" --stage convert-cache-to-16bit --config "$CONFIG"
            ;;
        chunk-cache-stats)
            print_header "Chunk cache: chunks-per-song distribution"
            python "$SCRIPT" --stage chunk-cache-stats --config "$CONFIG"
            ;;
        prune-chunk-cache)
            print_header "Prune chunk cache (remove redundant chunks for short songs)"
            python "$SCRIPT" --stage prune-chunk-cache --config "$CONFIG" "$@"
            ;;
        clear-cache)
            print_header "Clearing Waveform Chunk Cache (preserves fingerprint DB)"
            python "$SCRIPT" --stage clear-chunk-cache --config "$CONFIG"
            ;;
        model-card)
            display_model_card
            ;;
        ab-history)
            display_ab_history
            ;;
        cache-stats)
            print_header "Cache Statistics"
            python << 'PYEOF'
from ml_skeleton.music.chunk_cache import get_cache_stats
stats = get_cache_stats()
if stats['exists']:
    print(f'  Directory: {stats["cache_dir"]}')
    print(f'  Files: {stats["num_files"]}')
    print(f'  Songs: {stats["num_songs"]}')
    print(f'  Size: {stats["size_gb"]:.2f} GB')
else:
    print('  No cache found')
PYEOF
            ;;
        *)
            echo "Usage: $0 {all|encoder|classifier|joint-finetune|recommend|rag-query|quick|hpo|...} [options]"
            echo ""
            echo "Stages:"
            echo "  all                 - Run complete pipeline (encoder + classifier + [joint-finetune if enabled] + recommend)"
            echo "  all --hpo            - Full HPO pipeline: build-cache → hpo-encoder → encoder (best) → hpo-classifier → classifier (best) → joint-finetune → recommend → promote"
            echo "  encoder             - Train MoCo v2 + Genre BCE encoder"
            echo "  classifier          - Train rating classifier"
            echo "  joint-finetune      - Unfreeze encoder+classifier, train on audio→rating (run after classifier; requires joint_finetune.enabled)"
            echo "  recommend           - Generate recommendations"
            echo "  rag-query PLAYLIST.xspf  → rag_<name>.xspf (prod embeddings)"
            echo "  quick               - Quick test (5 epochs, 500 songs)"
            echo "  hpo                 - Full hyperparameter optimization (encoder + classifier). Use --from-step N to start from step N (e.g. 2 = skip encoder HPO, train encoder with best params)."
            echo "  hpo-encoder         - Hyperparameter optimization for encoder only (-N trials, -N 0 = save best params only; --reset-study; if malloc crash use EXPLR_HPO_DATALOADER_WORKERS=0)"
            echo "  hpo-classifier      - Hyperparameter optimization for classifier only (supports -N trials, --reps, --reset-study)"
            echo "  build-cache         - Build 4-chunk waveform cache (~30GB)"
            echo "  convert-cache-16bit - Convert existing float32 chunk cache to int16 in place (faster than clear+rebuild)"
            echo "  chunk-cache-stats   - Print distribution of chunks per song (frequency counts)"
            echo "  prune-chunk-cache   - Delete chunks with index >= N per song (N from duration); use --dry-run to preview"
            echo "  fingerprint         - Extract acoustic fingerprints from original files (for AcoustID)"
            echo "  enrich              - Complete pipeline: fingerprint + enrich + stats (recommended)"
            echo "  enrich-metadata     - Enrich metadata via AcoustID/MusicBrainz (requires API key)"
            echo "  fingerprint-stats   - Display fingerprint database statistics"
            echo "  backfill-fingerprint-bits - Precompute bits column (run once so HPO can use chromaprints)"
            echo "  musicbrainz-stats   - Display MusicBrainz database statistics"
            echo "  clear-cache         - Delete waveform chunk cache only (keeps fingerprint DB)"
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
            echo "  recommend --prod --select-distant        - Re-rank high picks for diversity (cosine max-min)"
            echo "  recommend --prod --select-distant-pool-factor N  - Candidate pool multiplier for --select-distant (default 5)"
            echo "  recommend --prod --genre rock           - Recommendations for rock songs only"
            echo "  recommend --error-playlist-size 0        - Skip false_positives/false_negatives playlists"
            echo "  rag-query /Music/.../mix.xspf          - Similar songs (prod/embeddings.db; no --prod flag)"
            echo ""
            echo "Options:"
            echo "  --encoder-type TYPE        - Encoder: use TYPE (e.g. moco). Required for encoder/hpo-encoder if config has encoder_type: fingerprint_baseline"
            echo "  --reset-study             - HPO only: delete existing Optuna study and start fresh (tune-classifier / tune-encoder)"
            echo "  --train-frac FRAC         - Encoder HPO only: use fraction of training data (e.g. 0.5); validation set stays full"
            echo "  --resume-checkpoint PATH   - Resume training from checkpoint"
            echo "  --best-params PATH         - Path to best params JSON (from HPO)"
            echo "  --mlflow-run-id ID         - Classifier: load hyperparameters from MLflow run (e.g. HPO parent run ID)"
            echo "  --encoder-version VERSION  - Encoder version for embeddings (e.g., v2)"
            echo "  --classifier-version VER   - Classifier version (e.g., v2)"
            echo "  --exhaust                  - Process max songs for the day (500 for free tier)"
            echo "  --all                      - Fingerprint: process all missing songs (no max_songs limit)"
            echo "  --workers N                - Number of parallel workers for fingerprinting (default: 4)"
            echo "  --low-rating-ratio N       - Include N% predicted dislikes in recommendations (0.0-1.0)"
            echo "  --genre CATEGORY           - Filter recommendations by genre category"
            echo "  --error-playlist-size N    - Max songs per false_positives/false_negatives playlist (0 = disable)"
            echo "                               Categories: rock, pop, electronic, hiphop, jazz_classical, country, latin_world"
            echo "  --random-init              - Use random init instead of loading from prod model (default: prod init)"
            echo "  --vault-size N             - Number of ratings to reserve for A/B testing (default: 1000)"
            echo "  --rag-top-n N              - rag-query: number of similar tracks (default: 50)"
            echo "  --rag-num-pc N             - rag-query: PCA dimensions for tie-break (default: 5)"
            echo "  --rag-unrated-only         - rag-query: candidate pool = unrated songs only"
            echo "  --rag-likes-only           - rag-query: rated = training-positive (config binary_positive_threshold);"
            echo "                               unrated = pred>0.5, top max(100, rag-top-n×20) by pred, then cosine rank"
            echo "  --hpo                      - With 'all': run full HPO pipeline (build-cache, encoder/clf HPO + train with best, joint-finetune, recommend, promote)"
            echo ""
            echo "Environment Variables:"
            echo "  HPO_ENCODER_TRIALS=30      - Number of encoder HPO trials"
            echo "  HPO_CLASSIFIER_TRIALS=800  - Number of classifier HPO trials (default 800)"
            echo "  EXPLR_HPO_DATALOADER_WORKERS=0 - Encoder HPO: use 0 dataloader workers"
            echo "  EXPLR_HPO_DISABLE_CHROMAPRINT=1 - Encoder HPO: disable chromaprint loss (if malloc crash in first batch)"
            echo "  --reps N                  - Repetitions per trial (different init seeds); best value and seed reported"
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
            echo "  EXPLR_HPO_DISABLE_CHROMAPRINT=1 EXPLR_HPO_DATALOADER_WORKERS=0 $0 hpo-encoder -N 30 --encoder-type moco --reset-study  # If malloc in first batch"
            echo "  $0 rag-query /Music/playlists/mix.xspf --rag-top-n 50 --rag-num-pc 5"
            echo "  $0 rag-query /Music/playlists/mix.xspf --rag-unrated-only"
            echo "  $0 rag-query /Music/playlists/mix.xspf --rag-likes-only"
            echo "  $0 fingerprint --workers 8              # Fingerprint with 8 parallel workers"
            echo "  $0 fingerprint --workers 2 --all       # Fingerprint all missing songs (2 workers)"
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
