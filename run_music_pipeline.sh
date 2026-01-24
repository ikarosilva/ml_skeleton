#!/bin/bash
# Music Recommendation Pipeline Runner
#
# Usage:
#   ./run_music_pipeline.sh all              # Run all 3 stages
#   ./run_music_pipeline.sh encoder          # Run Stage 1 only
#   ./run_music_pipeline.sh classifier       # Run Stage 2 only
#   ./run_music_pipeline.sh recommend        # Run Stage 3 only
#   ./run_music_pipeline.sh quick            # Quick test (5 epochs, 500 rated songs)
#   ./run_music_pipeline.sh hpo              # Full hyperparameter optimization pipeline
#   ./run_music_pipeline.sh clear-cache      # Delete waveform cache (prompts for confirmation)
#
# A/B Testing (SimSiam vs Simple encoder):
#   ./run_music_pipeline.sh encoder --encoder-type simsiam   # Train with SimSiam encoder
#   ./run_music_pipeline.sh all --encoder-type simple        # Train with Simple encoder
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
#   ENCODER_TYPE=simsiam                     # Override encoder type (simple or simsiam)
#   RESUME_CHECKPOINT=/path/to/checkpoint    # Resume from previous training
#   ENCODER_VERSION=v2                       # Encoder version for embeddings
#   CLASSIFIER_VERSION=v2                    # Classifier version

set -e  # Exit on error

CONFIG="${CONFIG:-configs/music_recommendation.yaml}"
SCRIPT="examples/music_recommendation.py"
HPO_ENCODER_TRIALS="${HPO_ENCODER_TRIALS:-30}"
HPO_CLASSIFIER_TRIALS="${HPO_CLASSIFIER_TRIALS:-20}"
ENCODER_TYPE="${ENCODER_TYPE:-}"  # Optional: "simple" or "simsiam"
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
    echo ""
}

run_encoder() {
    local extra_args="$1"
    local encoder_type_arg=""
    local resume_arg=""
    local version_arg=""

    # Add encoder type if specified
    if [ -n "$ENCODER_TYPE" ]; then
        encoder_type_arg="--encoder-type $ENCODER_TYPE"
    fi

    # Add resume checkpoint if specified
    if [ -n "$RESUME_CHECKPOINT" ]; then
        resume_arg="--resume-checkpoint $RESUME_CHECKPOINT"
    fi

    # Add encoder version if specified
    if [ -n "$ENCODER_VERSION" ]; then
        version_arg="--encoder-version $ENCODER_VERSION"
    fi

    # Print header with relevant info
    local header="Stage 1: Training Audio Encoder"
    [ -n "$ENCODER_TYPE" ] && header="$header (type: $ENCODER_TYPE)"
    [ -n "$RESUME_CHECKPOINT" ] && header="$header [RESUMING from $RESUME_CHECKPOINT]"
    [ -n "$ENCODER_VERSION" ] && header="$header [encoder: $ENCODER_VERSION]"
    print_header "$header"

    python "$SCRIPT" --stage encoder --config "$CONFIG" $encoder_type_arg $resume_arg $version_arg $extra_args
    print_success "Encoder training complete!"
    echo ""
}

run_classifier() {
    local extra_args="$1"
    local classifier_version_arg=""

    # Add classifier version if specified
    if [ -n "$CLASSIFIER_VERSION" ]; then
        classifier_version_arg="--classifier-version $CLASSIFIER_VERSION"
    fi

    # Print header with relevant info
    local header="Stage 2: Training Rating Classifier"
    [ -n "$CLASSIFIER_VERSION" ] && header="$header [classifier: $CLASSIFIER_VERSION]"
    print_header "$header"

    python "$SCRIPT" --stage classifier --config "$CONFIG" $classifier_version_arg $extra_args
    print_success "Classifier training complete!"
    echo ""
}

run_recommend() {
    print_header "Stage 3: Generating Recommendations"
    python "$SCRIPT" --stage recommend --config "$CONFIG"
    print_success "Recommendations generated!"
    echo ""
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
    
    print_header "HPO Step 3: Classifier Hyperparameter Tuning"
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

run_hpo_pipeline() {
    print_header "FULL HYPERPARAMETER OPTIMIZATION PIPELINE"
    echo "Steps:"
    echo "  1. Tune encoder ($HPO_ENCODER_TRIALS trials × 20 epochs each)"
    echo "  2. AUTOMATED: Train encoder with best parameters (50 epochs)"
    echo "  3. Tune classifier ($HPO_CLASSIFIER_TRIALS trials × 20 epochs each)"
    echo "  4. AUTOMATED: Train classifier with best parameters (50 epochs)"
    echo "  5. Display model card"
    echo "  6. Generate recommendations"
    echo ""
    echo "This is fully automated! Best parameters are applied automatically."
    echo "WARNING: This takes many hours!"
    echo ""

    # Step 1: Encoder HPO
    run_encoder_hpo "$HPO_ENCODER_TRIALS"

    # Step 2: Train encoder with best params (50 epochs) - AUTOMATED
    BEST_ENCODER_PARAMS="checkpoints/best_encoder_params.json"
    if [ -f "$BEST_ENCODER_PARAMS" ]; then
        print_header "HPO Step 2: Training Encoder with Best Parameters (50 epochs)"
        print_success "Using best parameters from: $BEST_ENCODER_PARAMS"
        python "$SCRIPT" --stage encoder --config "$CONFIG" \
            --final-training --best-params "$BEST_ENCODER_PARAMS"
        print_success "Encoder training with best params complete!"
        echo ""
    else
        print_error "Best encoder parameters not found: $BEST_ENCODER_PARAMS"
        echo "Running encoder HPO should have created this file."
        exit 1
    fi

    # Step 3: Classifier HPO
    run_classifier_hpo "$HPO_CLASSIFIER_TRIALS"

    # Step 4: Train classifier with best params (50 epochs) - AUTOMATED
    BEST_CLASSIFIER_PARAMS="checkpoints/best_classifier_params.json"
    if [ -f "$BEST_CLASSIFIER_PARAMS" ]; then
        print_header "HPO Step 4: Training Classifier with Best Parameters (50 epochs)"
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
    TEMP_CONFIG="/tmp/music_recommendation_quick.yaml"
    cp "$CONFIG" "$TEMP_CONFIG"
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

    # Parse additional arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --encoder-type)
                ENCODER_TYPE="$2"
                shift 2
                ;;
            --encoder-type=*)
                ENCODER_TYPE="${1#*=}"
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
            *)
                # Unknown option, ignore
                shift
                ;;
        esac
    done

    check_prerequisites

    case "$STAGE" in
        all)
            [ "${MIN_RATED_SONGS}" = "500" ] && export MIN_RATED_SONGS=60000
            run_encoder
            run_classifier
            run_recommend
            display_model_card
            print_header "Pipeline Complete!"
            ;;
        encoder)
            run_encoder
            ;;
        classifier)
            run_classifier
            ;;
        recommend)
            run_recommend
            ;;
        quick)
            run_quick_test
            ;;
        hpo)
            run_hpo_pipeline
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
        *)
            echo "Usage: $0 {all|encoder|classifier|recommend|quick|hpo|clear-cache|model-card} [options]"
            echo ""
            echo "Stages:"
            echo "  all         - Run complete pipeline"
            echo "  encoder     - Train audio encoder"
            echo "  classifier  - Train rating classifier"
            echo "  recommend   - Generate recommendations"
            echo "  quick       - Quick test (5 epochs, 500 songs)"
            echo "  hpo         - Full hyperparameter optimization"
            echo "  clear-cache - Delete waveform cache (use when audio files change)"
            echo "  model-card  - Display model card"
            echo ""
            echo "Options:"
            echo "  --encoder-type TYPE        - Encoder type: 'simple' or 'simsiam' (A/B testing)"
            echo "  --resume-checkpoint PATH   - Resume training from checkpoint"
            echo "  --encoder-version VERSION  - Encoder version for embeddings (e.g., v2)"
            echo "  --classifier-version VER   - Classifier version (e.g., v2)"
            echo ""
            echo "Environment Variables:"
            echo "  HPO_ENCODER_TRIALS=30"
            echo "  HPO_CLASSIFIER_TRIALS=20"
            echo "  ENCODER_TYPE=simsiam        - Override encoder type"
            echo "  RESUME_CHECKPOINT=/path/to  - Resume from checkpoint"
            echo "  ENCODER_VERSION=v2          - Encoder version for embeddings"
            echo "  CLASSIFIER_VERSION=v2       - Classifier version"
            echo ""
            echo "Examples:"
            echo "  $0 all                          # Run with default (simple) encoder"
            echo "  $0 encoder --encoder-type simsiam  # Train with SimSiam encoder"
            echo "  ENCODER_TYPE=simsiam $0 all     # Train with SimSiam via env var"
            echo "  HPO_ENCODER_TRIALS=50 $0 hpo    # Run HPO with 50 encoder trials"
            echo ""
            echo "Version Compatibility Rules:"
            echo "  - Encoder and Classifier have SEPARATE versions"
            echo "  - Classifier stores which encoder version it was trained with"
            echo "  - If Encoder is updated, Classifier MUST be retrained"
            echo "  - Deployment (recommend) FAILS if versions don't match"
            echo ""
            echo "Resume/Incremental Training Examples:"
            echo "  # Train v1 (initial training)"
            echo "  $0 all"
            echo ""
            echo "  # Update classifier only (encoder unchanged)"
            echo "  $0 classifier --classifier-version v2"
            echo ""
            echo "  # Update encoder (requires new classifier)"
            echo "  $0 encoder --resume-checkpoint checkpoints/encoder_best.pt --encoder-version v2"
            echo "  $0 classifier --classifier-version v1  # New classifier for encoder v2"
            echo "  $0 recommend"
            exit 1
            ;;
    esac
}

main "$@"
