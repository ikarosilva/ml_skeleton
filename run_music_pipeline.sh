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
#
# A/B Testing (SimSiam vs Simple encoder):
#   ./run_music_pipeline.sh encoder --encoder-type simsiam   # Train with SimSiam encoder
#   ./run_music_pipeline.sh all --encoder-type simple        # Train with Simple encoder
#
# Environment variables:
#   CONFIG=/path/to/config.yaml              # Override config file
#   CLEMENTINE_DB_PATH=/path/to/db           # Override database path
#   HPO_ENCODER_TRIALS=30                    # Number of encoder HPO trials
#   HPO_CLASSIFIER_TRIALS=20                 # Number of classifier HPO trials
#   ENCODER_TYPE=simsiam                     # Override encoder type (simple or simsiam)

set -e  # Exit on error

CONFIG="${CONFIG:-configs/music_recommendation.yaml}"
SCRIPT="examples/music_recommendation.py"
HPO_ENCODER_TRIALS="${HPO_ENCODER_TRIALS:-30}"
HPO_CLASSIFIER_TRIALS="${HPO_CLASSIFIER_TRIALS:-20}"
ENCODER_TYPE="${ENCODER_TYPE:-}"  # Optional: "simple" or "simsiam"

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

    # Add encoder type if specified
    if [ -n "$ENCODER_TYPE" ]; then
        encoder_type_arg="--encoder-type $ENCODER_TYPE"
        print_header "Stage 1: Training Audio Encoder (type: $ENCODER_TYPE)"
    else
        print_header "Stage 1: Training Audio Encoder"
    fi

    python "$SCRIPT" --stage encoder --config "$CONFIG" $encoder_type_arg $extra_args
    print_success "Encoder training complete!"
    echo ""
}

run_classifier() {
    local extra_args="$1"
    print_header "Stage 2: Training Rating Classifier"
    python "$SCRIPT" --stage classifier --config "$CONFIG" $extra_args
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
        model-card)
            display_model_card
            ;;
        *)
            echo "Usage: $0 {all|encoder|classifier|recommend|quick|hpo|model-card} [options]"
            echo ""
            echo "Stages:"
            echo "  all         - Run complete pipeline"
            echo "  encoder     - Train audio encoder"
            echo "  classifier  - Train rating classifier"
            echo "  recommend   - Generate recommendations"
            echo "  quick       - Quick test (5 epochs, 500 songs)"
            echo "  hpo         - Full hyperparameter optimization"
            echo "  model-card  - Display model card"
            echo ""
            echo "Options:"
            echo "  --encoder-type TYPE   - Encoder type: 'simple' or 'simsiam' (A/B testing)"
            echo ""
            echo "Environment Variables:"
            echo "  HPO_ENCODER_TRIALS=30"
            echo "  HPO_CLASSIFIER_TRIALS=20"
            echo "  ENCODER_TYPE=simsiam      - Override encoder type"
            echo ""
            echo "Examples:"
            echo "  $0 all                          # Run with default (simple) encoder"
            echo "  $0 encoder --encoder-type simsiam  # Train with SimSiam encoder"
            echo "  ENCODER_TYPE=simsiam $0 all     # Train with SimSiam via env var"
            echo "  HPO_ENCODER_TRIALS=50 $0 hpo    # Run HPO with 50 encoder trials"
            exit 1
            ;;
    esac
}

main "$@"
