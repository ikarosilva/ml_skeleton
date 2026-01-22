#!/bin/bash
# Music Recommendation Pipeline Runner
#
# Usage:
#   ./run_music_pipeline.sh all              # Run all 3 stages
#   ./run_music_pipeline.sh encoder          # Run Stage 1 only
#   ./run_music_pipeline.sh classifier       # Run Stage 2 only
#   ./run_music_pipeline.sh recommend        # Run Stage 3 only
#   ./run_music_pipeline.sh quick            # Quick test (5 epochs, 500 rated songs)
#
# Environment variables:
#   CONFIG=/path/to/config.yaml              # Override config file
#   CLEMENTINE_DB_PATH=/path/to/db           # Override database path

set -e  # Exit on error

CONFIG="${CONFIG:-configs/music_recommendation.yaml}"
SCRIPT="examples/music_recommendation.py"

# Allow database path override via environment variable
# Example: CLEMENTINE_DB_PATH=/Music/database/clementine_backup_2026-01.db ./run_music_pipeline.sh all
if [ -n "$CLEMENTINE_DB_PATH" ]; then
    export CLEMENTINE_DB_PATH
fi

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found"
        exit 1
    fi
    print_success "Python found: $(python --version)"

    # Check if in correct directory
    if [ ! -f "$SCRIPT" ]; then
        print_error "Script not found: $SCRIPT"
        print_error "Please run from ml_skeleton root directory"
        exit 1
    fi
    print_success "Script found: $SCRIPT"

    # Check config
    if [ ! -f "$CONFIG" ]; then
        print_error "Config not found: $CONFIG"
        exit 1
    fi
    print_success "Config found: $CONFIG"

    # Check database path in config
    DB_PATH=$(grep "database_path:" "$CONFIG" | head -1 | cut -d'"' -f2)
    if [ ! -f "$DB_PATH" ]; then
        print_warning "Database not found at: $DB_PATH"
        print_warning "Update database_path in $CONFIG"
    else
        print_success "Database found: $DB_PATH"
    fi

    echo ""
}

# Run encoder stage
run_encoder() {
    print_header "Stage 1: Training Audio Encoder"
    echo "This will:"
    echo "  - Load rated songs from Clementine DB"
    echo "  - Train 1D CNN encoder on 30-second audio clips"
    echo "  - Extract and store embeddings"
    echo "  - Save model to checkpoints/encoder_best.pt"
    echo ""

    python "$SCRIPT" --stage encoder --config "$CONFIG"

    print_success "Encoder training complete!"
    echo ""
}

# Run classifier stage
run_classifier() {
    print_header "Stage 2: Training Rating Classifier"
    echo "This will:"
    echo "  - Load pre-extracted embeddings"
    echo "  - Train MLP classifier to predict ratings"
    echo "  - Save model to checkpoints/classifier_best.pt"
    echo ""

    python "$SCRIPT" --stage classifier --config "$CONFIG"

    print_success "Classifier training complete!"
    echo ""
}

# Run recommendation generation
run_recommend() {
    print_header "Stage 3: Generating Recommendations"
    echo "This will:"
    echo "  - Load unrated songs"
    echo "  - Predict ratings using trained models"
    echo "  - Generate top-N recommendations"
    echo "  - Save to recommendations.txt"
    echo "  - Generate HITL playlists (recommender_help.xspf, recommender_best.xspf)"
    echo ""

    python "$SCRIPT" --stage recommend --config "$CONFIG"

    print_success "Recommendations generated!"
    echo ""
}

# Quick test mode (reduce epochs)
run_quick_test() {
    print_header "Quick Test Mode"
    print_warning "Creating temporary config with reduced epochs..."

    # Create temp config
    TEMP_CONFIG="/tmp/music_recommendation_quick.yaml"
    cp "$CONFIG" "$TEMP_CONFIG"

    # Reduce epochs using sed
    sed -i 's/epochs: 50/epochs: 5/g' "$TEMP_CONFIG"
    sed -i 's/epochs: 20/epochs: 5/g' "$TEMP_CONFIG"

    print_success "Temporary config created: $TEMP_CONFIG"

    # Set minimum rated songs for testing (avoids sparsity errors)
    export MIN_RATED_SONGS=500
    print_success "Set MIN_RATED_SONGS=500 (placeholder will generate 500 rated songs)"
    echo ""

    # Run all stages with temp config
    CONFIG="$TEMP_CONFIG" run_encoder
    CONFIG="$TEMP_CONFIG" run_classifier
    CONFIG="$TEMP_CONFIG" run_recommend

    print_success "Quick test complete!"
    print_warning "Remember to run full training for production"
}

# Main script
main() {
    STAGE="${1:-all}"

    check_prerequisites

    case "$STAGE" in
        all)
            run_encoder
            run_classifier
            run_recommend
            print_header "Pipeline Complete!"
            print_success "All 3 stages completed successfully"
            echo ""
            echo "Outputs:"
            echo "  - Encoder model: checkpoints/encoder_best.pt"
            echo "  - Classifier model: checkpoints/classifier_best.pt"
            echo "  - Embeddings: embeddings.db"
            echo "  - Recommendations: recommendations.txt"
            echo "  - HITL Playlists:"
            echo "    * recommender_help.xspf (uncertain songs for learning)"
            echo "    * recommender_best.xspf (top predictions for validation)"
            echo ""
            echo "Next: Open XSPF playlists in Clementine, rate songs, and re-run!"
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
        *)
            echo "Usage: $0 {all|encoder|classifier|recommend|quick}"
            echo ""
            echo "Stages:"
            echo "  all         - Run all 3 stages (encoder + classifier + recommend)"
            echo "  encoder     - Run Stage 1: Train audio encoder"
            echo "  classifier  - Run Stage 2: Train rating classifier"
            echo "  recommend   - Run Stage 3: Generate recommendations"
            echo "  quick       - Quick test with 5 epochs and 500 rated songs (for testing)"
            echo ""
            echo "Examples:"
            echo "  $0 all                    # Full pipeline"
            echo "  $0 encoder                # Just train encoder"
            echo "  $0 quick                  # Quick test"
            echo "  CONFIG=custom.yaml $0 all # Use custom config"
            exit 1
            ;;
    esac
}

main "$@"
