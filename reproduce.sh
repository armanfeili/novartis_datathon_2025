#!/bin/bash
# reproduce.sh - Full reproduction from raw data to submission
# Novartis Datathon 2025 - Generic Erosion Forecasting
#
# Usage:
#   ./reproduce.sh                    # Full pipeline
#   ./reproduce.sh --scenario 1       # Scenario 1 only
#   ./reproduce.sh --scenario 2       # Scenario 2 only
#   ./reproduce.sh --skip-tests       # Skip test verification
#   ./reproduce.sh --model lightgbm   # Use different model

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default settings
SCENARIO="both"
MODEL="catboost"
SEED=42
SKIP_TESTS=false
OUTPUT_DIR="artifacts"
SUBMISSION_DIR="submissions"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --scenario <1|2|both>   Scenario to run (default: both)"
            echo "  --model <model>         Model type: catboost, lightgbm, xgboost (default: catboost)"
            echo "  --seed <int>            Random seed (default: 42)"
            echo "  --skip-tests            Skip test verification"
            echo "  --output <dir>          Output directory (default: artifacts)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Novartis Datathon 2025 - Reproduction${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Settings:"
echo "  Scenario: $SCENARIO"
echo "  Model: $MODEL"
echo "  Seed: $SEED"
echo "  Output: $OUTPUT_DIR"
echo ""

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Change to project root
cd "$PROJECT_ROOT"

# Check for virtual environment
if [[ -d ".venv" ]]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source .venv/bin/activate
elif [[ -n "$VIRTUAL_ENV" ]]; then
    echo -e "${GREEN}Virtual environment already active: $VIRTUAL_ENV${NC}"
else
    echo -e "${YELLOW}Warning: No virtual environment found. Using system Python.${NC}"
fi

# Verify Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $PYTHON_VERSION"

# Check required data files
echo ""
echo -e "${GREEN}Checking data files...${NC}"
TRAIN_DIR="data/raw/TRAIN"
TEST_DIR="data/raw/TEST"

if [[ ! -d "$TRAIN_DIR" ]] || [[ ! -d "$TEST_DIR" ]]; then
    echo -e "${RED}Error: Data directories not found!${NC}"
    echo "Expected: $TRAIN_DIR and $TEST_DIR"
    echo "Please download competition data and place in data/raw/"
    exit 1
fi

REQUIRED_FILES=(
    "$TRAIN_DIR/df_volume_train.csv"
    "$TRAIN_DIR/df_generics_train.csv"
    "$TRAIN_DIR/df_medicine_info_train.csv"
    "$TEST_DIR/df_volume_test.csv"
    "$TEST_DIR/df_generics_test.csv"
    "$TEST_DIR/df_medicine_info_test.csv"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}Error: Missing file: $file${NC}"
        exit 1
    fi
done
echo "All data files present."

# Run tests if not skipped
if [[ "$SKIP_TESTS" = false ]]; then
    echo ""
    echo -e "${GREEN}Running tests...${NC}"
    pytest tests/ -v --tb=short
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}Tests failed! Fix issues before proceeding.${NC}"
        exit 1
    fi
    echo -e "${GREEN}All tests passed!${NC}"
fi

# Create timestamp for run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="${MODEL}_${TIMESTAMP}"

echo ""
echo -e "${GREEN}Starting training...${NC}"
echo "Run name: $RUN_NAME"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SUBMISSION_DIR"

# Train Scenario 1
if [[ "$SCENARIO" = "both" ]] || [[ "$SCENARIO" = "1" ]]; then
    echo ""
    echo -e "${GREEN}Training Scenario 1...${NC}"
    python -m src.train \
        --scenario 1 \
        --model "$MODEL" \
        --seed "$SEED" \
        --output "$OUTPUT_DIR/${RUN_NAME}_s1" \
        --run-name "${RUN_NAME}_s1"
    
    MODEL_S1="$OUTPUT_DIR/${RUN_NAME}_s1/model.bin"
    echo -e "${GREEN}Scenario 1 model saved to: $MODEL_S1${NC}"
fi

# Train Scenario 2
if [[ "$SCENARIO" = "both" ]] || [[ "$SCENARIO" = "2" ]]; then
    echo ""
    echo -e "${GREEN}Training Scenario 2...${NC}"
    python -m src.train \
        --scenario 2 \
        --model "$MODEL" \
        --seed "$SEED" \
        --output "$OUTPUT_DIR/${RUN_NAME}_s2" \
        --run-name "${RUN_NAME}_s2"
    
    MODEL_S2="$OUTPUT_DIR/${RUN_NAME}_s2/model.bin"
    echo -e "${GREEN}Scenario 2 model saved to: $MODEL_S2${NC}"
fi

# Generate submission if both models trained
if [[ "$SCENARIO" = "both" ]]; then
    echo ""
    echo -e "${GREEN}Generating submission...${NC}"
    
    SUBMISSION_FILE="$SUBMISSION_DIR/submission_${RUN_NAME}.csv"
    
    python -m src.inference \
        --model-s1 "$MODEL_S1" \
        --model-s2 "$MODEL_S2" \
        --output "$SUBMISSION_FILE" \
        --save-auxiliary
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Reproduction complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Outputs:"
    echo "  Scenario 1 model: $MODEL_S1"
    echo "  Scenario 2 model: $MODEL_S2"
    echo "  Submission: $SUBMISSION_FILE"
    echo "  Auxiliary: ${SUBMISSION_FILE%.csv}_auxiliary.csv"
    
    # Show submission statistics
    echo ""
    echo "Submission statistics:"
    wc -l "$SUBMISSION_FILE"
    head -5 "$SUBMISSION_FILE"
else
    echo ""
    echo -e "${GREEN}Training complete for Scenario $SCENARIO!${NC}"
    echo "To generate submission, train both scenarios."
fi

echo ""
echo "Logs available in: $OUTPUT_DIR/${RUN_NAME}_*/train.log"
