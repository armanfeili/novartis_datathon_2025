#!/bin/bash
# Script to train CatBoost with all bonus features and generate submissions

set -e  # Exit on error

cd "$(dirname "$0")"
source .venv/bin/activate

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="catboost_bonus_all_${TIMESTAMP}"
CONFIG_FILE="configs/run_bonus_all.yaml"

echo "=========================================="
echo "Training CatBoost with ALL Bonus Features"
echo "=========================================="
echo "Run Name: $RUN_NAME"
echo "Config: $CONFIG_FILE"
echo "Timestamp: $TIMESTAMP"
echo ""

# Create logs directory
mkdir -p logs

# Train Scenario 1
echo "Training Scenario 1..."
python -m src.train \
    --scenario 1 \
    --model catboost \
    --run-config "$CONFIG_FILE" \
    --run-name "${RUN_NAME}_s1" \
    2>&1 | tee "logs/training_s1_${TIMESTAMP}.log"

S1_ARTIFACTS="artifacts/${RUN_NAME}_s1"
echo "Scenario 1 artifacts: $S1_ARTIFACTS"
echo ""

# Train Scenario 2
echo "Training Scenario 2..."
python -m src.train \
    --scenario 2 \
    --model catboost \
    --run-config "$CONFIG_FILE" \
    --run-name "${RUN_NAME}_s2" \
    2>&1 | tee "logs/training_s2_${TIMESTAMP}.log"

S2_ARTIFACTS="artifacts/${RUN_NAME}_s2"
echo "Scenario 2 artifacts: $S2_ARTIFACTS"
echo ""

# Extract model paths
S1_MODEL=$(find "$S1_ARTIFACTS" -name "model*.bin" -o -name "bucket*_cat_model" -type d | head -1)
S2_MODEL=$(find "$S2_ARTIFACTS" -name "model*.bin" -o -name "bucket*_cat_model" -type d | head -1)

if [ -z "$S1_MODEL" ]; then
    echo "ERROR: Could not find Scenario 1 model"
    exit 1
fi

if [ -z "$S2_MODEL" ]; then
    echo "ERROR: Could not find Scenario 2 model"
    exit 1
fi

echo "Found models:"
echo "  S1: $S1_MODEL"
echo "  S2: $S2_MODEL"
echo ""

# Generate submission
echo "Generating submission..."
python -m src.inference \
    --model-s1 "$S1_MODEL" \
    --model-s2 "$S2_MODEL" \
    --run-config "$CONFIG_FILE" \
    --artifacts-dir-s1 "$S1_ARTIFACTS" \
    --artifacts-dir-s2 "$S2_ARTIFACTS" \
    --output submissions \
    --run-name "$RUN_NAME" \
    2>&1 | tee "logs/inference_${TIMESTAMP}.log"

echo ""
echo "=========================================="
echo "Training and Submission Complete!"
echo "=========================================="
echo "Results saved to:"
echo "  - Artifacts: artifacts/${RUN_NAME}_*"
echo "  - Submissions: submissions/"
echo "  - Logs: logs/*_${TIMESTAMP}.log"
echo ""

