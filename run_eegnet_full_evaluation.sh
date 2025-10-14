#!/bin/bash
# Full EEGNet Evaluation Pipeline
# This script runs the complete evaluation on all 10 subjects

set -e  # Exit on error

echo "=========================================="
echo "EEGNet Full Evaluation Pipeline"
echo "=========================================="
echo ""

# Configuration
SUBJECTS="1 2 3 4 5 6 7 8 9 10"
CONFIG="8-channel-motor"
OUTPUT_DIR="outputs_eegnet"
EPOCHS=100
BATCH_SIZE=16
LEARNING_RATE=0.001
CV_FOLDS=5

echo "Configuration:"
echo "  Subjects: $SUBJECTS"
echo "  Channel config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "=========================================="
echo ""

# Check CUDA availability
echo "Checking GPU..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

# Run batch evaluation
echo "Starting batch evaluation..."
echo ""

python3 src/evaluate_all_subjects_eegnet.py \
    --subjects $SUBJECTS \
    --config $CONFIG \
    --output $OUTPUT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --cv $CV_FOLDS

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""

# Generate comparison report
echo "Generating comparison with CSP+LDA..."
echo ""

python3 src/compare_methods.py \
    --eegnet_dir $OUTPUT_DIR \
    --output outputs_comparison

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - $OUTPUT_DIR/EEGNET_SUMMARY_REPORT.md"
echo "  - outputs_comparison/COMPARISON_REPORT.md"
echo ""
echo "View plots in: outputs_comparison/*.png"
echo "=========================================="