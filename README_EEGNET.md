# EEGNet for 8-Channel Motor Imagery BCI

Deep learning solution for improving motor imagery classification on ADS1299 hardware (8 channels).

## What's New

This project extends the baseline CSP+LDA implementation with:

- **EEGNet** - State-of-the-art deep learning model for EEG classification
- **8-channel optimization** - Specifically designed for ADS1299 hardware
- **Automated evaluation** - Batch processing across all 10 subjects
- **Comparison analysis** - Direct comparison with CSP+LDA baseline

## Project Structure

```
EEG_MI_CSP_LDA_Project/
├── src/
│   ├── models/
│   │   └── eegnet.py                      # EEGNet architecture
│   ├── train_eegnet.py                    # Single subject training
│   ├── evaluate_all_subjects_eegnet.py    # Batch evaluation
│   ├── compare_methods.py                 # CSP+LDA vs EEGNet comparison
│   └── [existing CSP+LDA scripts...]
├── data/
│   └── raw/                               # PhysioNet EEGBCI dataset
├── outputs_eegnet/                        # EEGNet results
├── outputs_comparison/                    # Comparison plots and reports
├── EEGNET_USAGE_GUIDE.md                  # Detailed usage instructions
├── requirements_eegnet.txt                # Python dependencies
└── run_eegnet_full_evaluation.sh          # One-click evaluation script
```

## Quick Start (RTX 5080)

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements_eegnet.txt

# For CUDA support (RTX 5080)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 2. Run Full Evaluation

```bash
# One command to train all 10 subjects and generate reports
bash run_eegnet_full_evaluation.sh

# Expected time: ~20-30 minutes on RTX 5080
```

### 3. View Results

```bash
# Summary report
cat outputs_eegnet/EEGNET_SUMMARY_REPORT.md

# Comparison with CSP+LDA
cat outputs_comparison/COMPARISON_REPORT.md

# Visualizations
open outputs_comparison/*.png
```

## Files Created

### Core Implementation

| File | Description | Lines |
|------|-------------|-------|
| `src/models/eegnet.py` | EEGNet model architecture | ~250 |
| `src/train_eegnet.py` | Single subject training script | ~300 |
| `src/evaluate_all_subjects_eegnet.py` | Batch evaluation | ~250 |
| `src/compare_methods.py` | Comparison analysis | ~350 |

### Documentation

- `EEGNET_USAGE_GUIDE.md` - Complete usage instructions
- `requirements_eegnet.txt` - Python dependencies
- `run_eegnet_full_evaluation.sh` - Automated pipeline
- `README_EEGNET.md` - This file

## Key Features

### 1. EEGNet Model

- **Lightweight**: ~2,000 parameters
- **Fast training**: 2-3 minutes per subject
- **Optimized for**: Small datasets, 8-channel EEG
- **Architecture**: Depthwise separable convolutions + temporal filters

### 2. Automated Pipeline

```bash
# Train single subject
python3 src/train_eegnet.py --subject 2

# Train all subjects
python3 src/evaluate_all_subjects_eegnet.py

# Compare methods
python3 src/compare_methods.py
```

### 3. Comprehensive Analysis

Generates:
- Per-subject accuracy reports
- Summary statistics
- Comparison visualizations
- Performance improvement metrics

## Expected Performance

Based on literature and your baseline:

| Method | Mean Accuracy | Subjects ≥70% |
|--------|---------------|---------------|
| CSP+LDA (baseline) | 62.0% ± 18.1% | 4/10 |
| **EEGNet (expected)** | **68-75%** | **6-8/10** |

## Usage Examples

### Train Best Subject

```bash
python3 src/train_eegnet.py \
    --subject 2 \
    --config 8-channel-motor \
    --epochs 100 \
    --batch_size 16
```

### Hyperparameter Tuning

```bash
# Lower learning rate
python3 src/train_eegnet.py --subject 5 --lr 0.0001

# Larger model
python3 src/train_eegnet.py --subject 5 --large_model

# More epochs
python3 src/train_eegnet.py --subject 5 --epochs 200
```

### Custom Subject List

```bash
python3 src/evaluate_all_subjects_eegnet.py \
    --subjects 1 2 7 10  # Only train best subjects
```

## Hardware Requirements

### Minimum (CPU only)

- Python 3.10-3.12
- 8GB RAM
- ~30GB disk space
- Training time: ~10-15 min/subject

### Recommended (GPU)

- **Your RTX 5080**: Perfect! ✅
- CUDA 11.8+
- 16GB VRAM
- Training time: ~2-3 min/subject

## Output Files

### Per-Subject Results

`outputs_eegnet/subject_XX/eegnet_results.json`:

```json
{
  "subject_id": 2,
  "accuracy_mean": 0.85,
  "accuracy_std": 0.05,
  "fold_accuracies": [0.82, 0.88, 0.84, 0.86, 0.85]
}
```

### Summary Reports

- `EEGNET_SUMMARY_REPORT.md` - Overall statistics
- `eegnet_summary.json` - Machine-readable summary

### Comparison Plots

- `comparison_barplot.png` - Main results
- `comparison_scatter.png` - Subject-wise comparison
- `comparison_boxplot.png` - Distribution comparison
- `comparison_improvement.png` - Improvement per subject

## Troubleshooting

### CUDA Out of Memory

```bash
python3 src/train_eegnet.py --subject 1 --batch_size 8
```

### Slow Training

```bash
# Check GPU usage
nvidia-smi

# Verify CUDA
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Poor Results (<55%)

Try:
1. Reduce learning rate: `--lr 0.0001`
2. More training: `--epochs 200`
3. Different architecture: `--large_model`

## Next Steps

If EEGNet shows good improvement (>5% over CSP+LDA):

1. **Data Augmentation** - Expand training data
2. **Transfer Learning** - Pre-train on all subjects
3. **Ensemble Methods** - Combine CSP+LDA + EEGNet
4. **More Data** - Download full 109 subjects
5. **Try Transformer** - If you have time and data

## Citation

```
Lawhern, V. J., et al. (2018).
EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces.
Journal of Neural Engineering, 15(5), 056013.
```

## Support

For detailed instructions, see: `EEGNET_USAGE_GUIDE.md`

---

**Ready to train?** Run `bash run_eegnet_full_evaluation.sh` on your RTX 5080! 🚀