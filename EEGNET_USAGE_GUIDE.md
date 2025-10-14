# EEGNet Training Guide for 8-Channel Motor Imagery BCI

This guide explains how to train and evaluate EEGNet on your PhysioNet dataset using your RTX 5080 GPU.

---

## Quick Start

### Prerequisites

On your RTX 5080 PC, ensure you have:

```bash
# Python 3.10, 3.11, or 3.12 (NOT 3.14!)
python3 --version

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn matplotlib mne
```

### Clone and Setup

```bash
# Clone from GitHub (or copy files)
cd /path/to/your/workspace
git clone <your-repo-url>
cd EEG_MI_CSP_LDA_Project

# Verify CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Training Workflow

### Option 1: Train Single Subject (Quick Test)

```bash
# Train EEGNet on Subject 2 (your best subject)
python3 src/train_eegnet.py \
    --subject 2 \
    --config 8-channel-motor \
    --output outputs_eegnet \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --cv 5

# Expected time: ~2-3 minutes on RTX 5080
```

**Output:**
- `outputs_eegnet/subject_02/eegnet_results.json` - Training results
- Console output showing fold-by-fold accuracy

### Option 2: Train All 10 Subjects (Full Evaluation)

```bash
# Batch training on all subjects
python3 src/evaluate_all_subjects_eegnet.py \
    --subjects 1 2 3 4 5 6 7 8 9 10 \
    --config 8-channel-motor \
    --output outputs_eegnet \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --cv 5

# Expected time: ~20-30 minutes total on RTX 5080
```

**Output:**
- `outputs_eegnet/subject_XX/eegnet_results.json` - Per-subject results
- `outputs_eegnet/EEGNET_SUMMARY_REPORT.md` - Summary report
- `outputs_eegnet/eegnet_summary.json` - JSON summary

### Option 3: Compare with CSP+LDA

After training EEGNet on all subjects:

```bash
# Generate comparison analysis
python3 src/compare_methods.py \
    --eegnet_dir outputs_eegnet \
    --output outputs_comparison

# Expected time: <1 minute
```

**Output:**
- `outputs_comparison/COMPARISON_REPORT.md` - Detailed comparison
- `outputs_comparison/comparison_barplot.png` - Bar chart
- `outputs_comparison/comparison_scatter.png` - Scatter plot
- `outputs_comparison/comparison_boxplot.png` - Box plot
- `outputs_comparison/comparison_improvement.png` - Improvement chart

---

## Understanding the Results

### EEGNet Results Format

Each subject's results file (`eegnet_results.json`) contains:

```json
{
  "subject_id": 2,
  "config_name": "8-channel-motor",
  "n_channels": 8,
  "matched_channels": ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4"],
  "n_timepoints": 320,
  "accuracy_mean": 0.85,
  "accuracy_std": 0.05,
  "fold_accuracies": [0.82, 0.88, 0.84, 0.86, 0.85],
  "hyperparameters": {...}
}
```

### Performance Interpretation

| Accuracy | Interpretation |
|----------|----------------|
| ≥80% | Excellent - Clinical grade |
| 70-79% | Good - Usable for BCI |
| 60-69% | Fair - Needs improvement |
| <60% | Poor - Not usable |

---

## Hyperparameter Tuning

If initial results are not satisfactory, try these adjustments:

### 1. Learning Rate

```bash
# Lower learning rate (more stable)
python3 src/train_eegnet.py --subject 5 --lr 0.0001

# Higher learning rate (faster convergence)
python3 src/train_eegnet.py --subject 5 --lr 0.01
```

### 2. Batch Size

```bash
# Smaller batch (better for small datasets)
python3 src/train_eegnet.py --subject 5 --batch_size 8

# Larger batch (faster training)
python3 src/train_eegnet.py --subject 5 --batch_size 32
```

### 3. Larger Model

```bash
# Use EEGNetLarge (more parameters)
python3 src/train_eegnet.py --subject 5 --large_model
```

### 4. More Epochs

```bash
# Extended training
python3 src/train_eegnet.py --subject 5 --epochs 200
```

---

## Expected Performance

Based on literature and your current CSP+LDA results:

| Method | Expected Mean Accuracy | Subjects ≥70% |
|--------|------------------------|---------------|
| CSP+LDA (baseline) | 62% ± 18% | 4/10 |
| EEGNet (conservative) | 68% ± 15% | 6/10 |
| EEGNet (optimistic) | 75% ± 12% | 7-8/10 |

**Factors affecting performance:**
- Small dataset (~135 trials/subject)
- High inter-subject variability
- 8-channel limitation vs 64-channel data

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python3 src/train_eegnet.py --subject 1 --batch_size 8
```

### Training Too Slow

```bash
# Check GPU usage
nvidia-smi

# Verify CUDA is being used
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Poor Accuracy (<50%)

Possible causes:
1. **Model overfitting** - Try smaller learning rate or more dropout
2. **Data quality** - Check preprocessing parameters
3. **Subject is BCI-illiterate** - Expected for some subjects

---

## File Structure

After running all scripts:

```
EEG_MI_CSP_LDA_Project/
├── src/
│   ├── models/
│   │   └── eegnet.py                      # EEGNet model
│   ├── train_eegnet.py                    # Single subject training
│   ├── evaluate_all_subjects_eegnet.py    # Batch training
│   └── compare_methods.py                 # Comparison analysis
├── outputs_eegnet/
│   ├── subject_01/
│   │   └── eegnet_results.json
│   ├── ...
│   ├── subject_10/
│   │   └── eegnet_results.json
│   ├── EEGNET_SUMMARY_REPORT.md
│   └── eegnet_summary.json
├── outputs_comparison/
│   ├── COMPARISON_REPORT.md
│   └── *.png (comparison plots)
└── EEGNET_USAGE_GUIDE.md                  # This file
```

---

## Command Reference

### Test Model Architecture

```bash
# Test EEGNet with dummy data
python3 src/models/eegnet.py
```

### Train with Custom Settings

```bash
python3 src/train_eegnet.py \
    --subject 2 \
    --config 8-channel-motor \
    --output outputs_eegnet \
    --epochs 150 \
    --batch_size 16 \
    --lr 0.001 \
    --cv 5 \
    --seed 42
```

### Skip Training, Only Generate Reports

```bash
python3 src/evaluate_all_subjects_eegnet.py \
    --subjects 1 2 3 4 5 6 7 8 9 10 \
    --skip_training
```

---

## Next Steps After EEGNet

If EEGNet shows good improvement (>5% over CSP+LDA):

### 1. Data Augmentation

Create `src/train_eegnet_augmented.py` with:
- Time jittering
- Noise injection
- Segment recombination

### 2. Try Transformer

Implement a lightweight Transformer:
- Requires more data (use augmentation)
- May provide additional 2-5% improvement

### 3. Download More Data

```bash
# Download all 109 subjects from PhysioNet
python3 src/data_download.py --subjects $(seq 1 109)
```

### 4. Ensemble Methods

Combine predictions from:
- CSP+LDA (classical)
- EEGNet (deep learning)
- Majority voting or weighted average

---

## For Your Final Report

### Recommended Reporting Format

**Within-Subject Evaluation Results:**

"We evaluated EEGNet on 10 subjects using 5-fold cross-validation.
The model achieved X.XX% ± X.XX% accuracy, with X/10 subjects exceeding
the 70% clinical threshold. This represents a X.X% improvement over the
traditional CSP+LDA baseline (62.00% ± 18.09%)."

**Key Metrics to Report:**
- Mean ± std accuracy (both methods)
- Number of subjects ≥70%
- Per-subject improvements
- Model complexity (~2K parameters for EEGNet)
- Training time (~2-3 min/subject)

### Visualizations for Paper

Use these plots from `outputs_comparison/`:
1. `comparison_barplot.png` - Main results figure
2. `comparison_scatter.png` - Subject-wise comparison
3. `comparison_improvement.png` - Show which subjects benefited

---

## Contact & Support

If you encounter issues:

1. Check GPU usage: `nvidia-smi`
2. Verify PyTorch CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`
3. Review error messages in console output
4. Check file paths and data availability

---

## Citation

If you use EEGNet in your paper:

```
Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018).
EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces.
Journal of Neural Engineering, 15(5), 056013.
```

---

**Good luck with your training! 🚀**

Your RTX 5080 should make quick work of this. The entire evaluation
should take less than 30 minutes total.