# Git Commit Guide - EEGNet Implementation

## Files to Commit

### New Files Created (8 files)

```bash
# Core implementation
src/models/eegnet.py                      # 6.1KB - EEGNet model
src/train_eegnet.py                       # 12KB  - Training script
src/evaluate_all_subjects_eegnet.py       # 9.3KB - Batch evaluation
src/compare_methods.py                    # 13KB  - Comparison analysis

# Documentation
EEGNET_USAGE_GUIDE.md                     # 8.1KB - Detailed usage
README_EEGNET.md                          # 5.9KB - Project overview
requirements_eegnet.txt                   # 469B  - Dependencies
run_eegnet_full_evaluation.sh             # 1.9KB - Automated pipeline
GIT_COMMIT_GUIDE.md                       # This file
```

**Total: ~55KB of new code and documentation**

## Commit Commands

### On your M1 Mac (current machine):

```bash
cd "/Users/patrick/Desktop/EEE/Fourth Year/Final Report/EEG_MI_CSP_LDA_Project"

# Check git status
git status

# Add all new EEGNet files
git add src/models/eegnet.py
git add src/train_eegnet.py
git add src/evaluate_all_subjects_eegnet.py
git add src/compare_methods.py
git add EEGNET_USAGE_GUIDE.md
git add README_EEGNET.md
git add requirements_eegnet.txt
git add run_eegnet_full_evaluation.sh
git add GIT_COMMIT_GUIDE.md

# Or add all at once
git add src/models/ src/*eegnet* src/compare*.py *.md requirements_eegnet.txt run_eegnet*.sh

# Commit with descriptive message
git commit -m "Add EEGNet deep learning implementation for 8-channel BCI

- Implement EEGNet model architecture (src/models/eegnet.py)
- Add single-subject training script (src/train_eegnet.py)
- Add batch evaluation for all 10 subjects (src/evaluate_all_subjects_eegnet.py)
- Add comparison analysis vs CSP+LDA (src/compare_methods.py)
- Include comprehensive documentation and automation scripts
- Support CUDA/MPS acceleration
- Optimized for ADS1299 8-channel hardware

Expected improvement: 5-10% over CSP+LDA baseline (62%)
Model size: ~2K parameters
Training time: ~2-3 min/subject on RTX 5080"

# Push to GitHub
git push origin master
```

## On Your RTX 5080 PC:

```bash
# Clone or pull latest
git clone <your-repo-url>
# OR
git pull origin master

# Setup environment
pip install -r requirements_eegnet.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Run full evaluation (one command!)
bash run_eegnet_full_evaluation.sh
```

## Expected Git Workflow

```
Your M1 Mac                           GitHub                          RTX 5080 PC
    │                                    │                                  │
    ├─[Implement code]──────────────────▶│                                  │
    │                                    │                                  │
    ├─[git add/commit/push]─────────────▶│                                  │
    │                                    │                                  │
    │                                    │◀──────[git clone/pull]──────────┤
    │                                    │                                  │
    │                                    │                                  ├─[Train models]
    │                                    │                                  │
    │                                    │                                  ├─[Generate results]
    │                                    │                                  │
    │                                    │◀──────[git add results]─────────┤
    │                                    │                                  │
    │◀──[git pull to view results]──────┤                                  │
    │                                    │                                  │
```

## Optional: Commit Results After Training

After training on RTX 5080:

```bash
cd /path/to/project

# Add result files
git add outputs_eegnet/
git add outputs_comparison/

# Commit results
git commit -m "Add EEGNet training results for 10 subjects

Results:
- Mean accuracy: XX.XX% ± XX.XX%
- Subjects ≥70%: X/10
- Improvement over CSP+LDA: +X.X%
- Training time: ~XX minutes on RTX 5080

Generated:
- Per-subject results (outputs_eegnet/)
- Comparison analysis (outputs_comparison/)
- Summary reports and visualizations"

# Push back to GitHub
git push origin master
```

## What NOT to Commit

```bash
# Exclude these from git
.venv/                    # Virtual environment
__pycache__/              # Python cache
*.pyc                     # Compiled Python
.DS_Store                 # Mac system files
*.pkl                     # Trained model weights (too large)
data/raw/                 # Raw data files (too large)
```

## Verify Before Pushing

```bash
# Check what will be committed
git status

# Review changes
git diff --staged

# Check commit message
git log -1

# Verify no sensitive data
git show

# If everything looks good
git push origin master
```

## Quick Commands Reference

```bash
# See what changed
git status

# Stage new files
git add <file>

# Commit
git commit -m "message"

# Push to GitHub
git push

# Pull latest
git pull

# View commit history
git log --oneline

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard local changes
git checkout -- <file>
```

## GitHub Repository Structure (After Push)

```
your-repo/
├── src/
│   ├── models/
│   │   └── eegnet.py              ← NEW
│   ├── train_eegnet.py            ← NEW
│   ├── evaluate_all_subjects_eegnet.py  ← NEW
│   ├── compare_methods.py         ← NEW
│   └── [existing CSP+LDA files]
├── EEGNET_USAGE_GUIDE.md          ← NEW
├── README_EEGNET.md               ← NEW
├── requirements_eegnet.txt        ← NEW
├── run_eegnet_full_evaluation.sh  ← NEW
└── [existing project files]
```

## Next Steps After Pushing

1. **On RTX 5080**: Clone/pull repository
2. **Install dependencies**: `pip install -r requirements_eegnet.txt`
3. **Run training**: `bash run_eegnet_full_evaluation.sh`
4. **Wait 20-30 minutes** for results
5. **Review reports**: Check `outputs_eegnet/` and `outputs_comparison/`
6. **(Optional)** Commit and push results back to GitHub

---

**Ready to commit?** Copy the commands above and you're good to go! 🚀