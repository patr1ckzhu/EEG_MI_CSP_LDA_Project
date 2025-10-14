# EEG Motor Imagery (MI) - CSP + LDA Pipeline (Python)

This project gives you the Day 1-2 skeleton:
- Install dependencies (MNE, scikit-learn, PyRiemann, pyLSL, etc.)
- Download and read PhysioNet EEG Motor Movement/Imagery (EEGBCI) data
- Preprocess: band-pass (8-30 Hz) + 50 Hz notch, re-reference
- Feature: CSP
- Classifier: LDA (cross-validation)
- Goal: Offline binary classification (left vs right) target accuracy >= 70% (varies by subject)

Runs mapping (per MNE docs):
- 4, 8, 12: Motor imagery - left vs right hand
- 6, 10, 14: Motor imagery - hands vs feet

## 1. Install
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Download data (EEGBCI)
```bash
python src/data_download.py --subjects 1 2 --runs 4 8 12 --out data/raw
```

## 3. Train & evaluate (CSP + LDA)
```bash
python src/train_csp_lda.py --subjects 1 2 --runs 4 8 12   --datadir data/raw --band 8 30 --notch 50 --sfreq 160   --cv 5 --tmin 1.0 --tmax 3.0
```

## 4. Structure
EEG_MI_CSP_LDA_Project/
- requirements.txt
- README.md
- data/raw/              (downloaded .edf files)
- outputs/               (figures)
- src/
  - data_download.py
  - train_csp_lda.py
  - utils.py
