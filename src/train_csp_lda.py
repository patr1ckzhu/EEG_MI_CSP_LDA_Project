"""
Train/evaluate CSP + LDA on EEGBCI motor imagery (left vs right hand).

Steps:
- Load EDFs
- Filter: band-pass + notch
- Epoch around MI events
- CSP features
- LDA classification (Stratified K-Fold CV)
Saves confusion matrix to outputs/confusion_matrix.png
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.io import read_raw_edf
from mne.channels import make_standard_montage
from mne.decoding import CSP

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from utils import ensure_dir, edf_paths_in

def load_concat_raw(edf_files, sfreq_expected=160):
    raws = []
    for f in edf_files:
        raw = read_raw_edf(f, preload=True, stim_channel=None, verbose=False)
        if int(raw.info['sfreq']) != sfreq_expected:
            raw.resample(sfreq_expected)
        raws.append(raw)
    raw = mne.concatenate_raws(raws)
    return raw

def epoch_mi_left_right(raw, tmin=1.0, tmax=3.0):
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore')

    raw.notch_filter(freqs=[50], picks='eeg')
    raw.filter(l_freq=8., h_freq=30., picks='eeg')

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    keep = {}
    for k, v in event_id.items():
        if k in ('T1', 'T2'):
            keep[k] = v
    if len(keep) < 2:
        raise RuntimeError("Did not find both T1 and T2 events. Use MI left/right runs 4/8/12.")

    event_id = keep
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')

    epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=0.0, tmax=tmax, proj=True, picks=picks,
                        baseline=None, preload=True, verbose=False)
    if tmin > 0.0:
        epochs.crop(tmin=tmin, tmax=tmax)

    y = epochs.events[:, 2]
    X = epochs.get_data()
    return X, y, epochs

def run_pipeline(edf_dir, band=(8,30), notch=50, sfreq=160, tmin=1.0, tmax=3.0, cv=5, csp_components=6):
    edf_files = edf_paths_in(edf_dir)
    if not edf_files:
        raise FileNotFoundError(f"No .edf files found under {edf_dir}. Run data_download.py first.")

    raw = load_concat_raw(edf_files, sfreq_expected=sfreq)
    X, y, epochs = epoch_mi_left_right(raw, tmin=tmin, tmax=tmax)

    csp = CSP(n_components=csp_components, reg='ledoit_wolf', log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in cv_split.split(X, y):
        csp.fit(X[train_idx], y[train_idx])
        X_train_csp = csp.transform(X[train_idx])
        X_test_csp  = csp.transform(X[test_idx])

        lda.fit(X_train_csp, y[train_idx])
        y_pred = lda.predict(X_test_csp)

        acc = accuracy_score(y[test_idx], y_pred)
        scores.append(acc)
        y_true_all.extend(list(y[test_idx]))
        y_pred_all .extend(list(y_pred))

    acc_mean = np.mean(scores)
    acc_std  = np.std(scores)

    print(f"[RESULT] CV accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}% ({cv}-fold)")
    outputs_dir = ensure_dir("outputs")

    classes_sorted = sorted(np.unique(y))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes_sorted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['T1(left)','T2(right)'])
    fig, ax = plt.subplots(figsize=(4,4))
    disp.plot(ax=ax, colorbar=False)
    plt.title(f"CSP+LDA Confusion Matrix (acc={acc_mean*100:.1f}%)")
    fig.tight_layout()
    fig_path = Path(outputs_dir) / "confusion_matrix.png"
    plt.savefig(fig_path, dpi=150)
    print(f"[SAVED] {fig_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", type=int, default=[1])
    ap.add_argument("--runs", nargs="+", type=int, default=[4,8,12],
                    help="For reference only (used during download).")
    ap.add_argument("--datadir", type=str, default="data/raw")
    ap.add_argument("--band", nargs=2, type=float, default=[8., 30.])
    ap.add_argument("--notch", type=float, default=50.0)
    ap.add_argument("--sfreq", type=int, default=160)
    ap.add_argument("--tmin", type=float, default=1.0)
    ap.add_argument("--tmax", type=float, default=3.0)
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    run_pipeline(edf_dir=args.datadir,
                 band=tuple(args.band),
                 notch=args.notch,
                 sfreq=args.sfreq,
                 tmin=args.tmin, tmax=args.tmax,
                 cv=args.cv)

if __name__ == "__main__":
    main()
