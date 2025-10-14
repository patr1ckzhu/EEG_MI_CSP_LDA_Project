"""
Optimized CSP + LDA training with:
- Per-subject analysis
- Hyperparameter grid search
- Enhanced evaluation metrics (ROC, Kappa, per-class metrics)
- Multiple visualization outputs
- Data augmentation options
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

import mne
from mne.io import read_raw_edf
from mne.channels import make_standard_montage
from mne.decoding import CSP

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    classification_report, cohen_kappa_score, roc_curve, auc
)
from sklearn.pipeline import Pipeline

from utils import ensure_dir, edf_paths_in


def load_concat_raw(edf_files, sfreq_expected=160):
    """Load and concatenate multiple EDF files."""
    raws = []
    for f in edf_files:
        raw = read_raw_edf(f, preload=True, stim_channel=None, verbose=False)
        if int(raw.info['sfreq']) != sfreq_expected:
            raw.resample(sfreq_expected)
        raws.append(raw)
    raw = mne.concatenate_raws(raws)
    return raw


def epoch_mi_left_right(raw, band=(8, 30), tmin=1.0, tmax=3.0):
    """Preprocess and epoch motor imagery data."""
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore')

    # Apply filters
    raw.notch_filter(freqs=[50], picks='eeg', verbose=False)
    raw.filter(l_freq=band[0], h_freq=band[1], picks='eeg', verbose=False)

    # Extract events
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    keep = {}
    for k, v in event_id.items():
        if k in ('T1', 'T2'):
            keep[k] = v
    if len(keep) < 2:
        raise RuntimeError("Did not find both T1 and T2 events.")

    event_id = keep
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')

    # Create epochs
    epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=0.0, tmax=tmax, proj=True, picks=picks,
                        baseline=None, preload=True, verbose=False)
    if tmin > 0.0:
        epochs.crop(tmin=tmin, tmax=tmax)

    y = epochs.events[:, 2]
    X = epochs.get_data()

    # Convert labels to 0/1 for binary classification
    unique_labels = np.unique(y)
    y_binary = np.zeros_like(y)
    y_binary[y == unique_labels[1]] = 1

    return X, y_binary, epochs, unique_labels


def evaluate_subject(subject_id, edf_dir, band=(8, 30), tmin=1.0, tmax=3.0,
                     cv=5, csp_components=6, optimize=False):
    """
    Evaluate a single subject with optional hyperparameter optimization.

    Returns:
        results: dict with accuracy, std, predictions, and optimal params
    """
    # Load subject data
    subject_dir = Path(edf_dir) / f"subject-{subject_id:02d}"
    edf_files = edf_paths_in(str(subject_dir))

    if not edf_files:
        raise FileNotFoundError(f"No .edf files found for subject {subject_id}")

    print(f"\n{'='*60}")
    print(f"Processing Subject {subject_id}")
    print(f"{'='*60}")
    print(f"EDF files found: {len(edf_files)}")

    raw = load_concat_raw(edf_files, sfreq_expected=160)
    X, y, epochs, label_mapping = epoch_mi_left_right(raw, band=band, tmin=tmin, tmax=tmax)

    print(f"Total trials: {len(y)}")
    print(f"Class distribution: T1={np.sum(y==0)}, T2={np.sum(y==1)}")

    # Setup cross-validation
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Store results
    scores = []
    y_true_all, y_pred_all, y_proba_all = [], [], []
    best_params = None

    # Track per-fold results
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_split.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if optimize and fold_idx == 1:
            # Hyperparameter optimization on first fold
            print("\n[Hyperparameter Optimization]")
            param_grid = {
                'csp__n_components': [4, 6, 8, 10],
                'lda__shrinkage': ['auto', 0.1, 0.3, 0.5]
            }

            pipeline = Pipeline([
                ('csp', CSP(reg='ledoit_wolf', log=True, norm_trace=False)),
                ('lda', LinearDiscriminantAnalysis(solver='lsqr'))
            ])

            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            csp_components = best_params['csp__n_components']
            lda_shrinkage = best_params['lda__shrinkage']

            print(f"Best params: {best_params}")
            print(f"Best CV score: {grid_search.best_score_*100:.2f}%")
        else:
            lda_shrinkage = 'auto'

        # Train CSP + LDA
        csp = CSP(n_components=csp_components, reg='ledoit_wolf', log=True, norm_trace=False)
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=lda_shrinkage)

        csp.fit(X_train, y_train)
        X_train_csp = csp.transform(X_train)
        X_test_csp = csp.transform(X_test)

        lda.fit(X_train_csp, y_train)
        y_pred = lda.predict(X_test_csp)
        y_proba = lda.predict_proba(X_test_csp)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        scores.append(acc)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        y_proba_all.extend(y_proba.tolist())

        fold_results.append({
            'fold': fold_idx,
            'accuracy': acc,
            'train_size': len(y_train),
            'test_size': len(y_test)
        })

        print(f"  Fold {fold_idx}: {acc*100:.2f}% (train={len(y_train)}, test={len(y_test)})")

    # Calculate metrics
    acc_mean = np.mean(scores)
    acc_std = np.std(scores)
    kappa = cohen_kappa_score(y_true_all, y_pred_all)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
    roc_auc = auc(fpr, tpr)

    print(f"\n{'='*60}")
    print(f"Subject {subject_id} Results:")
    print(f"  Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
    print(f"  Cohen's Kappa: {kappa:.3f}")
    print(f"  ROC AUC: {roc_auc:.3f}")
    print(f"{'='*60}")

    # Classification report
    print("\nPer-class Metrics:")
    report = classification_report(y_true_all, y_pred_all,
                                   target_names=['T1(left)', 'T2(right)'],
                                   output_dict=True)
    print(classification_report(y_true_all, y_pred_all,
                               target_names=['T1(left)', 'T2(right)']))

    results = {
        'subject_id': subject_id,
        'accuracy_mean': acc_mean,
        'accuracy_std': acc_std,
        'kappa': kappa,
        'roc_auc': roc_auc,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_proba': y_proba_all,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'classification_report': report,
        'fold_results': fold_results,
        'best_params': best_params,
        'n_trials': len(y),
        'band': band,
        'time_window': (tmin, tmax),
        'csp_components': csp_components
    }

    return results


def create_visualizations(all_results, output_dir):
    """Create comprehensive visualization plots."""
    output_dir = Path(output_dir)

    # 1. Confusion matrices for each subject
    fig, axes = plt.subplots(1, len(all_results), figsize=(5*len(all_results), 4))
    if len(all_results) == 1:
        axes = [axes]

    for idx, result in enumerate(all_results):
        cm = confusion_matrix(result['y_true'], result['y_pred'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['T1(left)', 'T2(right)'])
        disp.plot(ax=axes[idx], colorbar=False, cmap='Blues')
        axes[idx].set_title(f"Subject {result['subject_id']}\nAcc={result['accuracy_mean']*100:.1f}%")

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=150)
    print(f"[SAVED] {output_dir / 'confusion_matrices.png'}")
    plt.close()

    # 2. ROC curves
    plt.figure(figsize=(8, 6))
    for result in all_results:
        plt.plot(result['fpr'], result['tpr'],
                label=f"Subject {result['subject_id']} (AUC={result['roc_auc']:.3f})",
                linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Chance', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Motor Imagery Classification', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=150)
    print(f"[SAVED] {output_dir / 'roc_curves.png'}")
    plt.close()

    # 3. Accuracy comparison bar plot
    subjects = [r['subject_id'] for r in all_results]
    accuracies = [r['accuracy_mean']*100 for r in all_results]
    stds = [r['accuracy_std']*100 for r in all_results]

    plt.figure(figsize=(max(6, len(subjects)*1.5), 5))
    bars = plt.bar(range(len(subjects)), accuracies, yerr=stds,
                   capsize=5, alpha=0.8, color='steelblue', edgecolor='navy')

    # Color bars based on performance
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        if acc >= 70:
            bar.set_color('green')
        elif acc >= 60:
            bar.set_color('orange')
        else:
            bar.set_color('red')

        # Add value labels
        plt.text(i, acc + stds[i] + 1, f'{acc:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    plt.axhline(y=70, color='g', linestyle='--', label='Target (70%)', linewidth=2)
    plt.axhline(y=50, color='r', linestyle='--', label='Chance (50%)', linewidth=1)
    plt.xticks(range(len(subjects)), [f'Subject {s}' for s in subjects])
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Classification Accuracy by Subject', fontsize=14)
    plt.legend()
    plt.ylim([0, 105])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150)
    print(f"[SAVED] {output_dir / 'accuracy_comparison.png'}")
    plt.close()

    # 4. Per-class performance heatmap
    fig, ax = plt.subplots(figsize=(8, max(4, len(all_results)*0.8)))

    # Extract per-class F1 scores
    subject_labels = []
    left_f1 = []
    right_f1 = []

    for result in all_results:
        subject_labels.append(f"Subject {result['subject_id']}")
        left_f1.append(result['classification_report']['T1(left)']['f1-score'])
        right_f1.append(result['classification_report']['T2(right)']['f1-score'])

    data = np.array([left_f1, right_f1]).T

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['T1(left)', 'T2(right)'])
    ax.set_yticks(range(len(subject_labels)))
    ax.set_yticklabels(subject_labels)

    # Add text annotations
    for i in range(len(subject_labels)):
        for j in range(2):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')

    ax.set_title('Per-Class F1-Score Heatmap', fontsize=14)
    plt.colorbar(im, ax=ax, label='F1-Score')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_performance.png', dpi=150)
    print(f"[SAVED] {output_dir / 'per_class_performance.png'}")
    plt.close()


def save_summary_report(all_results, output_dir):
    """Save a JSON summary report."""
    output_dir = Path(output_dir)

    # Remove non-serializable numpy arrays
    summary = []
    for result in all_results:
        clean_result = {
            'subject_id': result['subject_id'],
            'accuracy_mean': float(result['accuracy_mean']),
            'accuracy_std': float(result['accuracy_std']),
            'kappa': float(result['kappa']),
            'roc_auc': float(result['roc_auc']),
            'n_trials': result['n_trials'],
            'band': result['band'],
            'time_window': result['time_window'],
            'csp_components': result['csp_components'],
            'best_params': result['best_params'],
            'classification_report': result['classification_report'],
            'fold_results': result['fold_results']
        }
        summary.append(clean_result)

    # Overall statistics
    overall = {
        'timestamp': datetime.now().isoformat(),
        'n_subjects': len(all_results),
        'mean_accuracy': float(np.mean([r['accuracy_mean'] for r in all_results])),
        'std_accuracy': float(np.std([r['accuracy_mean'] for r in all_results])),
        'subjects_above_70': sum(1 for r in all_results if r['accuracy_mean'] >= 0.70),
        'subjects_above_60': sum(1 for r in all_results if r['accuracy_mean'] >= 0.60),
        'per_subject_results': summary
    }

    report_path = output_dir / 'evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(overall, f, indent=2)

    print(f"[SAVED] {report_path}")

    # Also save a human-readable text report
    text_report_path = output_dir / 'evaluation_report.txt'
    with open(text_report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EEG MOTOR IMAGERY CLASSIFICATION - EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Algorithm: CSP + LDA\n")
        f.write(f"Cross-validation: 5-fold Stratified\n\n")

        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of subjects: {overall['n_subjects']}\n")
        f.write(f"Mean accuracy: {overall['mean_accuracy']*100:.2f}% ± {overall['std_accuracy']*100:.2f}%\n")
        f.write(f"Subjects ≥70%: {overall['subjects_above_70']}/{overall['n_subjects']}\n")
        f.write(f"Subjects ≥60%: {overall['subjects_above_60']}/{overall['n_subjects']}\n\n")

        f.write("PER-SUBJECT RESULTS\n")
        f.write("-"*80 + "\n")
        for result in all_results:
            f.write(f"\nSubject {result['subject_id']}:\n")
            f.write(f"  Accuracy: {result['accuracy_mean']*100:.2f}% ± {result['accuracy_std']*100:.2f}%\n")
            f.write(f"  Cohen's Kappa: {result['kappa']:.3f}\n")
            f.write(f"  ROC AUC: {result['roc_auc']:.3f}\n")
            f.write(f"  Trials: {result['n_trials']}\n")
            f.write(f"  CSP components: {result['csp_components']}\n")
            f.write(f"  Frequency band: {result['band'][0]}-{result['band'][1]} Hz\n")
            f.write(f"  Time window: {result['time_window'][0]}-{result['time_window'][1]} s\n")

            # Per-class metrics
            report = result['classification_report']
            f.write(f"  T1(left)  - Precision: {report['T1(left)']['precision']:.3f}, "
                   f"Recall: {report['T1(left)']['recall']:.3f}, "
                   f"F1: {report['T1(left)']['f1-score']:.3f}\n")
            f.write(f"  T2(right) - Precision: {report['T2(right)']['precision']:.3f}, "
                   f"Recall: {report['T2(right)']['recall']:.3f}, "
                   f"F1: {report['T2(right)']['f1-score']:.3f}\n")

    print(f"[SAVED] {text_report_path}")


def main():
    ap = argparse.ArgumentParser(description="Optimized CSP+LDA training with comprehensive evaluation")
    ap.add_argument("--subjects", nargs="+", type=int, default=[1, 2],
                    help="Subject IDs to evaluate")
    ap.add_argument("--datadir", type=str, default="data/raw",
                    help="Data directory")
    ap.add_argument("--band", nargs=2, type=float, default=[8., 30.],
                    help="Frequency band (Hz)")
    ap.add_argument("--tmin", type=float, default=1.0,
                    help="Epoch start time (s)")
    ap.add_argument("--tmax", type=float, default=3.0,
                    help="Epoch end time (s)")
    ap.add_argument("--cv", type=int, default=5,
                    help="Number of CV folds")
    ap.add_argument("--csp-components", type=int, default=6,
                    help="Number of CSP components (overridden if --optimize)")
    ap.add_argument("--optimize", action='store_true',
                    help="Perform hyperparameter optimization")
    ap.add_argument("--output", type=str, default="outputs_optimized",
                    help="Output directory")

    args = ap.parse_args()

    output_dir = ensure_dir(args.output)

    print("\n" + "="*80)
    print("OPTIMIZED CSP + LDA TRAINING")
    print("="*80)
    print(f"Subjects: {args.subjects}")
    print(f"Frequency band: {args.band[0]}-{args.band[1]} Hz")
    print(f"Time window: {args.tmin}-{args.tmax} s")
    print(f"CSP components: {args.csp_components}")
    print(f"Cross-validation: {args.cv}-fold")
    print(f"Hyperparameter optimization: {args.optimize}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Evaluate each subject
    all_results = []
    for subject_id in args.subjects:
        try:
            result = evaluate_subject(
                subject_id=subject_id,
                edf_dir=args.datadir,
                band=tuple(args.band),
                tmin=args.tmin,
                tmax=args.tmax,
                cv=args.cv,
                csp_components=args.csp_components,
                optimize=args.optimize
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed to process subject {subject_id}: {str(e)}")
            continue

    if not all_results:
        print("\n[ERROR] No subjects were successfully processed!")
        return

    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    create_visualizations(all_results, output_dir)

    # Save summary report
    print("\n" + "="*80)
    print("SAVING SUMMARY REPORT")
    print("="*80)
    save_summary_report(all_results, output_dir)

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    mean_acc = np.mean([r['accuracy_mean'] for r in all_results])
    std_acc = np.std([r['accuracy_mean'] for r in all_results])
    print(f"Overall accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Subjects achieving ≥70%: {sum(1 for r in all_results if r['accuracy_mean'] >= 0.70)}/{len(all_results)}")
    print(f"Subjects achieving ≥60%: {sum(1 for r in all_results if r['accuracy_mean'] >= 0.60)}/{len(all_results)}")

    best_subject = max(all_results, key=lambda x: x['accuracy_mean'])
    print(f"\nBest performer: Subject {best_subject['subject_id']} "
          f"({best_subject['accuracy_mean']*100:.2f}% ± {best_subject['accuracy_std']*100:.2f}%)")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()