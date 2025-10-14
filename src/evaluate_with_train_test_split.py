"""
Proper train/test split evaluation for BCI models.

This script implements the correct machine learning evaluation methodology:
- Training set (70%): Subjects 1-7
- Test set (30%): Subjects 8-10

Evaluates both:
1. Within-subject performance (traditional 5-fold CV on each subject)
2. Cross-subject generalization (train on 1-7, test on 8-10)
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from scipy import stats

import mne
from mne.io import read_raw_edf
from mne.channels import make_standard_montage
from mne.decoding import CSP

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    classification_report, cohen_kappa_score, roc_curve, auc
)

from utils import ensure_dir, edf_paths_in


# Channel configuration
CHANNEL_CONFIG_8CH = {
    'channels': ['FC3', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4'],
    'csp_components': 4,
    'description': 'ADS1299 optimal configuration'
}


def normalize_channel_name(ch_name):
    """Normalize channel names."""
    ch_clean = ch_name.rstrip('.').upper()
    ch_clean = ch_clean.replace('..', '')
    ch_clean = ch_clean.replace('Z', 'z')
    return ch_clean


def find_matching_channels(available_channels, requested_channels):
    """Find matching channels."""
    channel_map = {}
    for avail_ch in available_channels:
        norm_name = normalize_channel_name(avail_ch)
        channel_map[norm_name] = avail_ch

    matched = []
    for req_ch in requested_channels:
        norm_req = normalize_channel_name(req_ch)
        if norm_req in channel_map:
            matched.append(channel_map[norm_req])

    return matched


def load_subject_data(subject_id, edf_dir, channel_config, band=(8, 30), tmin=1.0, tmax=3.0):
    """Load and preprocess data for one subject."""
    subject_dir = Path(edf_dir) / f"subject-{subject_id:02d}"
    edf_files = edf_paths_in(str(subject_dir))

    if not edf_files:
        raise FileNotFoundError(f"No EDF files for subject {subject_id}")

    # Load and concatenate runs
    raws = []
    for f in edf_files:
        raw = read_raw_edf(f, preload=True, stim_channel=None, verbose=False)
        if int(raw.info['sfreq']) != 160:
            raw.resample(160)
        raws.append(raw)
    raw = mne.concatenate_raws(raws)

    # Select channels
    matched_channels = find_matching_channels(raw.ch_names, channel_config['channels'])
    raw.pick_channels(matched_channels)

    # Apply montage
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore')

    # Filters
    raw.notch_filter(freqs=[50], picks='eeg', verbose=False)
    raw.filter(l_freq=band[0], h_freq=band[1], picks='eeg', verbose=False)

    # Extract events
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    keep = {k: v for k, v in event_id.items() if k in ('T1', 'T2')}

    if len(keep) < 2:
        raise RuntimeError(f"Subject {subject_id} missing T1 or T2 events")

    picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')

    # Create epochs
    epochs = mne.Epochs(raw, events, event_id=keep,
                        tmin=0.0, tmax=tmax, proj=True, picks=picks,
                        baseline=None, preload=True, verbose=False)
    if tmin > 0.0:
        epochs.crop(tmin=tmin, tmax=tmax)

    y = epochs.events[:, 2]
    X = epochs.get_data()

    # Binary labels
    unique_labels = np.unique(y)
    y_binary = np.zeros_like(y)
    y_binary[y == unique_labels[1]] = 1

    return X, y_binary


def within_subject_evaluation(subjects, edf_dir, channel_config, cv=5):
    """
    Within-subject evaluation using 5-fold CV on each subject.
    This is the standard BCI evaluation (subject-specific models).
    """
    print("\n" + "="*80)
    print("WITHIN-SUBJECT EVALUATION (Subject-Specific Models)")
    print("="*80)
    print("Training and testing on the SAME subject (5-fold CV)")
    print("This represents the best-case scenario for BCI performance.\n")

    all_results = []

    for subject_id in subjects:
        print(f"\n[Subject {subject_id}]")
        try:
            X, y = load_subject_data(subject_id, edf_dir, channel_config)
            print(f"  Trials: {len(y)} (T1={np.sum(y==0)}, T2={np.sum(y==1)})")

            # 5-fold CV
            cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            scores = []

            for fold_idx, (train_idx, test_idx) in enumerate(cv_split.split(X, y), 1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Train CSP + LDA
                csp = CSP(n_components=channel_config['csp_components'],
                          reg='ledoit_wolf', log=True, norm_trace=False)
                lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

                csp.fit(X_train, y_train)
                X_train_csp = csp.transform(X_train)
                X_test_csp = csp.transform(X_test)

                lda.fit(X_train_csp, y_train)
                y_pred = lda.predict(X_test_csp)

                acc = accuracy_score(y_test, y_pred)
                scores.append(acc)

            acc_mean = np.mean(scores)
            acc_std = np.std(scores)

            print(f"  Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")

            all_results.append({
                'subject_id': subject_id,
                'accuracy': acc_mean,
                'std': acc_std,
                'cv_scores': scores
            })

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue

    return all_results


def cross_subject_evaluation(train_subjects, test_subjects, edf_dir, channel_config):
    """
    Cross-subject evaluation: Train on one set of subjects, test on another.
    This represents the generalization capability (can the model work on new users?).
    """
    print("\n" + "="*80)
    print("CROSS-SUBJECT EVALUATION (Generalization Test)")
    print("="*80)
    print(f"Training on: Subjects {train_subjects}")
    print(f"Testing on:  Subjects {test_subjects}")
    print("This tests if the model can work on NEW users without re-training.\n")

    # Load all training data
    print("[Loading Training Data...]")
    X_train_all, y_train_all = [], []

    for subject_id in train_subjects:
        try:
            X, y = load_subject_data(subject_id, edf_dir, channel_config)
            X_train_all.append(X)
            y_train_all.append(y)
            print(f"  Subject {subject_id}: {len(y)} trials")
        except Exception as e:
            print(f"  Subject {subject_id}: ERROR - {str(e)}")
            continue

    if not X_train_all:
        raise RuntimeError("No training data loaded!")

    X_train_all = np.concatenate(X_train_all, axis=0)
    y_train_all = np.concatenate(y_train_all, axis=0)

    print(f"\nTotal training samples: {len(y_train_all)}")
    print(f"  T1 (left):  {np.sum(y_train_all==0)}")
    print(f"  T2 (right): {np.sum(y_train_all==1)}")

    # Train a global model
    print("\n[Training Global Model...]")
    csp_global = CSP(n_components=channel_config['csp_components'],
                     reg='ledoit_wolf', log=True, norm_trace=False)
    lda_global = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    csp_global.fit(X_train_all, y_train_all)
    X_train_csp = csp_global.transform(X_train_all)
    lda_global.fit(X_train_csp, y_train_all)

    train_acc = accuracy_score(y_train_all, lda_global.predict(X_train_csp))
    print(f"Training set accuracy: {train_acc*100:.2f}%")

    # Test on each test subject
    print("\n[Testing on New Subjects...]")
    test_results = []

    for subject_id in test_subjects:
        print(f"\n[Test Subject {subject_id}]")
        try:
            X_test, y_test = load_subject_data(subject_id, edf_dir, channel_config)
            print(f"  Trials: {len(y_test)}")

            # Apply global model
            X_test_csp = csp_global.transform(X_test)
            y_pred = lda_global.predict(X_test_csp)
            y_proba = lda_global.predict_proba(X_test_csp)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            kappa = cohen_kappa_score(y_test, y_pred)

            print(f"  Accuracy: {acc*100:.2f}%")
            print(f"  Kappa: {kappa:.3f}")

            test_results.append({
                'subject_id': subject_id,
                'accuracy': acc,
                'kappa': kappa,
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist()
            })

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue

    return test_results, {
        'csp': csp_global,
        'lda': lda_global,
        'train_acc': train_acc
    }


def create_comprehensive_visualizations(within_results, cross_results, output_dir):
    """Create comprehensive comparison visualizations."""
    output_dir = Path(output_dir)

    # Extract data
    within_subjects = [r['subject_id'] for r in within_results]
    within_acc = [r['accuracy']*100 for r in within_results]
    within_std = [r['std']*100 for r in within_results]

    cross_subjects = [r['subject_id'] for r in cross_results]
    cross_acc = [r['accuracy']*100 for r in cross_results]

    # 1. Within vs Cross-Subject Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Within-subject plot
    ax = axes[0]
    bars = ax.bar(range(len(within_subjects)), within_acc, yerr=within_std,
                   capsize=5, alpha=0.8, color='steelblue', edgecolor='navy')

    # Color bars
    for i, (bar, acc) in enumerate(zip(bars, within_acc)):
        if acc >= 70:
            bar.set_color('green')
        elif acc >= 60:
            bar.set_color('orange')
        else:
            bar.set_color('red')

        ax.text(i, acc + within_std[i] + 2, f'{acc:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    ax.axhline(y=70, color='g', linestyle='--', label='Target (70%)', linewidth=2)
    ax.axhline(y=50, color='r', linestyle='--', label='Chance (50%)', linewidth=1)
    ax.set_xticks(range(len(within_subjects)))
    ax.set_xticklabels([f'S{s}' for s in within_subjects])
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Within-Subject Performance\n(Subject-Specific Models)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Cross-subject plot
    ax = axes[1]
    bars = ax.bar(range(len(cross_subjects)), cross_acc,
                   alpha=0.8, color='coral', edgecolor='darkred')

    # Color bars
    for i, (bar, acc) in enumerate(zip(bars, cross_acc)):
        if acc >= 70:
            bar.set_color('green')
        elif acc >= 60:
            bar.set_color('orange')
        else:
            bar.set_color('red')

        ax.text(i, acc + 2, f'{acc:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    ax.axhline(y=70, color='g', linestyle='--', label='Target (70%)', linewidth=2)
    ax.axhline(y=50, color='r', linestyle='--', label='Chance (50%)', linewidth=1)
    ax.set_xticks(range(len(cross_subjects)))
    ax.set_xticklabels([f'S{s}' for s in cross_subjects])
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Cross-Subject Performance\n(Trained on S1-7, Tested on S8-10)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'within_vs_cross_subject.png', dpi=150)
    print(f"[SAVED] {output_dir / 'within_vs_cross_subject.png'}")
    plt.close()

    # 2. Box plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = [within_acc, cross_acc]
    bp = ax.boxplot(data_to_plot, labels=['Within-Subject\n(Train & Test Same)', 'Cross-Subject\n(Train S1-7, Test S8-10)'],
                    patch_artist=True, widths=0.6)

    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')

    ax.axhline(y=70, color='g', linestyle='--', label='Target (70%)', linewidth=2)
    ax.axhline(y=50, color='r', linestyle='--', label='Chance (50%)', linewidth=1)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Within-Subject vs Cross-Subject Performance Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add mean markers
    means = [np.mean(within_acc), np.mean(cross_acc)]
    ax.plot([1, 2], means, 'D', markersize=10, color='red', label='Mean', zorder=3)

    # Add text annotations
    for i, (mean, std) in enumerate([(np.mean(within_acc), np.std(within_acc)),
                                      (np.mean(cross_acc), np.std(cross_acc))], 1):
        ax.text(i, mean + 5, f'μ={mean:.1f}%\nσ={std:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_distribution.png', dpi=150)
    print(f"[SAVED] {output_dir / 'performance_distribution.png'}")
    plt.close()


def generate_statistical_report(within_results, cross_results, global_model, output_dir):
    """Generate comprehensive statistical analysis report."""
    output_dir = Path(output_dir)

    # Calculate statistics
    within_acc = [r['accuracy']*100 for r in within_results]
    cross_acc = [r['accuracy']*100 for r in cross_results]

    within_mean = np.mean(within_acc)
    within_std = np.std(within_acc)
    within_median = np.median(within_acc)

    cross_mean = np.mean(cross_acc)
    cross_std = np.std(cross_acc)
    cross_median = np.median(cross_acc)

    # Statistical test
    if len(within_acc) >= 3 and len(cross_acc) >= 3:
        t_stat, p_value = stats.ttest_ind(within_acc, cross_acc)
    else:
        t_stat, p_value = None, None

    # Write report
    report_path = output_dir / 'STATISTICAL_ANALYSIS_REPORT.md'

    with open(report_path, 'w') as f:
        f.write("# Statistical Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Dataset Overview\n\n")
        f.write(f"- Total subjects: 10\n")
        f.write(f"- Training set: Subjects 1-7 (70%)\n")
        f.write(f"- Test set: Subjects 8-10 (30%)\n")
        f.write(f"- Configuration: 8-channel motor cortex\n")
        f.write(f"- Algorithm: CSP (4 components) + LDA\n\n")

        f.write("---\n\n")

        f.write("## Within-Subject Performance (Subject-Specific Models)\n\n")
        f.write("**Methodology**: 5-fold cross-validation on each subject independently\n\n")
        f.write(f"**Summary Statistics**:\n")
        f.write(f"- Mean: {within_mean:.2f}% ± {within_std:.2f}%\n")
        f.write(f"- Median: {within_median:.2f}%\n")
        f.write(f"- Min: {min(within_acc):.2f}%\n")
        f.write(f"- Max: {max(within_acc):.2f}%\n")
        f.write(f"- Subjects ≥70%: {sum(1 for x in within_acc if x >= 70)}/{len(within_acc)}\n")
        f.write(f"- Subjects ≥60%: {sum(1 for x in within_acc if x >= 60)}/{len(within_acc)}\n\n")

        f.write("**Per-Subject Results**:\n\n")
        f.write("| Subject | Accuracy | Std Dev | Status |\n")
        f.write("|---------|----------|---------|--------|\n")
        for r in within_results:
            status = "✅ Good" if r['accuracy']*100 >= 70 else ("⚠️ Fair" if r['accuracy']*100 >= 60 else "❌ Poor")
            f.write(f"| {r['subject_id']} | {r['accuracy']*100:.2f}% | ±{r['std']*100:.2f}% | {status} |\n")

        f.write("\n---\n\n")

        f.write("## Cross-Subject Performance (Generalization Test)\n\n")
        f.write("**Methodology**: Train on subjects 1-7, test on subjects 8-10 (unseen data)\n\n")
        f.write(f"**Summary Statistics**:\n")
        f.write(f"- Mean: {cross_mean:.2f}% ± {cross_std:.2f}%\n")
        f.write(f"- Median: {cross_median:.2f}%\n")
        f.write(f"- Min: {min(cross_acc):.2f}%\n")
        f.write(f"- Max: {max(cross_acc):.2f}%\n")
        f.write(f"- Training set accuracy: {global_model['train_acc']*100:.2f}%\n")
        f.write(f"- Subjects ≥70%: {sum(1 for x in cross_acc if x >= 70)}/{len(cross_acc)}\n")
        f.write(f"- Subjects ≥60%: {sum(1 for x in cross_acc if x >= 60)}/{len(cross_acc)}\n\n")

        f.write("**Per-Subject Results**:\n\n")
        f.write("| Test Subject | Accuracy | Kappa | Status |\n")
        f.write("|--------------|----------|-------|--------|\n")
        for r in cross_results:
            status = "✅ Good" if r['accuracy']*100 >= 70 else ("⚠️ Fair" if r['accuracy']*100 >= 60 else "❌ Poor")
            f.write(f"| {r['subject_id']} | {r['accuracy']*100:.2f}% | {r['kappa']:.3f} | {status} |\n")

        f.write("\n---\n\n")

        f.write("## Comparison: Within vs Cross-Subject\n\n")
        f.write(f"| Metric | Within-Subject | Cross-Subject | Difference |\n")
        f.write(f"|--------|----------------|---------------|------------|\n")
        f.write(f"| Mean Accuracy | {within_mean:.2f}% | {cross_mean:.2f}% | {within_mean-cross_mean:.2f}% |\n")
        f.write(f"| Std Dev | {within_std:.2f}% | {cross_std:.2f}% | - |\n")
        f.write(f"| Median | {within_median:.2f}% | {cross_median:.2f}% | {within_median-cross_median:.2f}% |\n")

        if t_stat is not None:
            f.write(f"\n**Statistical Significance Test** (Independent t-test):\n")
            f.write(f"- t-statistic: {t_stat:.3f}\n")
            f.write(f"- p-value: {p_value:.4f}\n")
            if p_value < 0.05:
                f.write(f"- Result: **Significant difference** (p < 0.05)\n")
            else:
                f.write(f"- Result: No significant difference (p ≥ 0.05)\n")

        f.write("\n---\n\n")

        f.write("## Key Findings\n\n")
        f.write(f"1. **Within-Subject (Best Case)**: {within_mean:.1f}% average accuracy when training on each individual\n")
        f.write(f"   - This represents the performance when you collect your own data and train a personalized model\n\n")

        f.write(f"2. **Cross-Subject (Generalization)**: {cross_mean:.1f}% average accuracy on new users\n")
        f.write(f"   - This represents the performance when using a pre-trained model on new users without calibration\n\n")

        f.write(f"3. **Performance Gap**: {within_mean-cross_mean:.1f}% drop when applying to new users\n")
        f.write(f"   - This demonstrates the **subject-specificity** of BCI systems\n\n")

        f.write("4. **Practical Implications**:\n")
        if cross_mean >= 70:
            f.write("   - ✅ Cross-subject model works reasonably well (>70%)\n")
            f.write("   - ✅ Could be used as starting point for new users\n")
        elif cross_mean >= 60:
            f.write("   - ⚠️ Cross-subject model shows limited generalization (60-70%)\n")
            f.write("   - ⚠️ Short calibration session (10-20 trials) recommended for new users\n")
        else:
            f.write("   - ❌ Cross-subject model shows poor generalization (<60%)\n")
            f.write("   - ❌ Full calibration session (50-100 trials) REQUIRED for each new user\n")

        f.write("\n---\n\n")

        f.write("## Recommendations\n\n")
        f.write("### For Your ADS1299 Hardware\n\n")
        f.write("**Deployment Strategy**:\n")
        f.write("1. **For yourself** (after collecting calibration data):\n")
        f.write(f"   - Expected accuracy: ~{within_mean:.0f}% (based on within-subject average)\n")
        f.write("   - Calibration needed: 50-100 trials (15-20 minutes)\n\n")

        f.write("2. **For other users** (using your pre-trained model):\n")
        f.write(f"   - Expected accuracy: ~{cross_mean:.0f}% (without calibration)\n")
        if cross_mean >= 65:
            f.write("   - Quick calibration: 10-20 trials (5 minutes) to fine-tune\n")
        else:
            f.write("   - Full calibration: 50-100 trials (15-20 minutes) required\n")

        f.write("\n### For Academic Paper\n\n")
        f.write(f"**Recommended Reporting**:\n\n")
        f.write(f'"We evaluated our 8-channel BCI system on 10 subjects from the PhysioNet EEGBCI dataset. ')
        f.write(f'Within-subject evaluation (5-fold CV) achieved {within_mean:.2f}% ± {within_std:.2f}% accuracy, ')
        f.write(f'with {sum(1 for x in within_acc if x >= 70)}/{len(within_acc)} subjects exceeding 70%. ')
        f.write(f'Cross-subject evaluation (training on 7 subjects, testing on 3 held-out subjects) ')
        f.write(f'achieved {cross_mean:.2f}% ± {cross_std:.2f}% accuracy, demonstrating ')
        if cross_mean >= 65:
            f.write('reasonable generalization capability."\n\n')
        else:
            f.write('the subject-specific nature of BCI systems, consistent with prior literature."\n\n')

    print(f"[SAVED] {report_path}")


def main():
    ap = argparse.ArgumentParser(description="Train/test split evaluation")
    ap.add_argument("--datadir", type=str, default="data/raw")
    ap.add_argument("--output", type=str, default="outputs_train_test_split")
    ap.add_argument("--train-subjects", nargs="+", type=int, default=list(range(1, 8)))
    ap.add_argument("--test-subjects", nargs="+", type=int, default=[8, 9, 10])

    args = ap.parse_args()

    output_dir = Path(ensure_dir(args.output))

    print("\n" + "="*80)
    print("PROPER TRAIN/TEST SPLIT EVALUATION")
    print("="*80)
    print(f"Training subjects: {args.train_subjects}")
    print(f"Test subjects: {args.test_subjects}")
    print("="*80)

    # Within-subject evaluation
    all_subjects = args.train_subjects + args.test_subjects
    within_results = within_subject_evaluation(all_subjects, args.datadir, CHANNEL_CONFIG_8CH)

    # Cross-subject evaluation
    cross_results, global_model = cross_subject_evaluation(
        args.train_subjects, args.test_subjects, args.datadir, CHANNEL_CONFIG_8CH
    )

    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    create_comprehensive_visualizations(within_results, cross_results, output_dir)

    # Statistical report
    print("\n" + "="*80)
    print("GENERATING STATISTICAL REPORT")
    print("="*80)
    generate_statistical_report(within_results, cross_results, global_model, output_dir)

    # Final summary
    within_acc = [r['accuracy']*100 for r in within_results]
    cross_acc = [r['accuracy']*100 for r in cross_results]

    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Within-Subject:  {np.mean(within_acc):.2f}% ± {np.std(within_acc):.2f}%")
    print(f"Cross-Subject:   {np.mean(cross_acc):.2f}% ± {np.std(cross_acc):.2f}%")
    print(f"Performance Gap: {np.mean(within_acc) - np.mean(cross_acc):.2f}%")
    print("="*80)


if __name__ == "__main__":
    main()