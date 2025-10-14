"""
Train CSP+LDA models with reduced channel counts (3/8/16 channels).

This script extracts specific channels from 64-channel EEGBCI data
to simulate training with low-cost hardware like ADS1299 (8-16 channels).

Key features:
- Multiple predefined channel configurations
- Automatic channel name matching (handles '.' suffixes)
- Performance comparison across configurations
- Electrode layout visualization for hardware deployment
- Export models for real hardware use
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
from mne.viz import plot_topomap

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    classification_report, cohen_kappa_score, roc_curve, auc
)

from utils import ensure_dir, edf_paths_in


# ============================================================================
# CHANNEL CONFIGURATIONS
# ============================================================================

CHANNEL_CONFIGS = {
    '3-channel': {
        'channels': ['C3', 'Cz', 'C4'],
        'description': 'Minimal configuration - core motor cortex',
        'csp_components': 2,
        'hardware': 'Minimum viable BCI (Graz-BCI standard)'
    },

    '8-channel-motor': {
        'channels': ['FC3', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4'],
        'description': 'ADS1299 optimal - motor imagery focus',
        'csp_components': 4,
        'hardware': 'ADS1299 (8-channel) - recommended layout'
    },

    '8-channel-extended': {
        'channels': ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CPz'],
        'description': 'Extended lateral coverage',
        'csp_components': 4,
        'hardware': 'ADS1299 (8-channel) - alternative layout'
    },

    '16-channel-full': {
        'channels': [
            'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CPz'
        ],
        'description': 'Full sensorimotor coverage',
        'csp_components': 6,
        'hardware': 'Dual ADS1299 or OpenBCI 16-channel'
    },

    '16-channel-compact': {
        'channels': [
            'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2'
        ],
        'description': 'Compact sensorimotor array',
        'csp_components': 6,
        'hardware': 'Dual ADS1299 or OpenBCI 16-channel'
    },
}


def normalize_channel_name(ch_name):
    """
    Normalize channel names to handle different formats.
    EEGBCI dataset uses names like 'Fc5.', 'Fc3.', etc.
    """
    # Remove trailing dots and convert to standard case
    ch_clean = ch_name.rstrip('.').upper()

    # Handle common variations
    ch_clean = ch_clean.replace('..', '')
    ch_clean = ch_clean.replace('Z', 'z')  # Cz, FCz, CPz

    return ch_clean


def find_matching_channels(available_channels, requested_channels):
    """
    Find matching channels between requested and available.
    Handles naming variations in EEGBCI dataset.
    """
    # Create normalized mapping
    channel_map = {}
    for avail_ch in available_channels:
        norm_name = normalize_channel_name(avail_ch)
        channel_map[norm_name] = avail_ch

    # Match requested channels
    matched = []
    missing = []

    for req_ch in requested_channels:
        norm_req = normalize_channel_name(req_ch)
        if norm_req in channel_map:
            matched.append(channel_map[norm_req])
        else:
            missing.append(req_ch)

    return matched, missing


def load_and_select_channels(edf_files, channel_config, sfreq=160):
    """
    Load EDF files and extract specific channels.
    """
    print(f"\n[Channel Selection]")
    print(f"Requested: {channel_config['channels']}")

    # Load data
    raws = []
    for f in edf_files:
        raw = read_raw_edf(f, preload=True, stim_channel=None, verbose=False)
        if int(raw.info['sfreq']) != sfreq:
            raw.resample(sfreq)
        raws.append(raw)
    raw = mne.concatenate_raws(raws)

    print(f"Original channels ({len(raw.ch_names)}): {raw.ch_names[:10]}...")

    # Find matching channels
    matched_channels, missing_channels = find_matching_channels(
        raw.ch_names,
        channel_config['channels']
    )

    if missing_channels:
        print(f"⚠️  Missing channels: {missing_channels}")

    if not matched_channels:
        raise ValueError(f"No matching channels found!")

    print(f"✓ Matched channels ({len(matched_channels)}): {matched_channels}")

    # Select only matched channels
    raw.pick_channels(matched_channels)

    return raw, matched_channels


def epoch_mi_left_right(raw, band=(8, 30), tmin=1.0, tmax=3.0):
    """Preprocess and epoch motor imagery data."""
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
        raise RuntimeError("Did not find both T1 and T2 events.")

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

    return X, y_binary, epochs


def train_reduced_channel_model(subject_id, edf_dir, config_name,
                                 band=(8, 30), tmin=1.0, tmax=3.0, cv=5):
    """
    Train CSP+LDA with reduced channels.
    """
    config = CHANNEL_CONFIGS[config_name]

    print(f"\n{'='*70}")
    print(f"Training: {config_name.upper()}")
    print(f"{'='*70}")
    print(f"Description: {config['description']}")
    print(f"Hardware: {config['hardware']}")
    print(f"CSP components: {config['csp_components']}")

    # Load data with channel selection
    subject_dir = Path(edf_dir) / f"subject-{subject_id:02d}"
    edf_files = edf_paths_in(str(subject_dir))

    if not edf_files:
        raise FileNotFoundError(f"No EDF files for subject {subject_id}")

    raw, matched_channels = load_and_select_channels(edf_files, config)
    X, y, epochs = epoch_mi_left_right(raw, band=band, tmin=tmin, tmax=tmax)

    print(f"\nData shape: {X.shape}")
    print(f"Trials: {len(y)} (T1={np.sum(y==0)}, T2={np.sum(y==1)})")

    # Cross-validation
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = []
    y_true_all, y_pred_all, y_proba_all = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_split.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # CSP + LDA
        csp = CSP(n_components=config['csp_components'],
                  reg='ledoit_wolf', log=True, norm_trace=False)
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

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

        print(f"  Fold {fold_idx}: {acc*100:.2f}%")

    # Metrics
    acc_mean = np.mean(scores)
    acc_std = np.std(scores)
    kappa = cohen_kappa_score(y_true_all, y_pred_all)

    fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
    roc_auc = auc(fpr, tpr)

    print(f"\n{'='*70}")
    print(f"Results: {config_name}")
    print(f"  Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
    print(f"  Cohen's Kappa: {kappa:.3f}")
    print(f"  ROC AUC: {roc_auc:.3f}")
    print(f"{'='*70}")

    # Classification report
    report = classification_report(y_true_all, y_pred_all,
                                   target_names=['T1(left)', 'T2(right)'],
                                   output_dict=True)

    return {
        'config_name': config_name,
        'config': config,
        'subject_id': subject_id,
        'n_channels': len(matched_channels),
        'matched_channels': matched_channels,
        'accuracy_mean': acc_mean,
        'accuracy_std': acc_std,
        'kappa': kappa,
        'roc_auc': roc_auc,
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_proba': y_proba_all,
        'classification_report': report,
        'cv_scores': scores
    }


def create_comparison_visualizations(all_results, output_dir):
    """Create comprehensive comparison plots."""
    output_dir = Path(output_dir)

    # 1. Accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    configs = [r['config_name'] for r in all_results]
    accuracies = [r['accuracy_mean']*100 for r in all_results]
    stds = [r['accuracy_std']*100 for r in all_results]
    n_channels = [r['n_channels'] for r in all_results]

    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#5f27cd']
    bars = ax.bar(range(len(configs)), accuracies, yerr=stds,
                   capsize=5, alpha=0.8, color=colors[:len(configs)])

    # Add value labels
    for i, (bar, acc, std, n_ch) in enumerate(zip(bars, accuracies, stds, n_channels)):
        ax.text(i, acc + std + 1, f'{acc:.1f}%\n({n_ch}ch)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.axhline(y=70, color='g', linestyle='--', label='Target (70%)', linewidth=2)
    ax.axhline(y=50, color='r', linestyle='--', label='Chance (50%)', linewidth=1)

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([c.replace('-', '\n') for c in configs], fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Performance Comparison: Reduced Channel Configurations', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'channel_comparison.png', dpi=150)
    print(f"[SAVED] {output_dir / 'channel_comparison.png'}")
    plt.close()

    # 2. Performance vs Channel Count
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by channel count
    sorted_results = sorted(all_results, key=lambda x: x['n_channels'])
    n_ch_sorted = [r['n_channels'] for r in sorted_results]
    acc_sorted = [r['accuracy_mean']*100 for r in sorted_results]
    std_sorted = [r['accuracy_std']*100 for r in sorted_results]
    names_sorted = [r['config_name'] for r in sorted_results]

    ax.errorbar(n_ch_sorted, acc_sorted, yerr=std_sorted,
                marker='o', markersize=10, linewidth=2, capsize=5, capthick=2)

    # Annotate points
    for n_ch, acc, name in zip(n_ch_sorted, acc_sorted, names_sorted):
        ax.annotate(name.split('-')[0], (n_ch, acc),
                   textcoords="offset points", xytext=(0,10),
                   ha='center', fontsize=9)

    ax.axhline(y=70, color='g', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Number of Channels', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Channel Count', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_channels.png', dpi=150)
    print(f"[SAVED] {output_dir / 'accuracy_vs_channels.png'}")
    plt.close()

    # 3. Confusion matrices comparison
    n_configs = len(all_results)
    fig, axes = plt.subplots(1, n_configs, figsize=(5*n_configs, 4))
    if n_configs == 1:
        axes = [axes]

    for idx, result in enumerate(all_results):
        cm = confusion_matrix(result['y_true'], result['y_pred'])
        disp = ConfusionMatrixDisplay(cm, display_labels=['T1(left)', 'T2(right)'])
        disp.plot(ax=axes[idx], colorbar=False, cmap='Blues')
        axes[idx].set_title(f"{result['config_name']}\n{result['n_channels']}ch, "
                           f"Acc={result['accuracy_mean']*100:.1f}%",
                           fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices_comparison.png', dpi=150)
    print(f"[SAVED] {output_dir / 'confusion_matrices_comparison.png'}")
    plt.close()


def create_electrode_layout_diagram(config_name, output_dir):
    """
    Create electrode placement diagram for hardware deployment.
    """
    config = CHANNEL_CONFIGS[config_name]
    channels = config['channels']

    # Create simple 2D layout
    fig, ax = plt.subplots(figsize=(10, 12))

    # Standard 10-20 positions (approximate 2D coordinates)
    # Format: (x, y) where x is left-right, y is front-back
    channel_positions = {
        'Fp1': (-0.3, 0.9), 'Fpz': (0, 0.9), 'Fp2': (0.3, 0.9),
        'F7': (-0.7, 0.7), 'F5': (-0.5, 0.7), 'F3': (-0.3, 0.7),
        'F1': (-0.15, 0.7), 'Fz': (0, 0.7), 'F2': (0.15, 0.7),
        'F4': (0.3, 0.7), 'F6': (0.5, 0.7), 'F8': (0.7, 0.7),
        'FC5': (-0.5, 0.5), 'FC3': (-0.3, 0.5), 'FC1': (-0.15, 0.5),
        'FCz': (0, 0.5), 'FC2': (0.15, 0.5), 'FC4': (0.3, 0.5), 'FC6': (0.5, 0.5),
        'T7': (-0.8, 0.3), 'C5': (-0.5, 0.3), 'C3': (-0.3, 0.3),
        'C1': (-0.15, 0.3), 'Cz': (0, 0.3), 'C2': (0.15, 0.3),
        'C4': (0.3, 0.3), 'C6': (0.5, 0.3), 'T8': (0.8, 0.3),
        'CP5': (-0.5, 0.1), 'CP3': (-0.3, 0.1), 'CP1': (-0.15, 0.1),
        'CPz': (0, 0.1), 'CP2': (0.15, 0.1), 'CP4': (0.3, 0.1), 'CP6': (0.5, 0.1),
        'P7': (-0.7, -0.1), 'P5': (-0.5, -0.1), 'P3': (-0.3, -0.1),
        'P1': (-0.15, -0.1), 'Pz': (0, -0.1), 'P2': (0.15, -0.1),
        'P4': (0.3, -0.1), 'P6': (0.5, -0.1), 'P8': (0.7, -0.1),
        'O1': (-0.3, -0.3), 'Oz': (0, -0.3), 'O2': (0.3, -0.3),
    }

    # Draw head outline
    head = plt.Circle((0, 0.3), 1.0, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(head)

    # Draw nose
    nose = plt.Polygon([[-0.1, 1.3], [0, 1.4], [0.1, 1.3]],
                      closed=True, facecolor='gray', edgecolor='black')
    ax.add_patch(nose)

    # Draw ears
    left_ear = plt.Circle((-1.0, 0.3), 0.1, facecolor='gray', edgecolor='black')
    right_ear = plt.Circle((1.0, 0.3), 0.1, facecolor='gray', edgecolor='black')
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)

    # Plot all standard positions (light gray)
    for ch_name, (x, y) in channel_positions.items():
        ax.plot(x, y, 'o', color='lightgray', markersize=8, alpha=0.5)
        ax.text(x, y-0.05, ch_name, ha='center', va='top',
               fontsize=7, color='gray', alpha=0.5)

    # Highlight selected channels
    for i, ch_name in enumerate(channels, 1):
        ch_norm = normalize_channel_name(ch_name)
        if ch_norm in channel_positions:
            x, y = channel_positions[ch_norm]
            ax.plot(x, y, 'o', color='red', markersize=20, alpha=0.7)
            ax.plot(x, y, 'o', color='yellow', markersize=15)
            ax.text(x, y, str(i), ha='center', va='center',
                   fontweight='bold', fontsize=12, color='black')

    # Add legend
    legend_text = f"{config_name.upper()}\n"
    legend_text += f"{config['description']}\n"
    legend_text += f"Hardware: {config['hardware']}\n\n"
    legend_text += "Channel mapping:\n"
    for i, ch in enumerate(channels, 1):
        legend_text += f"{i}. {ch}\n"

    ax.text(1.5, 0.3, legend_text, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           verticalalignment='center', family='monospace')

    ax.set_xlim([-1.3, 2.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Electrode Placement: {config_name}',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / f'electrode_layout_{config_name}.png', dpi=150, bbox_inches='tight')
    print(f"[SAVED] {output_dir / f'electrode_layout_{config_name}.png'}")
    plt.close()


def save_deployment_guide(all_results, output_dir):
    """Save hardware deployment guide."""
    output_dir = Path(output_dir)

    guide_path = output_dir / 'HARDWARE_DEPLOYMENT_GUIDE.md'

    with open(guide_path, 'w') as f:
        f.write("# Hardware Deployment Guide\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Performance Summary\n\n")
        f.write("| Configuration | Channels | Accuracy | Kappa | Hardware |\n")
        f.write("|---------------|----------|----------|-------|----------|\n")

        for result in sorted(all_results, key=lambda x: x['n_channels']):
            f.write(f"| {result['config_name']} | "
                   f"{result['n_channels']} | "
                   f"{result['accuracy_mean']*100:.2f}% ± {result['accuracy_std']*100:.2f}% | "
                   f"{result['kappa']:.3f} | "
                   f"{result['config']['hardware']} |\n")

        f.write("\n---\n\n")

        # Detailed configurations
        for result in all_results:
            config = result['config']
            f.write(f"## {result['config_name'].upper()}\n\n")
            f.write(f"**Description**: {config['description']}\n\n")
            f.write(f"**Hardware**: {config['hardware']}\n\n")
            f.write(f"**Performance**:\n")
            f.write(f"- Accuracy: {result['accuracy_mean']*100:.2f}% ± {result['accuracy_std']*100:.2f}%\n")
            f.write(f"- Cohen's Kappa: {result['kappa']:.3f}\n")
            f.write(f"- ROC AUC: {result['roc_auc']:.3f}\n")
            f.write(f"- Number of channels: {result['n_channels']}\n\n")

            f.write(f"**Channel Mapping** (for ADS1299):\n")
            f.write("```\n")
            for i, ch in enumerate(result['matched_channels'], 1):
                f.write(f"Channel {i}: {ch}\n")
            f.write("```\n\n")

            f.write(f"**CSP Parameters**:\n")
            f.write(f"- Number of components: {config['csp_components']}\n")
            f.write(f"- Regularization: Ledoit-Wolf\n")
            f.write(f"- Log transform: Yes\n\n")

            f.write(f"**Electrode Layout**: See `electrode_layout_{result['config_name']}.png`\n\n")
            f.write("---\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        best_8ch = max([r for r in all_results if r['n_channels'] == 8],
                       key=lambda x: x['accuracy_mean'], default=None)

        if best_8ch:
            f.write(f"### For ADS1299 (8-channel)\n\n")
            f.write(f"**Recommended**: `{best_8ch['config_name']}`\n")
            f.write(f"- Accuracy: {best_8ch['accuracy_mean']*100:.2f}%\n")
            f.write(f"- Configuration: {best_8ch['matched_channels']}\n\n")

        best_16ch = max([r for r in all_results if r['n_channels'] == 16],
                        key=lambda x: x['accuracy_mean'], default=None)

        if best_16ch:
            f.write(f"### For Dual ADS1299 (16-channel)\n\n")
            f.write(f"**Recommended**: `{best_16ch['config_name']}`\n")
            f.write(f"- Accuracy: {best_16ch['accuracy_mean']*100:.2f}%\n")
            f.write(f"- Configuration: See electrode layout diagram\n\n")

        f.write("---\n\n")
        f.write("## Next Steps\n\n")
        f.write("1. Choose configuration based on your hardware\n")
        f.write("2. Follow electrode placement diagram\n")
        f.write("3. Ensure proper skin preparation (impedance <10kΩ)\n")
        f.write("4. Verify signal quality before recording\n")
        f.write("5. Collect calibration data (50-100 trials per class)\n")
        f.write("6. Train model using this exact channel configuration\n")
        f.write("7. Test online classification performance\n\n")

    print(f"[SAVED] {guide_path}")


def main():
    ap = argparse.ArgumentParser(description="Train reduced-channel BCI models")
    ap.add_argument("--subject", type=int, required=True,
                    help="Subject ID")
    ap.add_argument("--datadir", type=str, default="data/raw",
                    help="Data directory")
    ap.add_argument("--configs", nargs="+",
                    choices=list(CHANNEL_CONFIGS.keys()) + ['all'],
                    default=['all'],
                    help="Channel configurations to test")
    ap.add_argument("--output", type=str, default="outputs_reduced_channels",
                    help="Output directory")
    ap.add_argument("--band", nargs=2, type=float, default=[8., 30.],
                    help="Frequency band")
    ap.add_argument("--tmin", type=float, default=1.0)
    ap.add_argument("--tmax", type=float, default=3.0)
    ap.add_argument("--cv", type=int, default=5)

    args = ap.parse_args()

    output_dir = Path(ensure_dir(args.output))

    # Determine which configs to run
    if 'all' in args.configs:
        configs_to_run = list(CHANNEL_CONFIGS.keys())
    else:
        configs_to_run = args.configs

    print("\n" + "="*70)
    print("REDUCED CHANNEL BCI TRAINING")
    print("="*70)
    print(f"Subject: {args.subject}")
    print(f"Configurations: {configs_to_run}")
    print(f"Frequency band: {args.band[0]}-{args.band[1]} Hz")
    print(f"Output: {output_dir}")
    print("="*70)

    # Train all configurations
    all_results = []

    for config_name in configs_to_run:
        try:
            result = train_reduced_channel_model(
                subject_id=args.subject,
                edf_dir=args.datadir,
                config_name=config_name,
                band=tuple(args.band),
                tmin=args.tmin,
                tmax=args.tmax,
                cv=args.cv
            )
            all_results.append(result)

            # Create electrode layout for this config
            create_electrode_layout_diagram(config_name, output_dir)

        except Exception as e:
            print(f"\n[ERROR] Failed to process {config_name}: {str(e)}")
            continue

    if not all_results:
        print("\n[ERROR] No configurations succeeded!")
        return

    # Create comparison visualizations
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON VISUALIZATIONS")
    print(f"{'='*70}")
    create_comparison_visualizations(all_results, output_dir)

    # Save deployment guide
    save_deployment_guide(all_results, output_dir)

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    for result in sorted(all_results, key=lambda x: -x['accuracy_mean']):
        print(f"{result['config_name']:20s} ({result['n_channels']:2d}ch): "
              f"{result['accuracy_mean']*100:5.2f}% ± {result['accuracy_std']*100:4.2f}%")

    best_result = max(all_results, key=lambda x: x['accuracy_mean'])
    print(f"\nBest configuration: {best_result['config_name']} "
          f"({best_result['accuracy_mean']*100:.2f}%)")

    print(f"\n{'='*70}")
    print("ALL FILES SAVED TO:", output_dir)
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
