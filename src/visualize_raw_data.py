"""
Visualize raw EEG data from EEGBCI dataset.

Features:
- Plot raw EEG time series
- Show power spectral density (PSD)
- Display topographic maps
- Visualize event-related potentials (ERPs)
- Compare left vs right hand trials
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import mne
from mne.io import read_raw_edf
from mne.channels import make_standard_montage
from mne.viz import plot_topomap

from utils import edf_paths_in


def visualize_raw_signal(raw, duration=10.0, n_channels=10, output_path=None):
    """Plot raw EEG signal."""
    fig = raw.plot(duration=duration, n_channels=n_channels, scalings='auto',
                   title='Raw EEG Signal', show=False)
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"[SAVED] {output_path}")
    else:
        plt.show()
    plt.close()


def visualize_psd(raw, fmin=1, fmax=50, output_path=None):
    """Plot power spectral density."""
    # Apply montage first
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore')

    fig = raw.compute_psd(fmin=fmin, fmax=fmax).plot(picks='eeg',
                                                       show=False,
                                                       average=True)
    fig.suptitle('Power Spectral Density (1-50 Hz)')
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"[SAVED] {output_path}")
    else:
        plt.show()
    plt.close()


def visualize_topomap(raw, band=(8, 30), output_path=None):
    """Plot topographic map of power in specific frequency band."""
    try:
        montage = make_standard_montage('standard_1005')
        raw.set_montage(montage, on_missing='warn')

        # Filter to band of interest
        raw_filtered = raw.copy().filter(l_freq=band[0], h_freq=band[1], verbose=False)

        # Compute power
        data = raw_filtered.get_data(picks='eeg')
        power = np.mean(data ** 2, axis=1)

        # Get channel positions
        info = raw_filtered.info
        picks = mne.pick_types(info, meg=False, eeg=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        im, _ = plot_topomap(power, info, axes=ax, show=False,
                             cmap='RdBu_r', vlim=(power.min(), power.max()))
        ax.set_title(f'Topographic Map - Power in {band[0]}-{band[1]} Hz')
        plt.colorbar(im, ax=ax, label='Power (μV²)')

        if output_path:
            fig.savefig(output_path, dpi=150)
            print(f"[SAVED] {output_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"[WARNING] Could not create topomap: {e}")
        print("[INFO] Skipping topomap visualization")


def visualize_events(raw, output_path=None):
    """Plot event markers."""
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'],
                               event_id=event_id, show=False)
    fig.suptitle('Event Markers (T1=Left hand, T2=Right hand)')

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"[SAVED] {output_path}")
    else:
        plt.show()
    plt.close()


def visualize_epochs(raw, tmin=0.0, tmax=3.0, output_path=None):
    """Plot individual epochs for left vs right hand."""
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore')

    # Filter
    raw.filter(l_freq=8., h_freq=30., picks='eeg', verbose=False)

    # Extract events
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    keep = {k: v for k, v in event_id.items() if k in ('T1', 'T2')}

    if len(keep) < 2:
        print("[WARNING] Not enough event types found")
        return

    picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')

    # Create epochs
    epochs = mne.Epochs(raw, events, event_id=keep,
                        tmin=tmin, tmax=tmax, proj=True, picks=picks,
                        baseline=None, preload=True, verbose=False)

    # Plot average epochs
    fig = epochs.plot_image(picks=['C3', 'Cz', 'C4'], show=False,
                            combine='mean', title='Average Epochs (Motor Cortex)')

    if output_path:
        for idx, f in enumerate(fig):
            path = Path(output_path).parent / f"{Path(output_path).stem}_{idx}.png"
            f.savefig(path, dpi=150)
            print(f"[SAVED] {path}")
        plt.close('all')
    else:
        plt.show()


def compare_left_right(raw, tmin=1.0, tmax=3.0, output_path=None):
    """Compare left vs right hand imagery in motor cortex."""
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore')

    # Filter
    raw.filter(l_freq=8., h_freq=30., picks='eeg', verbose=False)

    # Extract events
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    keep = {k: v for k, v in event_id.items() if k in ('T1', 'T2')}

    if len(keep) < 2:
        print("[WARNING] Not enough event types found")
        return

    picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')

    # Create epochs
    epochs = mne.Epochs(raw, events, event_id=keep,
                        tmin=tmin, tmax=tmax, proj=True, picks=picks,
                        baseline=(None, 0), preload=True, verbose=False)

    # Select motor cortex channels
    motor_channels = ['C3', 'C4']  # C3=right hand area, C4=left hand area

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, channel in enumerate(motor_channels):
        if channel not in epochs.ch_names:
            print(f"[WARNING] Channel {channel} not found")
            continue

        # Get data for each condition
        try:
            left_data = epochs['T1'].get_data(picks=channel).squeeze()
            right_data = epochs['T2'].get_data(picks=channel).squeeze()
        except KeyError:
            print(f"[WARNING] Could not extract epoch data")
            continue

        # Plot average
        times = epochs.times
        axes[idx].plot(times, left_data.mean(axis=0), 'b-', linewidth=2, label='Left hand (T1)')
        axes[idx].plot(times, right_data.mean(axis=0), 'r-', linewidth=2, label='Right hand (T2)')

        # Add confidence intervals
        left_sem = left_data.std(axis=0) / np.sqrt(left_data.shape[0])
        right_sem = right_data.std(axis=0) / np.sqrt(right_data.shape[0])

        axes[idx].fill_between(times,
                               left_data.mean(axis=0) - left_sem,
                               left_data.mean(axis=0) + left_sem,
                               alpha=0.3, color='b')
        axes[idx].fill_between(times,
                               right_data.mean(axis=0) - right_sem,
                               right_data.mean(axis=0) + right_sem,
                               alpha=0.3, color='r')

        axes[idx].axhline(y=0, color='k', linestyle='--', linewidth=1)
        axes[idx].axvline(x=0, color='k', linestyle='--', linewidth=1)
        axes[idx].set_xlabel('Time (s)', fontsize=12)
        axes[idx].set_ylabel('Amplitude (μV)', fontsize=12)
        axes[idx].set_title(f'Channel {channel}', fontsize=14)
        axes[idx].legend(loc='best')
        axes[idx].grid(alpha=0.3)

    plt.suptitle('Left vs Right Hand Motor Imagery (8-30 Hz)', fontsize=16)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"[SAVED] {output_path}")
    else:
        plt.show()
    plt.close()


def visualize_frequency_bands(raw, output_path=None):
    """Compare power in different frequency bands."""
    try:
        montage = make_standard_montage('standard_1005')
        raw.set_montage(montage, on_missing='warn')

        bands = {
            'Delta (1-4 Hz)': (1, 4),
            'Theta (4-8 Hz)': (4, 8),
            'Alpha (8-13 Hz)': (8, 13),
            'Beta (13-30 Hz)': (13, 30),
            'Gamma (30-50 Hz)': (30, 50)
        }

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for idx, (band_name, (fmin, fmax)) in enumerate(bands.items()):
            raw_band = raw.copy().filter(l_freq=fmin, h_freq=fmax, verbose=False)
            data = raw_band.get_data(picks='eeg')
            power = np.mean(data ** 2, axis=1)

            info = raw_band.info

            im, _ = plot_topomap(power, info, axes=axes[idx], show=False,
                                 cmap='hot', vlim=(power.min(), power.max()))
            axes[idx].set_title(band_name, fontsize=12, fontweight='bold')

        # Remove extra subplot
        fig.delaxes(axes[-1])

        plt.suptitle('Power Distribution Across Frequency Bands', fontsize=16)
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150)
            print(f"[SAVED] {output_path}")
        else:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"[WARNING] Could not create frequency band topomaps: {e}")
        print("[INFO] Skipping frequency band visualization")


def main():
    ap = argparse.ArgumentParser(description="Visualize raw EEG data")
    ap.add_argument("--subject", type=int, required=True,
                    help="Subject ID (e.g., 1, 2)")
    ap.add_argument("--datadir", type=str, default="data/raw",
                    help="Data directory")
    ap.add_argument("--output", type=str, default="outputs_visualization",
                    help="Output directory for plots")
    ap.add_argument("--all", action='store_true',
                    help="Generate all visualizations")

    args = ap.parse_args()

    # Load data
    subject_dir = Path(args.datadir) / f"subject-{args.subject:02d}"
    edf_files = edf_paths_in(str(subject_dir))

    if not edf_files:
        print(f"[ERROR] No EDF files found for subject {args.subject}")
        return

    print(f"\n{'='*60}")
    print(f"VISUALIZING SUBJECT {args.subject}")
    print(f"{'='*60}")
    print(f"Loading {len(edf_files)} EDF files...")

    # Load first run for visualization
    raw = read_raw_edf(edf_files[0], preload=True, stim_channel=None, verbose=False)

    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.1f} seconds")
    print(f"Channels: {len(raw.ch_names)} (EEG: {len(mne.pick_types(raw.info, eeg=True))})")
    print(f"{'='*60}\n")

    # Create output directory
    output_dir = Path(args.output) / f"subject-{args.subject:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        print("Generating all visualizations...\n")

        print("[1/7] Raw signal...")
        visualize_raw_signal(raw.copy(), duration=10.0, n_channels=10,
                            output_path=output_dir / "01_raw_signal.png")

        print("[2/7] Power spectral density...")
        visualize_psd(raw.copy(), fmin=1, fmax=50,
                     output_path=output_dir / "02_psd.png")

        print("[3/7] Topographic map (8-30 Hz)...")
        visualize_topomap(raw.copy(), band=(8, 30),
                         output_path=output_dir / "03_topomap.png")

        print("[4/7] Event markers...")
        visualize_events(raw.copy(),
                        output_path=output_dir / "04_events.png")

        print("[5/7] Individual epochs...")
        visualize_epochs(raw.copy(), tmin=0.0, tmax=3.0,
                        output_path=output_dir / "05_epochs")

        print("[6/7] Left vs right comparison...")
        compare_left_right(raw.copy(), tmin=1.0, tmax=3.0,
                          output_path=output_dir / "06_left_vs_right.png")

        print("[7/7] Frequency bands comparison...")
        visualize_frequency_bands(raw.copy(),
                                 output_path=output_dir / "07_frequency_bands.png")

        print(f"\n{'='*60}")
        print("VISUALIZATION COMPLETE!")
        print(f"{'='*60}")
        print(f"All plots saved to: {output_dir}")
    else:
        print("Use --all flag to generate all visualizations")
        print("Or call specific visualization functions from Python")


if __name__ == "__main__":
    main()