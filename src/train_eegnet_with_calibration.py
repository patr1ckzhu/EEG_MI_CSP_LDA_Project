"""
EEGNet with Transfer Learning and Few-shot Calibration

This script evaluates the effectiveness of transfer learning with calibration:
1. Train base model on N-1 subjects (LOSO)
2. For the held-out subject, use few trials for calibration
3. Test on remaining trials
4. Compare: no calibration vs 5/10/20 trials calibration
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

sys.path.append(str(Path(__file__).parent))

from models.eegnet import EEGNet, EEGNetLarge
from train_reduced_channels import (
    CHANNEL_CONFIGS,
    load_and_select_channels,
    epoch_mi_left_right
)
from utils import ensure_dir, edf_paths_in


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device():
    """Automatically select best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def load_subject_data(subject_id, data_dir, config, band=(8, 30), tmin=1.0, tmax=3.0):
    """Load data for a single subject"""
    subject_dir = Path(data_dir) / f"subject-{subject_id:02d}"
    edf_files = edf_paths_in(str(subject_dir))

    if not edf_files:
        raise FileNotFoundError(f"No EDF files for subject {subject_id}")

    raw, matched_channels = load_and_select_channels(edf_files, config)
    X, y, _ = epoch_mi_left_right(raw, band=band, tmin=tmin, tmax=tmax)

    return X, y


def load_multi_subject_data(subject_ids, data_dir, config, band=(8, 30),
                           tmin=1.0, tmax=3.0, normalize=True):
    """Load and concatenate data from multiple subjects"""
    X_all = []
    y_all = []

    for subject_id in subject_ids:
        try:
            X, y = load_subject_data(subject_id, data_dir, config, band, tmin, tmax)

            # Normalize per subject
            if normalize:
                X = (X - X.mean()) / (X.std() + 1e-8)

            X_all.append(X)
            y_all.append(y)

            print(f"  Subject {subject_id:02d}: {len(y)} trials")

        except Exception as e:
            print(f"  Failed Subject {subject_id:02d}: {e}")
            continue

    if not X_all:
        raise RuntimeError("No subjects loaded successfully")

    X_concat = np.concatenate(X_all, axis=0)
    y_concat = np.concatenate(y_all, axis=0)

    return X_concat, y_concat


def train_base_model(model, train_loader, criterion, optimizer, device, epochs=100):
    """Train base model on multi-subject data"""
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}")

    return model


def fine_tune_model(model, calib_loader, criterion, optimizer, device, epochs=20):
    """Fine-tune model with calibration data"""
    model.train()

    for epoch in range(epochs):
        for X_batch, y_batch in calib_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy"""
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def loso_with_calibration(all_subjects, data_dir, config, n_channels, n_timepoints,
                          device, base_epochs=100, calib_epochs=20, batch_size=32,
                          learning_rate=0.001, band=(8, 30), tmin=1.0, tmax=3.0,
                          use_large_model=False, calib_trials_list=[0, 5, 10, 20]):
    """
    Leave-One-Subject-Out with calibration evaluation

    For each test subject:
    1. Train base model on other subjects
    2. Test with different calibration strategies:
       - No calibration (baseline)
       - 5 trials calibration
       - 10 trials calibration
       - 20 trials calibration
    """

    results = []

    print("="*70)
    print("LOSO WITH CALIBRATION EVALUATION")
    print("="*70)
    print(f"Total subjects: {len(all_subjects)}")
    print(f"Calibration settings: {calib_trials_list} trials")
    print("="*70)

    for test_subject in all_subjects:
        print(f"\n{'='*70}")
        print(f"Testing Subject {test_subject:02d}")
        print(f"{'='*70}")

        # Split subjects
        train_subjects = [s for s in all_subjects if s != test_subject]

        print(f"\nTrain subjects: {train_subjects}")
        print(f"Test subject: {test_subject}")

        # Load training data
        print("\nLoading training data...")
        X_train, y_train = load_multi_subject_data(
            train_subjects, data_dir, config, band, tmin, tmax, normalize=True
        )
        print(f"Training data: {X_train.shape}")

        # Load test subject data
        print("\nLoading test subject data...")
        X_test_full, y_test_full = load_subject_data(
            test_subject, data_dir, config, band, tmin, tmax
        )
        # Normalize
        X_test_full = (X_test_full - X_test_full.mean()) / (X_test_full.std() + 1e-8)
        print(f"Test subject data: {X_test_full.shape}")

        # Train base model
        print("\nTraining base model...")
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, drop_last=False)

        ModelClass = EEGNetLarge if use_large_model else EEGNet
        base_model = ModelClass(n_channels=n_channels, n_timepoints=n_timepoints,
                               n_classes=2, dropout_rate=0.5)
        base_model = base_model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(base_model.parameters(), lr=learning_rate)

        base_model = train_base_model(base_model, train_loader, criterion,
                                      optimizer, device, epochs=base_epochs)

        # Test with different calibration settings
        subject_results = {
            'subject': test_subject,
            'n_test_trials': len(y_test_full),
            'calibration_results': {}
        }

        for n_calib in calib_trials_list:
            print(f"\n--- Calibration: {n_calib} trials ---")

            # Split test data
            if n_calib == 0:
                # No calibration - use all data for testing
                X_test = X_test_full
                y_test = y_test_full
                model = deepcopy(base_model)
            else:
                # Use first n_calib trials for calibration
                X_calib = X_test_full[:n_calib]
                y_calib = y_test_full[:n_calib]
                X_test = X_test_full[n_calib:]
                y_test = y_test_full[n_calib:]

                print(f"Calibration set: {X_calib.shape}")
                print(f"Test set: {X_test.shape}")

                # Fine-tune model
                calib_dataset = TensorDataset(
                    torch.FloatTensor(X_calib),
                    torch.LongTensor(y_calib)
                )
                calib_loader = DataLoader(calib_dataset, batch_size=min(8, n_calib),
                                        shuffle=True)

                model = deepcopy(base_model)
                fine_tune_optimizer = optim.Adam(model.parameters(), lr=learning_rate*0.1)

                print("Fine-tuning...")
                model = fine_tune_model(model, calib_loader, criterion,
                                       fine_tune_optimizer, device, epochs=calib_epochs)

            # Evaluate
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test),
                torch.LongTensor(y_test)
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            accuracy = evaluate_model(model, test_loader, device)
            print(f"Accuracy: {accuracy*100:.2f}%")

            subject_results['calibration_results'][n_calib] = {
                'accuracy': accuracy,
                'n_test_trials': len(y_test)
            }

        results.append(subject_results)

        print(f"\n{'='*70}")
        print(f"Subject {test_subject:02d} Summary:")
        for n_calib in calib_trials_list:
            acc = subject_results['calibration_results'][n_calib]['accuracy']
            print(f"  {n_calib:2d} calib trials: {acc*100:.2f}%")
        print(f"{'='*70}")

    # Summary statistics
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)

    summary = {}
    for n_calib in calib_trials_list:
        accuracies = [r['calibration_results'][n_calib]['accuracy'] for r in results]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        summary[n_calib] = {
            'mean': mean_acc,
            'std': std_acc,
            'min': np.min(accuracies),
            'max': np.max(accuracies)
        }

        print(f"\nCalibration: {n_calib} trials")
        print(f"  Mean Accuracy: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
        print(f"  Range: [{np.min(accuracies)*100:.2f}%, {np.max(accuracies)*100:.2f}%]")

    # Improvement analysis
    baseline_acc = summary[0]['mean']
    print(f"\n{'='*70}")
    print("IMPROVEMENT OVER BASELINE")
    print("="*70)
    for n_calib in calib_trials_list[1:]:
        improvement = (summary[n_calib]['mean'] - baseline_acc) * 100
        print(f"{n_calib:2d} trials: {improvement:+.2f}%")

    print("="*70)

    return {
        'per_subject_results': results,
        'summary': summary
    }


def main():
    parser = argparse.ArgumentParser(
        description="EEGNet with Transfer Learning and Calibration"
    )
    parser.add_argument("--subjects", nargs="+", type=int,
                       default=list(range(1, 11)),
                       help="Subject IDs (default: 1-10)")
    parser.add_argument("--datadir", type=str, default="data/raw")
    parser.add_argument("--config", type=str, default="8-channel-motor",
                       choices=list(CHANNEL_CONFIGS.keys()))
    parser.add_argument("--output", type=str, default="outputs_eegnet_calibration")
    parser.add_argument("--base_epochs", type=int, default=150,
                       help="Epochs for base model training")
    parser.add_argument("--calib_epochs", type=int, default=20,
                       help="Epochs for calibration fine-tuning")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--large_model", action="store_true")
    parser.add_argument("--band", nargs=2, type=float, default=[8., 30.])
    parser.add_argument("--tmin", type=float, default=1.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--calib_trials", nargs="+", type=int,
                       default=[0, 5, 10, 20],
                       help="Number of calibration trials to test")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()

    output_dir = Path(ensure_dir(args.output))

    print("\n" + "="*70)
    print("EEGNET WITH CALIBRATION EVALUATION")
    print("="*70)
    print(f"Subjects: {args.subjects}")
    print(f"Configuration: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Base epochs: {args.base_epochs}")
    print(f"Calibration epochs: {args.calib_epochs}")
    print(f"Calibration trials: {args.calib_trials}")
    print("="*70)

    # Get data dimensions
    config = CHANNEL_CONFIGS[args.config]
    X_sample, _ = load_subject_data(args.subjects[0], args.datadir, config,
                                    tuple(args.band), args.tmin, args.tmax)
    n_channels = X_sample.shape[1]
    n_timepoints = X_sample.shape[2]

    print(f"\nData dimensions: {n_channels} channels x {n_timepoints} timepoints")

    # Run LOSO with calibration
    results = loso_with_calibration(
        all_subjects=args.subjects,
        data_dir=args.datadir,
        config=config,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        device=device,
        base_epochs=args.base_epochs,
        calib_epochs=args.calib_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        band=tuple(args.band),
        tmin=args.tmin,
        tmax=args.tmax,
        use_large_model=args.large_model,
        calib_trials_list=args.calib_trials
    )

    # Save results
    save_data = {
        'subjects': args.subjects,
        'config_name': args.config,
        'n_channels': n_channels,
        'n_timepoints': n_timepoints,
        'calibration_trials': args.calib_trials,
        'summary': {
            str(k): {
                'mean': float(v['mean']),
                'std': float(v['std']),
                'min': float(v['min']),
                'max': float(v['max'])
            }
            for k, v in results['summary'].items()
        },
        'per_subject_results': results['per_subject_results'],
        'hyperparameters': {
            'base_epochs': args.base_epochs,
            'calib_epochs': args.calib_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'large_model': args.large_model
        },
        'timestamp': datetime.now().isoformat()
    }

    results_file = output_dir / "calibration_results.json"
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nSaved: {results_file}")

    # Generate report
    report_file = output_dir / "CALIBRATION_REPORT.md"
    with open(report_file, 'w') as f:
        f.write("# EEGNet Transfer Learning with Calibration Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Subjects: {len(args.subjects)}\n")
        f.write(f"- Configuration: {args.config}\n")
        f.write(f"- Base model epochs: {args.base_epochs}\n")
        f.write(f"- Calibration epochs: {args.calib_epochs}\n\n")

        f.write("---\n\n")

        f.write("## Results by Calibration Trials\n\n")
        f.write("| Calibration Trials | Mean Accuracy | Std Dev | Min | Max |\n")
        f.write("|-------------------|---------------|---------|-----|-----|\n")

        for n_calib in args.calib_trials:
            stats = results['summary'][n_calib]
            f.write(f"| {n_calib} | {stats['mean']*100:.2f}% | "
                   f"{stats['std']*100:.2f}% | {stats['min']*100:.2f}% | "
                   f"{stats['max']*100:.2f}% |\n")

        f.write("\n---\n\n")

        f.write("## Improvement Analysis\n\n")
        baseline = results['summary'][0]['mean']
        f.write(f"Baseline (no calibration): {baseline*100:.2f}%\n\n")
        f.write("| Calibration Trials | Improvement |\n")
        f.write("|-------------------|-------------|\n")

        for n_calib in args.calib_trials[1:]:
            improvement = (results['summary'][n_calib]['mean'] - baseline) * 100
            f.write(f"| {n_calib} | {improvement:+.2f}% |\n")

        f.write("\n---\n\n")

        f.write("## Per-Subject Results\n\n")
        f.write("| Subject | 0 trials | 5 trials | 10 trials | 20 trials |\n")
        f.write("|---------|----------|----------|-----------|----------|\n")

        for subj_result in results['per_subject_results']:
            subj_id = subj_result['subject']
            accs = [subj_result['calibration_results'][n]['accuracy']*100
                   for n in args.calib_trials]
            f.write(f"| {subj_id:02d} | " + " | ".join([f"{a:.2f}%" for a in accs]) + " |\n")

    print(f"Saved: {report_file}")
    print("\nDone")


if __name__ == "__main__":
    main()
