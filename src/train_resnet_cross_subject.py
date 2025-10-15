"""
Cross-Subject ResNet-1D Training with Leave-One-Subject-Out (LOSO)

This script trains a ResNet-1D model for EEG motor imagery classification.
Uses LOSO cross-validation for proper cross-subject generalization evaluation.

Key optimizations for RTX 5080 GPU:
- Mixed precision training (torch.cuda.amp)
- Larger batch size (128 vs 32)
- pin_memory=True and num_workers=4 in DataLoader
- Expected 3-5x speedup compared to baseline EEGNet training
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent))

from models.resnet1d import ResNet1D, ResNet1DLarge
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
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU (training will be slower)")
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
    """
    Load and concatenate data from multiple subjects

    Returns:
    --------
    X : np.ndarray, shape (n_total_trials, n_channels, n_timepoints)
    y : np.ndarray, shape (n_total_trials,)
    subject_labels : np.ndarray, shape (n_total_trials,)
        Subject ID for each trial
    """
    X_all = []
    y_all = []
    subject_labels = []

    for subject_id in subject_ids:
        try:
            X, y = load_subject_data(subject_id, data_dir, config, band, tmin, tmax)

            # Normalize per subject to reduce inter-subject variability
            if normalize:
                X = (X - X.mean()) / (X.std() + 1e-8)

            X_all.append(X)
            y_all.append(y)
            subject_labels.extend([subject_id] * len(y))

            print(f"  Subject {subject_id:02d}: {len(y)} trials")

        except Exception as e:
            print(f"  ✗ Subject {subject_id:02d}: Failed ({e})")
            continue

    if not X_all:
        raise RuntimeError("No subjects loaded successfully!")

    X_concat = np.concatenate(X_all, axis=0)
    y_concat = np.concatenate(y_all, axis=0)
    subject_labels = np.array(subject_labels)

    return X_concat, y_concat, subject_labels


def train_model(model, train_loader, criterion, optimizer, device, epochs=100,
               val_loader=None, patience=20, verbose=True, use_amp=True):
    """Train ResNet-1D model with early stopping and mixed precision"""

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # Mixed precision training for GPU speedup (critical for performance!)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Mixed precision forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        # Validation
        if val_loader is not None:
            val_acc, val_loss = evaluate_model(model, val_loader, criterion, device)
            scheduler.step(val_acc)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d}: "
                      f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                      f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f} Acc={train_acc:.4f}")

    return model


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on given dataloader"""
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            _, predicted = torch.max(outputs.data, 1)

            total_loss += loss.item()
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy, avg_loss


def leave_one_subject_out_cv(all_subjects, data_dir, config, n_channels, n_timepoints,
                             device, epochs=100, batch_size=128, learning_rate=0.001,
                             band=(8, 30), tmin=1.0, tmax=3.0, use_large_model=False,
                             verbose=True):
    """
    Leave-One-Subject-Out Cross-Validation

    For each subject:
    - Train on all other subjects
    - Test on the left-out subject
    """

    results = []

    print("\n" + "="*70)
    print("LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION (ResNet-1D)")
    print("="*70)
    print(f"Total subjects: {len(all_subjects)}")
    print(f"Training subjects per fold: {len(all_subjects)-1}")
    print(f"Batch size: {batch_size}")
    print("="*70)

    for test_subject in all_subjects:
        print(f"\n{'='*70}")
        print(f"FOLD: Testing on Subject {test_subject:02d}")
        print(f"{'='*70}")

        # Split subjects
        train_subjects = [s for s in all_subjects if s != test_subject]

        print(f"\nTrain subjects: {train_subjects}")
        print(f"Test subject: {test_subject}")

        # Load training data (from multiple subjects)
        print("\nLoading training data...")
        X_train, y_train, _ = load_multi_subject_data(
            train_subjects, data_dir, config, band, tmin, tmax, normalize=True
        )
        print(f"Training data: {X_train.shape}")

        # Load test data (single subject)
        print("\nLoading test data...")
        X_test, y_test = load_subject_data(test_subject, data_dir, config, band, tmin, tmax)
        # Normalize test data
        X_test = (X_test - X_test.mean()) / (X_test.std() + 1e-8)
        print(f"Test data: {X_test.shape}")

        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )

        # Optimize DataLoader for GPU training (CRITICAL for performance!)
        num_workers = 4 if device.type == 'cuda' else 0
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, drop_last=False,
                                 num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, pin_memory=True)

        # Initialize model
        ModelClass = ResNet1DLarge if use_large_model else ResNet1D
        model = ModelClass(n_channels=n_channels, n_timepoints=n_timepoints,
                          n_classes=2, dropout=0.5)
        model = model.to(device)

        if verbose:
            print(f"\nModel parameters: {model.get_num_parameters():,}")

        # Train
        print("\nTraining...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model = train_model(model, train_loader, criterion, optimizer, device,
                           epochs=epochs, val_loader=None, patience=30, verbose=verbose)

        # Test
        print("\nTesting...")
        test_acc, test_loss = evaluate_model(model, test_loader, criterion, device)

        print(f"\n{'='*70}")
        print(f"Subject {test_subject:02d} Test Accuracy: {test_acc*100:.2f}%")
        print(f"{'='*70}")

        results.append({
            'test_subject': test_subject,
            'train_subjects': train_subjects,
            'test_accuracy': test_acc,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        })

    # Summary
    test_accuracies = [r['test_accuracy'] for r in results]
    mean_acc = np.mean(test_accuracies)
    std_acc = np.std(test_accuracies)

    print("\n" + "="*70)
    print("LOSO CROSS-VALIDATION RESULTS")
    print("="*70)
    print(f"Mean Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Min Accuracy: {min(test_accuracies)*100:.2f}%")
    print(f"Max Accuracy: {max(test_accuracies)*100:.2f}%")
    print("\nPer-subject results:")
    for r in results:
        print(f"  Subject {r['test_subject']:02d}: {r['test_accuracy']*100:.2f}%")
    print("="*70)

    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'per_subject_results': results,
        'test_accuracies': test_accuracies
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Subject ResNet-1D with Leave-One-Subject-Out CV"
    )
    parser.add_argument("--subjects", nargs="+", type=int,
                       default=list(range(1, 11)),
                       help="Subject IDs (default: 1-10)")
    parser.add_argument("--datadir", type=str, default="data/raw")
    parser.add_argument("--config", type=str, default="8-channel-motor",
                       choices=list(CHANNEL_CONFIGS.keys()))
    parser.add_argument("--output", type=str, default="outputs_resnet_cross_subject")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size (default: 128 for RTX 5080)")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--large_model", action="store_true",
                       help="Use larger ResNet1D model (~2M params)")
    parser.add_argument("--band", nargs=2, type=float, default=[8., 30.])
    parser.add_argument("--tmin", type=float, default=1.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()

    output_dir = Path(ensure_dir(args.output))

    print("\n" + "="*70)
    print("CROSS-SUBJECT RESNET-1D TRAINING (LOSO)")
    print("="*70)
    print(f"Subjects: {args.subjects}")
    print(f"Configuration: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("="*70)

    # Get data dimensions
    config = CHANNEL_CONFIGS[args.config]
    X_sample, _ = load_subject_data(args.subjects[0], args.datadir, config,
                                    tuple(args.band), args.tmin, args.tmax)
    n_channels = X_sample.shape[1]
    n_timepoints = X_sample.shape[2]

    print(f"\nData dimensions: {n_channels} channels × {n_timepoints} timepoints")

    # Run LOSO CV
    results = leave_one_subject_out_cv(
        all_subjects=args.subjects,
        data_dir=args.datadir,
        config=config,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        band=tuple(args.band),
        tmin=args.tmin,
        tmax=args.tmax,
        use_large_model=args.large_model,
        verbose=True
    )

    # Save results
    save_data = {
        'model': 'ResNet1D-Large' if args.large_model else 'ResNet1D',
        'subjects': args.subjects,
        'config_name': args.config,
        'n_channels': n_channels,
        'n_timepoints': n_timepoints,
        'mean_accuracy': float(results['mean_accuracy']),
        'std_accuracy': float(results['std_accuracy']),
        'test_accuracies': [float(x) for x in results['test_accuracies']],
        'per_subject_results': results['per_subject_results'],
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'large_model': args.large_model
        },
        'timestamp': datetime.now().isoformat()
    }

    results_file = output_dir / "loso_results.json"
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n[SAVED] {results_file}")

    # Generate report
    report_file = output_dir / "LOSO_REPORT.md"
    with open(report_file, 'w') as f:
        f.write("# Cross-Subject ResNet-1D Results (LOSO)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Summary\n\n")
        model_name = 'ResNet1D-Large' if args.large_model else 'ResNet1D'
        f.write(f"- **Model**: {model_name}\n")
        f.write(f"- **Mean Accuracy**: {results['mean_accuracy']*100:.2f}% ± {results['std_accuracy']*100:.2f}%\n")
        f.write(f"- **Subjects**: {len(args.subjects)}\n")
        f.write(f"- **Configuration**: {args.config}\n")
        f.write(f"- **Batch size**: {args.batch_size}\n")
        f.write(f"- **Training samples per fold**: ~{(len(args.subjects)-1)*45}\n\n")

        f.write("---\n\n")

        f.write("## Per-Subject Results\n\n")
        f.write("| Subject | Accuracy | Status |\n")
        f.write("|---------|----------|--------|\n")

        for r in results['per_subject_results']:
            acc = r['test_accuracy'] * 100
            status = "✅" if acc >= 70 else "⚠️" if acc >= 60 else "❌"
            f.write(f"| {r['test_subject']:02d} | {acc:.2f}% | {status} |\n")

        f.write("\n---\n\n")
        f.write("## Comparison with Baselines\n\n")
        f.write("| Model | Mean Accuracy | Std |\n")
        f.write("|-------|---------------|-----|\n")
        f.write("| CSP+LDA (cross-subject) | 51.85% | ± 10.48% |\n")
        f.write("| EEGNet (cross-subject) | 60.62% | ± 11.28% |\n")
        f.write(f"| **ResNet-1D (this)** | **{results['mean_accuracy']*100:.2f}%** | **± {results['std_accuracy']*100:.2f}%** |\n\n")

        improvement_csp = (results['mean_accuracy'] - 0.5185) * 100
        improvement_eegnet = (results['mean_accuracy'] - 0.6062) * 100
        f.write(f"**Improvement over CSP+LDA**: {improvement_csp:+.2f}%\n\n")
        f.write(f"**Improvement over EEGNet**: {improvement_eegnet:+.2f}%\n")

    print(f"[SAVED] {report_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()