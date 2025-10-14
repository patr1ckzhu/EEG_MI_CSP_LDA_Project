"""
Train EEGNet on motor imagery data with reduced channels (3/8/16)

This script trains EEGNet models compatible with ADS1299 hardware (8 channels)
and compares with traditional CSP+LDA approach.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

# Add parent directory to path
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
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU (training will be slower)")
    return device


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    y_proba = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            proba = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            total_loss += loss.item()
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_proba.extend(proba[:, 1].cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    return avg_loss, accuracy, kappa, y_true, y_pred, y_proba


def train_eegnet_cv(X, y, n_channels, n_timepoints, n_splits=5,
                   epochs=100, batch_size=16, learning_rate=0.001,
                   device='cpu', verbose=True, use_large_model=False):
    """
    Train EEGNet with cross-validation

    Parameters:
    -----------
    X : np.ndarray
        Shape (n_samples, n_channels, n_timepoints)
    y : np.ndarray
        Shape (n_samples,) with binary labels
    """
    cv_split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = []
    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_split.split(X, y), 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx}/{n_splits}")
            print(f"{'='*60}")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)

        # Create dataloaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                               shuffle=False)

        # Initialize model
        ModelClass = EEGNetLarge if use_large_model else EEGNet
        model = ModelClass(n_channels=n_channels,
                          n_timepoints=n_timepoints,
                          n_classes=2,
                          dropout_rate=0.5)
        model = model.to(device)

        if verbose and fold_idx == 1:
            print(f"Model parameters: {model.get_num_parameters():,}")

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

        # Training loop
        best_val_acc = 0
        patience_counter = 0
        early_stop_patience = 20

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader,
                                               criterion, optimizer, device)
            val_loss, val_acc, val_kappa, _, _, _ = evaluate(model, val_loader,
                                                             criterion, device)

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
                if patience_counter >= early_stop_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

        # Final evaluation
        _, final_acc, final_kappa, y_true, y_pred, y_proba = evaluate(
            model, val_loader, criterion, device
        )

        if verbose:
            print(f"Fold {fold_idx} Final: Acc={final_acc:.4f} Kappa={final_kappa:.4f}")

        fold_accuracies.append(final_acc)
        cv_results.append({
            'fold': fold_idx,
            'accuracy': final_acc,
            'kappa': final_kappa,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba
        })

    # Aggregate results
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Cross-Validation Results: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"{'='*60}")

    return {
        'cv_accuracy_mean': mean_acc,
        'cv_accuracy_std': std_acc,
        'fold_results': cv_results,
        'fold_accuracies': fold_accuracies
    }


def main():
    parser = argparse.ArgumentParser(description="Train EEGNet on motor imagery data")
    parser.add_argument("--subject", type=int, required=True,
                       help="Subject ID")
    parser.add_argument("--datadir", type=str, default="data/raw",
                       help="Data directory")
    parser.add_argument("--config", type=str, default="8-channel-motor",
                       choices=list(CHANNEL_CONFIGS.keys()),
                       help="Channel configuration")
    parser.add_argument("--output", type=str, default="outputs_eegnet",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--cv", type=int, default=5,
                       help="Number of cross-validation folds")
    parser.add_argument("--large_model", action="store_true",
                       help="Use larger EEGNet variant")
    parser.add_argument("--band", nargs=2, type=float, default=[8., 30.],
                       help="Frequency band")
    parser.add_argument("--tmin", type=float, default=1.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device()

    # Create output directory
    output_dir = Path(ensure_dir(args.output))
    subject_output = output_dir / f"subject_{args.subject:02d}"
    subject_output.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("EEGNET TRAINING")
    print("="*70)
    print(f"Subject: {args.subject}")
    print(f"Configuration: {args.config}")
    print(f"Output: {subject_output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("="*70)

    # Load data
    config = CHANNEL_CONFIGS[args.config]
    subject_dir = Path(args.datadir) / f"subject-{args.subject:02d}"
    edf_files = edf_paths_in(str(subject_dir))

    if not edf_files:
        raise FileNotFoundError(f"No EDF files for subject {args.subject}")

    raw, matched_channels = load_and_select_channels(edf_files, config)
    X, y, epochs_obj = epoch_mi_left_right(raw, band=tuple(args.band),
                                           tmin=args.tmin, tmax=args.tmax)

    print(f"\nData shape: {X.shape}")
    print(f"Channels: {matched_channels}")
    print(f"Trials: {len(y)} (T1={np.sum(y==0)}, T2={np.sum(y==1)})")

    n_channels = X.shape[1]
    n_timepoints = X.shape[2]

    # Train model
    print("\nStarting training...")
    results = train_eegnet_cv(
        X, y,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_splits=args.cv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        verbose=True,
        use_large_model=args.large_model
    )

    # Save results
    save_data = {
        'subject_id': args.subject,
        'config_name': args.config,
        'n_channels': n_channels,
        'matched_channels': matched_channels,
        'n_timepoints': n_timepoints,
        'accuracy_mean': float(results['cv_accuracy_mean']),
        'accuracy_std': float(results['cv_accuracy_std']),
        'fold_accuracies': [float(x) for x in results['fold_accuracies']],
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'large_model': args.large_model
        },
        'timestamp': datetime.now().isoformat()
    }

    results_file = subject_output / "eegnet_results.json"
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n[SAVED] {results_file}")

    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Subject {args.subject}: {results['cv_accuracy_mean']*100:.2f}% ± "
          f"{results['cv_accuracy_std']*100:.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()