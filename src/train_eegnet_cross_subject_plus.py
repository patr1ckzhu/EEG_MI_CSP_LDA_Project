"""
Cross-Subject EEGNet Training with Leave-One-Subject-Out (LOSO)
(Upgraded for better generalization, reduced overfitting, and large model support)

Key changes vs your original script:
- Adds validation split from the training subjects per fold (Stratified) to enable
  early stopping and ReduceLROnPlateau on *validation accuracy*.
- Adds weight decay, optional gradient clipping, and optional AMP (mixed precision)
  to stabilize training and reduce overfitting.
- Adds train-statistics normalization option (recommended for cross-subject),
  alongside the original per-subject normalization.
- Defaults to using EEGNetLarge (toggle with --large_model/--no-large_model).
- UTF-8 file writes to avoid Windows GBK emoji errors.
- Extra logging to verify GPU usage and model/device.

Usage example:
  python train_eegnet_cross_subject_plus.py \
      --subjects 1 2 ... 100 \
      --datadir data/raw \
      --config 8-channel-motor \
      --epochs 200 --batch_size 256 --lr 0.002 \
      --weight_decay 1e-4 --val_frac 0.15 --patience 25 \
      --norm train --amp --large_model

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
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append(str(Path(__file__).parent))

from models.eegnet import EEGNet, EEGNetLarge
from train_reduced_channels import (
    CHANNEL_CONFIGS,
    load_and_select_channels,
    epoch_mi_left_right,
)
from utils import ensure_dir, edf_paths_in


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device():
    """Automatically select best available device and log it"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
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
                            tmin=1.0, tmax=3.0, normalize="subject"):
    """
    Load and concatenate data from multiple subjects

    normalize:
        - 'subject': z-score within each subject separately (original behavior)
        - 'none'   : no normalization here
        (Note: train-statistics normalization is applied later if --norm train)

    Returns:
        X : (n_total_trials, n_channels, n_timepoints)
        y : (n_total_trials,)
        subject_labels : (n_total_trials,)
    """
    X_all, y_all, subject_labels = [], [], []

    for subject_id in subject_ids:
        try:
            X, y = load_subject_data(subject_id, data_dir, config, band, tmin, tmax)

            if normalize == "subject":
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


def evaluate_model(model, dataloader, criterion, device, amp=False):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    autocast = torch.cuda.amp.autocast if (amp and device.type == "cuda") else torch.cpu.amp.autocast
    with torch.no_grad():
        with autocast(enabled=amp and device.type == "cuda"):
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                _, predicted = torch.max(outputs.data, 1)
                total_loss += loss.item()
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

    avg_loss = total_loss / max(1, len(dataloader))
    acc = accuracy_score(y_true, y_pred)
    return acc, avg_loss


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                epochs=100, patience=20, amp=False, grad_clip=None):
    """Train with early stopping on validation accuracy and LR scheduler."""
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            if amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_acc = correct / max(1, total)
        train_loss = total_loss / max(1, len(train_loader))

        # Validation
        val_acc, val_loss = evaluate_model(model, val_loader, criterion, device, amp=amp)
        scheduler.step(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

        # Early stopping on best val acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (best val acc={best_val_acc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def loso_cv(all_subjects, data_dir, config, n_channels, n_timepoints, device,
            epochs=150, batch_size=128, learning_rate=0.002, weight_decay=1e-4,
            band=(8, 30), tmin=1.0, tmax=3.0, use_large_model=True,
            val_frac=0.15, patience=25, amp=False, grad_clip=None,
            norm_mode="train"):
    """
    Leave-One-Subject-Out Cross-Validation with validation split from training subjects.

    norm_mode:
        - 'subject': z-score each subject separately (train/test both using their own stats)
        - 'train'  : compute mean/std from *training data only* (across all training subjects)
                     and apply to both train and the left-out test subject (recommended)
    """
    results = []

    print("\n" + "="*70)
    print("LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION (Enhanced)")
    print("="*70)
    print(f"Total subjects: {len(all_subjects)}")
    print(f"Training subjects per fold: {len(all_subjects)-1}")
    print(f"Normalization mode: {norm_mode}")
    print("="*70)

    for test_subject in all_subjects:
        print(f"\n{'='*70}")
        print(f"FOLD: Testing on Subject {test_subject:02d}")
        print(f"{'='*70}")

        train_subjects = [s for s in all_subjects if s != test_subject]
        print(f"Train subjects: {train_subjects}")
        print(f"Test subject: {test_subject}")

        # Load training data (from multiple subjects)
        print("\nLoading training data...")
        # If norm_mode == 'train', we defer the actual normalization until after concatenation.
        X_train, y_train, _ = load_multi_subject_data(
            train_subjects, data_dir, config, band, tmin, tmax,
            normalize=("none" if norm_mode == "train" else "subject"),
        )
        print(f"Training data: {X_train.shape}")

        # Load test data (single subject)
        print("\nLoading test data...")
        X_test, y_test = load_subject_data(test_subject, data_dir, config, band, tmin, tmax)
        print(f"Test data (raw): {X_test.shape}")

        # Normalization
        if norm_mode == "train":
            # Compute statistics on training set only (channel-wise & time-aggregated)
            mu = X_train.mean(axis=(0, 2), keepdims=True)
            sigma = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
            X_train = (X_train - mu) / sigma
            X_test = (X_test - mu) / sigma
        else:
            # already subject-normalized inside loader for X_train; do subject z-score for test
            X_test = (X_test - X_test.mean()) / (X_test.std() + 1e-8)

        # Stratified validation split from training set
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
        (train_idx, val_idx), = sss.split(np.zeros_like(y_train), y_train)

        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
                                  batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader   = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
                                  batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
                                  batch_size=batch_size, shuffle=False)

        # Initialize model
        ModelClass = EEGNetLarge if use_large_model else EEGNet
        model = ModelClass(n_channels=n_channels, n_timepoints=n_timepoints,
                           n_classes=2, dropout_rate=0.5)
        model = model.to(device)
        print(model)
        print("Model device:", next(model.parameters()).device)
        # Optimizer and loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Train
        print("\nTraining...")
        model = train_model(
            model, train_loader, val_loader, criterion, optimizer, device,
            epochs=epochs, patience=patience, amp=amp, grad_clip=grad_clip,
        )

        # Test
        print("\nTesting...")
        test_acc, test_loss = evaluate_model(model, test_loader, criterion, device, amp=amp)

        print(f"\n{'='*70}")
        print(f"Subject {test_subject:02d} Test Accuracy: {test_acc*100:.2f}%")
        print(f"{'='*70}")

        results.append({
            "test_subject": int(test_subject),
            "train_subjects": list(map(int, train_subjects)),
            "test_accuracy": float(test_acc),
            "n_train_samples": int(len(X_train)),
            "n_val_samples": int(len(X_val)),
            "n_test_samples": int(len(X_test)),
        })

    test_accuracies = [r["test_accuracy"] for r in results]
    mean_acc = float(np.mean(test_accuracies))
    std_acc = float(np.std(test_accuracies))

    print("\n" + "="*70)
    print("LOSO CROSS-VALIDATION RESULTS (Enhanced)")
    print("="*70)
    print(f"Mean Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Min Accuracy: {min(test_accuracies)*100:.2f}%")
    print(f"Max Accuracy: {max(test_accuracies)*100:.2f}%")
    print("\nPer-subject results:")
    for r in results:
        print(f"  Subject {r['test_subject']:02d}: {r['test_accuracy']*100:.2f}%")
    print("="*70)

    return {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "per_subject_results": results,
        "test_accuracies": test_accuracies,
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-Subject EEGNet with LOSO (Enhanced)")
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 11)),
                        help="Subject IDs (default: 1-10)")
    parser.add_argument("--datadir", type=str, default="data/raw")
    parser.add_argument("--config", type=str, default="8-channel-motor",
                        choices=list(CHANNEL_CONFIGS.keys()))
    parser.add_argument("--output", type=str, default="outputs_eegnet_cross_subject_plus")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--large_model", action=argparse.BooleanOptionalAction, default=True,
                        help="Use EEGNetLarge (default: True). Pass --no-large_model to disable.")
    parser.add_argument("--band", nargs=2, type=float, default=[8., 30.])
    parser.add_argument("--tmin", type=float, default=1.0)
    parser.add_argument("--tmax", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction from training subjects")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience (epochs)")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (AMP) when on CUDA")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max grad-norm for clipping; 0 or <0 disables")
    parser.add_argument("--norm", type=str, choices=["subject", "train"], default="train",
                        help="Normalization mode: 'train' (recommended) or 'subject'")

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()

    output_dir = Path(ensure_dir(args.output))

    print("\n" + "="*70)
    print("CROSS-SUBJECT EEGNET TRAINING (LOSO, Enhanced)")
    print("="*70)
    print(f"Subjects: {args.subjects}")
    print(f"Configuration: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Large model: {args.large_model}")
    print(f"Normalization: {args.norm}")
    print(f"AMP: {args.amp}")
    print("="*70)

    # Get data dimensions robustly: find the first subject that loads
    config = CHANNEL_CONFIGS[args.config]
    X_sample = None
    for sid in args.subjects:
        try:
            X_tmp, _ = load_subject_data(sid, args.datadir, config, tuple(args.band), args.tmin, args.tmax)
            X_sample = X_tmp
            print(f"Sampled dimensions from subject {sid:02d}")
            break
        except Exception as e:
            print(f"Skip subject {sid:02d} for dimension probe: {e}")
            continue
    if X_sample is None:
        raise RuntimeError("Failed to probe data dimensions from any subject.")

    n_channels = X_sample.shape[1]
    n_timepoints = X_sample.shape[2]

    print(f"\nData dimensions: {n_channels} channels × {n_timepoints} timepoints")

    grad_clip = args.grad_clip if args.grad_clip and args.grad_clip > 0 else None

    # Run LOSO CV
    results = loso_cv(
        all_subjects=args.subjects,
        data_dir=args.datadir,
        config=config,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        band=tuple(args.band),
        tmin=args.tmin,
        tmax=args.tmax,
        use_large_model=args.large_model,
        val_frac=args.val_frac,
        patience=args.patience,
        amp=args.amp,
        grad_clip=grad_clip,
        norm_mode=args.norm,
    )

    # Save results (UTF-8)
    save_data = {
        "subjects": args.subjects,
        "config_name": args.config,
        "n_channels": int(n_channels),
        "n_timepoints": int(n_timepoints),
        "mean_accuracy": float(results["mean_accuracy"]),
        "std_accuracy": float(results["std_accuracy"]),
        "test_accuracies": [float(x) for x in results["test_accuracies"]],
        "per_subject_results": results["per_subject_results"],
        "hyperparameters": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "large_model": bool(args.large_model),
            "val_frac": float(args.val_frac),
            "patience": int(args.patience),
            "amp": bool(args.amp),
            "grad_clip": float(grad_clip) if grad_clip is not None else None,
            "norm": args.norm,
        },
        "timestamp": datetime.now().isoformat(),
    }

    results_file = output_dir / "loso_results_plus.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {results_file}")

    # Generate report (UTF-8; ASCII status to avoid emoji issues on GBK consoles)
    report_file = output_dir / "LOSO_REPORT_PLUS.md"
    with open(report_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Cross-Subject EEGNet Results (LOSO, Enhanced)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Mean Accuracy**: {results['mean_accuracy']*100:.2f}% ± {results['std_accuracy']*100:.2f}%\n")
        f.write(f"- **Subjects**: {len(args.subjects)}\n")
        f.write(f"- **Configuration**: {args.config}\n")
        f.write(f"- **Training samples per fold**: ~{(len(args.subjects)-1)*45}\n\n")
        f.write("---\n\n")
        f.write("## Per-Subject Results\n\n")
        f.write("| Subject | Accuracy | Status |\n")
        f.write("|---------|----------|--------|\n")
        for r in results["per_subject_results"]:
            acc = r["test_accuracy"] * 100
            status = "PASS" if acc >= 70 else "WARN" if acc >= 60 else "FAIL"
            f.write(f"| {r['test_subject']:02d} | {acc:.2f}% | {status} |\n")

    print(f"[SAVED] {report_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
