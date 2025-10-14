#!/usr/bin/env python3
"""
Full EEGNet Evaluation Pipeline (Cross-platform Python version)

This script runs the complete evaluation on all 10 subjects.
Works on Windows, macOS, and Linux.
"""

import subprocess
import sys
from pathlib import Path

def check_environment():
    """Check if PyTorch is installed and GPU is available"""
    print("="*70)
    print("Checking Environment")
    print("="*70)

    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            device = "CUDA"
        elif torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon GPU) available")
            device = "MPS"
        else:
            print("⚠️  Using CPU (training will be slower)")
            device = "CPU"

        return device
    except ImportError:
        print("❌ PyTorch not installed!")
        print("\nInstall with:")
        print("  pip install torch torchvision torchaudio")
        sys.exit(1)


def run_batch_evaluation(subjects, config="8-channel-motor", output_dir="outputs_eegnet",
                         epochs=100, batch_size=16, lr=0.001, cv=5, datadir="data/raw"):
    """Run batch evaluation on specified subjects"""

    print("\n" + "="*70)
    print("BATCH EEGNET EVALUATION")
    print("="*70)
    print(f"Subjects: {subjects}")
    print(f"Configuration: {config}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {epochs}")
    print("="*70)
    print()

    success_count = 0
    failed_subjects = []

    for subject_id in subjects:
        print(f"\n{'='*70}")
        print(f"Training Subject {subject_id:02d}")
        print(f"{'='*70}")

        cmd = [
            sys.executable,  # Use current Python interpreter
            "src/train_eegnet.py",
            "--subject", str(subject_id),
            "--config", config,
            "--output", output_dir,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(lr),
            "--cv", str(cv),
            "--datadir", datadir
        ]

        try:
            result = subprocess.run(cmd, check=True)
            success_count += 1
            print(f"✓ Subject {subject_id} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Subject {subject_id} failed with error code {e.returncode}")
            failed_subjects.append(subject_id)
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            break

    print(f"\n{'='*70}")
    print(f"Training completed: {success_count}/{len(subjects)} subjects")
    if failed_subjects:
        print(f"Failed subjects: {failed_subjects}")
    print(f"{'='*70}")

    return success_count, failed_subjects


def generate_comparison_report(eegnet_dir="outputs_eegnet", output_dir="outputs_comparison"):
    """Generate comparison report with CSP+LDA"""

    print("\n" + "="*70)
    print("GENERATING COMPARISON REPORT")
    print("="*70)

    cmd = [
        sys.executable,
        "src/compare_methods.py",
        "--eegnet_dir", eegnet_dir,
        "--output", output_dir
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✓ Comparison report generated")
    except subprocess.CalledProcessError:
        print("✗ Failed to generate comparison report")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch EEGNet evaluation")
    parser.add_argument("--subjects", nargs="+", type=int,
                       default=list(range(1, 11)),
                       help="Subject IDs (default: 1-10)")
    parser.add_argument("--config", type=str, default="8-channel-motor",
                       help="Channel configuration")
    parser.add_argument("--output", type=str, default="outputs_eegnet",
                       help="Output directory")
    parser.add_argument("--datadir", type=str, default="data/raw",
                       help="Data directory")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Maximum epochs per subject")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--cv", type=int, default=5,
                       help="Cross-validation folds")
    parser.add_argument("--skip_comparison", action="store_true",
                       help="Skip comparison report generation")

    args = parser.parse_args()

    # Check environment
    device = check_environment()

    # Run batch evaluation
    success_count, failed = run_batch_evaluation(
        subjects=args.subjects,
        config=args.config,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        cv=args.cv,
        datadir=args.datadir
    )

    # Generate comparison report
    if not args.skip_comparison and success_count > 0:
        generate_comparison_report(
            eegnet_dir=args.output,
            output_dir="outputs_comparison"
        )

    # Final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"Device used: {device}")
    print(f"Successful: {success_count}/{len(args.subjects)}")
    if failed:
        print(f"Failed subjects: {failed}")
    print(f"Results saved to: {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()