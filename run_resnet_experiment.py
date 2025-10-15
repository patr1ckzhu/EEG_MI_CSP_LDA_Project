"""
Convenience script to run ResNet-1D experiments

Usage examples:
--------------
# Quick test with 10 subjects
python run_resnet_experiment.py --quick

# Full 100 subjects (for Windows PC with RTX 5080)
python run_resnet_experiment.py --full

# Custom subjects
python run_resnet_experiment.py --subjects 1 2 3 4 5 --epochs 100

# Use larger model
python run_resnet_experiment.py --quick --large_model
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd):
    """Run command and print output in real-time"""
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='')

    process.wait()
    return process.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run ResNet-1D Cross-Subject Experiments"
    )

    # Preset configurations
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with subjects 1-10")
    parser.add_argument("--full", action="store_true",
                       help="Full experiment with all 100 subjects")

    # Custom configuration
    parser.add_argument("--subjects", nargs="+", type=int,
                       help="Custom subject list (e.g., 1 2 3 4 5)")
    parser.add_argument("--epochs", type=int, default=150,
                       help="Number of training epochs (default: 150)")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size (default: 128 for RTX 5080)")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--large_model", action="store_true",
                       help="Use larger ResNet1D model (~2M params)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (auto-generated if not specified)")

    args = parser.parse_args()

    # Determine subject list
    if args.quick:
        subjects = list(range(1, 11))
        output_dir = "outputs_resnet_quick_test"
    elif args.full:
        subjects = list(range(1, 101))
        output_dir = "outputs_resnet_full_100subjects"
    elif args.subjects:
        subjects = args.subjects
        output_dir = f"outputs_resnet_{len(subjects)}subjects"
    else:
        print("Error: Please specify --quick, --full, or --subjects")
        parser.print_help()
        sys.exit(1)

    # Override output if specified
    if args.output:
        output_dir = args.output

    # Build command
    cmd = [
        sys.executable,  # Use current Python interpreter
        "src/train_resnet_cross_subject.py",
        "--subjects"] + [str(s) for s in subjects] + [
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--output", output_dir,
        "--config", "8-channel-motor"
    ]

    if args.large_model:
        cmd.append("--large_model")

    # Print configuration
    print("\n" + "="*70)
    print("ResNet-1D Cross-Subject Experiment Configuration")
    print("="*70)
    print(f"Subjects: {subjects[:5]}{'...' if len(subjects) > 5 else ''} (total: {len(subjects)})")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Model: {'ResNet1D-Large (~2M params)' if args.large_model else 'ResNet1D (~1.2M params)'}")
    print(f"Output: {output_dir}")
    print("="*70)

    # Run training
    returncode = run_command(cmd)

    if returncode == 0:
        print("\n" + "="*70)
        print("✓ Training completed successfully!")
        print("="*70)
        print(f"\nResults saved to: {output_dir}/")
        print(f"  - loso_results.json")
        print(f"  - LOSO_REPORT.md")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("✗ Training failed!")
        print("="*70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()