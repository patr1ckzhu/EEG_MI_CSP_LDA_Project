"""
Batch evaluation script for EEGNet on all 10 subjects

This script trains and evaluates EEGNet on all subjects sequentially,
then generates comparison reports with CSP+LDA baseline.
"""

import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
import numpy as np

from utils import ensure_dir


def run_subject_training(subject_id, config, output_dir, epochs=100,
                        batch_size=16, lr=0.001, cv=5, datadir="data/raw"):
    """Train EEGNet on a single subject"""

    cmd = [
        "python3", "src/train_eegnet.py",
        "--subject", str(subject_id),
        "--config", config,
        "--output", output_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--cv", str(cv),
        "--datadir", datadir
    ]

    print(f"\n{'='*70}")
    print(f"Training Subject {subject_id:02d}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR training subject {subject_id}: {e}")
        return False


def collect_results(output_dir, n_subjects=10):
    """Collect results from all subjects"""

    output_dir = Path(output_dir)
    all_results = []

    for subject_id in range(1, n_subjects + 1):
        results_file = output_dir / f"subject_{subject_id:02d}" / "eegnet_results.json"

        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                all_results.append(data)
                print(f"✓ Subject {subject_id:02d}: {data['accuracy_mean']*100:.2f}% "
                      f"± {data['accuracy_std']*100:.2f}%")
        else:
            print(f"✗ Subject {subject_id:02d}: No results found")
            all_results.append(None)

    return all_results


def generate_summary_report(all_results, output_dir):
    """Generate summary report comparing all subjects"""

    output_dir = Path(output_dir)

    # Filter valid results
    valid_results = [r for r in all_results if r is not None]

    if not valid_results:
        print("ERROR: No valid results found!")
        return

    # Calculate statistics
    accuracies = [r['accuracy_mean'] for r in valid_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    median_acc = np.median(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)

    # Count subjects above thresholds
    above_70 = sum(1 for acc in accuracies if acc >= 0.70)
    above_60 = sum(1 for acc in accuracies if acc >= 0.60)

    # Create report
    report_file = output_dir / "EEGNET_SUMMARY_REPORT.md"

    with open(report_file, 'w') as f:
        f.write("# EEGNet Evaluation Summary Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Overview\n\n")
        f.write(f"- Total subjects: {len(all_results)}\n")
        f.write(f"- Successfully trained: {len(valid_results)}\n")
        f.write(f"- Configuration: {valid_results[0]['config_name']}\n")
        f.write(f"- Channels: {valid_results[0]['n_channels']}\n")
        f.write(f"- Model: EEGNet\n\n")

        f.write("---\n\n")

        f.write("## Summary Statistics\n\n")
        f.write(f"- **Mean Accuracy**: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%\n")
        f.write(f"- **Median Accuracy**: {median_acc*100:.2f}%\n")
        f.write(f"- **Min Accuracy**: {min_acc*100:.2f}%\n")
        f.write(f"- **Max Accuracy**: {max_acc*100:.2f}%\n")
        f.write(f"- **Subjects ≥70%**: {above_70}/{len(valid_results)}\n")
        f.write(f"- **Subjects ≥60%**: {above_60}/{len(valid_results)}\n\n")

        f.write("---\n\n")

        f.write("## Per-Subject Results\n\n")
        f.write("| Subject | Accuracy | Std Dev | Status |\n")
        f.write("|---------|----------|---------|--------|\n")

        for result in valid_results:
            sid = result['subject_id']
            acc = result['accuracy_mean']
            std = result['accuracy_std']

            if acc >= 0.70:
                status = "✅ Good"
            elif acc >= 0.60:
                status = "⚠️ Fair"
            else:
                status = "❌ Poor"

            f.write(f"| {sid} | {acc*100:.2f}% | ±{std*100:.2f}% | {status} |\n")

        f.write("\n---\n\n")

        f.write("## Hyperparameters\n\n")
        f.write("```json\n")
        f.write(json.dumps(valid_results[0]['hyperparameters'], indent=2))
        f.write("\n```\n\n")

        f.write("---\n\n")

        f.write("## Performance Distribution\n\n")

        ranges = [
            ("Excellent (≥80%)", sum(1 for acc in accuracies if acc >= 0.80)),
            ("Good (70-79%)", sum(1 for acc in accuracies if 0.70 <= acc < 0.80)),
            ("Fair (60-69%)", sum(1 for acc in accuracies if 0.60 <= acc < 0.70)),
            ("Poor (<60%)", sum(1 for acc in accuracies if acc < 0.60))
        ]

        for range_name, count in ranges:
            percentage = count / len(valid_results) * 100
            f.write(f"- {range_name}: {count}/{len(valid_results)} ({percentage:.1f}%)\n")

        f.write("\n---\n\n")

        f.write("## Conclusion\n\n")

        if mean_acc >= 0.70:
            f.write("✅ **Excellent Performance**: EEGNet achieves clinical-grade accuracy!\n\n")
        elif mean_acc >= 0.60:
            f.write("⚠️ **Moderate Performance**: EEGNet shows improvement but needs optimization.\n\n")
        else:
            f.write("❌ **Limited Performance**: Consider data augmentation or more subjects.\n\n")

        f.write(f"The model achieved {mean_acc*100:.2f}% ± {std_acc*100:.2f}% accuracy across "
                f"{len(valid_results)} subjects, with {above_70}/{len(valid_results)} subjects "
                f"reaching the 70% clinical threshold.\n\n")

    print(f"\n[SAVED] {report_file}")

    # Save JSON summary
    summary_json = output_dir / "eegnet_summary.json"
    summary_data = {
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'median_accuracy': float(median_acc),
        'min_accuracy': float(min_acc),
        'max_accuracy': float(max_acc),
        'subjects_above_70': int(above_70),
        'subjects_above_60': int(above_60),
        'total_subjects': len(valid_results),
        'individual_results': valid_results,
        'timestamp': datetime.now().isoformat()
    }

    with open(summary_json, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"[SAVED] {summary_json}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate EEGNet on all subjects")
    parser.add_argument("--subjects", nargs="+", type=int,
                       default=list(range(1, 11)),
                       help="Subject IDs to evaluate (default: 1-10)")
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
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training, only generate reports")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(ensure_dir(args.output))

    print("\n" + "="*70)
    print("BATCH EEGNET EVALUATION")
    print("="*70)
    print(f"Subjects: {args.subjects}")
    print(f"Configuration: {args.config}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print("="*70)

    # Train all subjects
    if not args.skip_training:
        success_count = 0
        for subject_id in args.subjects:
            success = run_subject_training(
                subject_id=subject_id,
                config=args.config,
                output_dir=str(output_dir),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                cv=args.cv,
                datadir=args.datadir
            )
            if success:
                success_count += 1

        print(f"\n{'='*70}")
        print(f"Training completed: {success_count}/{len(args.subjects)} subjects")
        print(f"{'='*70}")

    # Collect and summarize results
    print("\n" + "="*70)
    print("COLLECTING RESULTS")
    print("="*70)

    all_results = collect_results(output_dir, n_subjects=max(args.subjects))

    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)

    generate_summary_report(all_results, output_dir)

    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"All results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()