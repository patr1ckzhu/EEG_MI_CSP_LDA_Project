"""
Compare EEGNet vs CSP+LDA performance across all subjects

This script loads results from both methods and generates comparative visualizations
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils import ensure_dir


def load_eegnet_results(eegnet_dir):
    """Load EEGNet results for all subjects"""
    eegnet_dir = Path(eegnet_dir)
    summary_file = eegnet_dir / "eegnet_summary.json"

    if not summary_file.exists():
        print(f"ERROR: EEGNet summary not found at {summary_file}")
        print("Run evaluate_all_subjects_eegnet.py first!")
        return None

    with open(summary_file, 'r') as f:
        data = json.load(f)

    return data


def load_csp_lda_results(results_file):
    """
    Load CSP+LDA results from your existing evaluation

    Expected format: Text report or you can create a JSON from your existing results
    """
    # For now, manually input your CSP+LDA results from STATISTICAL_ANALYSIS_REPORT.md
    # You can automate this later

    csp_lda_results = {
        'mean_accuracy': 0.62,
        'std_accuracy': 0.1809,
        'median_accuracy': 0.6111,
        'individual_accuracies': {
            1: 0.7111,
            2: 0.8222,
            3: 0.3556,
            4: 0.6667,
            5: 0.4444,
            6: 0.5556,
            7: 0.9333,
            8: 0.4444,
            9: 0.4889,
            10: 0.7778
        }
    }

    return csp_lda_results


def create_comparison_plots(eegnet_data, csp_lda_data, output_dir):
    """Generate comprehensive comparison visualizations"""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Extract data
    eegnet_results = {r['subject_id']: r['accuracy_mean']
                     for r in eegnet_data['individual_results']}
    csp_lda_results = csp_lda_data['individual_accuracies']

    # Ensure same subjects
    common_subjects = sorted(set(eegnet_results.keys()) & set(csp_lda_results.keys()))

    eegnet_accs = [eegnet_results[s] for s in common_subjects]
    csp_lda_accs = [csp_lda_results[s] for s in common_subjects]

    # 1. Bar chart comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(common_subjects))
    width = 0.35

    bars1 = ax.bar(x - width/2, [a*100 for a in csp_lda_accs], width,
                   label='CSP+LDA', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, [a*100 for a in eegnet_accs], width,
                   label='EEGNet', color='#e74c3c', alpha=0.8)

    ax.axhline(y=70, color='green', linestyle='--', linewidth=2,
              label='Target (70%)', alpha=0.7)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1,
              label='Chance (50%)', alpha=0.5)

    ax.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: CSP+LDA vs EEGNet (8-channel)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s}' for s in common_subjects])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_barplot.png', dpi=150)
    print(f"[SAVED] {output_dir / 'comparison_barplot.png'}")
    plt.close()

    # 2. Scatter plot (CSP+LDA vs EEGNet)
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter([a*100 for a in csp_lda_accs],
              [a*100 for a in eegnet_accs],
              s=200, alpha=0.6, edgecolors='black', linewidth=2)

    # Annotate points
    for i, s in enumerate(common_subjects):
        ax.annotate(f'S{s}', (csp_lda_accs[i]*100, eegnet_accs[i]*100),
                   fontsize=10, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points')

    # Diagonal line (equal performance)
    ax.plot([0, 100], [0, 100], 'k--', linewidth=2, alpha=0.5,
           label='Equal performance')

    # Target lines
    ax.axhline(y=70, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=70, color='green', linestyle='--', alpha=0.5)

    ax.set_xlabel('CSP+LDA Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('EEGNet Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Subject-wise Comparison: CSP+LDA vs EEGNet',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([30, 100])
    ax.set_ylim([30, 100])
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_scatter.png', dpi=150)
    print(f"[SAVED] {output_dir / 'comparison_scatter.png'}")
    plt.close()

    # 3. Box plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = [
        [a*100 for a in csp_lda_accs],
        [a*100 for a in eegnet_accs]
    ]

    bp = ax.boxplot(data_to_plot, labels=['CSP+LDA', 'EEGNet'],
                   patch_artist=True, widths=0.6)

    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=70, color='green', linestyle='--', linewidth=2,
              label='Target (70%)', alpha=0.7)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1,
              label='Chance (50%)', alpha=0.5)

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Distribution: CSP+LDA vs EEGNet',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_boxplot.png', dpi=150)
    print(f"[SAVED] {output_dir / 'comparison_boxplot.png'}")
    plt.close()

    # 4. Improvement histogram
    improvements = [(eegnet_accs[i] - csp_lda_accs[i]) * 100
                   for i in range(len(common_subjects))]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors_bars = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.bar(range(len(common_subjects)), improvements,
                 color=colors_bars, alpha=0.7, edgecolor='black')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('Subject', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title('EEGNet Improvement over CSP+LDA (per subject)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(common_subjects)))
    ax.set_xticklabels([f'S{s}' for s in common_subjects])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        y_pos = val + (1 if val > 0 else -3)
        ax.text(i, y_pos, f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top',
               fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_improvement.png', dpi=150)
    print(f"[SAVED] {output_dir / 'comparison_improvement.png'}")
    plt.close()

    return improvements


def generate_comparison_report(eegnet_data, csp_lda_data, improvements, output_dir):
    """Generate detailed comparison report"""

    output_dir = Path(output_dir)
    report_file = output_dir / "COMPARISON_REPORT.md"

    eegnet_mean = eegnet_data['mean_accuracy'] * 100
    eegnet_std = eegnet_data['std_accuracy'] * 100
    csp_lda_mean = csp_lda_data['mean_accuracy'] * 100
    csp_lda_std = csp_lda_data['std_accuracy'] * 100

    mean_improvement = np.mean(improvements)
    positive_improvements = sum(1 for imp in improvements if imp > 0)

    with open(report_file, 'w') as f:
        f.write("# Method Comparison Report: CSP+LDA vs EEGNet\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Summary Statistics\n\n")
        f.write("| Method | Mean Accuracy | Std Dev | Median | Min | Max |\n")
        f.write("|--------|---------------|---------|--------|-----|-----|\n")
        f.write(f"| CSP+LDA | {csp_lda_mean:.2f}% | ±{csp_lda_std:.2f}% | "
                f"{csp_lda_data['median_accuracy']*100:.2f}% | - | - |\n")
        f.write(f"| EEGNet | {eegnet_mean:.2f}% | ±{eegnet_std:.2f}% | "
                f"{eegnet_data['median_accuracy']*100:.2f}% | "
                f"{eegnet_data['min_accuracy']*100:.2f}% | "
                f"{eegnet_data['max_accuracy']*100:.2f}% |\n\n")

        f.write("---\n\n")

        f.write("## Performance Comparison\n\n")
        f.write(f"- **Mean Improvement**: {mean_improvement:+.2f}%\n")
        f.write(f"- **Subjects Improved**: {positive_improvements}/{len(improvements)}\n\n")

        if mean_improvement > 5:
            f.write("✅ **Conclusion**: EEGNet shows significant improvement over CSP+LDA!\n\n")
        elif mean_improvement > 0:
            f.write("⚠️ **Conclusion**: EEGNet shows moderate improvement over CSP+LDA.\n\n")
        else:
            f.write("❌ **Conclusion**: EEGNet does not outperform CSP+LDA on average.\n\n")

        f.write("---\n\n")

        f.write("## Subjects Above 70% Threshold\n\n")
        f.write(f"- **CSP+LDA**: {sum(1 for acc in csp_lda_data['individual_accuracies'].values() if acc >= 0.70)}/10\n")
        f.write(f"- **EEGNet**: {eegnet_data['subjects_above_70']}/{len(eegnet_data['individual_results'])}\n\n")

        f.write("---\n\n")

        f.write("## Individual Subject Comparison\n\n")
        f.write("| Subject | CSP+LDA | EEGNet | Improvement | Better Method |\n")
        f.write("|---------|---------|--------|-------------|---------------|\n")

        eegnet_results = {r['subject_id']: r['accuracy_mean']
                         for r in eegnet_data['individual_results']}

        for subject_id in sorted(csp_lda_data['individual_accuracies'].keys()):
            csp_acc = csp_lda_data['individual_accuracies'][subject_id] * 100
            eeg_acc = eegnet_results.get(subject_id, 0) * 100
            diff = eeg_acc - csp_acc
            better = "EEGNet" if diff > 0 else "CSP+LDA" if diff < 0 else "Tie"

            f.write(f"| {subject_id} | {csp_acc:.2f}% | {eeg_acc:.2f}% | "
                   f"{diff:+.2f}% | {better} |\n")

        f.write("\n---\n\n")

        f.write("## Recommendations\n\n")
        f.write("### For Your ADS1299 Project\n\n")

        if eegnet_mean > csp_lda_mean:
            f.write("✅ **Use EEGNet** as your primary classification algorithm\n\n")
            f.write("Benefits:\n")
            f.write(f"- {mean_improvement:+.2f}% average accuracy improvement\n")
            f.write("- Better handling of individual variability\n")
            f.write("- Can benefit from data augmentation\n")
            f.write("- Potential for transfer learning\n\n")
        else:
            f.write("⚠️ **Consider CSP+LDA** as baseline\n\n")
            f.write("Reasons:\n")
            f.write("- Simpler and more interpretable\n")
            f.write("- Fewer hyperparameters\n")
            f.write("- Current data may be insufficient for deep learning\n\n")

        f.write("### Next Steps\n\n")
        f.write("1. **Data Augmentation**: Try augmenting data to improve EEGNet\n")
        f.write("2. **Hyperparameter Tuning**: Optimize learning rate, dropout, etc.\n")
        f.write("3. **Ensemble Methods**: Combine CSP+LDA and EEGNet predictions\n")
        f.write("4. **More Data**: Consider downloading BCI Competition IV datasets\n\n")

    print(f"[SAVED] {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare EEGNet vs CSP+LDA")
    parser.add_argument("--eegnet_dir", type=str, default="outputs_eegnet",
                       help="EEGNet results directory")
    parser.add_argument("--csp_lda_file", type=str, default=None,
                       help="CSP+LDA results file (optional)")
    parser.add_argument("--output", type=str, default="outputs_comparison",
                       help="Output directory")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(ensure_dir(args.output))

    print("\n" + "="*70)
    print("METHOD COMPARISON: CSP+LDA vs EEGNet")
    print("="*70)

    # Load results
    print("\nLoading EEGNet results...")
    eegnet_data = load_eegnet_results(args.eegnet_dir)

    if eegnet_data is None:
        return

    print(f"✓ Loaded {len(eegnet_data['individual_results'])} EEGNet results")

    print("\nLoading CSP+LDA results...")
    csp_lda_data = load_csp_lda_results(args.csp_lda_file)
    print(f"✓ Loaded {len(csp_lda_data['individual_accuracies'])} CSP+LDA results")

    # Generate plots
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)

    improvements = create_comparison_plots(eegnet_data, csp_lda_data, output_dir)

    # Generate report
    print("\n" + "="*70)
    print("GENERATING COMPARISON REPORT")
    print("="*70)

    generate_comparison_report(eegnet_data, csp_lda_data, improvements, output_dir)

    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"CSP+LDA: {csp_lda_data['mean_accuracy']*100:.2f}% ± {csp_lda_data['std_accuracy']*100:.2f}%")
    print(f"EEGNet:  {eegnet_data['mean_accuracy']*100:.2f}% ± {eegnet_data['std_accuracy']*100:.2f}%")
    print(f"Mean Improvement: {np.mean(improvements):+.2f}%")
    print("="*70)
    print(f"\nAll comparison files saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()