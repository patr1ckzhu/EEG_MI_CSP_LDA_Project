#!/usr/bin/env python3
"""
Download PhysioNet EEGBCI Dataset - All Subjects

This script downloads motor imagery data for all 109 subjects.
You can choose which subjects and runs to download.

Usage:
    # Download all 109 subjects (runs 4, 8, 12 for left/right hand MI)
    python download_all_subjects.py --subjects all

    # Download specific subjects
    python download_all_subjects.py --subjects 1 2 3 4 5

    # Download subjects 1-50
    python download_all_subjects.py --subjects 1-50

    # Download with different runs
    python download_all_subjects.py --subjects all --runs 4 8 12
"""

import argparse
import mne
from pathlib import Path
from tqdm import tqdm


def parse_subject_range(subjects_arg):
    """Parse subject argument (e.g., '1-50' or [1, 2, 3])"""
    if subjects_arg == ['all']:
        return list(range(1, 110))  # All 109 subjects

    subjects = []
    for arg in subjects_arg:
        if '-' in str(arg):
            # Range like "1-50"
            start, end = map(int, str(arg).split('-'))
            subjects.extend(range(start, end + 1))
        else:
            subjects.append(int(arg))

    return sorted(set(subjects))


def download_subjects(subjects, runs, data_dir="data/raw", verbose=True):
    """
    Download EEGBCI data for specified subjects and runs

    Parameters:
    -----------
    subjects : list of int
        Subject IDs to download (1-109)
    runs : list of int
        Run numbers to download
        - Runs 4, 8, 12: Motor imagery (left vs right hand)
        - Runs 6, 10, 14: Motor imagery (hands vs feet)
    data_dir : str
        Directory to save data
    """

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("PhysioNet EEGBCI Dataset Download")
    print("="*70)
    print(f"Subjects: {len(subjects)} ({min(subjects)} to {max(subjects)})")
    print(f"Runs: {runs}")
    print(f"Data directory: {data_path.absolute()}")
    print("="*70)
    print()

    # Download info
    total_files = len(subjects) * len(runs)
    print(f"Total files to download: {total_files}")
    print("Note: Each file is ~2.5MB, total ~{:.1f}MB".format(total_files * 2.5))
    print()

    # Confirm download
    response = input("Continue with download? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return

    print("\nStarting download...\n")

    success_count = 0
    failed_downloads = []

    # Use tqdm for progress bar
    for subject_id in tqdm(subjects, desc="Overall progress"):
        subject_dir = data_path / f"subject-{subject_id:02d}"
        subject_dir.mkdir(exist_ok=True)

        for run in runs:
            try:
                # Download using MNE
                mne.datasets.eegbci.load_data(
                    subjects=[subject_id],
                    runs=[run],
                    path=str(data_path),
                    update_path=False,
                    verbose=False
                )
                success_count += 1

            except Exception as e:
                failed_downloads.append((subject_id, run))
                if verbose:
                    print(f"\nFailed: Subject {subject_id}, Run {run}: {e}")

    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Successfully downloaded: {success_count}/{total_files} files")

    if failed_downloads:
        print(f"Failed downloads: {len(failed_downloads)}")
        print("Failed files:")
        for subj, run in failed_downloads[:10]:  # Show first 10
            print(f"  - Subject {subj}, Run {run}")
        if len(failed_downloads) > 10:
            print(f"  ... and {len(failed_downloads) - 10} more")
    else:
        print("✓ All files downloaded successfully!")

    print("="*70)

    # Show data statistics
    print("\nData Statistics:")
    print(f"  Subjects: {len(subjects)}")
    print(f"  Runs per subject: {len(runs)}")
    print(f"  Total trials per subject: ~45 × {len(runs)//3} = ~{45 * len(runs)//3}")
    print(f"  Total trials (all subjects): ~{len(subjects) * 45 * len(runs)//3}")
    print("\nYou can now train on this expanded dataset!")


def main():
    parser = argparse.ArgumentParser(
        description="Download PhysioNet EEGBCI motor imagery data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all 109 subjects
  python download_all_subjects.py --subjects all

  # Download first 50 subjects
  python download_all_subjects.py --subjects 1-50

  # Download specific subjects
  python download_all_subjects.py --subjects 1 2 7 10

  # Download with additional runs (hands vs feet)
  python download_all_subjects.py --subjects 1-20 --runs 4 8 12 6 10 14
        """
    )

    parser.add_argument("--subjects", nargs="+", required=True,
                       help="Subject IDs: 'all', '1-50', or '1 2 3 4'")
    parser.add_argument("--runs", nargs="+", type=int, default=[4, 8, 12],
                       help="Run numbers (default: 4 8 12 for left/right MI)")
    parser.add_argument("--datadir", type=str, default="data/raw",
                       help="Output directory")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress detailed error messages")

    args = parser.parse_args()

    # Parse subjects
    subjects = parse_subject_range(args.subjects)

    if not subjects:
        print("Error: No subjects specified!")
        return

    if max(subjects) > 109:
        print("Warning: PhysioNet EEGBCI has only 109 subjects!")
        subjects = [s for s in subjects if s <= 109]

    # Download
    download_subjects(
        subjects=subjects,
        runs=args.runs,
        data_dir=args.datadir,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()