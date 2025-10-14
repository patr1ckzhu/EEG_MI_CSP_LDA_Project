"""
Download EEGBCI (PhysioNet) EEG motor imagery data using MNE.

Runs mapping (per MNE docs):
- 4, 8, 12: Motor imagery - left vs right hand
- 6, 10, 14: Motor imagery - hands vs feet
"""

import argparse
from pathlib import Path
import shutil

from mne.datasets import eegbci

from utils import ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", type=int, default=[1],
                    help="Subject IDs to download (e.g., 1 2 3)")
    ap.add_argument("--runs", nargs="+", type=int, default=[4, 8, 12],
                    help="Run IDs (e.g., 4 8 12 for MI left/right)")
    ap.add_argument("--out", type=str, default="data/raw", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(ensure_dir(args.out))

    for sub in args.subjects:
        print(f"[INFO] Downloading subject {sub} runs {args.runs} ...")
        paths = eegbci.load_data(sub, args.runs)  # downloads to MNE data dir
        # Copy to project data/raw/subject-XX/
        subj_dir = out_dir / f"subject-{sub:02d}"
        subj_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            src_p = Path(src)
            dst = subj_dir / src_p.name
            if not dst.exists():
                shutil.copy2(src_p, dst)
                print(f"  - copied {src_p.name}")
            else:
                print(f"  - exists {src_p.name}")

    print(f"[DONE] All files organized under: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
