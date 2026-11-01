"""Split raw/<breed>/*.jpg into data/train and data/val.

Usage: python scripts/split_dataset.py --raw raw --out data --val-ratio 0.2
"""
import argparse
import random
import shutil
from pathlib import Path

EXT = {".jpg", ".jpeg", ".png"}

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="raw")
ap.add_argument("--out", default="data")
ap.add_argument("--val-ratio", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

random.seed(args.seed)
for breed_dir in sorted(p for p in Path(args.raw).iterdir() if p.is_dir()):
    files = [f for f in breed_dir.iterdir() if f.suffix.lower() in EXT]
    random.shuffle(files)
    n_val = max(1, int(len(files) * args.val_ratio))
    for i, f in enumerate(files):
        split = "val" if i < n_val else "train"
        dest = Path(args.out) / split / breed_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest / f.name)
    print(f"{breed_dir.name}: {len(files) - n_val} train / {n_val} val")
