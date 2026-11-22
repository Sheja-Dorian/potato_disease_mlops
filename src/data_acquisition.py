"""Step 1 - Data acquisition.

Downloads only the three potato classes of the PlantVillage dataset
(https://github.com/spMohanty/PlantVillage-Dataset, CC BY-SA 3.0) with a sparse
git checkout, then makes a reproducible stratified split:

    data/train/<class>     70 %  -> used to build the model
    data/test/<class>      15 %  -> held-out evaluation set (never trained on)
    data/incoming/<class>  15 %  -> simulates NEW field data a user uploads to trigger retraining

Usage:
    python -m src.data_acquisition            # clone + split
    python -m src.data_acquisition --src DIR  # split an already-downloaded copy
"""
import argparse
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.config import DATA_DIR, SEED

REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
SOURCE_FOLDERS = {
    "early_blight": "raw/color/Potato___Early_blight",
    "late_blight": "raw/color/Potato___Late_blight",
    "healthy": "raw/color/Potato___healthy",
}
SPLITS = {"train": 0.70, "test": 0.15, "incoming": 0.15}


def sparse_clone(dest: Path) -> Path:
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", *SOURCE_FOLDERS.values()], check=True)
    return dest


def split_dataset(src_root: Path, out_root: Path = DATA_DIR) -> dict:
    rng = random.Random(SEED)
    counts = {}
    for label, rel in SOURCE_FOLDERS.items():
        files = sorted(p for p in (src_root / rel).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        rng.shuffle(files)
        n = len(files)
        n_train, n_test = int(n * SPLITS["train"]), int(n * SPLITS["test"])
        parts = {
            "train": files[:n_train],
            "test": files[n_train:n_train + n_test],
            "incoming": files[n_train + n_test:],
        }
        for split, items in parts.items():
            d = out_root / split / label
            d.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, d / f.name.replace(" ", "_"))
            counts[(split, label)] = len(items)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, help="existing PlantVillage checkout")
    args = ap.parse_args()
    if args.src:
        counts = split_dataset(args.src)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            counts = split_dataset(sparse_clone(Path(tmp) / "pv"))
    for (split, label), n in sorted(counts.items()):
        print(f"{split:9s} {label:13s} {n:5d}")


if __name__ == "__main__":
    main()
